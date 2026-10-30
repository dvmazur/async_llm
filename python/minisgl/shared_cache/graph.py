"""Shared-session preparation hooks for the original engine.GraphRunner."""

import torch
import torch.nn.functional as F

from minisgl.core import Batch, get_global_ctx
from .attention import PrefillSpec
from .attention_graph import SharedDecodeAttentionBuffers, SharedPrefillAttentionBuffers
from .gdn_decode import GDNDecodeBuffers
from .gdn_prefill import GDNPrefillBuffers
from .shared_block import CacheBlock
from .worker_group import WorkerGroup


class _PrefillInputs:
    """Stable model inputs; logits scale with requests, not padded token rows."""

    def __init__(self, session, rows, workers):
        device, config = session.device, session.engine.config.model_config
        self.input_ids = torch.zeros(rows, device=device, dtype=torch.int32)
        self.positions = torch.zeros(rows, device=device, dtype=torch.int64)
        self.out_loc = torch.full((rows,), -1, device=device, dtype=torch.int32)
        self.mrope = (torch.zeros(3, rows, device=device, dtype=torch.int64)
                      if session.sc_attn.mrope_section is not None else None)
        self.types = torch.zeros(rows, device=device, dtype=torch.int64) if config.is_multimodal else None
        self.embeds = (torch.zeros(rows, config.hidden_size, device=device, dtype=session.engine.dtype)
                       if config.is_multimodal else None)
        self.logits = torch.empty(workers, config.vocab_size, device=device, dtype=torch.float32)

    def bind(self, batch, rows):
        batch.input_ids, batch.positions, batch.out_loc = (
            t[:rows] for t in (self.input_ids, self.positions, self.out_loc))
        if self.mrope is not None:
            batch.mrope_positions = self.mrope[:, :rows]
        if self.embeds is not None:
            batch.mm_token_type_ids = self.types[:rows]
            batch.image_embeds = self.embeds[:rows]
            batch.image_embeds_token_aligned = True

    def copy_from(self, session, batch, rows):
        n = batch.input_ids.numel()
        self.input_ids[:rows].zero_()
        self.positions[:rows].zero_()
        self.out_loc[:rows].fill_(-1)
        self.input_ids[:n].copy_(batch.input_ids)
        self.positions[:n].copy_(batch.positions)
        self.out_loc[:n].copy_(batch.out_loc)
        if self.mrope is not None:
            self.mrope[:, :rows].zero_()
            positions = batch.mrope_positions
            self.mrope[:, :n].copy_(batch.positions.expand(3, -1) if positions is None else positions)
        if self.embeds is not None:
            self.types[:rows].zero_()
            self.embeds[:rows].zero_()
            if batch.mm_token_type_ids is not None:
                self.types[:n].copy_(batch.mm_token_type_ids)
            embeds = batch.image_embeds
            if embeds is None and batch.pixel_values is not None:
                # Same tower and image grouping as before, but outside capture.
                embeds = session.engine.model.model.visual.forward(batch.pixel_values, batch.image_grid_thw)
            if embeds is not None:
                if batch.image_embeds_token_aligned:
                    self.embeds[:n].copy_(embeds)
                else:
                    self.embeds[:n][self.types[:n] == 1] = embeds.to(self.embeds.dtype)


class SharedGraphIO:
    def __init__(self, session, depth, prefill_rows=()):
        self.session, self.depth = session, depth
        self.attention, self.gdn = {}, {}
        self.dummy_slots = None
        self.prefill_rows = sorted(set(prefill_rows))
        self.prefill_attention, self.prefill_gdn, self.prefill_batches = {}, {}, {}
        self.prefill_inputs = None

    def retire_completed(self):
        # A profile may not be selected again for many steps. Its completed
        # pointer-input/output references must not pin an old whole topology.
        for buffers in (*self.gdn.values(), *self.prefill_gdn.values()):
            buffers.retire_completed()

    def init_capture_graph(self, max_seq_len, bs_list):
        s = self.session
        self.dummy_slots = s.page_allocator.alloc_pages(max(bs_list))
        # A distinct reserved page per padded worker avoids concurrent KV writes
        # to one dummy slot and never overwrites a live block's pages.
        for bs in bs_list:
            max_pages = bs * ((max_seq_len + s.page_size - 1) // s.page_size + self.depth)
            self.attention[bs] = SharedDecodeAttentionBuffers(
                s.sc_attn, bs, self.depth, max_pages, self.dummy_slots[:bs])
            if s.sc_gdn is not None:
                self.gdn[bs] = GDNDecodeBuffers(
                    s.sc_gdn, s._model_config.num_linear_layers, bs, self.depth, s.engine.dtype)
        if self.prefill_rows:
            self.prefill_inputs = _PrefillInputs(s, max(self.prefill_rows), max(bs_list))
            dummy_pages = self.dummy_slots.long() // s.page_size
            for layer in range(s.kv_cache.num_layers):
                s.kv_cache.k_cache(layer).index_fill_(0, dummy_pages, 0)
                s.kv_cache.v_cache(layer).index_fill_(0, dummy_pages, 0)
            for rows in self.prefill_rows:
                workers = min(rows, max(bs_list))
                max_pages = workers * ((max_seq_len+s.page_size-1)//s.page_size + self.depth)
                self.prefill_attention[rows] = SharedPrefillAttentionBuffers(
                    s.sc_attn, workers, self.depth, rows, max_pages, self.dummy_slots[:workers])
                if s.sc_gdn is not None:
                    self.prefill_gdn[rows] = GDNPrefillBuffers(
                        s.sc_gdn, s._model_config.num_linear_layers, workers, self.depth, s.engine.dtype, rows)
        # Reserve for the entire catalogue before the first capture, including
        # decode-only configurations and every profile's padded segment slots.
        s.sc_attn.prepare_workspace((*self.attention.values(), *self.prefill_attention.values()))

    def select_prefill(self, jobs):
        workers, rows = len(jobs), sum(job.input_ids.numel() for job in jobs)
        contexts = [[b for b in (job.context or []) if b.num_tokens > 0] for job in jobs]
        depth = max((len(c)+1 for c in contexts), default=1)
        if depth > self.depth:
            raise ValueError('Shared prefill exceeds graph depth capacity')
        for capacity in self.prefill_rows:
            buffers = self.prefill_attention[capacity]
            if rows <= capacity and workers <= buffers.workers:
                ctx_pages = buffers.workers*self.depth + sum(b.num_pages-1 for c in contexts for b in c)
                own_pages = buffers.workers-workers + sum(
                    (job.block.num_tokens+job.input_ids.numel()+self.session.page_size-1)//self.session.page_size
                    for job in jobs)
                if max(ctx_pages, own_pages) <= buffers.max_pages:
                    return buffers
        raise ValueError('Shared prefill exceeds graph request/token-row/page capacity')

    def prefill_capture_batch(self, rows):
        buffers = self.prefill_attention[rows]
        batch = Batch(reqs=[self.session.engine.dummy_req] * buffers.workers, phase='prefill')
        batch.padded_reqs = batch.reqs
        self.prefill_inputs.bind(batch, rows)
        self.prefill_batches[rows] = batch
        return batch, self.prefill_inputs.logits[:buffers.workers]

    def prepare_for_capture(self, batch):
        s, bs = self.session, batch.size
        if batch.is_prefill:
            rows = batch.input_ids.numel()
            batch.input_ids.zero_()
            batch.positions.zero_()
            batch.out_loc.fill_(-1)
            lengths = [rows // bs + (i < rows % bs) for i in range(bs)]
            blocks = [CacheBlock(s.device, s.page_size) for _ in range(bs)]
            contexts = [CacheBlock(s.device, s.page_size) for _ in range(bs)]
            for block, slot in zip(contexts, self.dummy_slots[:bs].cpu().tolist()):
                block.page_starts, block.num_tokens = [slot], 1
            # Prime both public wrappers at MAXIMUM rows; later plans contain
            # only actual queries, not fake padding requests. No KV writes here.
            specs = [PrefillSpec([contexts[i]] * self.depth,
                                self.dummy_slots[i:i+1].repeat((n+s.page_size-1)//s.page_size), n)
                     for i, n in enumerate(lengths)]
            batch.attn_metadata = s.sc_attn.prepare_prefill_batch(specs, self.prefill_attention[rows])
            batch.attn_metadata.sub_loc.zero_()  # safe dummy RoPE even at the position limit
            if s.sc_gdn is not None:
                s.sc_gdn.set_context([[b] for b in blocks], blocks, prefill_segments=lengths)
                s.sc_gdn.prepare_prefill(s._model_config.num_linear_layers, s.engine.dtype, rows,
                                        buffers=self.prefill_gdn[rows])
                get_global_ctx().gdn_ar = s.sc_gdn
            return
        batch.input_ids.zero_()
        batch.positions.zero_()
        batch.out_loc.copy_(self.dummy_slots[:bs])
        blocks = [CacheBlock(s.device, s.page_size) for _ in range(bs)]
        for block, slot in zip(blocks, self.dummy_slots[:bs].cpu().tolist()):
            block.page_starts = [slot]
        group = WorkerGroup(cache_structure=[[b] for b in blocks], write_to=blocks)
        batch.attn_metadata = s.sc_attn.prepare(
            group, {id(b): None for b in blocks}, self.dummy_slots[:bs], self.attention[bs])
        if s.sc_gdn is not None:
            s.sc_gdn.set_context(group.cache_structure, group.write_to)
            s.sc_gdn.prepare_decode(s._model_config.num_linear_layers, s.engine.dtype,
                                    buffers=self.gdn[bs])
            get_global_ctx().gdn_ar = s.sc_gdn

    def finish_capture(self, batch):
        if self.session.sc_gdn is not None:
            if batch.is_prefill:
                self.session.sc_gdn.finish_prefill(False)
                buffers = self.prefill_gdn[batch.input_ids.numel()]
            else:
                self.session.sc_gdn.finish_decode(False)
                buffers = self.gdn[batch.size]
            # Warmup output slabs are not persistent state. Retire them now,
            # including profiles that might never be replayed by a real batch.
            self.session.engine.stream.synchronize()
            buffers.retire_completed()
            buffers.write_ptrs.zero_()
        get_global_ctx().gdn_ar = None

    def prepare_for_replay(self, batch):
        # Session has already planned its real group before entering forward.
        if batch.is_prefill:
            rows = batch.attn_metadata.graph_buffers.rows
            assert batch.attn_metadata is self.prefill_attention[rows].meta
            self.prefill_inputs.copy_from(self.session, batch, rows)
            return
        assert batch.attn_metadata is self.attention[batch.padded_size].meta

    def pad_inputs(self, batch):
        n, padded = batch.size, batch.padded_size
        batch.input_ids = F.pad(batch.input_ids, (0, padded - n))
        batch.positions = F.pad(batch.positions, (0, padded - n))
        batch.out_loc = torch.cat([batch.out_loc, self.dummy_slots[n:padded]])

    def destroy_capture_graph(self):
        if self.dummy_slots is not None:
            self.session.engine.stream.synchronize()
            self.session.page_allocator.free_pages(self.dummy_slots)
            self.dummy_slots = None
        for buffers in [*self.gdn.values(), *self.prefill_gdn.values()]:
            buffers.retire_completed()
        ar = self.session.sc_gdn
        if ar is not None and ar.decode_buffers in self.gdn.values():
            ar.decode_buffers = None
        if ar is not None and ar.prefill_buffers in self.prefill_gdn.values():
            ar.prefill_buffers = None
        for buffers in [*self.attention.values(), *self.prefill_attention.values()]:
            buffers.meta.graph_buffers = None
        self.attention.clear()
        self.gdn.clear()
        self.prefill_gdn.clear()
        self.prefill_attention.clear()
        self.prefill_batches.clear()
        self.prefill_inputs = None
