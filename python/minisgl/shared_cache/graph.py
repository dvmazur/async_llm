"""Shared-session preparation hooks for the original engine.GraphRunner."""

import torch
import torch.nn.functional as F

from minisgl.core import get_global_ctx
from .attention_graph import SharedDecodeAttentionBuffers
from .gdn_decode import GDNDecodeBuffers
from .shared_block import CacheBlock
from .worker_group import WorkerGroup


class SharedGraphIO:
    def __init__(self, session, depth):
        self.session, self.depth = session, depth
        self.attention, self.gdn = {}, {}
        self.dummy_slots = None

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

    def prepare_for_capture(self, batch):
        s, bs = self.session, batch.size
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
            self.session.sc_gdn.finish_decode(False)
            # Warmup output slabs are not persistent state. Retire them now,
            # including profiles that might never be replayed by a real batch.
            self.session.engine.stream.synchronize()
            self.gdn[batch.size].retire_completed()
            self.gdn[batch.size].write_ptrs.zero_()
        get_global_ctx().gdn_ar = None

    def prepare_for_replay(self, batch):
        # Session has already planned its real group before entering forward.
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
        for buffers in self.gdn.values():
            buffers.retire_completed()
        ar = self.session.sc_gdn
        if ar is not None and ar.decode_buffers in self.gdn.values():
            ar.decode_buffers = None
        for buffers in self.attention.values():
            buffers.meta.graph_buffers = None
        self.attention.clear()
        self.gdn.clear()
