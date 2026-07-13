"""
High-level session that drives shared-cache inference on a mini-sglang Engine.

Usage::

    engine = Engine(config)
    session = SharedCacheSession(engine)

    prompt_block = session.create_block()
    w1_block = session.create_block()
    w2_block = session.create_block()

    # Prefill shared prompt (returns logits [P, vocab])
    prompt_logits = session.prefill_block(prompt_block, prompt_token_ids)

    group = WorkerGroup(
        cache_structure=[
            [prompt_block, w2_block, w1_block],
            [prompt_block, w1_block, w2_block],
        ],
        write_to=[w1_block, w2_block],
    )

    # Decode loop
    next_ids = first_tokens  # [num_workers, 1]
    for _ in range(max_steps):
        logits = session.decode_step(group, next_ids)
        next_ids = logits.argmax(dim=-1, keepdim=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.utils import div_ceil

from .attention import SharedCacheAttention, _PrefillSpec
from .shared_block import NULL_CACHE_HANDLE, SharedBlock
from .worker_group import WorkerGroup


@dataclass
class PrefillRequest:
    """One in-context prefill to run inside a :meth:`SharedCacheSession.mixed_step`.

    ``block`` must be a fresh (empty) :class:`SharedBlock`; ``input_ids`` is a
    1-D int tensor; ``context`` are the (stable, non-empty) blocks the new
    tokens attend to, ordered as if concatenated ``[ctx_0, ..., block]``.
    """

    block: SharedBlock
    input_ids: torch.Tensor
    context: List[SharedBlock] = field(default_factory=list)

if TYPE_CHECKING:
    from minisgl.engine import Engine

_DEFAULT_SAMPLING = SamplingParams(temperature=0.0, max_tokens=1)


def extract_cos_sin_cache(engine: Engine) -> torch.Tensor:
    """
    Extract the ``cos_sin_cache`` tensor from the model's ``RotaryEmbedding``.

    Works for Llama / Qwen / Mistral model families in mini-sglang.
    """
    layers = engine.model.model.layers.op_list
    return layers[0].self_attn.attn.rotary._cos_sin_cache


class SharedCacheSession:
    """
    Manages ``SharedBlock`` pages and drives batched forward passes for a
    ``WorkerGroup`` on a mini-sglang ``Engine``.

    Pages are **borrowed from the engine's main page cache**
    (``engine.page_allocator``) rather than from a private pool, so allocations
    are page-aligned and drawn from the same physical pages the engine owns.
    The session is designed for **standalone** use, bypassing the scheduler; to
    run alongside the live scheduler the two consumers would need to share one
    allocator instance.
    """

    def __init__(
        self,
        engine: Engine,
        cos_sin_cache: Optional[torch.Tensor] = None,
    ):
        self.engine = engine
        self.device = engine.device
        self.page_table = engine.page_table
        self.kv_cache = engine.kv_cache
        self.page_size: int = engine.ctx.page_size
        self.attn_backend = engine.attn_backend
        self.page_allocator = engine.page_allocator

        max_table = engine.page_table.shape[0] - 1  # last row is dummy
        self._free_table_indices: List[int] = list(range(max_table))

        self._token_pool = torch.zeros(
            max_table + 1,
            engine.page_table.shape[1],
            dtype=torch.int32,
            device=self.device,
        )

        if cos_sin_cache is not None:
            self._cos_sin_cache = cos_sin_cache.to(self.device)
        else:
            self._cos_sin_cache = extract_cos_sin_cache(engine).to(self.device)

        # Query-rotation attention op for decode (arXiv:2512.10931).
        attn0 = engine.model.model.layers.op_list[0].self_attn.attn
        self.sc_attn = SharedCacheAttention(
            kv_cache=self.kv_cache,
            cos_sin_cache=self._cos_sin_cache,
            num_qo_heads=attn0.num_qo_heads,
            num_kv_heads=attn0.num_kv_heads,
            head_dim=attn0.head_dim,
            page_size=self.page_size,
            dtype=self.kv_cache.dtype,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_block(self) -> SharedBlock:
        return SharedBlock(self.device, page_size=self.page_size)

    def free_block(self, block: SharedBlock) -> None:
        """Return a block's pages to the engine's page allocator and reset it."""
        page_starts = block.clear()
        if page_starts:
            self.page_allocator.free_pages(
                torch.tensor(page_starts, dtype=torch.int32, device=self.device)
            )

    @torch.inference_mode()
    def prefill_block(
        self,
        block: SharedBlock,
        input_ids: torch.Tensor,
        context: Optional[List[SharedBlock]] = None,
    ) -> torch.Tensor:
        """
        Prefill a single ``SharedBlock`` with *input_ids* and return logits.

        *input_ids* must be a 1-D CPU ``int32`` tensor.

        When *context* blocks are given, the new tokens attend to them as if
        the blocks were concatenated ``[ctx_0, ..., block]`` (mirrors the
        reference's ``prefill_cache_block(text, [ctx..., new])``); the stored
        KV stays block-relative either way.  Empty context blocks are skipped.
        """
        input_ids = input_ids.to(dtype=torch.int32).flatten().cpu()
        seq_len = len(input_ids)
        assert seq_len > 0

        context = [b for b in (context or []) if b.num_tokens > 0]
        if context:
            return self._prefill_block_in_context(block, input_ids, context)

        page_starts, token_slots = self._alloc_token_storage(seq_len)
        table_idx = self._allocate_table_idx()

        try:
            self.page_table[table_idx, : token_slots.numel()] = token_slots
            self._token_pool[table_idx, :seq_len] = input_ids.to(self.device)

            req = Req(
                input_ids=input_ids,
                table_idx=table_idx,
                cached_len=0,
                output_len=1,
                uid=-1,
                sampling_params=_DEFAULT_SAMPLING,
                cache_handle=NULL_CACHE_HANDLE,
            )
            batch = self._build_batch([req], phase="prefill")
            logits = self._forward(batch)

            block.grow_pages(page_starts, seq_len)

            # NOTE: ParallelLMHead.forward already extracts last-token logits
            # for prefill batches, so logits has shape [bs, vocab].
            return logits[:1]
        finally:
            self._free_table_idx(table_idx)

    def _prefill_block_in_context(
        self,
        block: SharedBlock,
        input_ids: torch.Tensor,
        context: List[SharedBlock],
    ) -> torch.Tensor:
        """Prefill *block* while attending to *context* blocks (all non-empty)."""
        seq_len = len(input_ids)
        page_starts, token_slots = self._alloc_token_storage(seq_len)
        out_loc = token_slots[:seq_len]

        req = Req(
            input_ids=input_ids,
            table_idx=self.page_table.shape[0] - 1,  # dummy row; page table bypassed
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=_DEFAULT_SAMPLING,
            cache_handle=NULL_CACHE_HANDLE,
        )
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = [req]
        # block-relative RoPE positions for the stored keys
        batch.positions = torch.arange(seq_len, dtype=torch.int64, device=self.device)
        batch.input_ids = input_ids.to(self.device)
        batch.out_loc = out_loc
        batch.attn_metadata = self.sc_attn.prepare_context_prefill(context, page_starts, seq_len)

        logits = self._forward(batch)

        block.grow_pages(page_starts, seq_len)
        return logits[:1]

    @torch.inference_mode()
    def decode_step(
        self,
        group: WorkerGroup,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run one decode step for every worker in *group*.

        Cached keys are stored at block-relative RoPE positions and are never
        re-rotated; instead, per-(worker, segment) query copies are rotated and
        partial attention outputs merged (see ``shared_cache.attention``).
        Matching the AsyncReasoning reference, a worker reading another
        worker's write block also sees that worker's current-step token.

        Args:
            group: the ``WorkerGroup`` defining the cache structure.
            input_ids: ``[num_workers]`` or ``[num_workers, 1]`` int tensor
                of per-worker input tokens (typically the last generated token).

        Returns:
            Logits ``[num_workers, vocab_size]``.
        """
        num_workers = group.num_workers
        input_ids = input_ids.to(dtype=torch.int32).reshape(num_workers).cpu()

        # Decide, per (distinct) write block, whether the new token starts a
        # fresh page, then borrow all needed pages from the engine in one shot.
        new_page_for_block, new_token_slots, write_pos = self._plan_decode_writes(group)

        # Reqs are bookkeeping only here (batch size / phase); the page table
        # is bypassed entirely, so they point at the engine's dummy row.
        dummy_table_idx = self.page_table.shape[0] - 1
        reqs: List[Req] = []
        for wi in range(num_workers):
            cached_len = group.worker_cache_length(wi)
            full_ids = torch.zeros(cached_len + 1, dtype=torch.int32)
            full_ids[cached_len] = input_ids[wi]
            reqs.append(
                Req(
                    input_ids=full_ids,
                    table_idx=dummy_table_idx,
                    cached_len=cached_len,
                    output_len=1,
                    uid=-(wi + 1),
                    sampling_params=_DEFAULT_SAMPLING,
                    cache_handle=NULL_CACHE_HANDLE,
                )
            )

        batch = Batch(reqs=reqs, phase="decode")
        batch.padded_reqs = reqs
        # block-relative RoPE positions for the new tokens' keys
        batch.positions = torch.tensor(write_pos, dtype=torch.int64, device=self.device)
        batch.input_ids = input_ids.to(self.device)
        batch.out_loc = new_token_slots
        batch.attn_metadata = self.sc_attn.prepare(group, new_page_for_block, new_token_slots)

        logits = self._forward(batch)

        # Commit growth now that the forward (which read post-append lengths) is done.
        for wt in group.write_to:
            wt.append_token(new_page_for_block[id(wt)])
        return logits[:num_workers]

    @torch.inference_mode()
    def mixed_step(
        self,
        decode_group: Optional[WorkerGroup],
        decode_input_ids: Optional[torch.Tensor],
        prefill_requests: Sequence[PrefillRequest],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one forward that decodes every worker in *decode_group* **and**
        prefills each of *prefill_requests* together, in a single batch.

        Output rows are laid out decode-first, so the returned logits split as
        ``(decode_logits [W, vocab], prefill_logits [n_prefill, vocab])`` where
        each prefill row is that request's last-token distribution (its first
        generated token), exactly like :meth:`prefill_block`.

        Pass ``decode_group=None`` (and ``decode_input_ids=None``) for a
        prefill-only batch of several requests.

        v1 constraints (asserted): every prefill block is fresh/empty and not
        referenced by any decode worker's view; every prefill context block is
        non-empty and not written this step.  These make the mixed result
        identical to running the decode and prefill phases as separate forwards.
        """
        prefill_requests = list(prefill_requests)
        W = decode_group.num_workers if decode_group is not None else 0
        assert W > 0 or prefill_requests, "mixed_step needs decode workers or prefills"
        assert (decode_group is None) == (decode_input_ids is None)

        # ---- validate v1 block-independence constraints ----
        decode_view_ids = (
            {id(b) for view in decode_group.cache_structure for b in view}
            if decode_group is not None
            else set()
        )
        decode_write_ids = (
            {id(wt) for wt in decode_group.write_to} if decode_group is not None else set()
        )
        prefill_self_ids = {id(r.block) for r in prefill_requests}
        assert len(prefill_self_ids) == len(prefill_requests), "duplicate prefill block"

        # ---- allocate + describe each prefill (decode-first row ordering) ----
        specs: List[_PrefillSpec] = []
        pf_ids: List[torch.Tensor] = []
        pf_out_locs: List[torch.Tensor] = []
        pf_positions: List[torch.Tensor] = []
        pf_page_starts: List[torch.Tensor] = []
        for r in prefill_requests:
            assert r.block.num_tokens == 0, "prefill block must be fresh (empty)"
            assert (
                id(r.block) not in decode_view_ids
            ), "prefill self block is referenced by a decode worker this step"
            ctx = [b for b in r.context if b.num_tokens > 0]
            assert ctx, "mixed prefill requires at least one non-empty context block"
            for b in ctx:
                assert (
                    id(b) not in decode_write_ids
                ), "prefill context block is written by a decode worker this step"
                assert id(b) not in prefill_self_ids, "prefill context block is a fresh block"
            ids = r.input_ids.to(dtype=torch.int32).flatten().cpu()
            S = int(ids.numel())
            assert S > 0
            page_starts, token_slots = self._alloc_token_storage(S)
            specs.append(_PrefillSpec(ctx, page_starts, S))
            pf_ids.append(ids)
            pf_out_locs.append(token_slots[:S])
            pf_positions.append(torch.arange(S, dtype=torch.int64, device=self.device))
            pf_page_starts.append(page_starts)

        # ---- plan decode writes ----
        if W > 0:
            dec_ids = decode_input_ids.to(dtype=torch.int32).reshape(W).cpu()
            new_page_for_block, new_token_slots, write_pos = self._plan_decode_writes(decode_group)
        else:
            new_page_for_block, new_token_slots, write_pos = {}, None, []

        # ---- assemble the combined batch: decode tokens first, then prefills ----
        dummy_table_idx = self.page_table.shape[0] - 1
        reqs: List[Req] = []
        input_id_parts: List[torch.Tensor] = []
        position_parts: List[torch.Tensor] = []
        out_loc_parts: List[torch.Tensor] = []

        if W > 0:
            input_id_parts.append(dec_ids.to(self.device))
            position_parts.append(torch.tensor(write_pos, dtype=torch.int64, device=self.device))
            out_loc_parts.append(new_token_slots)
            for wi in range(W):
                cached_len = decode_group.worker_cache_length(wi)
                full_ids = torch.zeros(cached_len + 1, dtype=torch.int32)
                full_ids[cached_len] = dec_ids[wi]
                reqs.append(self._bookkeeping_req(full_ids, dummy_table_idx, cached_len))

        for ids, out_loc, positions in zip(pf_ids, pf_out_locs, pf_positions):
            input_id_parts.append(ids.to(self.device))
            position_parts.append(positions)
            out_loc_parts.append(out_loc)
            reqs.append(self._bookkeeping_req(ids, dummy_table_idx, 0))

        batch = Batch(reqs=reqs, phase="prefill")
        batch.padded_reqs = reqs
        batch.input_ids = torch.cat(input_id_parts)
        batch.positions = torch.cat(position_parts)
        batch.out_loc = torch.cat(out_loc_parts)
        batch.attn_metadata = self.sc_attn.prepare_mixed(
            decode_group, new_page_for_block, new_token_slots, specs
        )

        logits = self._forward(batch)

        # ---- commit growth now that the forward read post-append lengths ----
        if W > 0:
            for wt in decode_group.write_to:
                wt.append_token(new_page_for_block[id(wt)])
        for r, page_starts, spec in zip(prefill_requests, pf_page_starts, specs):
            r.block.grow_pages(page_starts, spec.num_new)

        return logits[:W], logits[W:]

    def _bookkeeping_req(
        self, input_ids: torch.Tensor, table_idx: int, cached_len: int
    ) -> Req:
        """A minimal ``Req`` used only for batch size / phase; the page table is
        bypassed (metadata drives attention), so it points at the dummy row."""
        return Req(
            input_ids=input_ids,
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=1,
            uid=-1,
            sampling_params=_DEFAULT_SAMPLING,
            cache_handle=NULL_CACHE_HANDLE,
        )

    # ------------------------------------------------------------------
    # Page / table-index management
    # ------------------------------------------------------------------

    def _alloc_token_storage(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Borrow ``ceil(seq_len/P)`` pages; return ``(page_starts, token_slots)``
        where ``token_slots`` are the ``num_pages * P`` per-token slots (the
        first ``seq_len`` are the real write locations)."""
        n_pages = div_ceil(seq_len, self.page_size)
        page_starts = self.page_allocator.alloc_pages(n_pages)
        return page_starts, self.page_allocator.pages_to_tokens(page_starts)

    def _plan_decode_writes(
        self, group: WorkerGroup
    ) -> Tuple[Dict[int, Optional[int]], torch.Tensor, List[int]]:
        """For one decode step, choose the destination slot of each worker's new
        token and which write blocks need a freshly-allocated page.

        Returns ``(new_page_for_block, new_token_slots, write_pos)`` where
        ``new_page_for_block`` maps ``id(block)`` -> page-start slot (or None),
        ``new_token_slots`` is ``[num_workers]`` int32 destination slots, and
        ``write_pos`` is the per-worker block-relative RoPE position.
        """
        num_workers = group.num_workers
        new_page_for_block: Dict[int, Optional[int]] = {}
        blocks_needing_page: List[SharedBlock] = []
        for wt in group.write_to:
            if id(wt) in new_page_for_block:
                raise ValueError(
                    "WorkerGroup has two workers writing the same block in one step"
                )
            if wt.has_capacity:
                new_page_for_block[id(wt)] = None
            else:
                new_page_for_block[id(wt)] = None  # filled in below once allocated
                blocks_needing_page.append(wt)

        if blocks_needing_page:
            fresh = self.page_allocator.alloc_pages(len(blocks_needing_page))
            for k, wt in enumerate(blocks_needing_page):
                new_page_for_block[id(wt)] = int(fresh[k].item())

        out_loc: List[int] = []
        write_pos: List[int] = []
        for wt in group.write_to:
            t = wt.num_tokens
            new_page = new_page_for_block[id(wt)]
            page_start = new_page if new_page is not None else wt.page_starts[-1]
            out_loc.append(page_start + (t % self.page_size))
            write_pos.append(t)

        new_token_slots = torch.tensor(out_loc, dtype=torch.int32, device=self.device)
        return new_page_for_block, new_token_slots, write_pos

    def _allocate_table_idx(self) -> int:
        return self._free_table_indices.pop()

    def _free_table_idx(self, idx: int) -> None:
        self._free_table_indices.append(idx)

    # ------------------------------------------------------------------
    # Batch building & forward (prefill path)
    # ------------------------------------------------------------------

    def _build_batch(self, reqs: List[Req], phase: str) -> Batch:
        batch = Batch(reqs=reqs, phase=phase)  # type: ignore[arg-type]
        batch.padded_reqs = reqs

        batch.positions = self._make_positions(batch)

        table_idx_list: List[torch.Tensor] = []
        for req in reqs:
            length = req.extend_len
            table_idx_list.append(
                torch.full((length,), req.table_idx, dtype=torch.int64)
            )
        table_idxs = torch.cat(table_idx_list).to(self.device)
        position_idxs = batch.positions.to(torch.int64)

        batch.out_loc = self.page_table[table_idxs, position_idxs]
        batch.input_ids = self._token_pool[table_idxs, position_idxs]

        self.attn_backend.prepare_metadata(batch)
        return batch

    def _make_positions(self, batch: Batch) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for req in batch.padded_reqs:
            parts.append(
                torch.arange(req.cached_len, req.device_len, dtype=torch.int32)
            )
        return torch.cat(parts).to(self.device)

    def _forward(self, batch: Batch) -> torch.Tensor:
        with self.engine.ctx.forward_batch(batch):
            return self.engine.model.forward()
