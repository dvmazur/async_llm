"""
Query-rotation attention for shared-cache decode.

Implements the decoding scheme of "Asynchronous Reasoning: Training-Free
Interactive Thinking LLMs" (arXiv:2512.10931): cached keys are stored at
BLOCK-RELATIVE RoPE positions (0..len-1) and are never re-rotated.  At decode
time, each (worker, segment) pair gets its own copy of the query, rotated to
the query's position *relative to that segment's start* in the worker's view.
Per-segment attention outputs are then combined with log-sum-exp merging,
which is mathematically identical to one softmax over the concatenated view.

Because RoPE attention scores depend only on relative position,
``score(q @ pos_a, k @ pos_b) = f(a - b)``; rotating the query by
``loc = query_pos - segment_start`` against block-relative keys reproduces
exactly the scores the old implementation obtained by copying + re-rotating
every cached key on every step.

Paging
------
The KV pool is paged: ``kv_cache.k_cache(layer)`` has shape
``[num_pages, page_size, n_kv_heads, head_dim]``.  Each ``SharedBlock`` owns a
list of page-start slots, so a block of length ``L`` occupies ``ceil(L/P)``
pages whose last page holds ``L - (npages-1)*P`` valid tokens.  Per-segment
attention runs through a FlashInfer paged decode wrapper planned with the
pool's real ``page_size`` (page-number indices + per-segment ``last_page_len``);
partial outputs are merged per worker with ``flashinfer.merge_states``.  No
intra-segment masking is needed in decode: non-self segments lie entirely in
the query's past, and the self segment's newest key *is* the query token.

A worker whose write block is NOT in its own read view still has to attend to
its own freshly-written token (distance 0).  That single key sits at an
arbitrary page offset, so it cannot be read as a contiguous paged prefix; it is
served instead by an auxiliary ``page_size=1`` wrapper that addresses the one
slot directly through the flattened pool, and merged in alongside the paged
segments (identical LSE convention, so the merge is exact).

Following the reference implementation, all workers' new KV entries are
written to the cache *before* attention runs, and segment lengths are counted
post-append -- so a worker reading another worker's write block sees that
worker's current-step token as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
from minisgl.attention import BaseAttnMetadata

from .rope_correction import apply_rope_correction

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )
    from minisgl.core import Batch
    from minisgl.kvcache import BaseKVCachePool

    from .shared_block import SharedBlock
    from .worker_group import WorkerGroup


@dataclass
class SharedCacheAttnMetadata(BaseAttnMetadata):
    # ``AttentionLayer.forward`` duck-types on this attribute to divert the
    # batch away from the regular attention backend.
    shared_cache_op: SharedCacheAttention
    # Per output q-row gather/rotation plan.  Sub-requests are ordered
    # ``[main paged ..., aux single-token ...]``.  For decode: one row per
    # (worker, segment).  For context prefill: S rows (one per new token) per
    # sub-request.
    sub_worker: torch.Tensor  # int64 — source q-row per sub-request row
    sub_loc: torch.Tensor  # int64 — query RoPE position per sub-request row
    pad_slot: torch.Tensor  # [N_sub] int64 — scatter index into [W * max_segments]
    num_workers: int  # number of output rows (workers for decode, tokens for prefill)
    max_segments: int
    # Per-group q-row-copy counts, in the fixed q_sub layout order
    # ``[dec_main | pf_ctx | pf_self | dec_aux]``.  ``forward`` runs whichever
    # wrappers have a non-zero count and merges their outputs by ``pad_slot``.
    n_main: int = 0  # dec_main: decode segments served by the paged decode wrapper
    n_pf_ctx: int = 0  # prefill context segments (non-causal paged prefill wrapper)
    n_pf_self: int = 0  # prefill self segment (causal paged prefill wrapper)
    n_aux: int = 0  # dec_aux: decode implicit-self served by the page_size=1 wrapper
    phase: Literal["decode", "context_prefill", "mixed"] = "decode"
    # Precomputed last-token row per logical request; set for ``phase="mixed"``.
    last_indices: Optional[torch.Tensor] = None
    _plan_refs: tuple = field(default=(), repr=False)

    def get_last_indices(self, bs: int) -> torch.Tensor:
        if self.last_indices is not None:
            return self.last_indices
        if self.phase == "context_prefill":
            # one request of num_workers(=S) tokens; the LM head wants the last
            return torch.tensor([self.num_workers - 1], device=self.sub_worker.device)
        return torch.arange(bs, device=self.sub_worker.device)


_CPU = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}


@dataclass
class _PagedGroup:
    """Paged sub-requests destined for one FlashInfer wrapper, in q_sub order.

    Per q-row-copy: ``row`` (output row), ``loc`` (query RoPE rotation offset),
    ``slot`` (segment index within that output row).  Per sub-request:
    ``kv_parts`` (page-number tensor), ``page_count``, ``seq_len``,
    ``last_page`` and ``qo_len`` (query rows -- 1 for decode, S for prefill).
    """

    row: List[int] = field(default_factory=list)
    loc: List[int] = field(default_factory=list)
    slot: List[int] = field(default_factory=list)
    kv_parts: List[torch.Tensor] = field(default_factory=list)
    page_count: List[int] = field(default_factory=list)
    seq_len: List[int] = field(default_factory=list)
    last_page: List[int] = field(default_factory=list)
    qo_len: List[int] = field(default_factory=list)

    @property
    def n_copies(self) -> int:
        return len(self.row)

    def extend(self, other: "_PagedGroup") -> None:
        self.row += other.row
        self.loc += other.loc
        self.slot += other.slot
        self.kv_parts += other.kv_parts
        self.page_count += other.page_count
        self.seq_len += other.seq_len
        self.last_page += other.last_page
        self.qo_len += other.qo_len


@dataclass
class _AuxGroup:
    """Single-token (page_size=1) implicit-self decode sub-requests."""

    row: List[int] = field(default_factory=list)
    loc: List[int] = field(default_factory=list)
    slot: List[int] = field(default_factory=list)
    kv_slot: List[int] = field(default_factory=list)

    @property
    def n_copies(self) -> int:
        return len(self.row)


@dataclass
class _PrefillSpec:
    """One in-context prefill: ``num_new`` fresh tokens stored block-relative in
    ``new_page_starts``, attending fully to each (stable) ``context`` block."""

    context: List[SharedBlock]
    new_page_starts: torch.Tensor  # freshly-allocated page-start slots of the self block
    num_new: int


class SharedCacheAttention:
    """Owns the FlashInfer wrappers and runs shared-cache attention."""

    def __init__(
        self,
        kv_cache: BaseKVCachePool,
        cos_sin_cache: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

        self.kv_cache = kv_cache
        self.cos_sin_cache = cos_sin_cache
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        self.device = device

        self._workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        gqa = num_qo_heads // num_kv_heads
        # Primary paged decode wrapper (planned with the pool's real page_size).
        self.wrapper: BatchDecodeWithPagedKVCacheWrapper = BatchDecodeWithPagedKVCacheWrapper(
            self._workspace,
            kv_layout="NHD",
            use_tensor_cores=gqa >= 4,
            backend="fa2",
        )
        # Auxiliary page_size=1 decode wrapper for the implicit single new-token
        # self-segment of workers whose write block is not in their read view.
        self.aux_wrapper: BatchDecodeWithPagedKVCacheWrapper = BatchDecodeWithPagedKVCacheWrapper(
            self._workspace,
            kv_layout="NHD",
            use_tensor_cores=gqa >= 4,
            backend="fa2",
        )
        # Context prefill needs two simultaneously-planned prefill wrappers:
        # causal attention of the new block over itself, and full (non-causal)
        # attention of the new tokens over each context segment.  All wrappers
        # share the float workspace (runs are sequential on one stream); each
        # keeps its own int (plan-state) workspace.
        self.prefill_self_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD", backend="fa2"
        )
        self.prefill_ctx_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD", backend="fa2"
        )

        self._plan_event = torch.cuda.Event()
        self._plan_event.record()

    # ------------------------------------------------------------------
    # Sub-request enumeration (shared by decode / context-prefill / mixed)
    # ------------------------------------------------------------------

    def _decode_subrequests(
        self,
        group: WorkerGroup,
        new_page_for_block: Dict[int, Optional[int]],
        new_token_slots: torch.Tensor,
        row_offset: int = 0,
    ) -> tuple[_PagedGroup, _AuxGroup, int]:
        """Enumerate one decode step's sub-requests: a paged ``dec_main`` group
        (one segment per non-empty block in each worker's view) plus a page-1
        ``dec_aux`` group (the implicit self token for workers whose write block
        is not in their view).  Output rows are ``row_offset + worker_index``.
        Returns ``(dec_main, dec_aux, max_segments)`` over the enumerated workers.
        """
        P = self.page_size
        write_set = set(new_page_for_block.keys())
        main = _PagedGroup()
        aux = _AuxGroup()
        max_segments = 0

        for w in range(group.num_workers):
            row = row_offset + w
            view = group.cache_structure[w]
            wt = group.write_to[w]
            self_in_view = any(b is wt for b in view)

            # post-append segment lengths: every block written this step grows by 1
            lengths = [b.num_tokens + (1 if id(b) in write_set else 0) for b in view]
            total = sum(lengths)

            n_seg = 0
            prefix = 0
            for b, length in zip(view, lengths):
                if length == 0:
                    continue
                page_nums = b.page_numbers_tensor()
                # If this block is written this step and the new token started a
                # fresh page, that page must be visible to the reader as well.
                new_page = new_page_for_block.get(id(b)) if id(b) in write_set else None
                if new_page is not None:
                    extra = torch.tensor([new_page // P], dtype=torch.int32, device=self.device)
                    page_nums = torch.cat([page_nums, extra])
                n_pages_seg = int(page_nums.numel())
                main.row.append(row)
                main.loc.append(total - prefix - 1)
                main.slot.append(n_seg)
                main.kv_parts.append(page_nums)
                main.page_count.append(n_pages_seg)
                main.seq_len.append(length)
                main.last_page.append(length - (n_pages_seg - 1) * P)
                main.qo_len.append(1)
                n_seg += 1
                prefix += length

            if not self_in_view:
                # The query must still attend to itself: a single-token segment
                # at distance 0 (query rotated to its own block-relative pos).
                aux.row.append(row)
                aux.loc.append(wt.num_tokens)
                aux.slot.append(n_seg)
                aux.kv_slot.append(int(new_token_slots[w].item()))
                n_seg += 1

            max_segments = max(max_segments, n_seg)

        return main, aux, max_segments

    def _prefill_subrequests(
        self, spec: _PrefillSpec, row_offset: int
    ) -> tuple[_PagedGroup, _PagedGroup, int, int]:
        """Enumerate one in-context prefill's sub-requests: a non-causal
        ``pf_ctx`` group (one segment per context block, ``S`` query rows each)
        and a causal ``pf_self`` group (the fresh block).  Output rows are
        ``row_offset + token_index``.  For context block ``j`` at view offset
        ``O_j``, token ``i`` is rotated to ``(O_self + i) - O_j``; the self
        segment rotates token ``i`` to its block-relative position ``i``.
        Returns ``(pf_ctx, pf_self, max_segments=n_ctx+1, last_row)``.
        """
        P = self.page_size
        S = int(spec.num_new)
        context = spec.context
        n_ctx = len(context)
        M = n_ctx + 1
        self_offset = sum(b.num_tokens for b in context)

        ctx = _PagedGroup()
        prefix = 0
        for j, b in enumerate(context):
            for i in range(S):  # copies grouped block-major to match qo_indptr
                ctx.row.append(row_offset + i)
                ctx.loc.append(self_offset + i - prefix)
                ctx.slot.append(j)
            ctx.kv_parts.append(b.page_numbers_tensor())
            ctx.page_count.append(b.num_pages)
            ctx.seq_len.append(b.num_tokens)
            ctx.last_page.append(b.last_page_len)
            ctx.qo_len.append(S)
            prefix += b.num_tokens

        self_g = _PagedGroup()
        self_pages = (spec.new_page_starts.to(self.device) // P).to(torch.int32)
        n_self_pages = int(self_pages.numel())
        for i in range(S):
            self_g.row.append(row_offset + i)
            self_g.loc.append(i)
            self_g.slot.append(n_ctx)
        self_g.kv_parts.append(self_pages)
        self_g.page_count.append(n_self_pages)
        self_g.seq_len.append(S)
        self_g.last_page.append(S - (n_self_pages - 1) * P)
        self_g.qo_len.append(S)

        return ctx, self_g, M, row_offset + S - 1

    # ------------------------------------------------------------------
    # Wrapper planning (each returns the pinned/device tensors to keep alive)
    # ------------------------------------------------------------------

    def _plan_decode(self, group: _PagedGroup) -> List[torch.Tensor]:
        """Plan the paged decode wrapper (implicit qo_len=1 per sub-request)."""
        kv_indices = torch.cat(group.kv_parts).to(torch.int32)
        kv_indptr = torch.tensor([0] + group.page_count, **_CPU).cumsum_(0).to(torch.int32)
        seq_lens = torch.tensor(group.seq_len, **_CPU)
        last_page = torch.tensor(group.last_page, **_CPU)
        self.wrapper.plan(
            indptr=kv_indptr,
            indices=kv_indices,
            last_page_len=last_page,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            pos_encoding_mode="NONE",
            seq_lens=seq_lens,
            data_type=self.dtype,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            non_blocking=True,
        )
        return [kv_indices, kv_indptr, seq_lens, last_page]

    def _plan_aux(self, group: _AuxGroup) -> List[torch.Tensor]:
        """Plan the page_size=1 wrapper for single-token implicit-self segments."""
        n = group.n_copies
        indices = torch.tensor(group.kv_slot, dtype=torch.int32, device=self.device)
        indptr = torch.arange(0, n + 1, **_CPU)
        seq_lens = torch.ones(n, **_CPU)
        last_page = torch.ones(n, **_CPU)
        self.aux_wrapper.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=last_page,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens=seq_lens,
            data_type=self.dtype,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            non_blocking=True,
        )
        return [indices, indptr, seq_lens, last_page]

    def _plan_prefill(
        self, wrapper: BatchPrefillWithPagedKVCacheWrapper, group: _PagedGroup, causal: bool
    ) -> List[torch.Tensor]:
        """Plan a paged prefill wrapper (per-sub-request qo_len via qo_indptr)."""
        kv_indices = torch.cat(group.kv_parts).to(torch.int32)
        kv_indptr = torch.tensor([0] + group.page_count, **_CPU).cumsum_(0).to(torch.int32)
        qo_indptr = torch.tensor([0] + group.qo_len, **_CPU).cumsum_(0).to(torch.int32)
        seq_lens = torch.tensor(group.seq_len, **_CPU)
        last_page = torch.tensor(group.last_page, **_CPU)
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page,
            seq_lens=seq_lens,
            causal=causal,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            non_blocking=True,
        )
        return [kv_indices, kv_indptr, qo_indptr, seq_lens, last_page]

    def _assemble_metadata(
        self,
        dec_main: _PagedGroup,
        pf_ctx: _PagedGroup,
        pf_self: _PagedGroup,
        dec_aux: _AuxGroup,
        max_segments: int,
        num_rows: int,
        phase: Literal["decode", "context_prefill", "mixed"],
        last_indices: Optional[torch.Tensor],
        plan_refs: List[torch.Tensor],
    ) -> SharedCacheAttnMetadata:
        """Concatenate the four groups in q_sub layout order
        ``[dec_main | pf_ctx | pf_self | dec_aux]`` and build the metadata."""
        rows = dec_main.row + pf_ctx.row + pf_self.row + dec_aux.row
        locs = dec_main.loc + pf_ctx.loc + pf_self.loc + dec_aux.loc
        slots = dec_main.slot + pf_ctx.slot + pf_self.slot + dec_aux.slot
        pad = [r * max_segments + s for r, s in zip(rows, slots)]
        meta = SharedCacheAttnMetadata(
            shared_cache_op=self,
            sub_worker=torch.tensor(rows, dtype=torch.int64, device=self.device),
            sub_loc=torch.tensor(locs, dtype=torch.int64, device=self.device),
            pad_slot=torch.tensor(pad, dtype=torch.int64, device=self.device),
            num_workers=num_rows,
            max_segments=max_segments,
            n_main=dec_main.n_copies,
            n_pf_ctx=pf_ctx.n_copies,
            n_pf_self=pf_self.n_copies,
            n_aux=dec_aux.n_copies,
            phase=phase,
            last_indices=last_indices,
        )
        # Keep plan inputs alive through the forward: the async plan copy reads
        # the pinned tensors after prepare() returns.
        meta._plan_refs = tuple(plan_refs)
        return meta

    # ------------------------------------------------------------------
    # Public prepare entry points
    # ------------------------------------------------------------------

    def prepare(
        self,
        group: WorkerGroup,
        new_page_for_block: Dict[int, Optional[int]],
        new_token_slots: torch.Tensor,
    ) -> SharedCacheAttnMetadata:
        """
        Build per-(worker, segment) sub-requests for one decode step and plan
        the FlashInfer wrappers.

        Args:
            group: the worker group being decoded.
            new_page_for_block: maps ``id(write_block)`` -> the page-start slot
                of a freshly-allocated page when this step's token starts a new
                page in that block, else ``None``.  Its keys identify the blocks
                written this step.
            new_token_slots: ``[num_workers]`` physical slot of each worker's
                new token (used for the implicit-self aux segment).
        """
        assert group.num_workers > 0
        main, aux, max_segments = self._decode_subrequests(
            group, new_page_for_block, new_token_slots
        )
        plan_refs: List[torch.Tensor] = []
        self._plan_event.synchronize()
        if main.n_copies:
            plan_refs += self._plan_decode(main)
        if aux.n_copies:
            plan_refs += self._plan_aux(aux)
        self._plan_event.record()
        return self._assemble_metadata(
            main, _PagedGroup(), _PagedGroup(), aux,
            max_segments, group.num_workers, "decode", None, plan_refs,
        )

    def prepare_context_prefill(
        self, context: List[SharedBlock], new_page_starts: torch.Tensor, num_new: int
    ) -> SharedCacheAttnMetadata:
        """
        Plan one prefill of ``num_new`` new tokens (stored block-relative at
        0..num_new-1 in the pages ``new_page_starts``) that attend causally to
        themselves and fully to each *context* block, as if the blocks were
        concatenated ``[ctx_0, ..., ctx_{n-1}, new]``.

        Mirrors the reference's ``prefill_cache_block(text, [ctx..., new])``.
        """
        assert context and all(b.num_tokens > 0 for b in context)
        spec = _PrefillSpec(list(context), new_page_starts, int(num_new))
        ctx, self_g, max_segments, _ = self._prefill_subrequests(spec, row_offset=0)
        plan_refs: List[torch.Tensor] = []
        self._plan_event.synchronize()
        # context segments are entirely in the new tokens' past -> no masking;
        # the new block attends to itself causally (standard prefill).
        plan_refs += self._plan_prefill(self.prefill_ctx_wrapper, ctx, causal=False)
        plan_refs += self._plan_prefill(self.prefill_self_wrapper, self_g, causal=True)
        self._plan_event.record()
        return self._assemble_metadata(
            _PagedGroup(), ctx, self_g, _AuxGroup(),
            max_segments, int(num_new), "context_prefill", None, plan_refs,
        )

    def prepare_mixed(
        self,
        group: Optional[WorkerGroup],
        new_page_for_block: Dict[int, Optional[int]],
        new_token_slots: torch.Tensor,
        prefill_specs: List[_PrefillSpec],
    ) -> SharedCacheAttnMetadata:
        """
        Plan one forward that decodes ``group``'s workers *and* prefills each of
        ``prefill_specs`` together.  Output rows are laid out decode-first:
        rows ``[0, W)`` are the decode workers, then each prefill request's
        ``S`` tokens.  Decode segments run on the decode/aux wrappers and
        prefill segments on the prefill wrappers; all partials share one LSE
        merge (see :meth:`_run_and_merge`).

        v1 constraints (caller-enforced): each prefill's self block is fresh and
        not referenced by any decode worker's view or another prefill's context
        this step, and prefill context blocks are not written this step -- so
        segment lengths match running the phases separately.
        """
        W = group.num_workers if group is not None else 0
        if W > 0:
            dec_main, dec_aux, max_segments = self._decode_subrequests(
                group, new_page_for_block, new_token_slots
            )
        else:
            dec_main, dec_aux, max_segments = _PagedGroup(), _AuxGroup(), 0

        pf_ctx = _PagedGroup()
        pf_self = _PagedGroup()
        last_pf_rows: List[int] = []
        row = W
        for spec in prefill_specs:
            ctx, self_g, m_pf, last_row = self._prefill_subrequests(spec, row)
            pf_ctx.extend(ctx)
            pf_self.extend(self_g)
            max_segments = max(max_segments, m_pf)
            last_pf_rows.append(last_row)
            row += int(spec.num_new)

        num_rows = row
        assert num_rows > 0, "mixed batch must contain at least one row"

        plan_refs: List[torch.Tensor] = []
        self._plan_event.synchronize()
        if dec_main.n_copies:
            plan_refs += self._plan_decode(dec_main)
        if pf_ctx.n_copies:
            plan_refs += self._plan_prefill(self.prefill_ctx_wrapper, pf_ctx, causal=False)
        if pf_self.n_copies:
            plan_refs += self._plan_prefill(self.prefill_self_wrapper, pf_self, causal=True)
        if dec_aux.n_copies:
            plan_refs += self._plan_aux(dec_aux)
        self._plan_event.record()

        # LM head extracts one row per logical request: decode workers (their own
        # row) then each prefill request's last token.
        last_indices = torch.tensor(
            list(range(W)) + last_pf_rows, dtype=torch.int64, device=self.device
        )
        return self._assemble_metadata(
            dec_main, pf_ctx, pf_self, dec_aux,
            max_segments, num_rows, "mixed", last_indices, plan_refs,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        Args:
            q: ``[num_workers, num_qo_heads * head_dim]`` — *unrotated* queries.
            k: ``[num_workers, num_kv_heads * head_dim]`` — unrotated keys.
            v: ``[num_workers, num_kv_heads * head_dim]``.
            batch.positions: block-relative write positions per worker.

        Returns ``[num_workers, num_qo_heads * head_dim]``.
        """
        meta = batch.attn_metadata
        assert isinstance(meta, SharedCacheAttnMetadata)
        W, Hq, D = meta.num_workers, self.num_qo_heads, self.head_dim

        # Store the new token's KV with block-relative key rotation.
        k_rot = apply_rope_correction(
            k.reshape(W, self.num_kv_heads, D), batch.positions, self.cos_sin_cache
        )
        # k_rot is freshly materialized (contiguous); the store kernel needs
        # v in the same layout, so detach v from its strided qkv slice too.
        self.kv_cache.store_kv(k_rot.view(W, -1), v.contiguous(), batch.out_loc, layer_id)

        # One query copy per (row, segment), rotated to its segment-relative position.
        q_sub = q.reshape(W, Hq, D)[meta.sub_worker]
        q_sub = apply_rope_correction(q_sub, meta.sub_loc, self.cos_sin_cache)

        return self._run_and_merge(q_sub, meta, layer_id)

    def _run_and_merge(
        self, q_sub: torch.Tensor, meta: SharedCacheAttnMetadata, layer_id: int
    ) -> torch.Tensor:
        """Run each non-empty wrapper on its slice of ``q_sub`` and LSE-merge the
        per-segment partials into one output row per query row.

        This is the shared attention core for decode, context-prefill and mixed
        batches.  ``q_sub`` is laid out in the fixed group order
        ``[dec_main | pf_ctx | pf_self | dec_aux]`` and each group's row count is
        carried on ``meta``; the four groups map 1:1 to the four FlashInfer
        wrappers.  Partial ``(out, lse)`` for group row ``r`` scatters to
        ``meta.pad_slot[r] = output_row * max_segments + segment_slot``.
        """
        R, Hq, D = meta.num_workers, self.num_qo_heads, self.head_dim
        k_paged = self.kv_cache.k_cache(layer_id)
        v_paged = self.kv_cache.v_cache(layer_id)
        paged = (k_paged, v_paged)

        # (wrapper, num_q_rows, paged_kv_cache) in the fixed q_sub layout order.
        runs: List[tuple] = []
        if meta.n_main > 0:
            runs.append((self.wrapper, meta.n_main, paged))
        if meta.n_pf_ctx > 0:
            runs.append((self.prefill_ctx_wrapper, meta.n_pf_ctx, paged))
        if meta.n_pf_self > 0:
            runs.append((self.prefill_self_wrapper, meta.n_pf_self, paged))
        if meta.n_aux > 0:
            # The single new-token self-segment lives at an arbitrary page
            # offset -> read it through the flattened (page_size=1) pool.
            flat = (
                k_paged.view(-1, 1, self.num_kv_heads, D),
                v_paged.view(-1, 1, self.num_kv_heads, D),
            )
            runs.append((self.aux_wrapper, meta.n_aux, flat))

        outs: List[torch.Tensor] = []
        lses: List[torch.Tensor] = []
        off = 0
        for wrapper, n, kv in runs:
            out_r, lse_r = wrapper.run(q_sub[off : off + n], kv, return_lse=True)
            outs.append(out_r)
            lses.append(lse_r)
            off += n

        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        lse = lses[0] if len(lses) == 1 else torch.cat(lses, dim=0)

        M = meta.max_segments
        if len(runs) == 1 and M == 1:
            # Every output row has exactly one segment, emitted in row order.
            return out.view(R, -1)

        from flashinfer import merge_states

        v_pad = out.new_zeros(R * M, Hq, D)
        # finite "minus infinity": exp(pad - max) underflows to 0 for any real lse
        s_pad = torch.full((R * M, Hq), -5.0e4, dtype=torch.float32, device=self.device)
        v_pad[meta.pad_slot] = out
        s_pad[meta.pad_slot] = lse
        merged, _ = merge_states(v_pad.view(R, M, Hq, D), s_pad.view(R, M, Hq))
        return merged.view(R, -1)
