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
``[num_pages, page_size, n_kv_heads, head_dim]``.  Each ``CacheBlock`` owns a
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
from .attention_metadata import page_numbers, upload_page_indices

if TYPE_CHECKING:
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from minisgl.core import Batch
    from minisgl.kvcache import BaseKVCachePool

    from .shared_block import CacheBlock
    from .worker_group import WorkerGroup


@dataclass
class SharedCacheAttnMetadata(BaseAttnMetadata):
    # ``AttentionLayer.forward`` duck-types on this attribute to divert the
    # batch away from the regular attention backend.
    shared_cache_op: SharedCacheAttention
    # Per output q-row gather/rotation plan.  For decode, sub-requests are
    # ordered ``[main paged ..., aux single-token ...]``, one row per
    # (worker, segment).  For prefill, ``[context ..., causal self ...]``, with
    # one row per new token per segment.
    sub_worker: torch.Tensor  # int64 — source q-row per sub-request row
    # int64 query RoPE position per sub-request row: ``[N_sub]`` for 1-D RoPE (and
    # for text mRoPE rows, where all three axes coincide), or ``[3, N_sub]`` when
    # the rows carry genuine 3-D mRoPE positions (image tokens in a prefill).
    sub_loc: torch.Tensor
    pad_slot: torch.Tensor  # [N_sub] int64 — scatter index into [W * max_segments]
    num_workers: int  # number of output rows (workers for decode, tokens for prefill)
    max_segments: int
    n_main: int = 0  # sub-requests served by the paged wrapper
    n_aux: int = 0  # sub-requests served by the page_size=1 (implicit-self) wrapper
    # ``prefill_batch``: q-rows served by the (non-causal) context wrapper; the
    # rest, ``[n_ctx_rows:]``, are the causal self segments.
    n_ctx_rows: int = 0
    # ``prefill_batch``: last-token output row of each request, for the LM head.
    last_indices: Optional[torch.Tensor] = None
    phase: Literal["decode", "prefill_batch"] = "decode"
    _plan_refs: tuple = field(default=(), repr=False)
    graph_buffers: object | None = field(default=None, repr=False)

    def get_last_indices(self, bs: int) -> torch.Tensor:
        if self.phase == "prefill_batch":
            assert self.last_indices is not None
            return self.last_indices
        return torch.arange(bs, device=self.sub_worker.device)


@dataclass
class PrefillSpec:
    """One prefill inside a batched shared-cache prefill forward.

    *context* are the (non-empty) blocks the new tokens attend to in view
    order — possibly empty, for a plain prefill —
    *self_page_starts* is the write block's complete post-write page list,
    *self_prefix_len* how many tokens it already held, *self_prefix_span* that
    prefix's mRoPE advance, and *mrope_rel* the new tokens' zero-based 3-D
    mRoPE positions ``[3, num_new]`` when they contain an image.
    """

    context: List[CacheBlock]
    self_page_starts: torch.Tensor
    num_new: int
    self_prefix_len: int = 0
    self_prefix_span: Optional[int] = None
    mrope_rel: Optional[torch.Tensor] = None


def _rel_positions(mrope_rel: Optional[torch.Tensor], num_new: int, use_3d: bool) -> torch.Tensor:
    """Zero-based positions of a request's new tokens: ``[S]`` for text, or
    ``[3, S]`` once any request in the batch carries genuine 3-D mRoPE positions
    (a text token's position is the same on all three axes)."""
    if mrope_rel is None:
        rel = torch.arange(num_new, dtype=torch.int64)
        return rel.view(1, -1).expand(3, -1) if use_3d else rel
    rel = mrope_rel.to(dtype=torch.int64, device="cpu")
    assert rel.shape == (3, num_new), f"mrope_rel must be [3, {num_new}], got {tuple(rel.shape)}"
    return rel


def _rotate_half_last(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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
        rotary_dim: int | None = None,
        mrope_section: tuple | None = None,
        rope_base: float | None = None,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.kv_cache = kv_cache
        self.cos_sin_cache = cos_sin_cache
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # Partial RoPE (Qwen3.5): rotate only the first `rotary_dim` head dims,
        # pass the rest through.  Defaults to full-head RoPE (standard models).
        self.rotary_dim = rotary_dim if rotary_dim is not None else head_dim
        # Interleaved mRoPE (Qwen3.5 multimodal): when set, the query/key rotation
        # uses 3-axis interleaved mRoPE (positions broadcast all-axes for text decode
        # tokens, so scores against image keys stored 3D remain exact).  cos/sin are
        # computed on the fly from a rope base rather than the 1D cos_sin_cache.
        self.mrope_section = mrope_section
        self._mrope_inv_freq = None
        if mrope_section is not None:
            self._mrope_inv_freq = 1.0 / (
                rope_base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / self.rotary_dim)
            )
        self.page_size = page_size
        self.dtype = dtype
        self.device = device

        self._workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._workspace_prepared = False
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
        # Prefill needs two simultaneously-planned prefill wrappers: causal
        # attention of each new block over itself, and full (non-causal)
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

    def prepare_workspace(self, profiles) -> None:
        """Reserve the complete fixed-buffer catalogue before planning/capture.

        Both decode and prefill wrappers share one float arena but own their int
        plan storage. Never resize/rebind a captured profile. Existing eager
        wrappers retain their original storage and are not invalidated here.
        """
        profiles = tuple(profiles)
        if not profiles:
            return
        if self._workspace_prepared or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Attention workspace must be prepared once before capture')
        if any(p.attention is not self for p in profiles):
            raise ValueError('Attention workspace profiles must belong to this instance')
        requirements = [r for p in profiles for r in p.workspace_requirements()]
        size = max(self._workspace.numel(), *(r[1] for r in requirements))
        workspace = self._workspace
        if size > workspace.numel():
            workspace = torch.empty(size, device=self.device, dtype=torch.uint8)
        # Allocate before rebinding any wrapper, so a device OOM cannot leave
        # only half the catalogue pointing at the new arena.
        bindings = [(wrapper, torch.empty(max(16, int_bytes), device=self.device,
                                         dtype=torch.uint8))
                    for wrapper, _, int_bytes in requirements]
        for wrapper, ints in bindings:
            wrapper.reset_workspace_buffer(workspace, ints)
        self._workspace = workspace
        self._workspace_prepared = True

    def _rope(self, x: torch.Tensor, corrections: torch.Tensor) -> torch.Tensor:
        """Block-relative RoPE correction on ``x [N, heads, head_dim]``.

        For partial RoPE only the first ``rotary_dim`` dims are rotated; the tail
        is passed through unchanged (Qwen3.5).  ``cos_sin_cache`` is sized
        ``[max_pos, rotary_dim]`` in the partial case.
        """
        if self.mrope_section is not None:
            return self._rope_mrope(x, corrections)
        if self.rotary_dim == self.head_dim:
            return apply_rope_correction(x, corrections, self.cos_sin_cache)
        x_rot = apply_rope_correction(
            x[..., : self.rotary_dim].contiguous(), corrections, self.cos_sin_cache
        )
        return torch.cat([x_rot, x[..., self.rotary_dim :]], dim=-1)

    def _rope_mrope(self, x: torch.Tensor, corrections: torch.Tensor) -> torch.Tensor:
        """Interleaved-mRoPE partial rotation of ``x [N, heads, head_dim]``.

        ``corrections`` are the tokens' mRoPE positions (block-relative for keys,
        segment-relative for query copies), either ``[N]`` -- a text token, whose
        position is the same on all three axes -- or ``[3, N]`` for genuine 3-D
        (image) positions.  Because a whole block is shifted by one scalar on all
        three axes, subtracting a segment's mRoPE prefix from a 3-D position stays
        exact, and text rows broadcast against image keys stored 3-D.
        """
        pos = corrections.to(self.device).float()  # [N] or [3, N]
        pos3 = pos if pos.dim() == 2 else pos[None, :].expand(3, -1)  # (3, N)
        freqs3 = pos3[:, :, None] * self._mrope_inv_freq[None, None, :]  # (3, N, rd/2)
        freqs = freqs3[0].clone()
        for dim, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs[..., idx] = freqs3[dim][..., idx]
        emb = torch.cat((freqs, freqs), dim=-1)  # (N, rotary_dim)
        cos = emb.cos().to(x.dtype)[:, None, :]
        sin = emb.sin().to(x.dtype)[:, None, :]
        x_rot, x_pass = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        x_rot = x_rot * cos + _rotate_half_last(x_rot) * sin
        return torch.cat([x_rot, x_pass], dim=-1)

    def prepare(
        self,
        group: WorkerGroup,
        new_page_for_block: Dict[int, Optional[int]],
        new_token_slots: torch.Tensor,
        graph_buffers=None,
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
        num_workers = group.num_workers
        assert num_workers > 0
        P = self.page_size
        write_set = set(new_page_for_block.keys())

        # Main (paged) sub-requests.
        main_sub_worker: List[int] = []
        main_sub_loc: List[int] = []
        main_sub_slot: List[int] = []
        main_kv_parts: List[List[int]] = []
        page_snapshot: Dict[int, List[int]] = {}
        token_slots_cpu = None
        main_page_counts: List[int] = []
        main_seq_lens: List[int] = []
        main_last_page: List[int] = []

        # Auxiliary (single-token, page_size=1) sub-requests.
        aux_sub_worker: List[int] = []
        aux_sub_loc: List[int] = []
        aux_sub_slot: List[int] = []
        aux_kv_slots: List[int] = []

        max_segments = 0

        for w in range(num_workers):
            view = group.cache_structure[w]
            wt = group.write_to[w]
            self_in_view = any(b is wt for b in view)

            # post-append segment lengths: every block written this step grows by 1
            lengths = [b.num_tokens + (1 if id(b) in write_set else 0) for b in view]
            # RoPE positions use each block's mRoPE *span* (== num_tokens for text,
            # but compressed for an image-bearing block), so the query rotates at
            # its true mRoPE position.  Identical to `lengths` for text/standard models.
            mspans = [b.mrope_span + (1 if id(b) in write_set else 0) for b in view]
            mtotal = sum(mspans)

            n_seg = 0
            mprefix = 0
            for b, length, mspan in zip(view, lengths, mspans):
                if length == 0:
                    mprefix += mspan
                    continue
                # If this block is written this step and the new token started a
                # fresh page, that page must be visible to the reader as well.
                page_nums = page_numbers(b, page_snapshot, new_page_for_block.get(id(b)))
                n_pages_seg = len(page_nums)
                main_sub_worker.append(w)
                main_sub_loc.append(mtotal - mprefix - 1)
                main_sub_slot.append(n_seg)
                main_kv_parts.append(page_nums)
                main_page_counts.append(n_pages_seg)
                main_seq_lens.append(length)
                main_last_page.append(length - (n_pages_seg - 1) * P)
                n_seg += 1
                mprefix += mspan

            if not self_in_view:
                # The query must still attend to itself: a single-token segment
                # at distance 0 (query rotated to its own block-relative pos).
                aux_sub_worker.append(w)
                aux_sub_loc.append(wt.mrope_span)
                aux_sub_slot.append(n_seg)
                if token_slots_cpu is None:
                    token_slots_cpu = new_token_slots.cpu().tolist()
                aux_kv_slots.append(token_slots_cpu[w])
                n_seg += 1

            max_segments = max(max_segments, n_seg)

        n_main = len(main_sub_worker)
        n_aux = len(aux_sub_worker)
        if graph_buffers is not None:
            return graph_buffers.plan(
                main_sub_worker, main_sub_loc, main_sub_slot, main_kv_parts,
                main_page_counts, main_seq_lens, main_last_page,
                aux_sub_worker, aux_sub_loc, aux_kv_slots)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        plan_refs: List[torch.Tensor] = []

        self._plan_event.synchronize()
        if n_main > 0:
            kv_indices = upload_page_indices(main_kv_parts, self.device, plan_refs)
            kv_indptr_cpu = (
                torch.tensor([0] + main_page_counts, **CPU_KWARGS).cumsum_(0).to(torch.int32)
            )
            seq_lens_cpu = torch.tensor(main_seq_lens, **CPU_KWARGS)
            last_page_cpu = torch.tensor(main_last_page, **CPU_KWARGS)
            self.wrapper.plan(
                indptr=kv_indptr_cpu,
                indices=kv_indices,
                last_page_len=last_page_cpu,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=P,
                pos_encoding_mode="NONE",
                seq_lens=seq_lens_cpu,
                data_type=self.dtype,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                non_blocking=True,
            )
            plan_refs += [kv_indices, kv_indptr_cpu, seq_lens_cpu, last_page_cpu]

        if n_aux > 0:
            aux_indices = upload_page_indices([aux_kv_slots], self.device, plan_refs)
            aux_indptr_cpu = torch.arange(0, n_aux + 1, **CPU_KWARGS)
            aux_seq_lens_cpu = torch.ones(n_aux, **CPU_KWARGS)
            aux_last_page_cpu = torch.ones(n_aux, **CPU_KWARGS)
            self.aux_wrapper.plan(
                indptr=aux_indptr_cpu,
                indices=aux_indices,
                last_page_len=aux_last_page_cpu,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=1,
                pos_encoding_mode="NONE",
                seq_lens=aux_seq_lens_cpu,
                data_type=self.dtype,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                non_blocking=True,
            )
            plan_refs += [aux_indices, aux_indptr_cpu, aux_seq_lens_cpu, aux_last_page_cpu]
        self._plan_event.record()

        sub_worker = main_sub_worker + aux_sub_worker
        sub_loc = main_sub_loc + aux_sub_loc
        sub_slot = main_sub_slot + aux_sub_slot
        pad_slot = [w * max_segments + s for w, s in zip(sub_worker, sub_slot)]

        meta = SharedCacheAttnMetadata(
            shared_cache_op=self,
            sub_worker=torch.tensor(sub_worker, dtype=torch.int64, device=self.device),
            sub_loc=torch.tensor(sub_loc, dtype=torch.int64, device=self.device),
            pad_slot=torch.tensor(pad_slot, dtype=torch.int64, device=self.device),
            num_workers=num_workers,
            max_segments=max_segments,
            n_main=n_main,
            n_aux=n_aux,
        )
        # Keep plan inputs alive through the forward: the async plan copy reads
        # the pinned tensors after prepare() returns.
        meta._plan_refs = tuple(plan_refs)
        return meta

    def prepare_prefill_batch(self, specs: List[PrefillSpec], graph_buffers=None) -> SharedCacheAttnMetadata:
        """
        Plan several prefills as one forward, laid out request-major: request
        ``r`` owns output rows ``[offset_r, offset_r + num_new_r)``.

        A request's new tokens attend causally to their own block (prefix
        included, so a non-empty write block is extended) and fully to each of
        its own context blocks, as if the blocks were concatenated
        ``[ctx_0, ..., block]``.  The two wrappers carry one sub-request per
        (request, context block) and one per request respectively, so requests'
        segment counts may differ and a request may have no context at all
        (a plain prefill).  This is the only prefill planner: a single request
        is simply a batch of one.

        Requests are independent: nothing planned here lets one request's
        queries reach another's freshly-written KV.  Context segments are sized
        from their block's *pre-write* token count, so co-batching a request
        that reads a block another request writes this step would silently drop
        those tokens — callers must not form such a batch (the async engine's
        grouping and ``SharedCacheSession.prefill_batch`` both refuse it).
        """
        assert specs, "prepare_prefill_batch needs at least one request"
        if graph_buffers is not None:
            return graph_buffers.plan(specs)
        P = self.page_size
        # A single 3-D request forces every row onto the 3-axis mRoPE path, since
        # ``sub_loc`` is one tensor; text rows simply repeat their position thrice.
        use_3d = any(spec.mrope_rel is not None for spec in specs)
        cat_dim = 1 if use_3d else 0

        max_segments = max(len(spec.context) + 1 for spec in specs)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        # Context sub-requests (non-causal), then self sub-requests (causal) —
        # one contiguous q_sub run per wrapper.
        ctx_gather: List[int] = []
        ctx_slot: List[int] = []
        ctx_loc: List[torch.Tensor] = []
        ctx_kv_parts: List[List[int]] = []
        page_snapshot: Dict[int, List[int]] = {}
        ctx_page_counts: List[int] = []
        ctx_q_lens: List[int] = []
        ctx_seq_lens: List[int] = []
        ctx_last_page: List[int] = []

        self_gather: List[int] = []
        self_slot: List[int] = []
        self_loc: List[torch.Tensor] = []
        self_kv_parts: List[List[int]] = []
        self_page_counts: List[int] = []
        self_q_lens: List[int] = []
        self_seq_lens: List[int] = []
        self_last_page: List[int] = []

        last_indices: List[int] = []
        offset = 0

        for spec in specs:
            S = int(spec.num_new)
            assert S > 0
            T = int(spec.self_prefix_len)
            T_span = T if spec.self_prefix_span is None else int(spec.self_prefix_span)
            ctx_spans = [b.mrope_span for b in spec.context]
            # Running mRoPE position of this request's first new token.
            self_offset = sum(ctx_spans) + T_span
            rows = list(range(offset, offset + S))
            rel = _rel_positions(spec.mrope_rel, S, use_3d)

            prefix = 0
            for j, (block, span) in enumerate(zip(spec.context, ctx_spans)):
                assert block.num_tokens > 0, "context blocks must be non-empty"
                ctx_gather.extend(rows)
                ctx_slot.extend([j] * S)
                ctx_loc.append(rel + (self_offset - prefix))
                ctx_kv_parts.append(page_numbers(block, page_snapshot))
                ctx_page_counts.append(block.num_pages)
                ctx_q_lens.append(S)
                ctx_seq_lens.append(block.num_tokens)
                ctx_last_page.append(block.last_page_len)
                prefix += span

            pages = [start // P for start in spec.self_page_starts.cpu().tolist()]
            n_pages = len(pages)
            self_len = T + S
            self_gather.extend(rows)
            self_slot.extend([len(spec.context)] * S)
            self_loc.append(rel + T_span)
            self_kv_parts.append(pages)
            self_page_counts.append(n_pages)
            self_q_lens.append(S)
            self_seq_lens.append(self_len)
            self_last_page.append(self_len - (n_pages - 1) * P)

            last_indices.append(offset + S - 1)
            offset += S

        n_rows = offset
        sub_gather = ctx_gather + self_gather
        sub_slot = ctx_slot + self_slot
        sub_loc_t = torch.cat(ctx_loc + self_loc, dim=cat_dim)
        pad_slot = [r * max_segments + s for r, s in zip(sub_gather, sub_slot)]

        plan_common = dict(
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=P,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            non_blocking=True,
        )
        plan_refs: List[torch.Tensor] = []

        self._plan_event.synchronize()
        if ctx_q_lens:
            # context segments lie entirely in the new tokens' past -> no masking
            ctx_kv_indices = upload_page_indices(ctx_kv_parts, self.device, plan_refs)
            ctx_kv_indptr = (
                torch.tensor([0] + ctx_page_counts, **CPU_KWARGS).cumsum_(0).to(torch.int32)
            )
            ctx_qo_indptr = torch.tensor([0] + ctx_q_lens, **CPU_KWARGS).cumsum_(0).to(torch.int32)
            ctx_seq_cpu = torch.tensor(ctx_seq_lens, **CPU_KWARGS)
            ctx_last_cpu = torch.tensor(ctx_last_page, **CPU_KWARGS)
            self.prefill_ctx_wrapper.plan(
                qo_indptr=ctx_qo_indptr,
                paged_kv_indptr=ctx_kv_indptr,
                paged_kv_indices=ctx_kv_indices,
                paged_kv_last_page_len=ctx_last_cpu,
                seq_lens=ctx_seq_cpu,
                max_token_per_sequence=max(ctx_q_lens),
                causal=False,
                **plan_common,
            )
            plan_refs += [ctx_kv_indices, ctx_kv_indptr, ctx_qo_indptr, ctx_seq_cpu, ctx_last_cpu]

        # Each request attends to its own block causally; with qo_len = S <=
        # kv_len = T + S FlashInfer aligns the mask to the end, i.e. an extend prefill.
        self_kv_indices = upload_page_indices(self_kv_parts, self.device, plan_refs)
        self_kv_indptr = (
            torch.tensor([0] + self_page_counts, **CPU_KWARGS).cumsum_(0).to(torch.int32)
        )
        self_qo_indptr = torch.tensor([0] + self_q_lens, **CPU_KWARGS).cumsum_(0).to(torch.int32)
        self_seq_cpu = torch.tensor(self_seq_lens, **CPU_KWARGS)
        self_last_cpu = torch.tensor(self_last_page, **CPU_KWARGS)
        self.prefill_self_wrapper.plan(
            qo_indptr=self_qo_indptr,
            paged_kv_indptr=self_kv_indptr,
            paged_kv_indices=self_kv_indices,
            paged_kv_last_page_len=self_last_cpu,
            seq_lens=self_seq_cpu,
            max_token_per_sequence=max(self_q_lens),
            causal=True,
            **plan_common,
        )
        plan_refs += [self_kv_indices, self_kv_indptr, self_qo_indptr, self_seq_cpu, self_last_cpu]
        self._plan_event.record()

        meta = SharedCacheAttnMetadata(
            shared_cache_op=self,
            sub_worker=torch.tensor(sub_gather, dtype=torch.int64, device=self.device),
            sub_loc=sub_loc_t.to(self.device),
            pad_slot=torch.tensor(pad_slot, dtype=torch.int64, device=self.device),
            num_workers=n_rows,
            max_segments=max_segments,
            n_ctx_rows=len(ctx_gather),
            last_indices=torch.tensor(last_indices, dtype=torch.int64, device=self.device),
            phase="prefill_batch",
        )
        # Keep plan inputs alive through the forward: the async plan copy reads
        # the pinned tensors after prepare() returns.
        meta._plan_refs = tuple(plan_refs)
        return meta

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        Args:
            q: ``[num_workers, num_qo_heads * head_dim]`` — *unrotated* queries.
            k: ``[num_workers, num_kv_heads * head_dim]`` — unrotated keys.
            v: ``[num_workers, num_kv_heads * head_dim]``.
            batch.positions: block-relative write positions per worker.
            batch.mrope_positions: ``[3, num_workers]`` block-relative mRoPE
                positions, when the write does not sit at ``batch.positions`` on
                all three axes (an image among the new tokens, or a write block
                whose mRoPE span already differs from its token count).

        Returns ``[num_workers, num_qo_heads * head_dim]``.
        """
        meta = batch.attn_metadata
        assert isinstance(meta, SharedCacheAttnMetadata)
        W, Hq, D = meta.num_workers, self.num_qo_heads, self.head_dim

        # Store the new token's KV with block-relative key rotation.
        key_pos = batch.positions if batch.mrope_positions is None else batch.mrope_positions
        k_rot = self._rope(k.reshape(W, self.num_kv_heads, D), key_pos)
        # k_rot is freshly materialized (contiguous); the store kernel needs
        # v in the same layout, so detach v from its strided qkv slice too.
        self.kv_cache.store_kv(k_rot.reshape(W, -1), v.contiguous(), batch.out_loc, layer_id)

        # One query copy per (row, segment), rotated to its segment-relative position.
        q_sub = q.reshape(W, Hq, D)[meta.sub_worker]
        q_sub = self._rope(q_sub, meta.sub_loc)

        if meta.phase == "prefill_batch":
            return self._forward_prefill_batch(q_sub, meta, layer_id)

        # Paged views of the KV pool: [num_pages, page_size, n_kv_heads, head_dim].
        k_paged = self.kv_cache.k_cache(layer_id)
        v_paged = self.kv_cache.v_cache(layer_id)

        outs: List[torch.Tensor] = []
        lses: List[torch.Tensor] = []
        main_wrapper = self.wrapper if meta.graph_buffers is None else meta.graph_buffers.main
        aux_wrapper = self.aux_wrapper if meta.graph_buffers is None else meta.graph_buffers.aux
        if meta.n_main > 0:
            out_m, lse_m = main_wrapper.run(
                q=q_sub[: meta.n_main], paged_kv_cache=(k_paged, v_paged), return_lse=True
            )
            outs.append(out_m)
            lses.append(lse_m)
        if meta.n_aux > 0:
            # The single new-token self-segment lives at an arbitrary page
            # offset -> read it through the flattened (page_size=1) pool.
            kflat = k_paged.view(-1, 1, self.num_kv_heads, D)
            vflat = v_paged.view(-1, 1, self.num_kv_heads, D)
            out_a, lse_a = aux_wrapper.run(
                q=q_sub[meta.n_main :], paged_kv_cache=(kflat, vflat), return_lse=True
            )
            outs.append(out_a)
            lses.append(lse_a)

        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        lse = lses[0] if len(lses) == 1 else torch.cat(lses, dim=0)
        if meta.graph_buffers is not None:
            mask = meta.graph_buffers.valid
            out = torch.where(mask[:, None, None], out, 0)
            lse = torch.where(mask[:, None], lse, -5.0e4)

        if meta.max_segments == 1 and meta.n_aux == 0:
            # Every worker has exactly one paged segment, emitted in worker order.
            return out.view(W, -1)

        from flashinfer import merge_states

        M = meta.max_segments
        v_pad = q.new_zeros(W * M, Hq, D)
        # Zero-weight filler; see ``_forward_prefill_batch`` for why this value.
        s_pad = torch.full((W * M, Hq), -5.0e4, dtype=torch.float32, device=self.device)
        v_pad[meta.pad_slot] = out
        s_pad[meta.pad_slot] = lse
        merged, _ = merge_states(v_pad.view(W, M, Hq, D), s_pad.view(W, M, Hq))
        return merged.view(W, -1)

    def _forward_prefill_batch(
        self, q_sub: torch.Tensor, meta: SharedCacheAttnMetadata, layer_id: int
    ) -> torch.Tensor:
        """Run the planned prefill wrappers over a whole batch of requests and
        merge each token row's segments.

        *q_sub*: ``[n_ctx_rows + N, Hq, D]`` rotated queries — every request's
        context segments first (request-major), then every request's causal self
        segment.  Returns ``[N, Hq * D]``, one row per new token of the batch.
        """
        N, Hq, D = meta.num_workers, self.num_qo_heads, self.head_dim
        n_ctx_rows = meta.n_ctx_rows

        kv = (self.kv_cache.k_cache(layer_id), self.kv_cache.v_cache(layer_id))
        buffers = meta.graph_buffers
        self_wrapper = self.prefill_self_wrapper if buffers is None else buffers.aux
        ctx_wrapper = self.prefill_ctx_wrapper if buffers is None else buffers.main
        out_self, lse_self = (self_wrapper.run(q_sub[n_ctx_rows:], kv, return_lse=True)
                             if buffers is None else buffers.run(
                                 self_wrapper, q_sub[n_ctx_rows:], kv, buffers.used_self_rows))
        if n_ctx_rows == 0:
            # No request has context: the causal self segment is the whole answer.
            return out_self.view(N, -1)
        out_ctx, lse_ctx = (ctx_wrapper.run(q_sub[:n_ctx_rows], kv, return_lse=True)
                           if buffers is None else buffers.run(
                               ctx_wrapper, q_sub[:n_ctx_rows], kv, buffers.used_ctx_rows))

        from flashinfer import merge_states

        # Requests may have different segment counts, so scatter into a padded
        # [N, max_segments] grid (as decode does) instead of a fixed reshape.
        M = meta.max_segments
        v_pad = q_sub.new_zeros(N * M, Hq, D)
        # Rows with fewer than M segments leave slots unwritten; they must carry
        # no weight, and the merge's ``exp(s - s_max)`` underflows to an exact 0
        # this far down.  -5e4 rather than -inf: it is what FlashInfer's own merge
        # kernel seeds its accumulator with (``triton/kernels/cascade.py``), and
        # staying finite keeps an all-padding row from going NaN.
        s_pad = torch.full((N * M, Hq), -5.0e4, dtype=torch.float32, device=self.device)
        out = torch.cat([out_ctx, out_self], dim=0)
        lse = torch.cat([lse_ctx, lse_self], dim=0)
        if buffers is not None:
            out = torch.where(buffers.valid[:, None, None], out, 0)
            lse = torch.where(buffers.valid[:, None], lse, -5.0e4)
        v_pad[meta.pad_slot] = out
        s_pad[meta.pad_slot] = lse
        merged, _ = merge_states(v_pad.view(N, M, Hq, D), s_pad.view(N, M, Hq))
        return merged.view(N, -1)
