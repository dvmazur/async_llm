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
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
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
    n_main: int = 0  # sub-requests served by the paged wrapper
    n_aux: int = 0  # sub-requests served by the page_size=1 (implicit-self) wrapper
    phase: Literal["decode", "context_prefill"] = "decode"
    _plan_refs: tuple = field(default=(), repr=False)

    def get_last_indices(self, bs: int) -> torch.Tensor:
        if self.phase == "context_prefill":
            # one request of num_workers(=S) tokens; the LM head wants the last
            return torch.tensor([self.num_workers - 1], device=self.sub_worker.device)
        return torch.arange(bs, device=self.sub_worker.device)


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
    ) -> None:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

        self.kv_cache = kv_cache
        self.cos_sin_cache = cos_sin_cache
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # Partial RoPE (Qwen3.5): rotate only the first `rotary_dim` head dims,
        # pass the rest through.  Defaults to full-head RoPE (standard models).
        self.rotary_dim = rotary_dim if rotary_dim is not None else head_dim
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

    def _rope(self, x: torch.Tensor, corrections: torch.Tensor) -> torch.Tensor:
        """Block-relative RoPE correction on ``x [N, heads, head_dim]``.

        For partial RoPE only the first ``rotary_dim`` dims are rotated; the tail
        is passed through unchanged (Qwen3.5).  ``cos_sin_cache`` is sized
        ``[max_pos, rotary_dim]`` in the partial case.
        """
        if self.rotary_dim == self.head_dim:
            return apply_rope_correction(x, corrections, self.cos_sin_cache)
        x_rot = apply_rope_correction(
            x[..., : self.rotary_dim].contiguous(), corrections, self.cos_sin_cache
        )
        return torch.cat([x_rot, x[..., self.rotary_dim :]], dim=-1)

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
        num_workers = group.num_workers
        assert num_workers > 0
        P = self.page_size
        write_set = set(new_page_for_block.keys())

        # Main (paged) sub-requests.
        main_sub_worker: List[int] = []
        main_sub_loc: List[int] = []
        main_sub_slot: List[int] = []
        main_kv_parts: List[torch.Tensor] = []  # page-number tensors
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
                main_sub_worker.append(w)
                main_sub_loc.append(total - prefix - 1)
                main_sub_slot.append(n_seg)
                main_kv_parts.append(page_nums)
                main_page_counts.append(n_pages_seg)
                main_seq_lens.append(length)
                main_last_page.append(length - (n_pages_seg - 1) * P)
                n_seg += 1
                prefix += length

            if not self_in_view:
                # The query must still attend to itself: a single-token segment
                # at distance 0 (query rotated to its own block-relative pos).
                aux_sub_worker.append(w)
                aux_sub_loc.append(wt.num_tokens)
                aux_sub_slot.append(n_seg)
                aux_kv_slots.append(int(new_token_slots[w].item()))
                n_seg += 1

            max_segments = max(max_segments, n_seg)

        n_main = len(main_sub_worker)
        n_aux = len(aux_sub_worker)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        plan_refs: List[torch.Tensor] = []

        self._plan_event.synchronize()
        if n_main > 0:
            kv_indices = torch.cat(main_kv_parts).to(dtype=torch.int32)
            kv_indptr_cpu = torch.tensor([0] + main_page_counts, **CPU_KWARGS).cumsum_(0).to(
                torch.int32
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
            aux_indices = torch.tensor(aux_kv_slots, dtype=torch.int32, device=self.device)
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
        P = self.page_size
        S = int(num_new)
        n_ctx = len(context)
        ctx_lens = [b.num_tokens for b in context]
        self_offset = sum(ctx_lens)

        # q-row gather + per-row rotation positions: for context segment j at
        # view offset O_j, token i is rotated to (O_self + i) - O_j; for the
        # self segment, to its block-relative position i.
        sub_gather: List[int] = []
        sub_loc: List[int] = []
        prefix = 0
        for length in ctx_lens:
            sub_gather.extend(range(S))
            sub_loc.extend(self_offset + i - prefix for i in range(S))
            prefix += length
        sub_gather.extend(range(S))
        sub_loc.extend(range(S))

        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        # Context segments (non-causal): paged page-number indices + last_page_len.
        ctx_kv_indices = torch.cat([b.page_numbers_tensor() for b in context]).to(torch.int32)
        ctx_kv_indptr = torch.tensor(
            [0] + [b.num_pages for b in context], **CPU_KWARGS
        ).cumsum_(0).to(torch.int32)
        ctx_qo_indptr = torch.arange(0, (n_ctx + 1) * S, S, **CPU_KWARGS)
        ctx_seq_lens = torch.tensor(ctx_lens, **CPU_KWARGS)
        ctx_last_page = torch.tensor([b.last_page_len for b in context], **CPU_KWARGS)

        # Self segment (causal): the new block's freshly-allocated pages.
        self_kv_indices = (new_page_starts.to(self.device) // P).to(torch.int32)
        n_self_pages = int(self_kv_indices.numel())
        self_indptr = torch.tensor([0, n_self_pages], **CPU_KWARGS)
        self_qo_indptr = torch.tensor([0, S], **CPU_KWARGS)
        self_seq_lens = torch.tensor([S], **CPU_KWARGS)
        self_last_page = torch.tensor([S - (n_self_pages - 1) * P], **CPU_KWARGS)

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
        self._plan_event.synchronize()
        # context segments are entirely in the new tokens' past -> no masking
        self.prefill_ctx_wrapper.plan(
            qo_indptr=ctx_qo_indptr,
            paged_kv_indptr=ctx_kv_indptr,
            paged_kv_indices=ctx_kv_indices,
            paged_kv_last_page_len=ctx_last_page,
            seq_lens=ctx_seq_lens,
            causal=False,
            **plan_common,
        )
        # the new block attends to itself causally (standard prefill)
        self.prefill_self_wrapper.plan(
            qo_indptr=self_qo_indptr,
            paged_kv_indptr=self_indptr,
            paged_kv_indices=self_kv_indices,
            paged_kv_last_page_len=self_last_page,
            seq_lens=self_seq_lens,
            causal=True,
            **plan_common,
        )
        self._plan_event.record()

        meta = SharedCacheAttnMetadata(
            shared_cache_op=self,
            sub_worker=torch.tensor(sub_gather, dtype=torch.int64, device=self.device),
            sub_loc=torch.tensor(sub_loc, dtype=torch.int64, device=self.device),
            pad_slot=torch.empty(0, dtype=torch.int64, device=self.device),  # unused
            num_workers=S,
            max_segments=n_ctx + 1,
            phase="context_prefill",
        )
        meta._plan_refs = (
            ctx_kv_indptr, ctx_qo_indptr, ctx_seq_lens, ctx_last_page, ctx_kv_indices,
            self_indptr, self_qo_indptr, self_seq_lens, self_last_page, self_kv_indices,
        )
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

        Returns ``[num_workers, num_qo_heads * head_dim]``.
        """
        meta = batch.attn_metadata
        assert isinstance(meta, SharedCacheAttnMetadata)
        W, Hq, D = meta.num_workers, self.num_qo_heads, self.head_dim

        # Store the new token's KV with block-relative key rotation.
        k_rot = self._rope(k.reshape(W, self.num_kv_heads, D), batch.positions)
        # k_rot is freshly materialized (contiguous); the store kernel needs
        # v in the same layout, so detach v from its strided qkv slice too.
        self.kv_cache.store_kv(k_rot.reshape(W, -1), v.contiguous(), batch.out_loc, layer_id)

        # One query copy per (row, segment), rotated to its segment-relative position.
        q_sub = q.reshape(W, Hq, D)[meta.sub_worker]
        q_sub = self._rope(q_sub, meta.sub_loc)

        if meta.phase == "context_prefill":
            return self._forward_context_prefill(q_sub, meta, layer_id)

        # Paged views of the KV pool: [num_pages, page_size, n_kv_heads, head_dim].
        k_paged = self.kv_cache.k_cache(layer_id)
        v_paged = self.kv_cache.v_cache(layer_id)

        outs: List[torch.Tensor] = []
        lses: List[torch.Tensor] = []
        if meta.n_main > 0:
            out_m, lse_m = self.wrapper.run(
                q=q_sub[: meta.n_main], paged_kv_cache=(k_paged, v_paged), return_lse=True
            )
            outs.append(out_m)
            lses.append(lse_m)
        if meta.n_aux > 0:
            # The single new-token self-segment lives at an arbitrary page
            # offset -> read it through the flattened (page_size=1) pool.
            kflat = k_paged.view(-1, 1, self.num_kv_heads, D)
            vflat = v_paged.view(-1, 1, self.num_kv_heads, D)
            out_a, lse_a = self.aux_wrapper.run(
                q=q_sub[meta.n_main :], paged_kv_cache=(kflat, vflat), return_lse=True
            )
            outs.append(out_a)
            lses.append(lse_a)

        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        lse = lses[0] if len(lses) == 1 else torch.cat(lses, dim=0)

        if meta.max_segments == 1 and meta.n_aux == 0:
            # Every worker has exactly one paged segment, emitted in worker order.
            return out.view(W, -1)

        from flashinfer import merge_states

        M = meta.max_segments
        v_pad = q.new_zeros(W * M, Hq, D)
        # finite "minus infinity": exp(pad - max) underflows to 0 for any real lse
        s_pad = torch.full((W * M, Hq), -5.0e4, dtype=torch.float32, device=self.device)
        v_pad[meta.pad_slot] = out
        s_pad[meta.pad_slot] = lse
        merged, _ = merge_states(v_pad.view(W, M, Hq, D), s_pad.view(W, M, Hq))
        return merged.view(W, -1)

    def _forward_context_prefill(
        self, q_sub: torch.Tensor, meta: SharedCacheAttnMetadata, layer_id: int
    ) -> torch.Tensor:
        """Run the two planned prefill wrappers and merge per token.

        *q_sub*: ``[(n_ctx + 1) * S, Hq, D]`` rotated queries, context segments
        first, the causal self segment last.
        """
        from flashinfer import merge_states

        S, Hq, D = meta.num_workers, self.num_qo_heads, self.head_dim
        n_ctx = meta.max_segments - 1

        kv = (self.kv_cache.k_cache(layer_id), self.kv_cache.v_cache(layer_id))
        out_ctx, lse_ctx = self.prefill_ctx_wrapper.run(
            q_sub[: n_ctx * S], kv, return_lse=True
        )
        out_self, lse_self = self.prefill_self_wrapper.run(
            q_sub[n_ctx * S :], kv, return_lse=True
        )

        v_states = torch.cat([out_ctx.view(n_ctx, S, Hq, D), out_self.view(1, S, Hq, D)])
        s_states = torch.cat([lse_ctx.view(n_ctx, S, Hq), lse_self.view(1, S, Hq)])
        merged, _ = merge_states(
            v_states.permute(1, 0, 2, 3).contiguous(),  # [S, n_ctx + 1, Hq, D]
            s_states.permute(1, 0, 2).contiguous(),  # [S, n_ctx + 1, Hq]
        )
        result = merged.view(S, -1)

        return result
