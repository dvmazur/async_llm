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

Per-segment attention runs through a FlashInfer paged decode wrapper with
``return_lse=True``; partial outputs are merged per worker with
``flashinfer.merge_states``.  No intra-segment masking is needed in decode:
non-self segments lie entirely in the query's past, and the self segment's
newest key *is* the query token.

Following the reference implementation, all workers' new KV entries are
written to the cache *before* attention runs, and segment lengths are counted
post-append — so a worker reading another worker's write block sees that
worker's current-step token as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

import torch
from minisgl.attention import BaseAttnMetadata

from .rope_correction import apply_rope_correction

if TYPE_CHECKING:
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from minisgl.core import Batch
    from minisgl.kvcache import BaseKVCachePool

    from .worker_group import WorkerGroup


@dataclass
class SharedCacheAttnMetadata(BaseAttnMetadata):
    # ``AttentionLayer.forward`` duck-types on this attribute to divert the
    # batch away from the regular attention backend.
    shared_cache_op: SharedCacheAttention
    # Per output q-row gather/rotation plan.  For decode: one row per
    # (worker, segment) sub-request.  For context prefill: S rows (one per new
    # token) per sub-request.
    sub_worker: torch.Tensor  # int64 — source q-row per sub-request row
    sub_loc: torch.Tensor  # int64 — query RoPE position per sub-request row
    pad_slot: torch.Tensor  # [N_sub] int64 — scatter index into [W * max_segments]
    num_workers: int  # number of output rows (workers for decode, tokens for prefill)
    max_segments: int
    phase: Literal["decode", "context_prefill"] = "decode"

    def get_last_indices(self, bs: int) -> torch.Tensor:
        if self.phase == "context_prefill":
            # one request of num_workers(=S) tokens; the LM head wants the last
            return torch.tensor([self.num_workers - 1], device=self.sub_worker.device)
        return torch.arange(bs, device=self.sub_worker.device)


class SharedCacheAttention:
    """Owns the FlashInfer wrapper and runs shared-cache decode attention."""

    def __init__(
        self,
        kv_cache: BaseKVCachePool,
        cos_sin_cache: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

        self.kv_cache = kv_cache
        self.cos_sin_cache = cos_sin_cache
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self._workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        gqa = num_qo_heads // num_kv_heads
        self.wrapper: BatchDecodeWithPagedKVCacheWrapper = BatchDecodeWithPagedKVCacheWrapper(
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

    def prepare(self, group: WorkerGroup, new_pages: torch.Tensor) -> SharedCacheAttnMetadata:
        """
        Build per-(worker, segment) sub-requests for one decode step and plan
        the FlashInfer wrapper.  *new_pages* is the ``[num_workers]`` tensor of
        freshly allocated KV slots for this step's tokens.
        """
        num_workers = group.num_workers
        assert num_workers > 0

        # block id -> index of the worker writing to it this step
        write_map = {id(b): wi for wi, b in enumerate(group.write_to)}

        sub_worker: List[int] = []
        sub_loc: List[int] = []
        sub_slot: List[int] = []  # (worker, segment-within-worker) pairs
        kv_lens: List[int] = []
        kv_parts: List[torch.Tensor] = []
        max_segments = 0

        for w in range(num_workers):
            view = group.cache_structure[w]
            wt = group.write_to[w]
            # post-append segment lengths: every block written this step grows by 1
            lengths = [b.num_tokens + (1 if id(b) in write_map else 0) for b in view]
            total = sum(lengths)
            self_in_view = any(b is wt for b in view)

            n_seg = 0
            prefix = 0
            for b, length in zip(view, lengths):
                if length == 0:
                    continue
                pages = b.get_page_indices()
                if id(b) in write_map:
                    new_page = new_pages[write_map[id(b)] : write_map[id(b)] + 1]
                    pages = torch.cat([pages, new_page]) if b.num_tokens else new_page
                sub_worker.append(w)
                sub_loc.append(total - prefix - 1)
                sub_slot.append(n_seg)
                kv_lens.append(length)
                kv_parts.append(pages)
                n_seg += 1
                prefix += length

            if not self_in_view:
                # The query must still attend to itself: add an implicit
                # single-token segment so self-attention has distance 0.
                sub_worker.append(w)
                sub_loc.append(wt.num_tokens)
                sub_slot.append(n_seg)
                kv_lens.append(1)
                kv_parts.append(new_pages[w : w + 1])
                n_seg += 1

            max_segments = max(max_segments, n_seg)

        n_sub = len(sub_worker)
        kv_indices = torch.cat(kv_parts).to(dtype=torch.int32)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        kv_indptr_cpu = torch.tensor([0] + kv_lens, **CPU_KWARGS).cumsum_(dim=0).to(torch.int32)
        seq_lens_cpu = torch.tensor(kv_lens, **CPU_KWARGS)
        last_page_len_cpu = torch.ones(n_sub, **CPU_KWARGS)

        self._plan_event.synchronize()
        self.wrapper.plan(
            indptr=kv_indptr_cpu,
            indices=kv_indices,
            last_page_len=last_page_len_cpu,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens=seq_lens_cpu,
            data_type=self.dtype,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            non_blocking=True,
        )
        self._plan_event.record()

        pad_slot = [w * max_segments + s for w, s in zip(sub_worker, sub_slot)]
        meta = SharedCacheAttnMetadata(
            shared_cache_op=self,
            sub_worker=torch.tensor(sub_worker, dtype=torch.int64, device=self.device),
            sub_loc=torch.tensor(sub_loc, dtype=torch.int64, device=self.device),
            pad_slot=torch.tensor(pad_slot, dtype=torch.int64, device=self.device),
            num_workers=num_workers,
            max_segments=max_segments,
        )
        # Keep plan inputs alive through the forward: the async plan copy reads
        # the pinned tensors after prepare() returns.
        meta._plan_refs = (kv_indptr_cpu, seq_lens_cpu, last_page_len_cpu, kv_indices)
        return meta

    def prepare_context_prefill(
        self, context: List, new_pages: torch.Tensor
    ) -> SharedCacheAttnMetadata:
        """
        Plan one prefill of S new tokens (stored block-relative at 0..S-1 in
        *new_pages*) that attend causally to themselves and fully to each
        *context* block, as if the blocks were concatenated:
        ``[ctx_0, ..., ctx_{n-1}, new]``.

        Mirrors the reference's ``prefill_cache_block(text, [ctx..., new])``.
        """
        assert context and all(b.num_tokens > 0 for b in context)
        S = int(new_pages.numel())
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
        ctx_kv_parts = [b.get_page_indices() for b in context]
        ctx_kv_indices = torch.cat(ctx_kv_parts).to(dtype=torch.int32)
        ctx_kv_indptr = torch.tensor([0] + ctx_lens, **CPU_KWARGS).cumsum_(dim=0).to(torch.int32)
        ctx_qo_indptr = torch.arange(0, (n_ctx + 1) * S, S, **CPU_KWARGS)
        ctx_seq_lens = torch.tensor(ctx_lens, **CPU_KWARGS)
        ctx_last_page = torch.ones(n_ctx, **CPU_KWARGS)
        self_kv_indices = new_pages.to(dtype=torch.int32)
        self_indptr = torch.tensor([0, S], **CPU_KWARGS)
        self_seq_lens = torch.tensor([S], **CPU_KWARGS)
        self_last_page = torch.ones(1, **CPU_KWARGS)

        plan_common = dict(
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=1,
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
            qo_indptr=self_indptr,
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
            self_indptr, self_seq_lens, self_last_page, self_kv_indices,
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
        k_rot = apply_rope_correction(
            k.reshape(W, self.num_kv_heads, D), batch.positions, self.cos_sin_cache
        )
        # k_rot is freshly materialized (contiguous); the store kernel needs
        # v in the same layout, so detach v from its strided qkv slice too.
        self.kv_cache.store_kv(k_rot.view(W, -1), v.contiguous(), batch.out_loc, layer_id)

        # One query copy per (row, segment), rotated to its segment-relative position.
        q_sub = q.reshape(W, Hq, D)[meta.sub_worker]
        q_sub = apply_rope_correction(q_sub, meta.sub_loc, self.cos_sin_cache)

        if meta.phase == "context_prefill":
            return self._forward_context_prefill(q_sub, meta, layer_id)

        def _paged(cache: torch.Tensor) -> torch.Tensor:  # page_size = 1
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        kv = (_paged(self.kv_cache.k_cache(layer_id)), _paged(self.kv_cache.v_cache(layer_id)))
        out, lse = self.wrapper.run(q=q_sub, paged_kv_cache=kv, return_lse=True)

        if meta.max_segments == 1:
            result = out.view(W, -1)
        else:
            from flashinfer import merge_states

            M = meta.max_segments
            v_pad = q.new_zeros(W * M, Hq, D)
            # finite "minus infinity": exp(pad - max) underflows to 0 for any real lse
            s_pad = torch.full((W * M, Hq), -5.0e4, dtype=torch.float32, device=self.device)
            v_pad[meta.pad_slot] = out
            s_pad[meta.pad_slot] = lse
            merged, _ = merge_states(v_pad.view(W, M, Hq, D), s_pad.view(W, M, Hq))
            result = merged.view(W, -1)

        return result

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

        def _paged(cache: torch.Tensor) -> torch.Tensor:  # page_size = 1
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        kv = (_paged(self.kv_cache.k_cache(layer_id)), _paged(self.kv_cache.v_cache(layer_id)))
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
