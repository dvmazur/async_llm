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

from dataclasses import dataclass, field, replace
from itertools import zip_longest
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import div_ceil

from .attention import PrefillSpec, SharedCacheAttention
from .gdn import SharedCacheGDN
from .gdn_affine import compose_gdn_affines
from .shared_block import NULL_CACHE_HANDLE, CacheBlock
from .worker_group import WorkerGroup

if TYPE_CHECKING:
    from minisgl.engine import Engine


@dataclass
class PrefillJob:
    """One prefill to run inside a :meth:`SharedCacheSession.prefill_batch`.

    The fields mirror :meth:`SharedCacheSession.prefill_block`'s arguments:
    *input_ids* are appended to *block* (a non-empty block is extended) and
    attend to *context* in view order.
    """

    block: CacheBlock
    input_ids: torch.Tensor
    context: List[CacheBlock] = field(default_factory=list)
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    mm_token_type_ids: Optional[torch.Tensor] = None
    # Set by the chunked-prefill driver only (see ``_chunk_job``).
    image_embeds: Optional[torch.Tensor] = None  # tower output for this chunk
    mrope_rel: Optional[torch.Tensor] = None  # positions rel. to the block's span
    mrope_span: Optional[int] = None  # the block's span once this chunk lands


@dataclass
class PrefillPlan:
    """Whole-request state a chunked prefill slices per chunk.

    ``rel_pos`` and ``embeds`` are computed once: ``get_rope_index`` needs the whole
    sequence, and the vision tower needs whole images.
    """

    ids: torch.Tensor  # [L] int32 cpu
    types: Optional[torch.Tensor] = None  # [L] int64 cpu; None for text-only
    rel_pos: Optional[torch.Tensor] = None  # [3, L] int64 cpu
    embeds: Optional[torch.Tensor] = None  # [n_img_tokens, hidden]
    base_span: int = 0  # the write block's mRoPE span before the first chunk


def _job_rows(job: PrefillJob) -> int:
    """Query rows a job costs: one q copy per (token, segment)."""
    n_ctx = len([b for b in (job.context or []) if b.num_tokens > 0])
    return (n_ctx + 1) * job.input_ids.numel()


def _chunk_job(job: PrefillJob, plan: PrefillPlan, start: int, stop: int) -> PrefillJob:
    """Tokens ``[start, stop)`` of *job* as a standalone prefill of its own.

    Image tokens split like any others: the chunk carries the matching rows of the
    already-computed tower output, so no boundary is special.
    """
    ids = plan.ids[start:stop]
    if plan.types is None:
        return replace(job, input_ids=ids)
    types = plan.types[start:stop]
    n_before = int((plan.types[:start] == 1).sum())
    n_here = int((types == 1).sum())
    # Image positions cycle rather than grow with token index, so a block's span is a
    # cumulative max -- not derivable from one chunk's own positions.
    shift = int(plan.rel_pos[:, :start].max()) + 1 if start else 0
    return replace(
        job,
        input_ids=ids,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=types,
        image_embeds=plan.embeds[n_before : n_before + n_here],
        mrope_rel=plan.rel_pos[:, start:stop] - shift,
        mrope_span=plan.base_span + int(plan.rel_pos[:, :stop].max()) + 1,
    )


def extract_cos_sin_cache(engine: Engine) -> torch.Tensor:
    """
    Extract the ``cos_sin_cache`` tensor from the model's ``RotaryEmbedding``.

    Works for Llama / Qwen / Mistral model families in mini-sglang.
    """
    layers = engine.model.model.layers.op_list
    return layers[0].self_attn.attn.rotary._cos_sin_cache


def _is_hybrid_model(engine: Engine) -> bool:
    layers = engine.model.model.layers.op_list
    return any(getattr(layer, "_is_linear", False) for layer in layers)


def _first_full_attn(engine: Engine):
    """The first full-attention module (``Qwen3_5Attention``) of a hybrid model."""
    for layer in engine.model.model.layers.op_list:
        if not getattr(layer, "_is_linear", True):
            return layer.self_attn
    raise RuntimeError("hybrid model has no full-attention layer")


def _first_gdn(engine: Engine):
    """The first Gated-DeltaNet module (``Qwen3_5GatedDeltaNet``) of a hybrid model."""
    for layer in engine.model.model.layers.op_list:
        if getattr(layer, "_is_linear", False):
            return layer.linear_attn
    raise RuntimeError("hybrid model has no linear-attention layer")


def _build_partial_cos_sin_cache(
    base: float, rotary_dim: int, max_pos: int, device: torch.device
) -> torch.Tensor:
    """``[max_pos, rotary_dim]`` cos|sin cache for partial-RoPE (Qwen3.5), in the
    ``minisgl.layers.rotary`` format (first half cos, second half sin)."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


class SharedCacheSession:
    """
    Manages ``CacheBlock`` pages and drives batched forward passes for a
    ``WorkerGroup`` on a mini-sglang ``Engine``.

    Pages are **borrowed from the engine's main page cache**
    (``engine.page_allocator``) rather than from a private pool, so allocations
    are page-aligned and drawn from the same physical pages the engine owns.
    The session is designed for **standalone** use, bypassing the scheduler; to
    run alongside the live scheduler the two consumers would need to share one
    allocator instance.

    Set ``max_prefill_rows`` to chunk long prefills over several forwards, bounding
    peak activation memory; every prefill entry point honours it, including
    ``AsyncLLM.forward`` (``llm.async_engine.session.max_prefill_rows = n``).
    """

    def __init__(
        self,
        engine: Engine,
        cos_sin_cache: Optional[torch.Tensor] = None,
        max_prefill_rows: Optional[int] = None,
    ):
        # Chunked prefill: cap on query rows per forward (see ``_job_rows``); None
        # disables it.  Mutable, so callers can set it after construction.
        self.max_prefill_rows = max_prefill_rows
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

        self._is_hybrid = _is_hybrid_model(engine)
        # ModelConfig of the loaded model (Qwen3.5 exposes it on the inner module);
        # only used for the vision tower's spatial_merge_size.
        self._model_config = getattr(engine.model.model, "config", None)

        # Query-rotation attention op for decode (arXiv:2512.10931).  Hybrid
        # (Qwen3.5) models use partial RoPE and a custom full-attention module;
        # standard models keep the AttentionLayer with a full-head cos/sin cache.
        if self._is_hybrid:
            attn0 = _first_full_attn(engine)  # Qwen3_5Attention
            rotary_dim = attn0.rotary_dim
            if cos_sin_cache is not None:
                self._cos_sin_cache = cos_sin_cache.to(self.device)
            else:
                self._cos_sin_cache = _build_partial_cos_sin_cache(
                    base=attn0._rope_base,
                    rotary_dim=rotary_dim,
                    max_pos=engine.max_seq_len,
                    device=self.device,
                )
            num_qo_heads = attn0.num_qo_heads
            num_kv_heads = attn0.num_kv_heads
            head_dim = attn0.head_dim
            # Interleaved mRoPE in the shared-cache op for Qwen3.5 (needed so AR decode
            # attends correctly to image keys in the prompt; reduces to 1D for text).
            sc_mrope = attn0._mrope_section
            sc_rope_base = attn0._rope_base
            gdn0 = _first_gdn(engine)  # Qwen3_5GatedDeltaNet
            self.sc_gdn: SharedCacheGDN | None = SharedCacheGDN(
                num_heads=gdn0.num_v_heads,
                head_k_dim=gdn0.head_k_dim,
                head_v_dim=gdn0.head_v_dim,
                conv_dim=gdn0.conv_dim,
                conv_kernel=gdn0.conv_kernel,
                device=self.device,
            )
        else:
            attn0 = engine.model.model.layers.op_list[0].self_attn.attn
            if cos_sin_cache is not None:
                self._cos_sin_cache = cos_sin_cache.to(self.device)
            else:
                self._cos_sin_cache = extract_cos_sin_cache(engine).to(self.device)
            rotary_dim = attn0.head_dim
            num_qo_heads = attn0.num_qo_heads
            num_kv_heads = attn0.num_kv_heads
            head_dim = attn0.head_dim
            sc_mrope = None
            sc_rope_base = None
            self.sc_gdn = None

        self.sc_attn = SharedCacheAttention(
            kv_cache=self.kv_cache,
            cos_sin_cache=self._cos_sin_cache,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=self.page_size,
            dtype=self.kv_cache.dtype,
            device=self.device,
            rotary_dim=rotary_dim,
            mrope_section=sc_mrope,
            rope_base=sc_rope_base,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_block(self) -> CacheBlock:
        return CacheBlock(self.device, page_size=self.page_size)

    def free_block(self, block: CacheBlock) -> None:
        """Return a block's pages to the engine's page allocator and reset it."""
        self._validate_block(block)
        page_starts = block.clear()
        if page_starts:
            self.page_allocator.free_pages(
                torch.tensor(page_starts, dtype=torch.int32, device=self.device)
            )

    @torch.inference_mode()
    def merge_blocks(
        self,
        left: CacheBlock,
        right: CacheBlock,
    ) -> CacheBlock:
        """Copy ``left + right`` into a new compact block.

        Both input blocks remain valid and unchanged.  The result owns fresh
        pages, so its lifetime is independent from either input.
        """
        self._validate_block(left)
        self._validate_block(right)

        left_tokens = left.num_tokens
        right_tokens = right.num_tokens
        left_span = left.mrope_span
        right_span = right.mrope_span
        total_tokens = left_tokens + right_tokens

        merged = self.create_block()
        num_pages = div_ceil(total_tokens, self.page_size)
        if num_pages:
            merged.page_starts = self.page_allocator.alloc_pages(num_pages).tolist()
        merged.num_tokens = total_tokens

        if left_tokens:
            self._copy_block(left, merged, destination_start=0)
        if right_tokens:
            self._copy_block(right, merged, destination_start=left_tokens)
        if right_tokens and left_span:
            self._shift_keys_(
                block=merged,
                start=left_tokens,
                num_tokens=right_tokens,
                position_shift=left_span,
            )

        self._finish_block_merge(
            destination=merged,
            left=left,
            right=right,
            left_span=left_span,
            right_span=right_span,
            keep_left_state=False,
        )
        return merged

    @torch.inference_mode()
    def append_block(self, left: CacheBlock, right: CacheBlock) -> CacheBlock:
        """Append a copy of ``right`` to ``left`` and return ``left``.

        Free slots in the last page of ``left`` are filled first; only the
        remaining pages are allocated.  A distinct ``right`` remains valid and
        unchanged; ``append_block(block, block)`` duplicates ``block`` in place.
        """
        self._validate_block(left)
        self._validate_block(right)

        left_tokens = left.num_tokens
        right_tokens = right.num_tokens
        left_span = left.mrope_span
        right_span = right.mrope_span

        if right_tokens:
            # Capture the source layout before growing ``left``.  This matters
            # when both arguments are the same object: grow_pages changes its
            # num_tokens and therefore the slots returned by token_slots_tensor.
            source_slots = right.token_slots_tensor().to(torch.int64)
            new_pages = self.page_allocator.alloc_pages(left.pages_needed(right_tokens))
            left.grow_pages(new_pages, right_tokens)
            self._copy_slots(source_slots, left, destination_start=left_tokens)
            if left_span:
                self._shift_keys_(
                    block=left,
                    start=left_tokens,
                    num_tokens=right_tokens,
                    position_shift=left_span,
                )

        self._finish_block_merge(
            destination=left,
            left=left,
            right=right,
            left_span=left_span,
            right_span=right_span,
            keep_left_state=True,
        )
        return left

    def _finish_block_merge(
        self,
        *,
        destination: CacheBlock,
        left: CacheBlock,
        right: CacheBlock,
        left_span: int,
        right_span: int,
        keep_left_state: bool,
    ) -> None:
        """Merge token metadata and recurrent state after KV has been copied."""
        left_token_ids = list(left.token_ids)
        destination.token_ids = [*left_token_ids, *right.token_ids]
        merged_span = left_span + right_span
        destination.mrope_span_override = (
            None if merged_span == destination.num_tokens else merged_span
        )

        merged_affine: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        affine_layers = set(left.linear_affine) | set(right.linear_affine)
        for layer_idx in affine_layers:
            left_pair = left.linear_affine.get(layer_idx)
            right_pair = right.linear_affine.get(layer_idx)
            if left_pair is None:
                assert right_pair is not None
                merged_affine[layer_idx] = (right_pair[0].clone(), right_pair[1].clone())
            elif right_pair is None:
                merged_affine[layer_idx] = (
                    left_pair
                    if keep_left_state
                    else (left_pair[0].clone(), left_pair[1].clone())
                )
            else:
                merged_affine[layer_idx] = compose_gdn_affines(
                    A_first=left_pair[0],
                    B_first=left_pair[1],
                    A_second=right_pair[0],
                    B_second=right_pair[1],
                )
        destination.linear_affine = merged_affine

        merged_conv: Dict[int, torch.Tensor] = {}
        conv_layers = set(left.linear_conv_state) | set(right.linear_conv_state)
        for layer_idx in conv_layers:
            right_state = right.linear_conv_state.get(layer_idx)
            left_state = left.linear_conv_state.get(layer_idx)
            state = right_state if right_state is not None else left_state
            assert state is not None
            merged_conv[layer_idx] = (
                state if keep_left_state and right_state is None else state.clone()
            )
        destination.linear_conv_state = merged_conv

    def _validate_block(self, block: CacheBlock) -> None:
        if block.device != self.device:
            raise ValueError("block must belong to this session's device")
        if block.page_size != self.page_size:
            raise ValueError("block must use this session's page size")

    def _copy_block(
        self,
        source: CacheBlock,
        destination: CacheBlock,
        destination_start: int,
    ) -> None:
        """Copy all KV from ``source`` into one range of ``destination``."""
        source_slots = source.token_slots_tensor().to(torch.int64)
        self._copy_slots(source_slots, destination, destination_start)

    def _copy_slots(
        self,
        source_slots: torch.Tensor,
        destination: CacheBlock,
        destination_start: int,
    ) -> None:
        """Copy KV from captured physical slots into ``destination``."""
        destination_slots = destination.token_slots_tensor()[
            destination_start : destination_start + source_slots.numel()
        ].to(torch.int64)
        for layer_idx in range(self.kv_cache.num_layers):
            k_cache = self.kv_cache.k_cache(layer_idx)
            v_cache = self.kv_cache.v_cache(layer_idx)
            k_flat = k_cache.reshape(-1, *k_cache.shape[2:])
            v_flat = v_cache.reshape(-1, *v_cache.shape[2:])
            k_flat.index_copy_(0, destination_slots, k_flat.index_select(0, source_slots))
            v_flat.index_copy_(0, destination_slots, v_flat.index_select(0, source_slots))

    def _shift_keys_(
        self,
        *,
        block: CacheBlock,
        start: int,
        num_tokens: int,
        position_shift: int,
    ) -> None:
        """Move stored keys into the left block's position frame."""
        slots = block.token_slots_tensor()[start : start + num_tokens].to(torch.int64)
        corrections = torch.full(
            (num_tokens,), position_shift, dtype=torch.int64, device=self.device
        )
        for layer_idx in range(self.kv_cache.num_layers):
            k_cache = self.kv_cache.k_cache(layer_idx)
            k_flat = k_cache.reshape(-1, *k_cache.shape[2:])
            keys = self.sc_attn._rope(
                k_flat.index_select(0, slots).float(), corrections
            ).to(k_flat.dtype)
            k_flat.index_copy_(0, slots, keys)

    @torch.inference_mode()
    def prefill_block(
        self,
        block: CacheBlock,
        input_ids: torch.Tensor,
        context: Optional[List[CacheBlock]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prefill a single ``CacheBlock`` with *input_ids* and return logits.

        *input_ids* must be a 1-D CPU ``int32`` tensor.

        A **non-empty** *block* is **extended**: the new tokens are appended
        after its existing ones (at block-relative positions
        ``num_tokens ...``), attend causally to them, and their KV lands in the
        block's last page's free slots and then in fresh pages -- the same state
        a run of ``decode_step`` calls would leave, in one forward.

        When *context* blocks are given, the new tokens attend to them as if
        the blocks were concatenated ``[ctx_0, ..., block]`` (mirrors the
        reference's ``prefill_cache_block(text, [ctx..., new])``); the stored
        KV stays block-relative either way.  Empty context blocks are skipped.
        """
        self._validate_block(block)
        for context_block in context or []:
            self._validate_block(context_block)
        input_ids = input_ids.to(dtype=torch.int32).flatten().cpu()
        seq_len = len(input_ids)
        assert seq_len > 0

        cached_len = block.num_tokens
        # Zero-based interleaved-mRoPE positions of the new tokens (None => text,
        # position == index).  They are offset into *block*'s own frame below, so
        # an image may be appended to a non-empty block and a block that already
        # holds an image may be extended.
        mrope_rel = (
            self._mrope_rel(input_ids, mm_token_type_ids, image_grid_thw)
            if pixel_values is not None
            else None
        )

        context = [b for b in (context or []) if b.num_tokens > 0]
        if any(b is block for b in context):
            # The block's own prefix is already covered by the causal self segment;
            # listing it in *context* as well would count its KV twice.  (A fresh
            # block passed as part of a whole view is filtered out above, so this
            # only fires for a genuine extension.)
            raise ValueError(
                "the write block must not appear in context: pass the blocks before "
                "it (its own tokens are attended to by the self segment)"
            )
        job = PrefillJob(
            block=block,
            input_ids=input_ids,
            context=context,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        if self.max_prefill_rows is not None and _job_rows(job) > self.max_prefill_rows:
            return self._prefill_waves([job], self.max_prefill_rows)[0]
        if context:
            # An in-context prefill is just a one-request batch: ``prefill_batch``
            # plans exactly the same two wrappers for a single spec (bit-identical
            # results, and the extra scatter-merge is lost in the noise), so there
            # is no separate single-request implementation to keep in step.
            return self._prefill_batch_fused([job])[0]

        assert cached_len + seq_len <= self.page_table.shape[1], (
            f"prefill of {seq_len} tokens into a block of {cached_len} exceeds the "
            f"engine's max sequence length ({self.page_table.shape[1]})"
        )
        page_starts, token_slots = self._alloc_token_storage(seq_len, write_to=block)
        table_idx = self._allocate_table_idx()

        try:
            # An extension is a plain extend-prefill over the block's own prefix:
            # the page-table row holds the block's existing per-token slots first,
            # then the new ones, and ``cached_len`` makes the backend attend the new
            # tokens to that prefix causally (keys stay block-relative, since the
            # write positions continue the block's own frame).
            if cached_len:
                self.page_table[table_idx, :cached_len] = block.token_slots_tensor()
            self.page_table[table_idx, cached_len : cached_len + seq_len] = token_slots
            self._token_pool[table_idx, cached_len : cached_len + seq_len] = input_ids.to(
                self.device
            )

            # Only positions >= cached_len are read (input ids come from the token
            # pool), so the prefix ids are placeholders -- as in ``decode_step``.
            full_ids = input_ids
            if cached_len:
                full_ids = torch.cat([torch.zeros(cached_len, dtype=torch.int32), input_ids])
            req = Req(
                input_ids=full_ids,
                table_idx=table_idx,
                cached_len=cached_len,
                output_len=1,
                uid=-1,
                sampling_params=self.engine.config.get_default_sampling_params(),
                cache_handle=NULL_CACHE_HANDLE,
            )
            batch = self._build_batch([req], phase="prefill")
            # Multimodal (Qwen3.5 vision): attach pixel_values/grid + 3D mRoPE positions
            # so the vision tower + interleaved mRoPE run in the model forward.
            if pixel_values is not None:
                batch.pixel_values = pixel_values.to(self.device)
                batch.image_grid_thw = image_grid_thw.to(self.device)
                batch.mm_token_type_ids = mm_token_type_ids.to(self.device)
            new_span = self._attach_mrope_positions(batch, block, seq_len, mrope_rel)
            logits = self._forward(batch, cache_structure=[[block]], write_to=[block])

            block.grow_pages(page_starts, seq_len)
            block.token_ids.extend(input_ids.tolist())
            self._commit_mrope_span(block, new_span)

            # NOTE: ParallelLMHead.forward already extracts last-token logits
            # for prefill batches, so logits has shape [bs, vocab].
            return logits[:1]
        finally:
            self._free_table_idx(table_idx)

    @torch.inference_mode()
    def prefill_batch(self, jobs: Sequence[PrefillJob]) -> List[torch.Tensor]:
        """
        Run several prefills in a **single forward** and return each one's
        last-token logits ``[1, vocab]``, in request order.

        Each job behaves exactly as the equivalent :meth:`prefill_block` call —
        same write semantics (a non-empty block is extended), same context
        handling, same multimodal inputs — the requests just share one batch.
        A single job is delegated to :meth:`prefill_block` unchanged.

        The jobs must be mutually independent: no two may write the same block,
        and no job's context may name another job's write block.  Such a job
        would have to read KV this very forward produces, which the batch does
        not expose (context segments are sized pre-write), so it is rejected
        rather than silently answered from stale KV — run it in a later batch.
        """
        jobs = list(jobs)
        assert jobs, "prefill_batch needs at least one request"
        for job in jobs:
            self._validate_block(job.block)
            for context_block in job.context:
                self._validate_block(context_block)
        budget = self.max_prefill_rows
        if budget is not None and sum(_job_rows(j) for j in jobs) > budget:
            return self._prefill_waves(jobs, budget)
        if len(jobs) == 1:
            job = jobs[0]
            return [
                self.prefill_block(
                    job.block,
                    job.input_ids,
                    context=list(job.context) or None,
                    pixel_values=job.pixel_values,
                    image_grid_thw=job.image_grid_thw,
                    mm_token_type_ids=job.mm_token_type_ids,
                )
            ]
        return self._prefill_batch_fused(jobs)

    @torch.inference_mode()
    def _prefill_batch_fused(self, jobs: Sequence[PrefillJob]) -> List[torch.Tensor]:
        """The one-forward implementation behind :meth:`prefill_batch`.

        Works for any number of requests, including one — a single job is
        normally routed to :meth:`prefill_block` instead (same result, but it
        keeps the well-trodden page-table path for the common case), so this is
        called directly only by the batched path and by tests isolating it.
        """
        jobs = list(jobs)
        write_ids = {id(job.block) for job in jobs}
        assert len(write_ids) == len(jobs), "two prefills write the same block in one batch"

        specs: List[PrefillSpec] = []
        chains: List[List[CacheBlock]] = []
        ids_list: List[torch.Tensor] = []
        positions: List[torch.Tensor] = []
        out_locs: List[torch.Tensor] = []
        new_pages: List[torch.Tensor] = []
        mrope_parts: List[torch.Tensor] = []
        new_spans: List[int] = []
        mm_type_parts: List[torch.Tensor] = []
        pixel_parts: List[torch.Tensor] = []
        grid_parts: List[torch.Tensor] = []
        embed_parts: List[torch.Tensor] = []
        need_mrope = False

        for job in jobs:
            block = job.block
            input_ids = job.input_ids.to(dtype=torch.int32).flatten().cpu()
            seq_len = len(input_ids)
            assert seq_len > 0
            cached_len = block.num_tokens
            assert cached_len + seq_len <= self.page_table.shape[1], (
                f"prefill of {seq_len} tokens into a block of {cached_len} exceeds the "
                f"engine's max sequence length ({self.page_table.shape[1]})"
            )

            if any(b is block for b in (job.context or [])):
                raise ValueError(
                    "the write block must not appear in context: pass the blocks before "
                    "it (its own tokens are attended to by the self segment)"
                )
            # Checked before the empty-block filter: naming a block another
            # request fills this very step is an ordering dependency whether or
            # not it happens to be empty right now.
            if any(id(b) in write_ids for b in (job.context or [])):
                raise ValueError(
                    "a batched prefill's context names another request's write block; "
                    "run it in a later batch so it can see that block's new tokens"
                )
            context = [b for b in (job.context or []) if b.num_tokens > 0]

            # A chunk arrives with its positions already sliced out of the request's
            # whole-sequence layout; a whole job computes them here.
            mrope_rel = job.mrope_rel
            if mrope_rel is None and job.pixel_values is not None:
                mrope_rel = self._mrope_rel(input_ids, job.mm_token_type_ids, job.image_grid_thw)
            page_starts, token_slots = self._alloc_token_storage(seq_len, write_to=block)
            # The self segment reads the block's whole post-write page list.
            self_pages = torch.cat([block.page_starts_tensor(), page_starts.to(self.device)])
            specs.append(
                PrefillSpec(
                    context=context,
                    self_page_starts=self_pages,
                    num_new=seq_len,
                    self_prefix_len=cached_len,
                    self_prefix_span=block.mrope_span,
                    mrope_rel=mrope_rel,
                )
            )
            chains.append([*context, block])
            ids_list.append(input_ids)
            new_pages.append(page_starts)
            out_locs.append(token_slots)
            # Keys are stored block-relative, continuing the block's own frame.
            positions.append(
                torch.arange(
                    cached_len, cached_len + seq_len, dtype=torch.int64, device=self.device
                )
            )

            # mRoPE bookkeeping, per request, in the block's own frame — same
            # rule as ``_attach_mrope_positions`` applies to a single prefill.
            offset = block.mrope_span
            rel = mrope_rel
            if rel is None:
                rel = torch.arange(seq_len, dtype=torch.int64).view(1, -1).expand(3, -1)
            else:
                need_mrope = True
            need_mrope |= block.mrope_span_override is not None
            mrope_parts.append(rel.to(torch.int64) + offset)
            if job.mrope_span is not None:
                new_spans.append(job.mrope_span)
            else:
                span_advance = seq_len if mrope_rel is None else int(mrope_rel.max()) + 1
                new_spans.append(offset + span_advance)

            if job.image_embeds is not None:
                embed_parts.append(job.image_embeds)
            elif job.pixel_values is not None:
                pixel_parts.append(job.pixel_values.to(self.device))
                grid_parts.append(job.image_grid_thw.to(self.device))
            if job.mm_token_type_ids is not None:
                # Normalized to int64: a text-only request contributes a zero
                # filler and the whole batch's type ids must concatenate.
                mm_type_parts.append(
                    job.mm_token_type_ids.flatten().to(dtype=torch.int64, device=self.device)
                )
            else:
                mm_type_parts.append(torch.zeros(seq_len, dtype=torch.int64, device=self.device))

        # Reqs are bookkeeping only (batch size for the LM head); the page table
        # is bypassed, so they point at the engine's dummy row.
        dummy_table_idx = self.page_table.shape[0] - 1
        reqs = [
            Req(
                input_ids=ids,
                table_idx=dummy_table_idx,
                cached_len=0,
                output_len=1,
                uid=-1,
                sampling_params=self.engine.config.get_default_sampling_params(),
                cache_handle=NULL_CACHE_HANDLE,
            )
            for ids in ids_list
        ]
        batch = Batch(reqs=reqs, phase="prefill")
        batch.padded_reqs = reqs
        batch.input_ids = torch.cat(ids_list).to(self.device)
        batch.positions = torch.cat(positions)
        batch.out_loc = torch.cat(out_locs)
        batch.attn_metadata = self.sc_attn.prepare_prefill_batch(specs)
        if need_mrope:
            batch.mrope_positions = torch.cat(mrope_parts, dim=1).to(self.device)
        if pixel_parts or embed_parts:
            # Image embeddings are scattered into the image-token positions in order,
            # so plain concatenation is the batched form of a single request -- whether
            # the rows come from the tower in the forward or were precomputed.
            batch.mm_token_type_ids = torch.cat(mm_type_parts)
            if pixel_parts:
                batch.pixel_values = torch.cat(pixel_parts)
                batch.image_grid_thw = torch.cat(grid_parts)
            if embed_parts:
                batch.image_embeds = torch.cat(embed_parts)

        logits = self._forward(
            batch,
            cache_structure=chains,
            write_to=[job.block for job in jobs],
            prefill_segments=[len(ids) for ids in ids_list],
        )

        for job, ids, pages, span in zip(jobs, ids_list, new_pages, new_spans):
            job.block.grow_pages(pages, len(ids))
            job.block.token_ids.extend(ids.tolist())
            self._commit_mrope_span(job.block, span)

        # ParallelLMHead already extracted the per-request last-token rows.
        return [logits[i : i + 1] for i in range(len(jobs))]

    # ------------------------------------------------------------------
    # Chunked prefill
    # ------------------------------------------------------------------

    def _plan_job(self, job: PrefillJob) -> PrefillPlan:
        """Run the per-request work that cannot be done per chunk: the mRoPE layout
        (needs the whole sequence) and the vision tower (needs whole images)."""
        ids = job.input_ids.to(dtype=torch.int32).flatten().cpu()
        if job.pixel_values is None:
            return PrefillPlan(ids=ids)
        types = job.mm_token_type_ids.flatten().to(dtype=torch.int64).cpu()
        rel_pos = self._mrope_rel(ids, types, job.image_grid_thw)
        visual = self.engine.model.model.visual
        embeds = visual.forward(
            job.pixel_values.to(self.device), job.image_grid_thw.to(self.device)
        )
        assert embeds.shape[0] == int((types == 1).sum()), (
            f"tower produced {embeds.shape[0]} embeddings for "
            f"{int((types == 1).sum())} image tokens"
        )
        return PrefillPlan(
            ids=ids,
            types=types,
            rel_pos=rel_pos,
            embeds=embeds,
            base_span=job.block.mrope_span,
        )

    def _split_job(self, job: PrefillJob, max_rows: int) -> List[PrefillJob]:
        n_ctx = len([b for b in (job.context or []) if b.num_tokens > 0])
        size = max(1, max_rows // (n_ctx + 1))
        if self._is_hybrid and size >= 64:
            # Align to the GDN scan's own chunk where the budget allows; a smaller
            # budget is honoured as given rather than silently overrun.
            size = size // 64 * 64
        total = int(job.input_ids.numel())
        if size >= total and job.pixel_values is None:
            return [job]
        # A multimodal job is always planned, even unsplit: a batch cannot mix jobs
        # carrying pixels with jobs carrying tower output.
        plan = self._plan_job(job)
        return [_chunk_job(job, plan, s, min(s + size, total)) for s in range(0, total, size)]

    @torch.inference_mode()
    def _prefill_waves(self, jobs: Sequence[PrefillJob], max_rows: int) -> List[torch.Tensor]:
        """Run *jobs* as several forwards, each within *max_rows* query rows.

        Wave ``i`` holds the i-th chunk of every job that still has tokens, so requests
        stay batched together.  Only the last chunk's logits are returned.
        """
        share = max(1, max_rows // len(jobs))
        per_job = [self._split_job(job, share) for job in jobs]
        last: Dict[int, torch.Tensor] = {}
        for wave in zip_longest(*per_job):
            chunks = [c for c in wave if c is not None]
            try:
                rows = self._prefill_batch_fused(chunks)
            except Exception:
                # A block holding a prefix cannot be truncated and a GDN affine cannot
                # be un-folded, so there is no rollback: reset and let the caller retry.
                for job in jobs:
                    self.free_block(job.block)
                raise
            for chunk, row in zip(chunks, rows):
                last[id(chunk.block)] = row
        return [last[id(job.block)] for job in jobs]

    # ------------------------------------------------------------------
    # Interleaved mRoPE (Qwen3.5 multimodal)
    # ------------------------------------------------------------------

    def _mrope_rel(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Zero-based interleaved-mRoPE positions ``[3, S]`` of the new tokens.

        Kept on the CPU: the values feed both the query-rotation plan and the
        key-rotation positions, and the sequence is short.
        """
        from minisgl.models.qwen3_5_mrope import get_rope_index

        assert mm_token_type_ids is not None and image_grid_thw is not None
        assert self._model_config is not None and self._model_config.is_multimodal, (
            "multimodal prefill on a model without a vision config"
        )
        return get_rope_index(
            input_ids.cpu(),
            mm_token_type_ids.cpu(),
            self._model_config.vision_config.spatial_merge_size,
            image_grid_thw.cpu(),
        )

    def _attach_mrope_positions(
        self,
        batch: Batch,
        block: CacheBlock,
        seq_len: int,
        mrope_rel: Optional[torch.Tensor],
    ) -> int:
        """Put the new tokens' mRoPE positions, shifted into *block*'s own frame,
        on *batch* and return the block's post-write mRoPE span.

        Keys are stored block-relative, so an image prefilled into a non-empty
        block -- or any token following one -- must be rotated at
        ``block.mrope_span + rel``, not at its zero-based grid position.  The
        positions are only attached when they actually differ from
        ``batch.positions`` (an image now, or an image already in the block), so
        text-only prefills keep the plain 1-D RoPE path.
        """
        span_advance = seq_len if mrope_rel is None else int(mrope_rel.max()) + 1
        offset = block.mrope_span
        if mrope_rel is not None or block.mrope_span_override is not None:
            rel = mrope_rel
            if rel is None:
                rel = torch.arange(seq_len, dtype=torch.int64).view(1, -1).expand(3, -1)
            batch.mrope_positions = (rel.to(torch.int64) + offset).to(self.device)
        return offset + span_advance

    @staticmethod
    def _commit_mrope_span(block: CacheBlock, new_span: int) -> None:
        """Record the block's mRoPE span after a write (no-op while it tracks the
        token count, i.e. for every text-only block)."""
        if new_span != block.num_tokens or block.mrope_span_override is not None:
            block.mrope_span_override = new_span

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
        for context in group:
            self._validate_block(context.output_block)
            for block in context.cache_view:
                self._validate_block(block)
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
                    sampling_params=self.engine.config.get_default_sampling_params(),
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

        logits = self._forward(
            batch, cache_structure=group.cache_structure, write_to=group.write_to
        )

        # Commit growth now that the forward (which read post-append lengths) is done.
        for wi, wt in enumerate(group.write_to):
            wt.append_token(new_page_for_block[id(wt)])
            wt.token_ids.append(int(input_ids[wi]))
        return logits[:num_workers]

    # ------------------------------------------------------------------
    # Page / table-index management
    # ------------------------------------------------------------------

    def _alloc_token_storage(
        self, seq_len: int, write_to: Optional[CacheBlock] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Borrow the pages needed to write ``seq_len`` tokens into *write_to* and
        return ``(new_page_starts, token_slots)``, where ``token_slots`` are the
        ``seq_len`` write locations in block order.

        For a fresh (or absent) block that is ``ceil(seq_len/P)`` pages worth of
        slots; when *write_to* is non-empty the free slots of its current last page
        are used first, so only ``pages_needed(seq_len)`` new pages are borrowed
        (mirrors how ``append_token`` grows a block during decode)."""
        free_tail = write_to.free_tail if write_to is not None else 0
        n_pages = div_ceil(max(0, seq_len - free_tail), self.page_size)
        page_starts = self.page_allocator.alloc_pages(n_pages)
        new_slots = self.page_allocator.pages_to_tokens(page_starts)
        if free_tail == 0:
            return page_starts, new_slots[:seq_len]
        assert write_to is not None
        tail_start = write_to.page_starts[-1] + write_to.last_page_len
        tail_slots = torch.arange(
            tail_start, tail_start + free_tail, dtype=new_slots.dtype, device=self.device
        )
        return page_starts, torch.cat([tail_slots, new_slots])[:seq_len]

    def _plan_decode_writes(
        self, group: WorkerGroup
    ) -> Tuple[Dict[int, Optional[int]], torch.Tensor, List[int]]:
        """For one decode step, choose the destination slot of each worker's new
        token and which write blocks need a freshly-allocated page.

        Returns ``(new_page_for_block, new_token_slots, write_pos)`` where
        ``new_page_for_block`` maps ``id(block)`` -> page-start slot (or None),
        ``new_token_slots`` is ``[num_workers]`` int32 destination slots, and
        ``write_pos`` is the per-worker block-relative RoPE position -- the
        block's mRoPE *span*, which equals its token count unless the block holds
        an image (whose compressed positions advance the frame by less).
        """
        new_page_for_block: Dict[int, Optional[int]] = {}
        blocks_needing_page: List[CacheBlock] = []
        for wt in group.write_to:
            if id(wt) in new_page_for_block:
                raise ValueError("WorkerGroup has two workers writing the same block in one step")
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
            write_pos.append(wt.mrope_span)

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
            table_idx_list.append(torch.full((length,), req.table_idx, dtype=torch.int64))
        table_idxs = torch.cat(table_idx_list).to(self.device)
        position_idxs = batch.positions.to(torch.int64)

        batch.out_loc = self.page_table[table_idxs, position_idxs]
        batch.input_ids = self._token_pool[table_idxs, position_idxs]

        self.attn_backend.prepare_metadata(batch)
        return batch

    def _make_positions(self, batch: Batch) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for req in batch.padded_reqs:
            parts.append(torch.arange(req.cached_len, req.device_len, dtype=torch.int32))
        return torch.cat(parts).to(self.device)

    def _forward(
        self,
        batch: Batch,
        cache_structure: "List[List[CacheBlock]] | None" = None,
        write_to: "List[CacheBlock] | None" = None,
        prefill_segments: "List[int] | None" = None,
    ) -> torch.Tensor:
        ctx = self.engine.ctx
        # For hybrid (Qwen3.5) models, hand the GDN layers the worker chains so
        # they can compose the initial recurrent state and capture per-token
        # affine updates.  ``prefill_segments`` splits a batched prefill's rows
        # per request, since a GDN chain is inherently sequential and each
        # request runs its own scan.  No-op for standard models (sc_gdn is None).
        if self.sc_gdn is not None and cache_structure is not None:
            self.sc_gdn.set_context(
                cache_structure,
                write_to or [c[-1] for c in cache_structure],
                prefill_segments=prefill_segments,
            )
            ctx.gdn_ar = self.sc_gdn
        try:
            with ctx.forward_batch(batch):
                return self.engine.model.forward()
        finally:
            ctx.gdn_ar = None
