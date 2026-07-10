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

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.utils import div_ceil

from .attention import SharedCacheAttention
from .gdn import SharedCacheGDN
from .shared_block import NULL_CACHE_HANDLE, SharedBlock
from .worker_group import WorkerGroup

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

        self._is_hybrid = _is_hybrid_model(engine)

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
        capture_affine: bool = True,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mrope_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prefill a single ``SharedBlock`` with *input_ids* and return logits.

        *input_ids* must be a 1-D CPU ``int32`` tensor.

        When *context* blocks are given, the new tokens attend to them as if
        the blocks were concatenated ``[ctx_0, ..., block]`` (mirrors the
        reference's ``prefill_cache_block(text, [ctx..., new])``); the stored
        KV stays block-relative either way.  Empty context blocks are skipped.

        ``capture_affine`` only applies to hybrid (Qwen3.5) models with no
        context.  Set it ``False`` for throwaway prefills whose block is read
        once and freed (e.g. the mode-switching probe): the GDN layers then skip
        the O(seq) affine capture (a large, otherwise-wasted cost) and take the
        numerically-identical from-zero path.  Ignored for standard models.
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
            # Multimodal (Qwen3.5 vision): attach pixel_values/grid + 3D mRoPE positions
            # so the vision tower + interleaved mRoPE run in the model forward.
            if pixel_values is not None:
                batch.pixel_values = pixel_values.to(self.device)
                batch.image_grid_thw = image_grid_thw.to(self.device)
            if mrope_positions is not None:
                batch.mrope_positions = mrope_positions.to(self.device)
            # Throwaway prefills (capture_affine=False) skip the AR path so the
            # GDN layers don't pay the O(seq) affine capture; a from-zero
            # standalone prefill is identical to composing an empty chain.
            cs = [[block]] if capture_affine else None
            logits = self._forward(batch, cache_structure=cs, write_to=[block])

            block.grow_pages(page_starts, seq_len)
            if mrope_positions is not None:
                # image tokens compress positions: record the block's mRoPE span so
                # later decode queries rotate at their true (continued) mRoPE position.
                block.mrope_span_override = int(mrope_positions.max().item()) + 1

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

        logits = self._forward(batch, cache_structure=[[*context, block]], write_to=[block])

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

        logits = self._forward(
            batch, cache_structure=group.cache_structure, write_to=group.write_to
        )

        # Commit growth now that the forward (which read post-append lengths) is done.
        for wt in group.write_to:
            wt.append_token(new_page_for_block[id(wt)])
        return logits[:num_workers]

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
        cache_structure: "List[List[SharedBlock]] | None" = None,
        write_to: "List[SharedBlock] | None" = None,
    ) -> torch.Tensor:
        ctx = self.engine.ctx
        # For hybrid (Qwen3.5) models, hand the GDN layers the worker chains so
        # they can compose the initial recurrent state and capture per-token
        # affine updates.  No-op for standard models (sc_gdn is None).
        if self.sc_gdn is not None and cache_structure is not None:
            self.sc_gdn.set_context(cache_structure, write_to or [c[-1] for c in cache_structure])
            ctx.gdn_ar = self.sc_gdn
        try:
            with ctx.forward_batch(batch):
                return self.engine.model.forward()
        finally:
            ctx.gdn_ar = None
