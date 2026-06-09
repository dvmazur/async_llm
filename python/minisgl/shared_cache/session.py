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

from typing import TYPE_CHECKING, List, Optional

import torch
from minisgl.core import Batch, Req, SamplingParams

from .attention import SharedCacheAttention
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


class SharedCacheSession:
    """
    Manages ``SharedBlock`` pages and drives batched forward passes for a
    ``WorkerGroup`` on a mini-sglang ``Engine``.

    The session owns its own page pool (taken from the Engine's KV-cache
    capacity) and table-index pool.  It is designed for **standalone** use,
    bypassing the scheduler.  To integrate with the live scheduler, the page
    pools would need to be coordinated.
    """

    def __init__(
        self,
        engine: Engine,
        cos_sin_cache: Optional[torch.Tensor] = None,
        max_pages: Optional[int] = None,
    ):
        self.engine = engine
        self.device = engine.device
        self.page_table = engine.page_table
        self.kv_cache = engine.kv_cache
        self.page_size: int = engine.ctx.page_size
        self.attn_backend = engine.attn_backend

        owned_pages = max_pages if max_pages is not None else engine.num_pages
        num_token_slots = owned_pages * self.page_size
        self._free_pages = torch.arange(num_token_slots, dtype=torch.int32, device=self.device)

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
            dtype=self.kv_cache.dtype,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_block(self) -> SharedBlock:
        return SharedBlock(self.device)

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

        pages = self._allocate_pages(seq_len)
        table_idx = self._allocate_table_idx()

        try:
            self.page_table[table_idx, :seq_len] = pages
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

            positions = torch.arange(seq_len, dtype=torch.int64)
            block.grow(pages.cpu(), positions)

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
        pages = self._allocate_pages(seq_len)

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
        batch.out_loc = pages
        batch.attn_metadata = self.sc_attn.prepare_context_prefill(context, pages)

        logits = self._forward(batch)

        block.grow(pages.cpu(), torch.arange(seq_len, dtype=torch.int64))
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
        new_pages = self._allocate_pages(num_workers)
        write_pos = [group.write_to[wi].num_tokens for wi in range(num_workers)]

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
        batch.out_loc = new_pages
        batch.attn_metadata = self.sc_attn.prepare(group, new_pages)

        logits = self._forward(batch)

        for wi in range(num_workers):
            group.write_to[wi].grow(
                new_pages[wi : wi + 1].cpu(),
                torch.tensor([write_pos[wi]], dtype=torch.int64),
            )
        return logits[:num_workers]

    # ------------------------------------------------------------------
    # Page / table-index management
    # ------------------------------------------------------------------

    def _allocate_pages(self, n: int) -> torch.Tensor:
        assert len(self._free_pages) >= n, (
            f"Out of pages: requested {n}, available {len(self._free_pages)}"
        )
        allocated = self._free_pages[:n].clone()
        self._free_pages = self._free_pages[n:]
        return allocated

    def _free_pages_back(self, pages: torch.Tensor) -> None:
        self._free_pages = torch.cat([self._free_pages, pages.to(self.device)])

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
