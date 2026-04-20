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

from .rope_correction import CorrectionKey, build_correction_plan, correct_kv_pages
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

    def __init__(self, engine: Engine, cos_sin_cache: Optional[torch.Tensor] = None):
        self.engine = engine
        self.device = engine.device
        self.page_table = engine.page_table
        self.kv_cache = engine.kv_cache
        self.page_size: int = engine.ctx.page_size
        self.attn_backend = engine.attn_backend

        num_token_slots = engine.num_pages * self.page_size
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

        self._temp_pages: List[torch.Tensor] = []
        self._correction_cache: Dict[CorrectionKey, torch.Tensor] = {}

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
    ) -> torch.Tensor:
        """
        Prefill a single ``SharedBlock`` with *input_ids* and return logits.

        *input_ids* must be a 1-D CPU ``int32`` tensor.
        """
        input_ids = input_ids.to(dtype=torch.int32).flatten().cpu()
        seq_len = len(input_ids)
        assert seq_len > 0

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

    @torch.inference_mode()
    def decode_step(
        self,
        group: WorkerGroup,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run one decode step for every worker in *group*.

        Args:
            group: the ``WorkerGroup`` defining the cache structure.
            input_ids: ``[num_workers]`` or ``[num_workers, 1]`` int tensor
                of per-worker input tokens (typically the last generated token).

        Returns:
            Logits ``[num_workers, vocab_size]``.
        """
        input_ids = input_ids.to(dtype=torch.int32).reshape(group.num_workers).cpu()
        table_indices = [self._allocate_table_idx() for _ in range(group.num_workers)]

        try:
            self._apply_corrections(group)
            new_pages = self._fill_page_tables(group, table_indices, input_ids)

            reqs: List[Req] = []
            for wi in range(group.num_workers):
                cached_len = group.worker_cache_length(wi)
                full_ids = torch.zeros(cached_len + 1, dtype=torch.int32)
                full_ids[cached_len] = input_ids[wi]
                reqs.append(
                    Req(
                        input_ids=full_ids,
                        table_idx=table_indices[wi],
                        cached_len=cached_len,
                        output_len=1,
                        uid=-(wi + 1),
                        sampling_params=_DEFAULT_SAMPLING,
                        cache_handle=NULL_CACHE_HANDLE,
                    )
                )

            batch = self._build_batch(reqs, phase="decode")
            logits = self._forward(batch)

            self._record_writes(group, reqs, new_pages)
            return logits[: group.num_workers]
        finally:
            self._cleanup_corrections()
            for ti in table_indices:
                self._free_table_idx(ti)

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
    # RoPE correction
    # ------------------------------------------------------------------

    def _apply_corrections(self, group: WorkerGroup) -> None:
        """Allocate temp pages and write RoPE-corrected KV for all blocks that need it."""
        blocks_with_targets: List[Tuple[SharedBlock, int]] = []
        for worker_seq in group.cache_structure:
            pos = 0
            for block in worker_seq:
                blocks_with_targets.append((block, pos))
                pos += block.num_tokens

        plan = build_correction_plan(blocks_with_targets)
        if not plan:
            return

        for key, (block, target_start, corrections) in plan.items():
            n = block.num_tokens
            temp = self._allocate_pages(n)
            self._temp_pages.append(temp)

            src = block.get_page_indices()
            correct_kv_pages(
                self.kv_cache,
                source_pages=src,
                dest_pages=temp,
                corrections=corrections.to(self.device),
                cos_sin_cache=self._cos_sin_cache,
            )
            self._correction_cache[key] = temp

    def _cleanup_corrections(self) -> None:
        for pages in self._temp_pages:
            self._free_pages_back(pages)
        self._temp_pages.clear()
        self._correction_cache.clear()

    # ------------------------------------------------------------------
    # Page-table construction
    # ------------------------------------------------------------------

    def _fill_page_tables(
        self,
        group: WorkerGroup,
        table_indices: List[int],
        input_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Fill the global ``page_table`` for every worker and allocate a new page
        per worker for the incoming decode token.

        Returns the list of new-page tensors (one per worker).
        """
        new_pages_list: List[torch.Tensor] = []

        for wi in range(group.num_workers):
            table_idx = table_indices[wi]
            pos = 0

            for block in group.cache_structure[wi]:
                n = block.num_tokens
                if n == 0:
                    continue

                key: CorrectionKey = (block.block_id, pos)
                if key in self._correction_cache:
                    pages = self._correction_cache[key]
                else:
                    pages = block.get_page_indices()

                self.page_table[table_idx, pos : pos + n] = pages
                pos += n

            new_page = self._allocate_pages(1)
            self.page_table[table_idx, pos] = new_page
            self._token_pool[table_idx, pos] = input_ids[wi].to(self.device)
            new_pages_list.append(new_page)

        return new_pages_list

    # ------------------------------------------------------------------
    # Batch building & forward
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

    # ------------------------------------------------------------------
    # Post-forward bookkeeping
    # ------------------------------------------------------------------

    def _record_writes(
        self,
        group: WorkerGroup,
        reqs: List[Req],
        new_pages: List[torch.Tensor],
    ) -> None:
        """After forward, record new pages and stored positions in write targets."""
        for wi, (req, new_page) in enumerate(zip(reqs, new_pages)):
            target_block = group.write_to[wi]
            stored_pos = torch.tensor(
                [req.cached_len], dtype=torch.int64
            )
            target_block.grow(new_page.cpu(), stored_pos)
