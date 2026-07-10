"""Scheduler-side mechanism for shared-cache (async-reasoning) work.

This is the *mechanism-only* surface from PLAN.md: block page lifecycle plus
``WorkerGroup`` forwards that return **logprobs**. It holds **no** reasoning
state — the external driver owns the state machine, forbidden-token masking,
the mode-switch probe, streaming, and termination. The scheduler exposes it so
async-reasoning chains draw from the same engine + page pool as normal traffic.

Returning logprobs (rather than the engine's sampled token) is what lets the
driver do per-worker forbidden-masking + argmax and the probe's yes/no compare
itself: ``log_softmax`` is monotonic and shares one normalizer per row, so
argmax-after-masking and yes/no comparison on logprobs are identical to raw
logits.

Resource ownership follows the scheduler's managers: page-table rows come from
``TableManager`` (borrowed transiently during standalone prefill), and block
pages are lent by ``CacheManager.borrow_pages`` / returned via ``return_pages``
so ``check_integrity`` accounts for them (and borrows can evict prefix-cache
entries under pressure). The service never touches the ``PageAllocator``
except for the read-only ``pages_to_tokens``.

Forwards run on the caller's current stream (the scheduler's stream), on a
dedicated path bypassing batching/sampling/CUDA graphs. Do not interleave
these calls with ``LLM.generate()``; co-scheduling with normal traffic is a
later phase (PLAN.md P2/P3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.shared_cache import SharedBlock, build_shared_cache_attention
from minisgl.shared_cache.shared_block import NULL_CACHE_HANDLE
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from minisgl.engine import Engine
    from minisgl.shared_cache import WorkerGroup

    from .cache import CacheManager
    from .table import TableManager

_DEFAULT_SAMPLING = SamplingParams(temperature=0.0, max_tokens=1)


class SharedCacheService:
    """Block lifecycle + ``WorkerGroup``/block forwards returning logprobs."""

    def __init__(
        self, engine: Engine, cache_manager: CacheManager, table_manager: TableManager
    ) -> None:
        self.engine = engine
        self.device = engine.device
        self.page_table = engine.page_table
        self.page_size: int = engine.ctx.page_size
        self.attn_backend = engine.attn_backend
        self._cache_manager = cache_manager
        self._table_manager = table_manager
        self._page_allocator = engine.page_allocator  # read-only: pages_to_tokens
        self._dummy_table_idx = engine.dummy_req.table_idx

        # Query-rotation attention op for decode (arXiv:2512.10931).
        self.sc_attn = build_shared_cache_attention(engine)

        # Engine init / warmup ran on other streams; order our forwards after it.
        # (FlashInfer requires plan + forward on one stream — hereafter both run
        # on the caller's current stream.)
        torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------
    # Block page lifecycle (accounted via CacheManager.borrowed_block_pages)
    # ------------------------------------------------------------------

    def create_block(self) -> SharedBlock:
        return SharedBlock(self.device, page_size=self.page_size)

    def free_block(self, block: SharedBlock) -> None:
        """Return a block's pages to the cache manager and reset the block."""
        page_starts = block.clear()
        if page_starts:
            self._cache_manager.return_pages(
                torch.tensor(page_starts, dtype=torch.int32, device=self.device)
            )

    # ------------------------------------------------------------------
    # Forwards (return logits, or logprobs when return_logprobs=True)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def prefill_block(
        self,
        block: SharedBlock,
        input_ids: torch.Tensor,
        context: Optional[List[SharedBlock]] = None,
        return_logprobs: bool = False,
    ) -> torch.Tensor:
        """
        Prefill a single ``SharedBlock`` with *input_ids*; return the
        last-token distribution ``[1, vocab]``.

        *input_ids* must be a 1-D CPU ``int32`` tensor.

        When *context* blocks are given, the new tokens attend to them as if
        the blocks were concatenated ``[ctx_0, ..., block]``; the stored KV
        stays block-relative either way. Empty context blocks are skipped.
        """
        input_ids = input_ids.to(dtype=torch.int32).flatten().cpu()
        seq_len = len(input_ids)
        assert seq_len > 0

        context = [b for b in (context or []) if b.num_tokens > 0]
        if context:
            logits = self._prefill_in_context(block, input_ids, context)
        else:
            logits = self._prefill_standalone(block, input_ids)
        return self._finish(logits, return_logprobs)

    @torch.inference_mode()
    def decode_group(
        self,
        group: WorkerGroup,
        input_ids: torch.Tensor,
        return_logprobs: bool = False,
    ) -> torch.Tensor:
        """
        Run one decode step for every worker in *group*; return per-worker
        distributions ``[num_workers, vocab]``. No sampling/masking here —
        the driver does that on the returned rows.

        Cached keys are stored at block-relative RoPE positions and are never
        re-rotated; instead, per-(worker, segment) query copies are rotated and
        partial attention outputs merged (see ``shared_cache.attention``).
        Matching the AsyncReasoning reference, a worker reading another
        worker's write block also sees that worker's current-step token.

        Args:
            group: the ``WorkerGroup`` defining the cache structure.
            input_ids: ``[num_workers]`` or ``[num_workers, 1]`` int tensor
                of per-worker input tokens (typically the last generated token).
        """
        num_workers = group.num_workers
        input_ids = input_ids.to(dtype=torch.int32).reshape(num_workers).cpu()

        # Decide, per (distinct) write block, whether the new token starts a
        # fresh page, then borrow all needed pages in one shot.
        new_page_for_block, new_token_slots, write_pos = self._plan_decode_writes(group)

        # Reqs are bookkeeping only here (batch size / phase); the page table
        # is bypassed entirely, so they point at the engine's dummy row.
        reqs: List[Req] = []
        for wi in range(num_workers):
            cached_len = group.worker_cache_length(wi)
            full_ids = torch.zeros(cached_len + 1, dtype=torch.int32)
            full_ids[cached_len] = input_ids[wi]
            reqs.append(
                Req(
                    input_ids=full_ids,
                    table_idx=self._dummy_table_idx,
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
        return self._finish(logits[:num_workers], return_logprobs)

    # ------------------------------------------------------------------
    # Prefill paths
    # ------------------------------------------------------------------

    def _prefill_standalone(self, block: SharedBlock, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = len(input_ids)
        page_starts, token_slots = self._alloc_token_storage(seq_len)
        table_idx = self._table_manager.allocate()

        try:
            self.page_table[table_idx, : token_slots.numel()] = token_slots

            req = Req(
                input_ids=input_ids,
                table_idx=table_idx,
                cached_len=0,
                output_len=1,
                uid=-1,
                sampling_params=_DEFAULT_SAMPLING,
                cache_handle=NULL_CACHE_HANDLE,
            )
            batch = Batch(reqs=[req], phase="prefill")
            batch.padded_reqs = [req]
            batch.positions = torch.arange(seq_len, dtype=torch.int32).to(self.device)
            batch.input_ids = input_ids.to(self.device)
            batch.out_loc = token_slots[:seq_len]
            self.attn_backend.prepare_metadata(batch)

            logits = self._forward(batch)

            block.grow_pages(page_starts, seq_len)

            # NOTE: ParallelLMHead.forward already extracts last-token logits
            # for prefill batches, so logits has shape [bs, vocab].
            return logits[:1]
        finally:
            self._table_manager.free(table_idx)

    def _prefill_in_context(
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
            table_idx=self._dummy_table_idx,  # page table bypassed
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

    # ------------------------------------------------------------------
    # Page planning & forward
    # ------------------------------------------------------------------

    def _alloc_token_storage(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Borrow ``ceil(seq_len/P)`` pages; return ``(page_starts, token_slots)``
        where ``token_slots`` are the ``num_pages * P`` per-token slots (the
        first ``seq_len`` are the real write locations)."""
        n_pages = div_ceil(seq_len, self.page_size)
        page_starts = self._cache_manager.borrow_pages(n_pages)
        return page_starts, self._page_allocator.pages_to_tokens(page_starts)

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
            fresh = self._cache_manager.borrow_pages(len(blocks_needing_page))
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

    def _forward(self, batch: Batch) -> torch.Tensor:
        with self.engine.ctx.forward_batch(batch):
            return self.engine.model.forward()

    @staticmethod
    def _finish(logits: torch.Tensor, return_logprobs: bool) -> torch.Tensor:
        return logits.float().log_softmax(dim=-1) if return_logprobs else logits
