"""
Asyncio frontend for the async-cache engine (Phase 2 of
docs/async_sched_impl_plan.md; user API of ASYNC_SCHED_DESIGN.md).

``AsyncLLM`` lets per-agent coroutines drive shared-cache inference
concurrently; batching is handled by the engine tick, which runs as a task in
the caller's event loop::

    llm = AsyncLLM(model_path)

    prompt = await llm.prefill_block(prompt_ids)
    agent = await llm.prefill_block(prefix_ids, context=[prompt.block])
    ctx = AsyncContext(cache_view=[prompt.block, agent.block])

    async for token in llm.async_generate(ctx, first_token_id=sep_id):
        ...

Everything runs single-threaded: the GPU forward blocks the loop for one tick,
then every consumer woken by that tick gets to enqueue its next request before
the following batch is formed (one ``sleep(0)`` round).  Concurrent agents
therefore decode in the same ``WorkerGroup`` step, with same-step cross-worker
visibility, exactly like the lock-step ``SharedCacheSession`` API.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Optional, Sequence, Union

import torch
from minisgl.scheduler.async_engine import AsyncCacheEngine
from minisgl.shared_cache import AsyncContext, CacheBlock, CacheView
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from minisgl.core import SamplingParams
    from minisgl.engine import Engine

logger = init_logger(__name__)

TokenIds = Union[torch.Tensor, Sequence[int]]


@dataclass
class PrefillResult:
    block: CacheBlock  # the filled block
    # Last-token logits [vocab]; only populated when the prefill was submitted
    # with return_logits=True (e.g. the mode-switching probe), else None.
    logits: Optional[torch.Tensor]


def _as_token_tensor(token_ids: TokenIds) -> torch.Tensor:
    if isinstance(token_ids, torch.Tensor):
        return token_ids.to(dtype=torch.int32).flatten().cpu()
    return torch.tensor(list(token_ids), dtype=torch.int32)


class AsyncLLM:
    """
    Async user API over ``AsyncCacheEngine``.

    Construct from a ``model_path`` (builds an ``Engine``; extra kwargs go to
    ``EngineConfig``), or inject a prebuilt ``engine`` / ``async_engine``.
    The engine-tick task is started lazily on the first request and stopped by
    ``close()``; call ``close()`` only after all streams are done.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        engine: "Engine | None" = None,
        async_engine: Optional[AsyncCacheEngine] = None,
        dtype: torch.dtype = torch.bfloat16,
        **engine_kwargs,
    ):
        self._owns_engine = False
        if async_engine is None:
            if engine is None:
                assert model_path is not None, (
                    "AsyncLLM needs a model_path, an engine, or an async_engine"
                )
                from minisgl.distributed import DistributedInfo
                from minisgl.engine import Engine, EngineConfig

                engine = Engine(
                    EngineConfig(
                        model_path=model_path,
                        tp_info=DistributedInfo(rank=0, size=1),
                        dtype=dtype,
                        **engine_kwargs,
                    )
                )
                self._owns_engine = True
            async_engine = AsyncCacheEngine(engine)
        self.engine = engine
        self.async_engine = async_engine
        self.tokenizer = None
        if model_path is not None:
            from minisgl.utils import load_tokenizer

            self.tokenizer = load_tokenizer(model_path)

        self._loop_task: Optional[asyncio.Task] = None
        self._work_event = asyncio.Event()
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_block(self) -> CacheBlock:
        return self.async_engine.create_block()

    async def free_block(self, block: CacheBlock) -> None:
        self.async_engine.free_block(block)

    async def prefill_block(
        self,
        token_ids: TokenIds,
        *,
        context: Optional[CacheView] = None,
        into: Optional[CacheBlock] = None,
        capture_affine: bool = True,
        return_logits: bool = False,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mrope_positions: Optional[torch.Tensor] = None,
    ) -> PrefillResult:
        """Prefill a fresh block (created here unless *into* is given) with
        *token_ids*, attending to *context*; returns the block, plus the
        last-token logits when *return_logits* is set.  Pass ``pixel_values`` /
        ``image_grid_thw`` / ``mrope_positions`` for a multimodal (image) block."""
        self._ensure_loop()
        block = into if into is not None else self.async_engine.create_block()
        future = self.async_engine.submit_prefill(
            _as_token_tensor(token_ids),
            into=block,
            context=context,
            capture_affine=capture_affine,
            return_logits=return_logits,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mrope_positions=mrope_positions,
        )
        self._work_event.set()
        logits = await future
        return PrefillResult(block=block, logits=logits)

    async def refresh_block(
        self,
        block: CacheBlock,
        token_ids: TokenIds,
        *,
        context: Optional[CacheView] = None,
        capture_affine: bool = True,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mrope_positions: Optional[torch.Tensor] = None,
    ) -> PrefillResult:
        """Re-encode an existing *block* in place (frees its pages, keeps its
        identity so live contexts stay valid), through the engine tick so it is
        serialized with decodes.  The hook for an updatable image: pass fresh
        ``pixel_values`` / ``image_grid_thw`` / ``mrope_positions`` to swap the
        image while surrounding blocks (prompt, generated reasoning) are kept."""
        self._ensure_loop()
        future = self.async_engine.submit_prefill(
            _as_token_tensor(token_ids),
            into=block,
            context=context,
            capture_affine=capture_affine,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mrope_positions=mrope_positions,
            refresh=True,
        )
        self._work_event.set()
        await future
        return PrefillResult(block=block, logits=None)

    async def async_generate(
        self,
        context: AsyncContext,
        *,
        forbid_ids: Sequence[int] = (),
        sampling_params: "SamplingParams | None" = None,
        max_steps: Optional[int] = None,
        first_token_id: Optional[int] = None,
        return_logits: bool = False,
    ) -> AsyncIterator["int | tuple[int, torch.Tensor]"]:
        """Decode step-by-step for *context*, yielding each sampled token id —
        or ``(token_id, logits)`` pairs when *return_logits* is set, where
        *logits* is the raw ``[vocab]`` row before forbid masking.

        The token fed to a step is ``context.next_input_id`` — seeded by
        *first_token_id* (or a previous stream over the same context) and
        advanced to each newly-sampled token.  Breaking out of the generator
        stops the stream cleanly; a later ``async_generate`` over the same
        context resumes where it left off.
        """
        if first_token_id is not None:
            context.next_input_id = int(first_token_id)
        steps = 0
        while max_steps is None or steps < max_steps:
            input_id = context.next_input_id
            if input_id is None:
                raise ValueError(
                    "context has no pending input token; pass first_token_id "
                    "(or set context.next_input_id) to seed the stream"
                )
            self._ensure_loop()
            future = self.async_engine.submit_decode(
                context,
                input_id,
                forbid_ids=forbid_ids,
                sampling_params=sampling_params,
                return_logits=return_logits,
            )
            self._work_event.set()
            result = await future
            token = result[0] if return_logits else result
            context.next_input_id = token
            steps += 1
            yield result

    async def close(self) -> None:
        """Stop the engine-tick task (call after all streams are finished)."""
        self._closed = True
        self._work_event.set()
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None
        if self._owns_engine and self.engine is not None:
            self.engine.shutdown()

    # ------------------------------------------------------------------
    # Engine loop
    # ------------------------------------------------------------------

    def _ensure_loop(self) -> None:
        """Lazily start the tick task; must run before a submit so requests
        get asyncio futures from the running loop."""
        if self._loop_task is None or self._loop_task.done():
            loop = asyncio.get_running_loop()
            self.async_engine.future_factory = loop.create_future
            self._loop_task = loop.create_task(self._engine_loop(), name="minisgl-async-tick")

    async def _engine_loop(self) -> None:
        while not self._closed:
            if not self.async_engine.has_work:
                self._work_event.clear()
                await self._work_event.wait()
                if self._closed:
                    return
            # One yield so every consumer woken by the previous tick (or by
            # the wake-up above) gets to enqueue before the batch is formed —
            # this is what keeps concurrent agents in the same WorkerGroup.
            await asyncio.sleep(0)
            try:
                self.async_engine.tick()
            except Exception:
                # tick() already failed the affected futures; keep serving.
                logger.exception("async-cache tick failed")
