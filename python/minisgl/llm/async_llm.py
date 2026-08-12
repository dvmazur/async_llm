"""
Asyncio frontend for the async-cache engine

``AsyncLLM`` lets per-agent coroutines drive shared-cache inference
concurrently; batching is handled by the engine tick, which runs as a task in
the caller's event loop::

    llm = AsyncLLM(model_path)

    prompt = await llm.prefill_block(prompt_ids)
    agent = await llm.prefill_block(prefix_ids, context=[prompt.block])
    ctx = AsyncContext(cache_view=[prompt.block, agent.block])

    async for token in llm.async_generate(ctx, first_token_id=sep_id):
        ...

``forward`` is the unified single-pass method (prefill / decode / conditional
prefill selected by its arguments) for building custom scaffolding on top of
raw logits; ``prefill_block`` and ``async_generate`` remain the convenience
paths with engine-side sampling.

Everything runs single-threaded: the GPU forward blocks the loop for one tick,
then every consumer woken by that tick gets to enqueue its next request before
the following batch is formed (one ``sleep(0)`` round).  Concurrent agents
therefore decode in the same ``WorkerGroup`` step, with same-step cross-worker
visibility, exactly like the lock-step ``SharedCacheSession`` API — and their
prefill-mode requests likewise share one prefill forward.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, AsyncIterator, Optional, Sequence, Union

import torch
import transformers

from minisgl.scheduler.async_engine import AsyncCacheEngine
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.shared_cache import AsyncContext, CacheBlock, CacheView
from minisgl.utils import init_logger, load_tokenizer
from minisgl.utils.hf import load_processor

if TYPE_CHECKING:
    from minisgl.core import SamplingParams
    from minisgl.engine import Engine

logger = init_logger(__name__)

TokenIds = Union[torch.Tensor, Sequence[int]]


@dataclass
class CausalLMOutput:
    """Result of one forward pass (``forward`` / ``prefill_block``) — and
    little else.

    ``logits`` is the last-position row ``[vocab]``, a device-side tensor;
    ``None`` unless the pass was submitted with ``return_logits=True``
    (e.g. the mode-switching probe).  ``block`` is the write block the pass
    appended its KV to.
    """

    logits: Optional[torch.Tensor]
    block: CacheBlock


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
                assert model_path is not None, "AsyncLLM needs a model_path, an engine, or an async_engine"
                if "generation_config" not in engine_kwargs:
                    try:
                        engine_kwargs["generation_config"] = transformers.GenerationConfig.from_pretrained(model_path)
                    except OSError:  # missing file
                        engine_kwargs["generation_config"] = transformers.GenerationConfig()
                        logger.warning(f"Model {model_path} has no generation_config, using model-agnostic defaults.")
                if "tp_info" not in engine_kwargs:
                    engine_kwargs["tp_info"] = DistributedInfo(rank=0, size=1)

                engine = Engine(EngineConfig(model_path=model_path, dtype=dtype, **engine_kwargs))
                self._owns_engine = True
            async_engine = AsyncCacheEngine(engine)
        self.engine = engine
        self.async_engine = async_engine
        self.processor = self.tokenizer = None
        if model_path is not None:
            if engine.config.model_config.is_multimodal:
                self.processor = load_processor(model_path)
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = load_tokenizer(model_path)

        self._loop_task: Optional[asyncio.Task] = None
        self._work_event = asyncio.Event()
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> EngineConfig:
        return self.engine.config

    async def create_block(self) -> CacheBlock:
        return self.async_engine.create_block()

    async def free_block(self, block: CacheBlock) -> None:
        self.async_engine.free_block(block)

    async def merge_blocks(
        self,
        left: CacheBlock,
        right: CacheBlock,
    ) -> CacheBlock:
        """Copy ``left + right`` into a newly allocated block."""
        return self.async_engine.merge_blocks(left, right)

    async def append_block(self, left: CacheBlock, right: CacheBlock) -> CacheBlock:
        """Append a copy of ``right`` to ``left`` and return ``left``."""
        return self.async_engine.append_block(left, right)

    async def prefill_block(
        self,
        input_ids: TokenIds,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        cache_view: Optional[CacheView] = None,
        write_to: Optional[CacheBlock] = None,
        return_logits: bool = False,
    ) -> CausalLMOutput:
        """Prefill a block (created here unless *into* is given) with
        *input_ids*, attending to *context*; returns the block, plus the
        last-token logits when *return_logits* is set.  A non-empty *into* is
        extended: the tokens are appended to what it already holds and attend to
        it causally.  Pass ``pixel_values`` / ``image_grid_thw`` for a multimodal (image) block."""
        self._ensure_loop()
        assert (pixel_values is None) == (image_grid_thw is None) == (mm_token_type_ids is None), "pass all or none"
        block = write_to if write_to is not None else self.async_engine.create_block()
        future = self.async_engine.submit_prefill(
            torch.as_tensor(input_ids).flatten(),
            write_to=block,
            cache_view=cache_view,
            return_logits=return_logits,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids.flatten() if mm_token_type_ids is not None else None,
        )
        self._work_event.set()
        logits = await future
        return CausalLMOutput(logits=logits, block=block)

    async def __call__(self, *args, **kwargs):
        """Alias for AsyncLLM.forward"""
        return await self.forward(*args, **kwargs)

    async def forward(
        self,
        input_ids: Optional[TokenIds] = None,
        cache_view: "AsyncContext | CacheView | None" = None,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        write_to: Optional[CacheBlock] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> CausalLMOutput:
        """Run a single forward pass on the LM with the specified cache view,
        adding the new KVs to *write_to* — the unified async forward, covering (conditional) prefill,
        action choice and custom generate.  The mode depends on the provided arguments:

        * ``forward(input_ids, write_to=block)`` — **prefill**: fill the fresh
          block *write_to* with *input_ids*.
        * ``forward(cache_view=ctx)`` — **decode**: one step for the context,
          feeding its pending ``next_input_id`` (consumed on success).  Token
          selection is the caller's job: sample from ``.logits`` and re-seed
          ``ctx.next_input_id`` (or pass ``input_ids``) for the next step.
        * ``forward(input_ids, cache_view)`` — **in-context / conditional
          prefill**: the new tokens attend to the view's blocks in order and
          their KV is appended to *write_to*.  *write_to* must be the last
          block of *cache_view* (earlier positions raise) or outside it.  A
          non-empty *write_to* is *extended* — the new tokens are appended after
          the ones it already holds and attend to them causally — which requires
          it to be the view's last block.

        *write_to* defaults to the view's last block (``ctx.output_block``
        when *cache_view* is an ``AsyncContext``); plain prefill requires it.
        Like every request, the pass is batched by the engine tick with
        whatever else is in flight.

        Returns a :class:`CausalLMOutput` whose ``.logits`` is the raw (un-
        sampled) last-position ``[vocab]`` row when *return_logits* is set,
        and whose ``.block`` is the resolved write block.
        """
        ctx = cache_view if isinstance(cache_view, AsyncContext) else None
        view: CacheView = list(ctx.cache_view if ctx is not None else (cache_view or []))
        if write_to is None:
            if ctx is not None:
                write_to = ctx.output_block
            elif view:
                write_to = view[-1]

        if input_ids is None:
            assert pixel_values is None and image_grid_thw is None and mm_token_type_ids is None, "requires input_ids"
            # Decode mode: the input token is the context's pending one.
            if cache_view is None:
                raise ValueError("forward needs input_ids and/or cache_view")
            if ctx is None or ctx.next_input_id is None:
                raise ValueError(
                    "decode mode needs a pending input token: pass an AsyncContext "
                    "with next_input_id set, or provide input_ids"
                )
            step_ctx = (
                ctx
                if write_to is ctx.output_block
                else AsyncContext(cache_view=view, output_block=write_to)
            )
            logits = await self._forward_decode_step(step_ctx, ctx.next_input_id, return_logits)
            ctx.next_input_id = None  # consumed: its KV now lives in write_to
            return CausalLMOutput(logits=logits, block=write_to)

        input_ids = torch.as_tensor(input_ids)
        if input_ids.numel() == 0:
            raise ValueError("empty input_ids")
        assert input_ids.ndim in (0, 1, 2) and input_ids.shape[-1] == input_ids.numel()
        assert (pixel_values is None) == (image_grid_thw is None) == (mm_token_type_ids is None), "pass all or none"
        assert mm_token_type_ids is None or mm_token_type_ids.shape == input_ids.shape
        assert pixel_values is None or (isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 2)
        assert image_grid_thw is None or (image_grid_thw.shape == (len(image_grid_thw), 3))
        assert attention_mask is None or torch.all(torch.as_tensor(attention_mask)), "attention masking not supported"
        input_ids = input_ids.flatten()
        if mm_token_type_ids is not None:
            mm_token_type_ids = torch.as_tensor(mm_token_type_ids).flatten()

        if ctx is not None and ctx.next_input_id is not None:
            raise ValueError(
                "cache_view has a pending next_input_id and input_ids were also given; "
                "feed the pending token first (forward(cache_view=ctx))"
            )
        if write_to is None:
            raise ValueError("prefill mode needs a write_to block")
        if any(b is write_to for b in view[:-1]):
            raise ValueError("write_to must be the last block of cache_view")
        in_view = bool(view) and view[-1] is write_to

        if write_to.num_tokens > 0 and not in_view:
            raise ValueError("cannot extend a non-empty write_to outside cache_view")

        # (Conditional) prefill / extension: one forward for all tokens.
        self._ensure_loop()
        future = self.async_engine.submit_prefill(
            input_ids,
            write_to=write_to,
            cache_view=(view[:-1] if in_view else view) or None,
            return_logits=return_logits,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        self._work_event.set()
        return CausalLMOutput(logits=await future, block=write_to)

    async def _forward_decode_step(
        self, ctx: AsyncContext, input_id: int, return_logits: bool
    ) -> Optional[torch.Tensor]:
        """One engine decode for *ctx* feeding *input_id*; returns the raw
        logits row (or ``None``).  The engine-sampled token is discarded —
        ``forward`` callers do their own selection on the logits."""
        self._ensure_loop()
        future = self.async_engine.submit_decode(ctx, input_id, return_logits=return_logits)
        self._work_event.set()
        result = await future
        return result[1] if return_logits else None

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

    async def sample(self, logits: torch.Tensor | CausalLMOutput, **param_overrides) -> torch.IntTensor:
        """sample with logits, use model's default generation config with optional user overrides"""
        logits = logits.logits if isinstance(logits, CausalLMOutput) else logits
        params = replace(self.engine.config.get_default_sampling_params(), **param_overrides)
        batch_params = self.engine.sampler.prepare_params([params])
        return self.engine.sampler.sample(logits.view(1, -1), batch_params).view(logits.shape[:-1])

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
