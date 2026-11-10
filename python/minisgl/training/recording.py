"""Independent recording scopes over a shared AsyncLLM scheduler."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextvars import ContextVar
from functools import wraps
from inspect import iscoroutinefunction
from types import TracebackType
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import torch

from minisgl.llm import AsyncLLM

from .trace import _IMAGE_INPUT_FIELDS, BlockUpdate, ForwardBatch, ForwardRequest, TensorData, Trace

if TYPE_CHECKING:
    from minisgl.core import SamplingParams
    from minisgl.llm import CausalLMOutput
    from minisgl.llm.async_llm import TokenIds
    from minisgl.shared_cache import AsyncContext, CacheBlock, CacheView

_P = ParamSpec("_P")
_T = TypeVar("_T")
_current: ContextVar[RecordedAsyncLLM | None] = ContextVar("minisgl_recording", default=None)


class _TraceRecorder:
    """Session proxy: install before submitting any work to AsyncLLM."""

    def __init__(self, session):
        self.session = session
        self.trace = Trace()
        self._contents = {}
        self._failed = False

    def __getattr__(self, name):
        return getattr(self.session, name)

    def _check(self, blocks):
        if self._failed:
            raise RuntimeError("Discard recorder after a failed inference call")
        for block in blocks:
            if block.num_tokens != self._contents.get(block.block_id, 0):
                raise ValueError("Unrecorded or modified block; attach recorder before prefill")

    def _call(self, call):
        try:
            return call()
        except Exception:
            self._failed = True
            raise

    def _run(self, batch, call):
        result = self._call(call)
        self._commit(batch)
        return result

    def _commit(self, batch):
        for request in batch.requests:
            self._contents[request.write_to] = request.write_start + len(request.input_ids)
        self.trace.append(batch)
        self.trace.steps += 1

    def _replace(self, block, refs):
        self._contents[block.block_id] = sum(length for _, length in refs)
        self.trace.updates.append(BlockUpdate(self.trace.steps, block.block_id, refs=refs))

    def merge_blocks(self, left, right):
        self._check([left, right])
        refs = ((left.block_id, left.num_tokens), (right.block_id, right.num_tokens))
        merged = self._call(lambda: self.session.merge_blocks(left, right))
        self._replace(merged, refs)
        return merged

    def append_block(self, left, right):
        self._check([left, right])
        refs = ((left.block_id, left.num_tokens), (right.block_id, right.num_tokens))
        result = self._call(lambda: self.session.append_block(left, right))
        self._replace(left, refs)
        return result

    def free_block(self, block):
        self._check([block])
        self._call(lambda: self.session.free_block(block))
        self._replace(block, ())

    def prefill_batch(self, jobs):
        jobs = list(jobs)
        batch = self._prefill_batch(jobs)
        return self._run(batch, lambda: self.session.prefill_batch(jobs))

    def _prefill_batch(self, jobs):
        requests = []
        for job in jobs:
            self._check([*job.context, job.block])
            if getattr(job, "image_embeds", None) is not None:
                raise ValueError(
                    "Record image requests with pixels before internal prefill chunking"
                )
            view = [*job.context, job.block]
            tensors = {
                name: TensorData.capture(getattr(job, name, None)) for name in _IMAGE_INPUT_FIELDS
            }
            positions = getattr(job, "mrope_rel", None)
            if positions is None and tensors["pixel_values"] is not None:
                derive = getattr(self.session, "_mrope_rel", None)
                if derive is not None:
                    positions = derive(
                        job.input_ids.flatten(), job.mm_token_type_ids, job.image_grid_thw
                    )
            span = getattr(job.block, "mrope_span", job.block.num_tokens)
            spans = tuple(getattr(block, "mrope_span", block.num_tokens) for block in view)
            multimodal = positions is not None or any(
                value is not None for value in tensors.values()
            )
            if positions is None and span != job.block.num_tokens:
                import torch

                positions = torch.arange(job.input_ids.numel()).expand(3, -1)
            requests.append(
                ForwardRequest(
                    tuple(int(token) for token in job.input_ids.flatten().tolist()),
                    job.block.block_id,
                    job.block.num_tokens,
                    tuple((block.block_id, block.num_tokens) for block in view),
                    **tensors,
                    mrope_positions=TensorData.capture(
                        None if positions is None else positions + span
                    ),
                    write_span=span if multimodal or span != job.block.num_tokens else None,
                    cache_spans=(
                        spans
                        if multimodal or any(n != block.num_tokens for n, block in zip(spans, view))
                        else None
                    ),
                )
            )
        return ForwardBatch(self.trace.steps, "prefill", tuple(requests))

    def prefill_block(
        self,
        block,
        input_ids,
        context=None,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
    ):
        from minisgl.shared_cache.session import PrefillJob

        job = PrefillJob(
            block=block, input_ids=input_ids, context=list(context or ()),
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        return self.prefill_batch([job])[0]

    def decode_step(self, group, input_ids):
        batch = self._decode_batch(group, input_ids)
        return self._run(batch, lambda: self.session.decode_step(group, input_ids))

    def _decode_batch(self, group, input_ids):
        contexts = list(group)
        requests = []
        for ctx, token_id in zip(contexts, input_ids.flatten().tolist(), strict=True):
            self._check([*ctx.cache_view, ctx.output_block])
            block = ctx.output_block
            span = getattr(block, "mrope_span", block.num_tokens)
            spans = tuple(getattr(b, "mrope_span", b.num_tokens) for b in ctx.cache_view)
            requests.append(
                ForwardRequest(
                    (int(token_id),),
                    block.block_id,
                    block.num_tokens,
                    tuple((b.block_id, b.num_tokens) for b in ctx.cache_view),
                    write_span=span if span != block.num_tokens else None,
                    cache_spans=(
                        spans
                        if any(n != b.num_tokens for n, b in zip(spans, ctx.cache_view))
                        else None
                    ),
                )
            )
        return ForwardBatch(self.trace.steps, "decode", tuple(requests))


class _RecordingEngine:
    """Route submissions and executed batches to independent recording scopes."""

    def __init__(self, engine):
        self.engine = engine
        self._session = engine.session
        self.owners = {}
        self.recorders = set()
        self.session = engine.session = self

    def __getattr__(self, name):
        try:
            return getattr(self.engine, name)
        except AttributeError:
            return getattr(self._session, name)

    @property
    def future_factory(self):
        return self.engine.future_factory

    @future_factory.setter
    def future_factory(self, value):
        self.engine.future_factory = value

    def claim(self, blocks):
        owner = _current.get()
        if owner is not None and owner._adapter is not self:
            raise ValueError("Recording scope belongs to a different AsyncLLM")
        blocks = list(blocks)
        for block in blocks:
            previous = self.owners.get(block.block_id)
            if previous is not None and previous is not owner:
                raise ValueError("Cache blocks cannot be shared across recording scopes")
            if owner is not None and previous is None:
                if block.num_tokens or self.engine._block_in_use(block):
                    raise ValueError(
                        "Unrecorded block history; record before submitting its prefill"
                    )
        if owner is not None:
            owner._recorder._check(blocks)
            for block in blocks:
                self.owners[block.block_id] = owner
        return owner

    def create_block(self):
        block = self._session.create_block()
        self.claim([block])
        return block

    def merge_blocks(self, left, right):
        owner = self.claim([left, right])
        if owner is None:
            return self._session.merge_blocks(left, right)
        block = owner._recorder.merge_blocks(left, right)
        self.owners[block.block_id] = owner
        return block

    def append_block(self, left, right):
        owner = self.claim([left, right])
        session = self._session if owner is None else owner._recorder
        return session.append_block(left, right)

    def free_block(self, block):
        if self.engine._block_in_use(block):
            raise RuntimeError(f"cannot free {block!r}: referenced by a queued request")
        owner = self.claim([block])
        if owner is None or block.block_id not in owner._recorder._contents:
            return self._session.free_block(block)
        return owner._recorder.free_block(block)

    def _groups(self, items, block):
        groups = {}
        for index, item in enumerate(items):
            owner = self.owners.get(block(item).block_id)
            if owner is not None:
                groups.setdefault(owner, []).append(index)
        return groups

    def _run(self, batches, call):
        try:
            result = call()
        except Exception:
            for owner, _ in batches:
                owner._recorder._failed = True
            raise
        for owner, batch in batches:
            owner._recorder._commit(batch)
        return result

    def prefill_batch(self, jobs):
        jobs = list(jobs)
        batches = [
            (owner, owner._recorder._prefill_batch([jobs[i] for i in indices]))
            for owner, indices in self._groups(jobs, lambda job: job.block).items()
        ]
        return self._run(batches, lambda: self._session.prefill_batch(jobs))

    def decode_step(self, group, input_ids):
        contexts = list(group)
        batches = [
            (
                owner,
                owner._recorder._decode_batch([contexts[i] for i in indices], input_ids[indices]),
            )
            for owner, indices in self._groups(contexts, lambda ctx: ctx.output_block).items()
        ]
        return self._run(batches, lambda: self._session.decode_step(group, input_ids))

    def submit_prefill(self, token_ids, *, write_to, cache_view=None, **kwargs):
        self.claim([*(cache_view or ()), write_to])
        return self.engine.submit_prefill(
            token_ids, write_to=write_to, cache_view=cache_view, **kwargs
        )

    def submit_decode(self, context, input_id, **kwargs):
        self.claim([*context.cache_view, context.output_block])
        return self.engine.submit_decode(context, input_id, **kwargs)

    def cancel_pending(self, owner):
        cancelled = False
        for name, output in (
            ("_prefill_queue", lambda request: request.write_to),
            ("_decode_queue", lambda request: request.context.output_block),
        ):
            kept = deque()
            for request in getattr(self.engine, name):
                if self.owners.get(output(request).block_id) is owner:
                    cancelled = True
                    request.future.cancel()
                else:
                    kept.append(request)
            setattr(self.engine, name, kept)
        return cancelled


class RecordedAsyncLLM(AsyncLLM):
    """Record only this wrapper's actions, while sharing the underlying engine.

    Use as a synchronous context manager and await all calls before leaving it.
    Each scope must record its own cache history: populated external blocks and
    blocks belonging to another active scope are rejected. Multiple agents in
    one scope may freely share blocks. Exiting does not close the underlying LLM.
    """

    def __init__(self, llm: "AsyncLLM") -> None:
        self.llm = llm
        self._recorder: _TraceRecorder | None = None
        self._adapter: _RecordingEngine | None = None
        self._pending = 0

    def __enter__(self) -> "RecordedAsyncLLM":
        if self._recorder is not None:
            raise RuntimeError("A RecordedAsyncLLM scope can only be entered once")
        engine = self.llm.async_engine
        if not isinstance(engine, _RecordingEngine):
            if isinstance(engine.session, _TraceRecorder):
                raise ValueError("Cannot wrap a session that is already recording")
            engine = _RecordingEngine(engine)
            self.llm.async_engine = engine
        self._adapter = engine
        self._recorder = _TraceRecorder(engine._session)
        engine.recorders.add(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        engine = self._adapter
        cancelled = engine.cancel_pending(self)
        self._adapter = None
        engine.owners = {
            block: owner for block, owner in engine.owners.items() if owner is not self
        }
        engine.recorders.remove(self)
        if not engine.recorders:
            engine.engine.session = engine._session
            self.llm.async_engine = engine.engine
        if exc_type is None and (self._pending or cancelled):
            raise RuntimeError("Await all recording calls before leaving RecordedAsyncLLM")

    def get_trace(self) -> Trace:
        """Return this scope's trace, available during recording and after exit."""
        if self._recorder is None:
            raise RuntimeError("Enter RecordedAsyncLLM before requesting its trace")
        return self._recorder.trace

    async def _invoke(
        self, method: Callable[_P, Awaitable[_T]], *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        if self._adapter is None:
            raise RuntimeError("Use RecordedAsyncLLM inside its context manager")
        token = _current.set(self)
        self._pending += 1
        try:
            return await method(*args, **kwargs)
        finally:
            self._pending -= 1
            _current.reset(token)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.llm, name)
        if not iscoroutinefunction(value):
            return value

        @wraps(value)
        async def invoke(*args: Any, **kwargs: Any) -> Any:
            return await self._invoke(value, *args, **kwargs)

        return invoke

    async def create_block(self) -> CacheBlock:
        return await self._invoke(self.llm.create_block)

    async def free_block(self, block: CacheBlock) -> None:
        await self._invoke(self.llm.free_block, block)

    async def merge_blocks(self, left: CacheBlock, right: CacheBlock) -> CacheBlock:
        return await self._invoke(self.llm.merge_blocks, left, right)

    async def append_block(self, left: CacheBlock, right: CacheBlock) -> CacheBlock:
        return await self._invoke(self.llm.append_block, left, right)

    async def prefill_block(
        self,
        input_ids: TokenIds,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        cache_view: CacheView | None = None,
        write_to: CacheBlock | None = None,
        return_logits: bool = False,
    ) -> CausalLMOutput:
        return await self._invoke(
            self.llm.prefill_block,
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            cache_view=cache_view,
            write_to=write_to,
            return_logits=return_logits,
        )

    async def forward(
        self,
        input_ids: TokenIds | None = None,
        cache_view: AsyncContext | CacheView | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        write_to: CacheBlock | None = None,
        attention_mask: torch.Tensor | None = None,
        return_logits: bool = True,
    ) -> CausalLMOutput:
        return await self._invoke(
            self.llm.forward,
            input_ids,
            cache_view,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            write_to=write_to,
            attention_mask=attention_mask,
            return_logits=return_logits,
        )

    async def __call__(
        self,
        input_ids: TokenIds | None = None,
        cache_view: AsyncContext | CacheView | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        write_to: CacheBlock | None = None,
        attention_mask: torch.Tensor | None = None,
        return_logits: bool = True,
    ) -> CausalLMOutput:
        return await self.forward(
            input_ids,
            cache_view,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            write_to=write_to,
            attention_mask=attention_mask,
            return_logits=return_logits,
        )

    async def sample(
        self, logits: torch.Tensor | CausalLMOutput, **param_overrides: float | int | bool
    ) -> torch.Tensor:
        return await self._invoke(self.llm.sample, logits, **param_overrides)

    async def close(self) -> None:
        """Close the shared LLM; leaving a recording context only detaches it."""
        await self._invoke(self.llm.close)

    async def async_generate(
        self,
        context: AsyncContext,
        *,
        forbid_ids: Sequence[int] = (),
        sampling_params: SamplingParams | None = None,
        max_steps: int | None = None,
        first_token_id: int | None = None,
        return_logits: bool = False,
    ) -> AsyncIterator[int | tuple[int, torch.Tensor]]:
        stream = self.llm.async_generate(
            context,
            forbid_ids=forbid_ids,
            sampling_params=sampling_params,
            max_steps=max_steps,
            first_token_id=first_token_id,
            return_logits=return_logits,
        )
        try:
            while True:
                try:
                    result = await self._invoke(anext, stream)
                except StopAsyncIteration:
                    return
                yield result
        finally:
            await stream.aclose()
