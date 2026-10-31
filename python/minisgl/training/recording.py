"""Independent recording scopes over a shared AsyncLLM scheduler."""

from collections import deque
from contextvars import ContextVar
from functools import wraps
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING

from .trace import RecordingSession, Trace

if TYPE_CHECKING:
    from minisgl.llm import AsyncLLM

_current = ContextVar("minisgl_recording", default=None)


class _RecordingSession:
    _recording_hub = True

    def __init__(self, engine):
        self.engine = engine
        self.session = engine.session
        self.owners = {}

    def __getattr__(self, name):
        return getattr(self.session, name)

    def claim(self, blocks):
        owner = _current.get()
        if owner is not None and owner._hub is not self:
            raise ValueError("Recording scope belongs to a different AsyncLLM")
        blocks = list(blocks)
        for block in blocks:
            previous = self.owners.get(block.block_id)
            if previous is not None and previous[0] is not owner:
                raise ValueError("Cache blocks cannot be shared across recording scopes")
            if owner is not None and previous is None:
                if block.num_tokens or self.engine._block_in_use(block):
                    raise ValueError(
                        "Unrecorded block history; record before submitting its prefill"
                    )
        if owner is not None:
            owner._recorder._check(blocks)
            for block in blocks:
                self.owners[block.block_id] = (owner, block)
        return owner

    def create_block(self):
        block = self.session.create_block()
        self.claim([block])
        return block

    def merge_blocks(self, left, right):
        owner = self.claim([left, right])
        if owner is None:
            return self.session.merge_blocks(left, right)
        block = owner._recorder.merge_blocks(left, right)
        self.owners[block.block_id] = (owner, block)
        return block

    def append_block(self, left, right):
        owner = self.claim([left, right])
        session = self.session if owner is None else owner._recorder
        return session.append_block(left, right)

    def free_block(self, block):
        owner = self.claim([block])
        if owner is None or block.block_id not in owner._recorder._contents:
            return self.session.free_block(block)
        return owner._recorder.free_block(block)

    def _groups(self, items, block):
        groups = {}
        for index, item in enumerate(items):
            owner = self.owners.get(block(item).block_id)
            if owner is not None:
                groups.setdefault(owner[0], []).append(index)
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
        return self._run(batches, lambda: self.session.prefill_batch(jobs))

    def decode_step(self, group, input_ids):
        contexts = list(group)
        batches = [
            (
                owner,
                owner._recorder._decode_batch([contexts[i] for i in indices], input_ids[indices]),
            )
            for owner, indices in self._groups(contexts, lambda ctx: ctx.output_block).items()
        ]
        return self._run(batches, lambda: self.session.decode_step(group, input_ids))


class _RecordingEngine:
    def __init__(self, llm):
        self.llm = llm
        self.engine = llm.async_engine
        self.hub = _RecordingSession(self.engine)
        self.recorders = set()
        self.engine.session = self.hub

    def __getattr__(self, name):
        return getattr(self.engine, name)

    @property
    def future_factory(self):
        return self.engine.future_factory

    @future_factory.setter
    def future_factory(self, value):
        self.engine.future_factory = value

    def submit_prefill(self, token_ids, *, write_to, cache_view=None, **kwargs):
        self.hub.claim([*(cache_view or ()), write_to])
        return self.engine.submit_prefill(
            token_ids, write_to=write_to, cache_view=cache_view, **kwargs
        )

    def submit_decode(self, context, input_id, **kwargs):
        self.hub.claim([*context.cache_view, context.output_block])
        return self.engine.submit_decode(context, input_id, **kwargs)

    def cancel_pending(self, owner):
        cancelled = False
        for name, output in (
            ("_prefill_queue", lambda request: request.write_to),
            ("_decode_queue", lambda request: request.context.output_block),
        ):
            kept = deque()
            for request in getattr(self.engine, name):
                entry = self.hub.owners.get(output(request).block_id)
                if entry is not None and entry[0] is owner:
                    cancelled = True
                    request.future.cancel()
                else:
                    kept.append(request)
            setattr(self.engine, name, kept)
        return cancelled


class RecordedAsyncLLM:
    """Record only this wrapper's actions, while sharing the underlying engine.

    Use as a synchronous context manager and await all calls before leaving it.
    Each scope must record its own cache history: populated external blocks and
    blocks belonging to another active scope are rejected. Multiple agents in
    one scope may freely share blocks. Exiting does not close the underlying LLM.
    """

    def __init__(self, llm: "AsyncLLM"):
        self.llm = llm
        self._recorder = None
        self._hub = None
        self._active = False
        self._pending = 0

    def __enter__(self):
        if self._recorder is not None:
            raise RuntimeError("A RecordedAsyncLLM scope can only be entered once")
        engine = self.llm.async_engine
        if not isinstance(engine, _RecordingEngine):
            if isinstance(engine.session, RecordingSession):
                raise ValueError("Cannot combine record(llm) with RecordedAsyncLLM")
            engine = _RecordingEngine(self.llm)
            self.llm.async_engine = engine
        self._hub = engine.hub
        self._recorder = RecordingSession(self._hub.session)
        self._active = True
        engine.recorders.add(self)
        return self

    def __exit__(self, exc_type, exc, traceback):
        engine = self.llm.async_engine
        blocks = [block for owner, block in self._hub.owners.values() if owner is self]
        cancelled = engine.cancel_pending(self)
        self._active = False
        for block in blocks:
            del self._hub.owners[block.block_id]
        engine.recorders.remove(self)
        if not engine.recorders:
            engine.engine.session = self._hub.session
            self.llm.async_engine = engine.engine
        if exc_type is None and (self._pending or cancelled):
            raise RuntimeError("Await all recording calls before leaving RecordedAsyncLLM")

    def get_trace(self) -> Trace:
        """Return this scope's trace, available during recording and after exit."""
        if self._recorder is None:
            raise RuntimeError("Enter RecordedAsyncLLM before requesting its trace")
        return self._recorder.trace

    async def _invoke(self, method, *args, **kwargs):
        if not self._active:
            raise RuntimeError("Use RecordedAsyncLLM inside its context manager")
        token = _current.set(self)
        self._pending += 1
        try:
            return await method(*args, **kwargs)
        finally:
            self._pending -= 1
            _current.reset(token)

    def __getattr__(self, name):
        value = getattr(self.llm, name)
        if not iscoroutinefunction(value):
            return value

        @wraps(value)
        async def invoke(*args, **kwargs):
            return await self._invoke(value, *args, **kwargs)

        return invoke

    async def __call__(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    async def async_generate(self, *args, **kwargs):
        stream = self.llm.async_generate(*args, **kwargs)
        try:
            while True:
                try:
                    result = await self._invoke(anext, stream)
                except StopAsyncIteration:
                    return
                yield result
        finally:
            await stream.aclose()
