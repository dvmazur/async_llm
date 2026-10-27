"""Small, nonblocking session traces; records contain text and plain metadata only."""

from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import aclosing, contextmanager
from contextvars import ContextVar
from functools import wraps

_CURRENT = ContextVar("tutor_trace_span", default=None)


class Trace:
    def __init__(self, queue=None, *, session=0, started=None):
        self.queue = queue
        self.session = session
        self.started = time.monotonic() if started is None else started
        self._next_id = 0

    @property
    def enabled(self):
        return self.queue is not None

    def add(self, event, **fields):
        """Queue plain fields without awaiting the transport or retaining model objects."""
        if not self.enabled:
            return
        scope = _CURRENT.get()
        record = {
            **(scope.fields if scope is not None and scope.trace is self else {}),
            **fields,
            "event": event,
            "session": self.session,
            "t_ms": round((time.monotonic() - self.started) * 1000, 3),
        }
        if self.queue.full():
            discarded = self.queue.get_nowait()
            self.queue.task_done()
            # Preserve the count even if a previous overflow notice is itself evicted.
            record["dropped"] = 1 + discarded.get("dropped", 0)
        self.queue.put_nowait(record)

    @contextmanager
    def span(self, stage, **fields):
        scope = _Span(self, stage, fields)
        error = None
        with scope.bind():
            scope.begin()
            try:
                yield scope
            except BaseException as exc:
                error = exc
                raise
            finally:
                scope.end(error)

    def first_text(self):
        scope = _CURRENT.get()
        if scope is not None and scope.trace is self and not scope.has_text:
            scope.has_text = True
            self.add("first_text", duration_ms=scope.elapsed())


class _Span:
    def __init__(self, trace, stage, fields):
        self.trace = trace
        self.started = time.monotonic()
        self.data = {}
        self.has_text = False
        parent = _CURRENT.get()
        inherited = {}
        if parent is not None and parent.trace is trace:
            inherited["parent_call_id"] = parent.fields["call_id"]
            if "frame" in parent.fields:
                inherited["frame"] = parent.fields["frame"]
        trace._next_id += 1
        self.fields = {**inherited, **fields, "stage": stage, "call_id": trace._next_id}

    @contextmanager
    def bind(self):
        token = _CURRENT.set(self)
        try:
            yield
        finally:
            _CURRENT.reset(token)

    def elapsed(self):
        return round((time.monotonic() - self.started) * 1000, 3)

    def begin(self):
        self.trace.add("begin")

    def end(self, error):
        status = "ok"
        details = dict(self.data)
        if isinstance(error, asyncio.CancelledError):
            status = "cancelled"
        elif isinstance(error, GeneratorExit):
            status = "interrupted"
        elif error is not None:
            status = "error"
            details["error"] = f"{type(error).__name__}: {error}"
        details.update(duration_ms=self.elapsed(), status=status)
        self.trace.add("end", **details)


def traced(stage, *, frame=None, result=None):
    """Time a backend method without changing its inference requests or return value."""

    def decorate(function):
        signature = inspect.signature(function) if frame else None

        def fields(args, kwargs):
            if signature is None:
                return {}
            value = signature.bind(*args, **kwargs).arguments.get(frame)
            return {} if value is None else {"frame": getattr(value, "revision", value)}

        if inspect.isasyncgenfunction(function):

            @wraps(function)
            async def stream(self, *args, **kwargs):
                trace = getattr(self, "trace", None)
                if trace is None or not trace.enabled:
                    async with aclosing(function(self, *args, **kwargs)) as iterator:
                        async for value in iterator:
                            yield value
                    return
                scope = _Span(trace, stage, fields((self, *args), kwargs))
                iterator = function(self, *args, **kwargs)
                error = None
                with scope.bind():
                    scope.begin()
                try:
                    while True:
                        # A generator shares its consumer's task. Never carry the scope
                        # across yield, including when the consumer closes it elsewhere.
                        with scope.bind():
                            try:
                                value = await anext(iterator)
                            except StopAsyncIteration:
                                break
                            if isinstance(value, str):
                                scope.data["text"] = value
                                if value:
                                    trace.first_text()
                        yield value
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    with scope.bind():
                        try:
                            await iterator.aclose()
                        except BaseException as exc:
                            error = exc
                            raise
                        finally:
                            scope.end(error)

            return stream

        @wraps(function)
        async def call(self, *args, **kwargs):
            trace = getattr(self, "trace", None)
            if trace is None or not trace.enabled:
                return await function(self, *args, **kwargs)
            with trace.span(stage, **fields((self, *args), kwargs)) as scope:
                value = await function(self, *args, **kwargs)
                if result == "notes" and isinstance(getattr(value, "notes", None), str):
                    scope.data["text"] = value.notes
                elif result is True and (value is None or isinstance(value, (str, int, float))):
                    scope.data["result"] = value
                return value

        return call

    return decorate
