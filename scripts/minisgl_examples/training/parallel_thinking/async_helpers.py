"""Concurrency and fresh-process execution for the parallel-thinking demo."""

import asyncio
import os
import pickle
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM

from minisgl_examples.training.recording import Samples, generate_with_targets

T = TypeVar("T")


async def together(*coroutines: Coroutine[Any, Any, T]) -> list[T]:
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def one_token(
    llm: AsyncLLM | RecordedAsyncLLM,
    context: AsyncContext,
    forbidden: Sequence[int],
    sampling: SamplingParams | None = None,
    *,
    samples: Samples | None = None,
) -> int:
    async with aclosing(
        generate_with_targets(
            llm,
            context,
            samples,
            max_steps=1,
            forbid_ids=forbidden,
            sampling_params=sampling or SamplingParams(temperature=0.0, max_tokens=1),
        )
    ) as stream:
        return await anext(stream)


class PublicationGate:
    """Allow concurrent forwards and exclusive cache publication."""

    def __init__(self) -> None:
        self.condition = asyncio.Condition()
        self.readers = 0
        self.writers_waiting = 0
        self.writing = False

    @asynccontextmanager
    async def forward(self) -> AsyncIterator[None]:
        async with self.condition:
            await self.condition.wait_for(lambda: not self.writing and not self.writers_waiting)
            self.readers += 1
        try:
            yield
        finally:
            async with self.condition:
                self.readers -= 1
                self.condition.notify_all()

    @asynccontextmanager
    async def publish(self) -> AsyncIterator[None]:
        async with self.condition:
            self.writers_waiting += 1
            try:
                await self.condition.wait_for(lambda: not self.readers and not self.writing)
                self.writing = True
            finally:
                self.writers_waiting -= 1
                self.condition.notify_all()
        try:
            yield
        finally:
            async with self.condition:
                self.writing = False
                self.condition.notify_all()


def isolated(function: Callable[..., T | Awaitable[T]], *args: Any, **kwargs: Any) -> T:
    """Run an importable demo function in a fresh process/CUDA context."""
    module = sys.modules[function.__module__]
    module_name = module.__spec__.name if module.__name__ == "__main__" else module.__name__
    worker = """
import asyncio, importlib, inspect, pickle, sys
from pathlib import Path
module, name, args, kwargs = pickle.loads(sys.stdin.buffer.read())
result = getattr(importlib.import_module(module), name)(*args, **kwargs)
if inspect.isawaitable(result):
    result = asyncio.run(result)
Path(sys.argv[1]).write_bytes(pickle.dumps(result))
"""
    env = os.environ.copy()
    with TemporaryDirectory(prefix="parallel-thinking-", dir=Path.cwd()) as temporary:
        result_path = Path(temporary) / "result.pkl"
        process = subprocess.run(
            [sys.executable, "-u", "-c", worker, str(result_path)],
            input=pickle.dumps((module_name, function.__name__, args, kwargs)),
            capture_output=True,
            env=env,
        )
        if process.returncode:
            details = (process.stdout + process.stderr).decode(errors="replace")
            raise RuntimeError(
                f"{function.__name__} exited with status {process.returncode}\n{details}"
            )
        return pickle.loads(result_path.read_bytes())
