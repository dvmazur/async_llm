"""Small async/cache helpers shared by the two notebooks."""

import asyncio
import os
import subprocess
import sys
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import cloudpickle
from minisgl.core import SamplingParams


async def together(*coroutines):
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def one_token(llm, context, forbidden):
    async with aclosing(
        llm.async_generate(
            context,
            max_steps=1,
            forbid_ids=forbidden,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        )
    ) as stream:
        return await anext(stream)


class PublicationGate:
    """Concurrent forwards; exclusive flush/append/clear/prefix publication.

    Writers stop new forwards, then wait for the existing ones to finish.
    There is no rendezvous after each token and no fixed participant count.
    """

    def __init__(self):
        self.condition = asyncio.Condition()
        self.readers = 0
        self.writers_waiting = 0
        self.writing = False

    @asynccontextmanager
    async def forward(self):
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
    async def publish(self):
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


def recorded_example(recorder):
    """The same row alignment as DemoRecorder.save, kept entirely in memory.

    DemoRecorder currently exposes samples through _samples, but has no export
    method other than save. Keep that dependency isolated here.
    """
    targets = [-100] * recorder.trace.num_tokens
    for row, (token, _) in recorder._samples.items():
        targets[row] = token
    return {"trace": recorder.trace, "targets": targets}


def isolated(function, *args, **kwargs):
    """Run a notebook function with a fresh CUDA context; no persistent artifacts.

    cloudpickle captures the executed notebook definitions, including unsaved
    edits. Only call this with our own trusted functions and arguments.
    """
    worker = """
import asyncio, inspect, sys, cloudpickle
from pathlib import Path
function, args, kwargs = cloudpickle.loads(sys.stdin.buffer.read())
result = function(*args, **kwargs)
if inspect.isawaitable(result):
    result = asyncio.run(result)
Path(sys.argv[1]).write_bytes(cloudpickle.dumps(result))
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    with TemporaryDirectory(prefix="parallel-thinking-") as temporary:
        result_path = Path(temporary) / "result.pkl"
        process = subprocess.run(
            [sys.executable, "-u", "-c", worker, str(result_path)],
            input=cloudpickle.dumps((function, args, kwargs)),
            capture_output=True,
            env=env,
        )
        if process.returncode:
            raise RuntimeError(process.stderr.decode(errors="replace"))
        return cloudpickle.loads(result_path.read_bytes())
