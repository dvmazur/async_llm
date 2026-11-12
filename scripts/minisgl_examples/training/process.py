"""Run demo stages in fresh processes while preserving the caller's environment."""

import pickle
import subprocess
import sys
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

T = TypeVar("T")


def isolated(
    function: Callable[..., T | Awaitable[T]],
    *args: Any,
    log_path: str | Path | None = None,
    **kwargs: Any,
) -> T:
    """Execute an importable function and return its result after the child exits."""
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
    log = Path(log_path) if log_path is not None else None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="minisgl-demo-", dir=Path.cwd()) as temporary:
        result_path = Path(temporary) / "result.pkl"
        with log.open("wb") if log is not None else nullcontext() as output:
            process = subprocess.run(
                [sys.executable, "-u", "-c", worker, str(result_path)],
                input=pickle.dumps((module_name, function.__name__, args, kwargs)),
                stdout=output if output is not None else subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        if process.returncode:
            details = (
                log.read_text(errors="replace")
                if log is not None
                else process.stdout.decode(errors="replace")
            )
            raise RuntimeError(
                f"{function.__name__} exited with status {process.returncode}\n{details[-2000:]}"
            )
        return pickle.loads(result_path.read_bytes())
