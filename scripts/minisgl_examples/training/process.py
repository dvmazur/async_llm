"""Run demo stages in fresh processes while preserving the caller's environment."""

import os
import pickle
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

T = TypeVar("T")


def run_process(root: Path, arguments: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-u", *arguments],
            cwd=root,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                output.write(line)
                output.flush()
                if line.startswith(("initial:", "step ")):
                    print(line.strip(), flush=True)
            if process.wait():
                raise RuntimeError(f"Process failed. See {log}\n" + log.read_text()[-2000:])
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def isolated(function: Callable[..., T | Awaitable[T]], *args: Any, **kwargs: Any) -> T:
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
    with TemporaryDirectory(prefix="minisgl-demo-", dir=Path.cwd()) as temporary:
        result_path = Path(temporary) / "result.pkl"
        process = subprocess.run(
            [sys.executable, "-u", "-c", worker, str(result_path)],
            input=pickle.dumps((module_name, function.__name__, args, kwargs)),
            capture_output=True,
            env=os.environ.copy(),
        )
        if process.returncode:
            details = (process.stdout + process.stderr).decode(errors="replace")
            raise RuntimeError(
                f"{function.__name__} exited with status {process.returncode}\n{details}"
            )
        return pickle.loads(result_path.read_bytes())
