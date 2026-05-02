#!/usr/bin/env python3
"""
Run the shared-cache standalone demo defined in ``tests/core/test_shared_cache.py``.

Usage (from the ``mini-sglang`` repo root, with dependencies installed)::

    MINISGL_E2E_MODEL=Qwen/Qwen2.5-0.5B python scripts/run_shared_cache_demo.py

Or from anywhere::

    MINISGL_E2E_MODEL=Qwen/Qwen2.5-0.5B \\
        python /path/to/mini-sglang/scripts/run_shared_cache_demo.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    py_dir = repo_root / "python"
    if str(py_dir) not in sys.path:
        sys.path.insert(0, str(py_dir))

    if not os.environ.get("MINISGL_E2E_MODEL"):
        print(
            "Error: set MINISGL_E2E_MODEL to a Hugging Face model id or local path "
            "(e.g. Qwen/Qwen2.5-0.5B).",
            file=sys.stderr,
        )
        return 2

    import torch

    if not torch.cuda.is_available():
        print("Error: CUDA is required for this demo.", file=sys.stderr)
        return 2

    test_path = repo_root / "tests" / "core" / "test_shared_cache.py"
    if not test_path.is_file():
        print(f"Error: expected test file at {test_path}", file=sys.stderr)
        return 1

    spec = importlib.util.spec_from_file_location(
        "_minisgl_shared_cache_demo_loader_",
        test_path,
    )
    if spec is None or spec.loader is None:
        print("Error: could not load test module.", file=sys.stderr)
        return 1

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._run_standalone_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
