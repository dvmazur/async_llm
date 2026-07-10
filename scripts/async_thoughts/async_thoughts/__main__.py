"""Command-line entry point for the async-thoughts demo.

Runnable as ``python -m async_thoughts`` or via the ``async-thoughts`` console
script (see pyproject.toml). Builds an offline scheduler (``minisgl.llm.LLM``)
and drives a ``ReasoningDriver`` against its ``shared_cache_service``, with
colored live streaming of the thinker/writer tokens.
"""

from __future__ import annotations

import argparse
import gc
import sys

import torch

from . import __doc__ as _PKG_DOC
from .driver import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MEMORY_RATIO,
    DEFAULT_MODEL,
    DEFAULT_PAGE_SIZE,
    DEFAULT_PROBE_PERIOD,
    DEFAULT_PROBLEM,
    ReasoningConfig,
    ReasoningDriver,
)

# ANSI colours keyed by role.
_C = {
    "thinker": "\033[2;36m",  # dim cyan
    "writer": "\033[1;32m",  # bold green
    "state": "\033[1;33m",  # bold yellow
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def _ansi(text: str, *keys: str) -> str:
    return "".join(_C[k] for k in keys) + text + _C["reset"]


def _print_header(text: str) -> None:
    bar = "─" * 64
    print(f"\n{bar}")
    print(_ansi(f"  {text}", "state"))
    print(f"{bar}\n", flush=True)


def _print_state_change(text: str) -> None:
    print(_ansi(f"\n  [{text}]", "state"), flush=True)


def _stream_token(text: str, role: str) -> None:
    sys.stdout.write(_ansi(text, role))
    sys.stdout.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="async-thoughts",
        description=_PKG_DOC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model path (default: {DEFAULT_MODEL}; override with MINISGL_DEMO_MODEL).",
    )
    p.add_argument(
        "--problem", default=DEFAULT_PROBLEM, help="The user problem the assistant should solve."
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Hard cap on total decode steps (default: {DEFAULT_MAX_STEPS}).",
    )
    p.add_argument(
        "--probe-period",
        type=int,
        default=DEFAULT_PROBE_PERIOD,
        help="Run the mode-switching probe every N decode steps "
        f"(default: {DEFAULT_PROBE_PERIOD}; plus on every thinker "
        "end-of-step).",
    )
    p.add_argument(
        "--memory-ratio",
        type=float,
        default=DEFAULT_MEMORY_RATIO,
        help="Fraction of free GPU memory to reserve for weights + KV "
        f"cache (default: {DEFAULT_MEMORY_RATIO}).",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="The number of tokens in a page.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(2)

    _print_header("Async Thoughts Demo (minisgl)")
    print(f"  model    : {args.model}")
    print(f"  problem  : {args.problem}")
    print(f"  max steps: {args.max_steps}")

    print("\nLoading engine...")
    from minisgl.llm import LLM

    llm = LLM(
        args.model,
        page_size=args.page_size,
        memory_ratio=args.memory_ratio,
        max_running_req=4,
        cuda_graph_bs=[1, 2],  # we only ever run 1- or 2-worker decode
        cuda_graph_max_bs=2,
        max_seq_len_override=4096,
    )
    try:
        _print_header("Generation")
        print(_ansi("  thinker (dim cyan) | writer (bold green)\n", "dim"))

        driver = ReasoningDriver(
            backend=llm.shared_cache_service,
            tokenizer=llm.tokenizer,
            problem=args.problem,
            config=ReasoningConfig(max_steps=args.max_steps, probe_period=args.probe_period),
            device=llm.engine.device,
            on_thinker_token=lambda text: _stream_token(text, "thinker"),
            on_writer_token=lambda text: _stream_token(text, "writer"),
            on_state_change=_print_state_change,
        )
        result = driver.run()

        _print_header("Final")
        print(_ansi("  Thinker:", "thinker"))
        print(_ansi(str(result["thinker_text"]), "thinker"))
        print()
        print(_ansi("  Writer:", "writer"))
        print(_ansi(str(result["writer_text"]), "writer"))
        print()
    finally:
        llm.engine.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
