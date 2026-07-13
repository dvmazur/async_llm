"""Command-line entry point for the async-thoughts demo.

Runnable as ``python -m async_thoughts`` or via the ``async-thoughts`` console
script (see pyproject.toml).
"""

from __future__ import annotations

import argparse

from . import __doc__ as _PKG_DOC
from .demo import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MEMORY_RATIO,
    DEFAULT_MODEL,
    DEFAULT_PAGE_SIZE,
    DEFAULT_PROBE_PERIOD,
    DEFAULT_PROBLEM,
    DemoConfig,
    run,
)


def parse_args(argv: list[str] | None = None) -> DemoConfig:
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
    args = p.parse_args(argv)
    return DemoConfig(
        model=args.model,
        problem=args.problem,
        max_steps=args.max_steps,
        probe_period=args.probe_period,
        memory_ratio=args.memory_ratio,
        page_size=args.page_size,
    )


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
