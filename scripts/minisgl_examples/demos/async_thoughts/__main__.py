"""Command-line entry point for the async-thoughts demo.

Run with ``uv run --no-sync async-thoughts`` after installing the package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from minisgl_examples.demos.async_thoughts import __doc__ as _PKG_DOC
from minisgl_examples.demos.async_thoughts.demo import (
    DEFAULT_LLAMA_MODEL,
    DEFAULT_MAX_ANSWER_TOKENS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MEMORY_RATIO,
    DEFAULT_MODEL,
    DEFAULT_PAGE_SIZE,
    DEFAULT_PROBE_PERIOD,
    DEFAULT_PROBLEM,
    DemoConfig,
    run,
)


def parse_args(argv: list[str] | None = None, *, llama: bool = False) -> DemoConfig:
    default_model = DEFAULT_LLAMA_MODEL if llama else DEFAULT_MODEL
    p = argparse.ArgumentParser(
        prog="async-reasoning-llama" if llama else "async-thoughts",
        description=_PKG_DOC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=default_model,
        help=f"Hugging Face checkpoint ID or local directory (default: {default_model}).",
    )
    p.add_argument(
        "--problem", default=DEFAULT_PROBLEM, help="The user problem the assistant should solve."
    )
    p.add_argument(
        "--max-think-tokens",
        "--max-steps",
        dest="max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Thinker token budget (default: {DEFAULT_MAX_STEPS}); the writer can finish afterward.",
    )
    p.add_argument(
        "--max-answer-tokens",
        type=int,
        default=DEFAULT_MAX_ANSWER_TOKENS,
        help=f"Writer token budget (default: {DEFAULT_MAX_ANSWER_TOKENS}).",
    )
    p.add_argument(
        "--probe-period",
        type=int,
        default=DEFAULT_PROBE_PERIOD,
        help=f"Check writer readiness every N thinking tokens (default: {DEFAULT_PROBE_PERIOD}) "
        "and at thinker paragraph boundaries.",
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
    p.add_argument(
        "--trace-output",
        type=Path,
        help="Save executed requests and sampled-token targets to a new training JSON file.",
    )
    p.add_argument("--max-context-len", type=int, default=8192, help="Engine context capacity.")
    p.add_argument(
        "--llama-protocol",
        choices=("working-notes", "native-chat"),
        default="working-notes",
        help="Use labeled working notes or encode the Llama roles with its native chat template.",
    )
    p.add_argument(
        "--retry-short-answer",
        action="store_true",
        help="On Llama, spend remaining writer tokens on a native retry after a short EOS draft.",
    )
    args = p.parse_args(argv)
    if min(args.max_steps, args.max_answer_tokens, args.probe_period, args.max_context_len) < 1:
        p.error("Token budgets, probe period and context capacity must be positive")
    return DemoConfig(
        model=args.model,
        problem=args.problem,
        max_steps=args.max_steps,
        max_answer_tokens=args.max_answer_tokens,
        probe_period=args.probe_period,
        memory_ratio=args.memory_ratio,
        page_size=args.page_size,
        trace_output=args.trace_output,
        llama_protocol=args.llama_protocol,
        retry_short_answer=args.retry_short_answer,
        max_context_len=args.max_context_len,
    )


def _run(config: DemoConfig) -> None:
    if config.retry_short_answer:
        from minisgl_examples.demos.async_thoughts.short_retry import run_short_retry

        run_short_retry(config)
    else:
        run(config)


def main(argv: list[str] | None = None) -> None:
    _run(parse_args(argv))


def llama_main(argv: list[str] | None = None) -> None:
    _run(parse_args(argv, llama=True))


if __name__ == "__main__":
    main()
