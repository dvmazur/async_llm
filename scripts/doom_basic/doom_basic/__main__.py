"""Command-line entry point for the basic Doom VLM demo.

Runnable as ``python -m doom_basic`` or via the ``doom-basic`` console script
(see pyproject.toml).  Flag defaults come from :class:`BasicConfig`, so the
dataclass stays the single place where the defaults are recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import sys

import torch

from .config import BasicConfig
from .demo import __doc__ as _DEMO_DOC
from .demo import run

_D = BasicConfig()  # defaults for the flags below


def parse_args(argv: list[str] | None = None) -> BasicConfig:
    p = argparse.ArgumentParser(prog="doom-basic", description=_DEMO_DOC,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=_D.model,
                   help=f"HF model path (default: {_D.model}; override with MINISGL_DEMO_MODEL).")
    p.add_argument("--env-id", default=_D.env_id)
    p.add_argument("--steps", type=int, default=_D.steps)
    p.add_argument("--seed", type=int, default=_D.seed, help="Env reset seed (fixes the spawn).")
    p.add_argument("--frame-skip", type=int, default=_D.frame_skip)
    p.add_argument("--max-pixels", type=int, default=_D.max_pixels,
                   help="Frame smart-resized to fit this (lower => can't localize the monster).")
    p.add_argument("--reason-tokens", type=int, default=_D.reason_tokens,
                   help="Cap on the per-step reasoning trace before the probe.")
    p.add_argument("--temperature", type=float, default=_D.temperature,
                   help="Reasoning sampling temperature (0 = greedy).")
    p.add_argument("--top-p", type=float, default=_D.top_p)
    p.add_argument("--history", type=int, default=_D.history,
                   help="Recent actions listed in the prompt (0 = fully stateless).")
    p.add_argument("--memory-ratio", type=float, default=_D.memory_ratio)
    p.add_argument("--out", default=_D.out, help="GIF of the episode (empty to skip).")
    a = p.parse_args(argv)
    return BasicConfig(model=a.model, env_id=a.env_id, steps=a.steps, seed=a.seed,
                       frame_skip=a.frame_skip, max_pixels=a.max_pixels,
                       reason_tokens=a.reason_tokens, temperature=a.temperature, top_p=a.top_p,
                       history=a.history, memory_ratio=a.memory_ratio, out=a.out)


def main(argv: list[str] | None = None) -> None:
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(2)
    cfg = parse_args(argv)
    try:
        asyncio.run(run(cfg))
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
