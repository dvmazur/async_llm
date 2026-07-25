"""Command-line entry point for the Doom demo.

Runnable as ``python -m doom_demo`` or via the ``doom-demo`` console script
(see pyproject.toml).  Flag defaults come from :class:`DoomConfig`, so the
dataclass stays the single place where the winning configuration is recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import sys

import torch

from .config import DoomConfig
from .demo import __doc__ as _DEMO_DOC
from .demo import run

_D = DoomConfig()  # defaults for the flags below


def parse_args(argv: list[str] | None = None) -> DoomConfig:
    p = argparse.ArgumentParser(prog="doom-demo", description=_DEMO_DOC,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=_D.model,
                   help=f"HF model path (default: {_D.model}; override with MINISGL_DEMO_MODEL).")
    p.add_argument("--env-id", default=_D.env_id)
    p.add_argument("--steps", type=int, default=_D.steps)
    p.add_argument("--seed", type=int, default=_D.seed, help="Env reset seed (fixes the spawn).")
    p.add_argument("--frame-skip", type=int, default=_D.frame_skip)
    p.add_argument("--max-pixels", type=int, default=_D.max_pixels,
                   help="Frame smart-resized to fit this (lower => can't localize the monster).")
    p.add_argument("--k-frames", type=int, default=_D.k_frames,
                   help="Keep the last K screen frames in context (image-block queue).")
    p.add_argument("--user-prompt", default=_D.user_prompt, help="Override the user request text.")
    p.add_argument("--no-reason", action="store_true",
                   help="Reactive baseline: single key with no reasoning (a constant 'fire').")
    p.add_argument("--reason-tokens", type=int, default=_D.reason_tokens,
                   help="Per-action reasoning length before the key-name action.")
    p.add_argument("--reason-temp", type=float, default=_D.reason_temp,
                   help="Reasoning sampling temperature (0 = greedy).")
    p.add_argument("--thinker", action="store_true",
                   help="Also run a separate persistent thinker whose plan the doer reads.")
    p.add_argument("--thinker-temp", type=float, default=_D.thinker_temp)
    p.add_argument("--thinker-tokens", type=int, default=_D.thinker_tokens)
    p.add_argument("--frame-hint", action="store_true",
                   help="Splice a 'Screen is updated' note into the thinker on each new frame.")
    p.add_argument("--memory-ratio", type=float, default=_D.memory_ratio)
    p.add_argument("--out", default=_D.out, help="GIF of the episode (empty to skip).")
    a = p.parse_args(argv)
    return DoomConfig(model=a.model, env_id=a.env_id, steps=a.steps, seed=a.seed,
                      frame_skip=a.frame_skip,
                      max_pixels=a.max_pixels, k_frames=a.k_frames, user_prompt=a.user_prompt,
                      reason=not a.no_reason, reason_tokens=a.reason_tokens, reason_temp=a.reason_temp,
                      thinker=a.thinker, thinker_temp=a.thinker_temp, thinker_tokens=a.thinker_tokens,
                      frame_hint=a.frame_hint, memory_ratio=a.memory_ratio, out=a.out)


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
