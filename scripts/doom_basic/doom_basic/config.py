"""Config + constants for the basic Doom VLM demo."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Model: env override, else a small default that runs anywhere.
DEFAULT_MODEL = os.environ.get("MINISGL_DEMO_MODEL", "Qwen/Qwen3.5-0.8B")

# Qwen3.5 vision special ids + spatial-merge (see minisgl.models config).
IMG_TOK, VSTART, VEND, MERGE = 248056, 248053, 248054, 2

# VizdoomBasic-v1 is Discrete(4).
ACTIONS = {0: "do nothing", 1: "fire", 2: "move right", 3: "move left"}
# The action *words* the probe scores.  "do nothing" is deliberately not among
# them — with an idle option on the menu the model never commits to a shot.
KEYNAMES = {1: "FIRE", 2: "RIGHT", 3: "LEFT"}


@dataclass
class BasicConfig:
    """Everything the basic demo needs, with sane defaults baked in.

    Per env step: one self-contained prompt (system + user + current frame), a
    short abstract reasoning trace generated from it, then a logit probe on top
    of that trace for the action.  Nothing is carried in the KV cache between
    steps; the only memory the model has is the ``history`` action list rendered
    into the prompt as text.
    """

    model: str = DEFAULT_MODEL
    env_id: str = "VizdoomBasic-v1"
    steps: int = 60
    seed: int | None = None  # env reset seed (fixes the monster spawn; None = random)
    frame_skip: int = 4
    max_pixels: int = 150 * 1000  # high enough to localize the monster (lower => can't aim)
    reason_tokens: int = 32  # cap on the reasoning trace (MAX_REASONING_STEPS)
    temperature: float = 0.0  # 0 => greedy reasoning (deterministic; no sampling kernel)
    top_p: float = 0.95
    history: int = 4  # recent actions shown in the prompt (0 = stateless)
    memory_ratio: float = 0.9
    out: str = "doom_basic.gif"
