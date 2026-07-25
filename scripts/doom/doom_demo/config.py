"""Config + constants for the Doom demo (scripts/doom/doom_demo.py)."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Model: env override, else a small default that runs anywhere.
DEFAULT_MODEL = os.environ.get("MINISGL_DEMO_MODEL", "Qwen/Qwen3.5-0.8B")

# Qwen3.5 vision special ids + spatial-merge (see minisgl.models config).
IMG_TOK, VSTART, VEND, MERGE = 248056, 248053, 248054, 2

# VizdoomBasic-v1 is Discrete(4).
ACTIONS = {0: "do nothing", 1: "fire", 2: "move right", 3: "move left"}
# The doer chooses a key *word* (strong prior) among these; "do nothing" is
# deliberately excluded — offering it made the doer idle instead of committing.
KEYNAMES = {1: "SPACE", 2: "RIGHT", 3: "LEFT"}


@dataclass
class DoomConfig:
    """Everything the Doom demo needs, with sane defaults baked in.

    Defaults are the config that actually clears VizdoomBasic on Qwen3.5-27B
    (mean reward ~+69 over seeds 0-3): the doer *reasons* about the frame, then
    presses a key-name action, at high resolution.  The single-token reactive
    doer (and prompt tweaks to it) is a constant "fire" — see doom_sweep*.py.
    """

    model: str = DEFAULT_MODEL
    env_id: str = "VizdoomBasic-v1"
    steps: int = 60
    seed: int | None = None  # env reset seed (fixes the monster spawn; None = random)
    frame_skip: int = 4
    max_pixels: int = 150 * 1000  # high enough to localize the monster (lower => can't aim)
    k_frames: int = 2  # keep the last K frames in context (a queue of image blocks)
    user_prompt: str = ""  # override the user request text (empty -> default USER_TEXT)
    reason: bool = True  # doer reasons about the frame before the key-name action (needed to aim)
    reason_tokens: int = 28  # per-action reasoning length
    reason_temp: float = 0.0  # 0 => greedy reasoning (deterministic; no sampling kernel)
    thinker: bool = False  # also run a separate persistent thinker whose plan the doer reads
    thinker_temp: float = 0.7
    thinker_tokens: int = 8
    frame_hint: bool = False  # splice a "screen updated" note into the thinker on each new frame
    memory_ratio: float = 0.9
    out: str = "doom.gif"
