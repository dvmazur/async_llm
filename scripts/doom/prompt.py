"""Prompt strings for the Doom demo.

Turn structure fed to the model:
    system role  ->  user request  ->  [frame image blocks]  ->  assistant reason -> action

These are the prompts that actually clear VizdoomBasic on Qwen3.5-27B (found via
scripts/doom/doom_sweep*.py).  Two things mattered most: a system prompt that
corrects the game mechanics (you cannot move forward/back — only strafe to centre
the monster under the fixed crosshair, then fire), and letting the doer *reason*
before it commits to a key-name action.
"""

from __future__ import annotations

# System: correct the mechanics the model gets wrong on its own, then open the user turn.
SYS_TEXT = (
    "<|im_start|>system\n"
    "You control the player in Doom. A crosshair is fixed at the CENTRE of the screen. "
    "You CANNOT move forward or back; you can only strafe LEFT/RIGHT, or fire (SPACE). "
    "The monster is already in range. To hit it, strafe toward the side it is on until "
    "it sits under the centre crosshair, then fire."
    "<|im_end|>\n<|im_start|>user\n"
)

# User request: how to read the frame queue (``{k}`` filled at runtime); the frame
# image blocks are appended right after this text, inside the user turn.
# NOTE: exact wording from the winning sweep (doom_sweep3.py, candidate "ra") — changing
# it shifts the greedy trajectory and the doer over-strafes / mis-times the shot.
USER_TEXT = "The last {k} frames (oldest first).\nFrames:"

# Doer, reason-then-act: a short frame-grounded analysis, then the key-name action.
# The exact wording matters — this phrasing commits to firing once centred, whereas a
# "Step 1 / Step 2" framing made the doer oscillate and never fire.
DOER_REASON_PREFIX = (
    "<|im_end|>\n<|im_start|>assistant\nRelative to the centre crosshair, the monster is"
)
DOER_ACT_QUERY = "\nTo put it under the crosshair and shoot, I press:"

# Baseline reactive doer (no reasoning) — kept for comparison; it is a constant "fire".
DOER_QUERY_BASE = "<|im_end|>\n<|im_start|>assistant\nPress "

# --- optional persistent thinker (secondary; the per-action reasoning above is the driver) ---
THINK_PREFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    "My plan to clear the room, based on the most recent frame:\n"
)
FRAME_HINT = "\n[Screen is updated]\n"
