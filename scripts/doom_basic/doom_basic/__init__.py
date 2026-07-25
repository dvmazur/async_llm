"""Basic VLM agent loop: Qwen3.5 vision reasons about a ViZDoom frame, then a
logit probe over the action words turns that reasoning into a keypress."""

from __future__ import annotations

from .config import BasicConfig
from .demo import action_probe, build_probe, reason_and_act, run

__all__ = ["BasicConfig", "action_probe", "build_probe", "reason_and_act", "run"]
