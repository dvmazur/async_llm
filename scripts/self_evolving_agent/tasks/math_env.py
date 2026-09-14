from __future__ import annotations

import re
from typing import Any, Optional

# The x=5,6,7,8 problems are async_thoughts/demo.py's DEFAULT_PROBLEM split
# into four separate exact-match problems.
DEFAULT_PROBLEMS: list[dict[str, Any]] = [
    {"problem": "Calculate x - x^2 + x^3 for x = 5. Return the final answer in \\boxed{}.", "answer": 105},
    {"problem": "Calculate x - x^2 + x^3 for x = 6. Return the final answer in \\boxed{}.", "answer": 186},
    {"problem": "Calculate x - x^2 + x^3 for x = 7. Return the final answer in \\boxed{}.", "answer": 301},
    {"problem": "Calculate x - x^2 + x^3 for x = 8. Return the final answer in \\boxed{}.", "answer": 456},
    {"problem": "What is 17 * 23? Return the final answer in \\boxed{}.", "answer": 391},
    {"problem": "What is 144 / 12 + 7? Return the final answer in \\boxed{}.", "answer": 19},
    {"problem": "A train travels 60 miles in 45 minutes. What is its speed in miles per hour? "
                 "Return the final answer in \\boxed{}.", "answer": 80},
    {"problem": "What is the sum of the first 10 positive integers? Return the final answer in \\boxed{}.",
     "answer": 55},
]

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def extract_final_number(text: str) -> Optional[float]:
    boxed = _BOXED_RE.findall(text)
    candidate = boxed[-1] if boxed else None
    if candidate is None:
        numbers = _NUMBER_RE.findall(text)
        if not numbers:
            return None
        candidate = numbers[-1]
    cleaned = re.sub(r"[^0-9.\-]", "", candidate)
    try:
        return float(cleaned)
    except ValueError:
        return None


class MathEnv:
    """A doom_basic-style GymEnv: reset() -> observation (problem text),
    step(action) -> (obs, reward, done, info). Each episode is one problem;
    reward is 1.0/0.0 for an exact numeric match."""

    name = "math"

    def __init__(self, problems: Optional[list[dict[str, Any]]] = None) -> None:
        self.problems = problems if problems is not None else DEFAULT_PROBLEMS
        self.max_episodes = len(self.problems)
        self._idx = -1

    def reset(self) -> Any:
        self._idx = (self._idx + 1) % len(self.problems)
        return self.problems[self._idx]["problem"]

    def restart(self) -> None:
        """Reset the episode cursor back to the beginning, so the next
        reset() starts again from problem 0 instead of resuming mid-cycle --
        used by SelfEditEnv.restart_task() for a clean, comparable re-run."""
        self._idx = -1

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        expected = self.problems[self._idx]["answer"]
        got = extract_final_number(str(action)) if action is not None else None
        correct = got is not None and abs(got - expected) < 1e-6
        reward = 1.0 if correct else 0.0
        return None, reward, True, {"expected": expected, "got": got}
