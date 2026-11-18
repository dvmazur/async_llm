"""Extract and symbolically grade every boxed answer in a correction rollout."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def boxed_answers(text: str) -> list[str]:
    """Extract complete ``\\boxed{...}`` expressions, including nested braces."""
    prefix = r"\boxed{"
    answers: list[str] = []
    cursor = 0
    while True:
        start = text.find(prefix, cursor)
        if start < 0:
            return answers
        depth = 1
        position = start + len(prefix)
        while position < len(text) and depth:
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
            position += 1
        if depth == 0:
            answers.append(text[start + len(prefix) : position - 1].strip())
            cursor = position
        else:
            cursor = start + len(prefix)


def _symbolically_equal(reference: str, candidate: str) -> bool:
    from math_verify import parse, verify

    try:
        return bool(
            verify(parse(r"\boxed{" + reference + "}"), parse(r"\boxed{" + candidate + "}"))
        )
    except Exception:
        return False


def grade_text(
    text: str,
    reference: str,
    *,
    equivalent: Callable[[str, str], bool] = _symbolically_equal,
) -> dict[str, Any]:
    """Return correct/incorrect/mixed while retaining every individual decision."""
    decisions = [
        {"answer": answer, "correct": equivalent(reference, answer)}
        for answer in boxed_answers(text)
    ]
    correct = sum(decision["correct"] for decision in decisions)
    incorrect = len(decisions) - correct
    label = "mixed" if correct and incorrect else "correct" if correct else "incorrect"
    return {
        "label": label,
        "answers": decisions,
        "correct_answers": correct,
        "incorrect_answers": incorrect,
        "has_answer": bool(decisions),
    }
