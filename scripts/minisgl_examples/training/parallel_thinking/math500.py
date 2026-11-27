"""Load the pinned MATH-500 test set and score the contents of a final answer box."""

from __future__ import annotations

import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from math_verify import LatexExtractionConfig, parse, verify

DATASET = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
DATASET_FILE = "test.jsonl"
GRADING_POLICY = (
    "Final complete LaTeX box; Math-Verify 0.9.0 strict symbolic equivalence, "
    "default six-decimal numeric tolerance, no string fallback. Text labels use "
    "case-insensitive equality. Missing/unparsed answers count as incorrect."
)


def load_math500() -> tuple[Path, list[dict]]:
    path = Path(hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 500:
        raise ValueError(f"Expected 500 MATH-500 test problems, found {len(rows)}")
    for index, row in enumerate(rows):
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("unique_id", "problem", "answer")):
            raise ValueError(f"Incomplete MATH-500 problem at index {index}")
    if len({row["unique_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate MATH-500 problem ID")
    return path, rows


def _label(answer: str) -> str | None:
    """Recognize unambiguous word or choice labels for text-valued gold answers."""
    wrapper = re.fullmatch(r"\\(?:text|mathrm|textrm)\{([^{}]+)\}", answer.strip())
    text = (wrapper.group(1) if wrapper else answer).strip()
    if re.fullmatch(r"\([A-Za-z]\)", text):
        text = text[1:-1]
    if re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z]+)*", text):
        return " ".join(text.casefold().split())
    return None


def _parse(answer: str) -> list:
    # Both inputs are already box contents. Restrict extraction to that one
    # expression so an unrelated number cannot become a fallback prediction.
    return parse(
        rf"\boxed{{{answer}}}",
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
        fallback_mode="no_fallback",
    )


def grade_math500(gold: str, prediction: str | None) -> str:
    """Return correct, incorrect, unparsed_gold, or unparsed_prediction."""
    if not isinstance(gold, str) or not gold.strip():
        return "unparsed_gold"
    if not isinstance(prediction, str) or not prediction.strip():
        return "unparsed_prediction"
    if gold.strip().startswith(r"\text{") and (gold_label := _label(gold)) is not None:
        predicted_label = _label(prediction)
        if predicted_label is not None:
            return "correct" if gold_label == predicted_label else "incorrect"
    expected = _parse(gold)
    if not expected:
        return "unparsed_gold"
    observed = _parse(prediction)
    if not observed:
        return "unparsed_prediction"
    return "correct" if verify(expected, observed, strict=True) else "incorrect"
