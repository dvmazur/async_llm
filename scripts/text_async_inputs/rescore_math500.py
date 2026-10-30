"""Recompute saved Math-500 scores without rerunning model generation."""

import argparse
import json
import os
from pathlib import Path

from math_verify import parse, verify


def find_last_boxed_answer(text: str):
    prefix = r"\boxed{"
    end = len(text)
    while True:
        start = text.rfind(prefix, 0, end)
        if start < 0:
            return None
        depth = 0
        for pos in range(start + len(r"\boxed"), len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    return text[start + len(prefix):pos].strip()
        end = start


def is_equal(predicted, reference):
    if predicted is None:
        return False
    try:
        return bool(verify(
            parse(r"\boxed{" + str(reference) + "}"),
            parse(r"\boxed{" + predicted + "}"),
        ))
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    for result_dir in sorted(args.results_dir.glob("k_*")):
        if not result_dir.is_dir():
            continue
        correct = total = 0
        for path in sorted(result_dir.glob("sample_*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            predicted = find_last_boxed_answer(data.get("generated_text", ""))
            score = is_equal(predicted, data.get("correct_answer", ""))
            data["predicted_answer"] = predicted
            data["is_equal"] = score
            data["grader"] = "math_verify"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, path)
            correct += int(score)
            total += 1
        summary = {
            "k_steps": int(result_dir.name.removeprefix("k_")),
            "accuracy": correct / total if total else None,
            "correct": correct,
            "total": total,
            "grader": "math_verify",
            "partial": total < 500,
        }
        (result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{result_dir.name}: {correct}/{total}")


if __name__ == "__main__":
    main()
