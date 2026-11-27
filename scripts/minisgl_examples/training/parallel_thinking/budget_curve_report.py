"""Report a frozen matched cohort from budget-curve JSONL results, without regrading.

Required fields: mode, workers, index, budget, correct, generated_tokens,
answer_tokens, source_trace_sha256, wall_seconds. Extra fields are retained.
``generated_tokens`` includes reasoning and answer samples; ``answer_tokens``
counts at most eight newly sampled tokens, including a sampled EOS.

Run ``python -m minisgl_examples.training.parallel_thinking.budget_curve_report
results.jsonl --output report``. Only newline-committed input rows are included.

An optional --native-reference accepts either native per-task JSONL rows
(mode=native, workers=1, index, correct), or JSON with accuracy and tasks.
The latter is explicitly labeled a separate-cohort 16k reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

BUDGETS = tuple(range(1000, 8001, 1000))
CONDITIONS = (("native", 1), *(("hogwild", n) for n in (1, 2, 4, 6, 8, 16)))
EXPECTED_TASKS = 500
REQUIRED = (
    "mode", "workers", "index", "budget", "correct", "generated_tokens",
    "answer_tokens", "source_trace_sha256", "wall_seconds",
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _jsonl(path):
    raw = path.read_bytes()
    rows, ignored = [], 0
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.endswith(b"\n"):
            ignored = len(line)
            break
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            _require(isinstance(row, dict), "Expected a JSON object")
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        rows.append((number, row))
    return rows, {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                  "ignored_uncommitted_trailing_bytes": ignored}


def verify_completion(root, rows, source):
    """Bind complete readouts to their saved plan, exact keys and immutable result bytes."""
    root = Path(root)
    plan_raw = (root / "plan.json").read_bytes()
    plan = json.loads(plan_raw)["config"]
    complete = json.loads((root / "complete.json").read_text())
    conditions = {f"{mode}-w{workers}": (mode, workers) for mode, workers in CONDITIONS}
    expected = {(*conditions[name], index, budget) for name in plan["conditions"]
                for index in plan["indices"] for budget in plan["budgets"]}
    _require(not source["ignored_uncommitted_trailing_bytes"] and set(rows) == expected,
             f"Incomplete or unexpected scored keys in {root}")
    _require(complete["plan_sha256"] == hashlib.sha256(plan_raw).hexdigest()
             and complete["completed"] == complete["target"] == len(expected),
             f"Completion record differs from the saved plan in {root}")
    expected_digest = complete.get("results_sha256")
    if expected_digest is None:
        # Earlier completed campaigns stored the same fingerprint in their CPU audit.
        audit = json.loads((root / "completion-verification.json").read_text())
        _require(audit["status"] == "passed", f"Prior completion audit failed in {root}")
        expected_digest = audit.get("results_sha256")
        for field in ("sha256_before_and_after", "source_result_report_sha256"):
            for path, digest in audit.get(field, {}).items():
                if path.endswith(f"/{root.name}/results.jsonl"):
                    expected_digest = digest
    _require(expected_digest == source["sha256"], f"Completed result bytes changed in {root}")
    return plan


def load_rows(path: Path):
    records, source = _jsonl(path)
    rows, traces = {}, {}
    for number, row in records:
        try:
            _require(all(key in row for key in REQUIRED), "Missing required result field")
            mode, workers, index, budget = (row[key] for key in ("mode", "workers", "index", "budget"))
            _require(type(workers) is int and (mode, workers) in CONDITIONS, "Unknown condition")
            _require(type(index) is int and 0 <= index < EXPECTED_TASKS, "Invalid MATH-500 index")
            _require(type(budget) is int and budget in BUDGETS, "Unknown budget")
            _require(type(row["correct"]) is bool, "correct must be a boolean")
            _require(type(row["generated_tokens"]) is int and row["generated_tokens"] >= 0,
                     "generated_tokens must be a nonnegative integer")
            _require(type(row["answer_tokens"]) is int and 0 <= row["answer_tokens"] <= 8
                     and row["answer_tokens"] <= row["generated_tokens"], "Invalid answer token count")
            seconds = row["wall_seconds"]
            _require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
                     "Invalid wall_seconds")
            digest = row["source_trace_sha256"]
            _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "Invalid source trace SHA256")
            task = (mode, workers, index)
            _require(traces.setdefault(task, digest) == digest, "Different source traces across task budgets")
            key = (*task, budget)
            _require(key not in rows, f"Duplicate result {key}")
            rows[key] = row
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return rows, source


def _reference(path, matched):
    if path is None:
        return None
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        _require(isinstance(data, dict), "Reference summary must be a JSON object")
        _require(data.get("mode", "native") == "native" and data.get("workers", 1) == 1,
                 "Reference summary must describe ordinary native Qwen")
        accuracy, tasks = data.get("accuracy"), data.get("tasks")
        _require(type(accuracy) in (int, float) and math.isfinite(accuracy) and 0 <= accuracy <= 1,
                 "Reference accuracy must lie between zero and one")
        _require(type(tasks) is int and 0 < tasks <= EXPECTED_TASKS, "Invalid reference task count")
        _require(data.get("budget", 16384) == 16384, "Reference must use the 16k budget")
        return {"accuracy": accuracy, "tasks": tasks, "budget": 16384, "paired": False,
                "label": f"Qwen 16k reference ({tasks} tasks; separate cohort)", "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    entries, source = _jsonl(path)
    rows = {}
    for _, row in entries:
        _require(row.get("mode") == "native" and type(row.get("workers")) is int and row["workers"] == 1,
                 "Reference JSONL must contain only ordinary native Qwen")
        index = row.get("index")
        _require(type(index) is int and 0 <= index < EXPECTED_TASKS and type(row.get("correct")) is bool,
                 "Invalid reference task or grade")
        _require(row.get("budget", 16384) == 16384, "Reference must use the 16k budget")
        _require(index not in rows, f"Duplicate reference task {index}")
        rows[index] = row
    _require(set(matched) <= rows.keys(), "Reference is missing tasks from the matched cohort")
    return {"accuracy": sum(rows[i]["correct"] for i in matched) / len(matched) if matched else None,
            "tasks": len(matched), "budget": 16384, "paired": True,
            "label": "Qwen 16k reference (same matched tasks)", **source}


def _csv(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return stream.getvalue()


def _write(path, contents):
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(contents)
            file.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)




def build_report(input_path: Path, output_dir: Path, *, native_reference: Path | None = None):
    rows, source = load_rows(Path(input_path))
    groups = {(mode, workers, budget): [] for mode, workers in CONDITIONS for budget in BUDGETS}
    for (mode, workers, _, budget), row in rows.items():
        groups[(mode, workers, budget)].append(row)
    matched = set.intersection(*({row["index"] for row in group} for group in groups.values()))
    cells = []
    for (mode, workers, budget), group in groups.items():
        cohort = [row for row in group if row["index"] in matched]
        n = len(cohort)
        cell = {"mode": mode, "workers": workers, "budget": budget, "completed_tasks": len(group),
                "pending_tasks": EXPECTED_TASKS - len(group), "matched_tasks": n,
                "correct": sum(row["correct"] for row in cohort) if n else None,
                "accuracy": sum(row["correct"] for row in cohort) / n if n else None}
        for field in ("generated_tokens", "answer_tokens", "wall_seconds"):
            total = sum(row[field] for row in cohort)
            cell[f"total_{field}"] = total if n else None
            cell[f"mean_{field}"] = total / n if n else None
        for field, value, metric in (("stop_reason", "max_tokens", "answer_cap_reached"),
                                     ("grading_status", "unparsed_prediction", "unparsed_prediction")):
            observed = [row for row in cohort if isinstance(row.get(field), str)]
            count = sum(row[field] == value for row in observed)
            cell[f"{field}_tasks"] = len(observed)
            cell[f"{metric}_count"] = count if observed else None
            cell[f"{metric}_rate"] = count / len(observed) if observed else None
        cells.append(cell)
    tasks = [{**row, "in_matched_cohort": row["index"] in matched} for _, row in sorted(
        rows.items(), key=lambda entry: (CONDITIONS.index(entry[0][:2]), entry[0][3], entry[0][2]))]
    payload = {
        "status": "complete" if len(matched) == EXPECTED_TASKS else "matched_partial" if matched else "progress_only",
        "complete": len(matched) == EXPECTED_TASKS, "completed_evaluations": len(rows),
        "expected_evaluations": EXPECTED_TASKS * len(groups), "matched_tasks": len(matched),
        "matched_indices": sorted(matched), "budgets": list(BUDGETS), "cells": cells, "source": source,
        "native_16k_reference": _reference(Path(native_reference), matched) if native_reference is not None else None,
        "metric_definitions": {
            "cohort": "Intersection of task indices across all seven conditions and all eight budgets.",
            "accuracy": "Correct divided by matched tasks; false and unparsed completed answers remain in the denominator.",
            "missing": "Missing evaluations remain pending; they are never converted to incorrect answers.",
            "generated_tokens": "Reported reasoning and answer sample count; no reconstruction or regrading is performed.",
            "answer_tokens": "At most eight new answer samples, including a sampled EOS.",
            "answer_cap_reached": "Matched rows with stop_reason=max_tokens: the cap was reached without a complete box. Rate uses matched rows with a recorded stop_reason; stop_reason_tasks gives coverage.",
            "unparsed_prediction": "Matched rows with grading_status=unparsed_prediction. Rate uses matched rows with a recorded grading_status; grading_status_tasks gives coverage. Missing optional statuses are not counted as zero.",
            "budget": "Reasoning rounds per worker, excluding prompt and answer-cue prefill.",
            "tasks_csv": "All committed input rows, with an in_matched_cohort flag and optional input fields preserved.",
        },
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extra = sorted(set().union(*(row.keys() for row in tasks)) - set(REQUIRED) - {"in_matched_cohort"})
    _write(output / "summary.csv", _csv(cells, cells[0].keys()))
    _write(output / "tasks.csv", _csv(tasks, (*REQUIRED, *extra, "in_matched_cohort")))
    _write(output / "summary.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = build_report(args.results, args.output, native_reference=args.native_reference)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Budget-curve report failed: {error}\n")
    print(f"{payload['completed_evaluations']}/{payload['expected_evaluations']} evaluations; "
          f"{payload['matched_tasks']} matched tasks; {payload['status']}; {args.output}")


if __name__ == "__main__":
    main()
