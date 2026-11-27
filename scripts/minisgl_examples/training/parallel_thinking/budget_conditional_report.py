"""Score completed MATH-500 traces with their original answers at each cutoff."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import budget_long_report as long_report
from .budget_curve_report import CONDITIONS, EXPECTED_TASKS, _csv, _write, verify_completion
from .budget_short_report import compare_plans, load_short_rows, read_protocol


SHORT = (256, 512, 1024)
BUDGET_RANGES = (
    ("accuracy-256-512-1024", SHORT),
    ("accuracy-1k-to-10k", tuple(range(1000, 10001, 1000))),
    ("accuracy-10k-to-16k", tuple(range(10000, 16001, 1000))),
)


def read_originals(path):
    originals = {}
    problem_ids = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            arm = row["condition"], int(row["workers"])
            index = int(row["index"])
            key = (*arm, index)
            if arm not in CONDITIONS or not 0 <= index < EXPECTED_TASKS or key in originals:
                raise ValueError(f"Unexpected or duplicate original trace {key}")
            if problem_ids.setdefault(index, row["unique_id"]) != row["unique_id"]:
                raise ValueError(f"Conflicting problem identity at index {index}")
            if row["correct"] not in {"True", "False"}:
                raise ValueError(f"Invalid original grade for {key}")
            row["correct"] = row["correct"] == "True"
            row["decoding_steps"] = int(row["decoding_steps"])
            if row["correct"] != (row["grading_status"] == "correct") or row["decoding_steps"] < 0:
                raise ValueError(f"Inconsistent original trace for {key}")
            originals[key] = row
    if len(originals) != EXPECTED_TASKS * len(CONDITIONS) or len(set(problem_ids.values())) != EXPECTED_TASKS:
        raise ValueError("Expected seven complete arms on the same 500 distinct problems")
    return originals


def conditional_row(key, original, forced):
    mode, workers, index, budget = key
    rounds = original["decoding_steps"]
    if (original["unique_id"] != forced["unique_id"] or original["gold"] != forced["gold"]
            or original["trace_sha256"] != forced["source_trace_sha256"]
            or min(budget, rounds) != forced["effective_rounds"]):
        raise ValueError(f"Trace identity or cutoff mismatch for {key}")
    complete = rounds <= budget
    selected = original if complete else forced
    return {
        "mode": mode, "workers": workers, "index": index, "unique_id": original["unique_id"],
        "budget": budget, "source_reasoning_rounds": rounds, "effective_rounds": forced["effective_rounds"],
        "answer_source": "original" if complete else "forced", "prediction": selected["prediction"],
        "correct": selected["correct"], "grading_status": selected["grading_status"], "gold": original["gold"],
        "original_prediction": original["prediction"], "original_correct": original["correct"],
        "original_used_finisher": original["used_finisher"], "forced_prediction": forced["prediction"],
        "forced_correct": forced["correct"], "forced_answer_tokens": forced["answer_tokens"],
        "source_trace_sha256": original["trace_sha256"],
    }


def verify_hashes(short, extended, digests):
    certified = {}
    for root, field in ((short, "sha256_before_and_after"), (extended, "source_result_report_sha256")):
        audit = json.loads((root / "completion-verification.json").read_text())
        if audit["status"] != "passed":
            raise ValueError(f"Input audit did not pass for {root}")
        for path, digest in audit[field].items():
            relative = path.rsplit("/runs/", 1)[-1]
            if certified.setdefault(relative, digest) != digest:
                raise ValueError(f"Conflicting completion audit hashes for {relative}")
    for path, digest in digests.items():
        if certified.get(path) != digest:
            raise ValueError(f"Archived input checksum mismatch for {path}")


def write_csv(path, rows):
    _write(path, _csv(rows, list(rows[0])))


def wide_table(by_cell, budgets):
    rows = []
    for mode, workers in CONDITIONS:
        name = "Ordinary Qwen" if mode == "native" else f"Hogwild {workers}w"
        row = {"setup": name, "matched_tasks": EXPECTED_TASKS}
        for budget in budgets:
            row[f"correct_{budget}"] = by_cell[mode, workers, budget]["correct"]
        for budget in budgets:
            row[f"accuracy_pct_{budget}"] = round(100 * by_cell[mode, workers, budget]["accuracy"], 1)
        rows.append(row)
    return rows


def write_range_reports(summary, output):
    cells = {(row["mode"], row["workers"], row["budget"]): row for row in summary}
    output.mkdir(parents=True, exist_ok=True)
    for basename, budgets in BUDGET_RANGES:
        write_csv(output / f"{basename}.csv", wide_table(cells, budgets))


def build_report(source, short, previous, extended, output):
    path = source / "report" / "tasks.csv"
    originals = read_originals(path)
    digests = {f"{source.name}/report/tasks.csv": hashlib.sha256(path.read_bytes()).hexdigest()}
    tasks, protocols, plans = [], [], []
    cells = {}
    for root, budgets in ((short, SHORT), (previous, long_report.LOW), (extended, long_report.HIGH)):
        path = root / "results.jsonl"
        if budgets == SHORT:
            rows, evidence = load_short_rows(path)
        else:
            cap = long_report.LOW_CAP if budgets == long_report.LOW else long_report.HIGH_CAP
            rows, evidence = long_report.load_rows(path, budgets, cap)
        if evidence["ignored_uncommitted_trailing_bytes"] or len(rows) != len(originals) * len(budgets):
            raise ValueError(f"Incomplete forced readout coverage in {path}")
        plan = verify_completion(root, rows, evidence)
        if plan["budgets"] != list(budgets):
            raise ValueError(f"Unexpected budget plan in {root}")
        plans.append(plan)
        protocol, _ = read_protocol(root, required=True)
        protocols.append({key: value for key, value in protocol.items() if key != "context_cap"})
        digests[f"{root.name}/results.jsonl"] = evidence["sha256"]
        for mode, workers in CONDITIONS:
            for budget in budgets:
                group = []
                for index in range(EXPECTED_TASKS):
                    key = (mode, workers, index, budget)
                    group.append(conditional_row(key, originals[key[:3]], rows[key]))
                correct = sum(row["correct"] for row in group)
                original_count = sum(row["answer_source"] == "original" for row in group)
                cells[mode, workers, budget] = {
                    "mode": mode, "workers": workers, "budget": budget, "matched_tasks": EXPECTED_TASKS,
                    "correct": correct, "accuracy": correct / EXPECTED_TASKS,
                    "original_count": original_count, "forced_count": EXPECTED_TASKS - original_count,
                }
                tasks.extend(group)
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("Readout protocol differs across budget studies")
    for other in (short, extended):
        if not compare_plans(previous, other)["verified"]:
            raise ValueError("All budget studies require their saved source and runtime plans")
    complete_path = source / "complete.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        source_hashes = {key: hashlib.sha256((source / filename).read_bytes()).hexdigest() for key, filename in (
            ("source_manifest_sha256", "source-files.json"),
            ("active_source_manifest_sha256", "active-source-files.json"))}
        if (complete["cases"] != len(originals) or complete["tasks_sha256"] != digests[f"{source.name}/report/tasks.csv"]
                or any(complete[key] != value or any(plan[key] != value for plan in plans)
                       for key, value in source_hashes.items())):
            raise ValueError("Source catalog or manifests differ from the completed experiment")
    else:
        verify_hashes(short, extended, digests)
    output.mkdir(parents=True, exist_ok=True)
    summary = list(cells.values())
    write_csv(output / "tasks.csv", tasks)
    write_csv(output / "summary.csv", summary)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": "Use the original answer and grade when source reasoning rounds <= cutoff. "
                  "Otherwise use the saved greedy eight-token readout and its grade.",
        "model": "Qwen/Qwen3.8-27B", "matched_tasks": EXPECTED_TASKS, "total_rows": len(tasks),
        "input_sha256": digests, "cells": summary,
    }
    _write(output / "summary.json", json.dumps(metadata, indent=2) + "\n")
    write_range_reports(summary, output / "three-ranges")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "short", "previous", "extended", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    build_report(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
