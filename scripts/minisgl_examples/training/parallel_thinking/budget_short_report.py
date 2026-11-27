"""Join MATH-500 readouts at 256, 512, 1000, 1024, 2000, 3000 and 4000 rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from . import budget_curve_report as frozen
from .report import COMMON

SHORT_BUDGETS = (256, 512, 1024)
BUDGETS = (256, 512, 1000, 1024, 2000, 3000, 4000)
CONDITIONS = frozen.CONDITIONS
SHA256 = re.compile(r"[0-9a-f]{64}")


def check_answer(row):
    count = row["answer_tokens"]
    if "answer_token_ids" in row:
        ids = row["answer_token_ids"]
        if not isinstance(ids, list) or len(ids) != count or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("answer_token_ids do not match answer_tokens")
    if "reasoning_tokens" in row and (type(row["reasoning_tokens"]) is not int or row["reasoning_tokens"] < 0
                                      or row["generated_tokens"] != row["reasoning_tokens"] + count):
        raise ValueError("generated_tokens must equal reasoning_tokens plus answer_tokens")
    if "effective_rounds" in row and (type(row["effective_rounds"]) is not int
                                      or not 0 <= row["effective_rounds"] <= row["budget"]):
        raise ValueError("effective_rounds exceed the requested budget")
    for name in ("context_sha256", "view_sha256"):
        if name in row and not (isinstance(row[name], str) and SHA256.fullmatch(row[name])):
            raise ValueError(f"Invalid {name}")
    if "grading_status" in row:
        grade = row["grading_status"]
        if grade not in ("correct", "incorrect", "unparsed_prediction", "unparsed_gold") or row["correct"] != (grade == "correct"):
            raise ValueError("grade and correct disagree")
    if "stop_reason" in row:
        reason = row["stop_reason"]
        if reason not in ("eos", "answer", "max_tokens") or (reason == "max_tokens" and count != 8):
            raise ValueError("Invalid readout stop_reason or answer token cap")
        if "prediction" in row and (reason == "answer") != (row["prediction"] is not None):
            raise ValueError("prediction and stop_reason disagree")


def load_short_rows(path):
    """Read only committed short-budget rows, using the frozen report's row keys."""
    records, source = frozen._jsonl(Path(path))
    rows, traces = {}, {}
    for number, row in records:
        try:
            key = (row["mode"], row["workers"], row["index"], row["budget"])
            mode, workers, index, budget = key
            if type(workers) is not int or (mode, workers) not in CONDITIONS:
                raise ValueError("Unknown condition")
            if type(index) is not int or not 0 <= index < frozen.EXPECTED_TASKS:
                raise ValueError("Invalid MATH-500 index")
            if type(budget) is not int or budget not in SHORT_BUDGETS:
                raise ValueError("Unknown short budget")
            if type(row["correct"]) is not bool:
                raise ValueError("correct must be a boolean")
            generated, answer = row["generated_tokens"], row["answer_tokens"]
            if type(generated) is not int or type(answer) is not int or not 0 <= answer <= min(8, generated):
                raise ValueError("Invalid generated or answer token count")
            wall = row["wall_seconds"]
            if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
                raise ValueError("Invalid wall_seconds")
            trace = row["source_trace_sha256"]
            if not isinstance(trace, str) or not SHA256.fullmatch(trace):
                raise ValueError("Invalid source trace SHA256")
            if traces.setdefault(key[:3], trace) != trace:
                raise ValueError("Different source traces across task budgets")
            if key in rows:
                raise ValueError(f"Duplicate result {key}")
            check_answer(row)
            rows[key] = row
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return rows, source


# Compare the inputs and numerical settings that determine the scored readout.
# Device assignments, file locations and requested budgets vary across the runs.
IDENTITY_FIELDS = (
    "protocol", "model", "model_revision", "dataset", "dataset_revision", "dataset_sha256", "seed",
    "answer_tokens", "grading", "readout_context", "budget_unit", "always_force_readout", "answer_sampling",
    "answer_cue", "close_thinking", "context_cap", "memory_ratio", "max_prefill_rows", "max_running_req",
    "cuda_graph_bs", "shared_cuda_graph_max_depth", "source_sha256", "metric_definitions",
)


def read_protocol(root, *, required):
    protocol, manifests = None, []
    for path in sorted(root.glob("lanes/*/config.json")):
        raw = path.read_bytes()
        config = json.loads(raw)
        missing = set(IDENTITY_FIELDS) - config.keys()
        if missing:
            raise ValueError(f"{path}: missing protocol fields {sorted(missing)}")
        observed = {name: config[name] for name in IDENTITY_FIELDS}
        for key in ("model", "model_revision", "dataset", "dataset_revision", "dataset_sha256", "seed"):
            if observed[key] != COMMON[key]:
                raise ValueError(f"{path}: unexpected pinned {key}")
        if observed["answer_tokens"] != 8 or observed["answer_sampling"] != {"temperature": 0.0}:
            raise ValueError(f"{path}: expected eight-token greedy readout")
        if protocol is not None and observed != protocol:
            fields = [key for key in IDENTITY_FIELDS if observed[key] != protocol[key]]
            raise ValueError(f"{path}: inconsistent lane protocol fields {fields}")
        protocol = observed
        manifests.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
    if required and not manifests:
        raise ValueError(f"{root}: no lane manifest to identify the scored results")
    return protocol, manifests


def answer_signature(row):
    return {key: row[key] for key in ("answer_token_ids", "prediction", "grading_status", "correct", "stop_reason")}


def compare_plans(previous, short):
    paths = [root / "plan.json" for root in (previous, short)]
    present = [path.exists() for path in paths]
    if not all(present):
        return {"verified": False, "previous_available": present[0], "short_available": present[1]}
    raw = [path.read_bytes() for path in paths]
    old, new = [json.loads(data)["config"] for data in raw]
    for name in ("source_manifest_sha256", "active_source_manifest_sha256"):
        if not old.get(name) or old[name] != new.get(name):
            raise ValueError(f"Different source run identity: {name}")
    old_code = old.get("code_sha256", {})
    new_code = new.get("code_sha256", {})
    if not old_code or any(new_code.get(path) != value for path, value in old_code.items()):
        raise ValueError("Previous runtime or engine file hashes differ in short study plan")
    return {"verified": True, "shared_runtime_files": len(old_code),
            "previous_plan_sha256": hashlib.sha256(raw[0]).hexdigest(),
            "short_plan_sha256": hashlib.sha256(raw[1]).hexdigest()}


def audit_contexts(old, short):
    """Compare independently scored answers when both runs used identical prefill tokens."""
    fields = {"context_sha256", "answer_token_ids", "prediction", "grading_status", "stop_reason"}
    prior = {}
    for key, row in old.items():
        if fields <= row.keys():
            context = (*key[:3], row["context_sha256"])
            prior.setdefault(context, []).append((key[3], answer_signature(row)))
    overlaps, different, examples = set(), set(), []
    overlap_rows = mismatch_rows = missing_metadata = 0
    for key, row in short.items():
        if not fields <= row.keys():
            missing_metadata += 1
            continue
        context = (*key[:3], row["context_sha256"])
        if context not in prior:
            continue
        overlaps.add(context)
        overlap_rows += 1
        answer = answer_signature(row)
        mismatches = [(budget, result) for budget, result in prior[context] if result != answer]
        if mismatches:
            different.add(context)
            mismatch_rows += 1
            if len(examples) < 10:
                examples.append({"mode": key[0], "workers": key[1], "index": key[2], "short_budget": key[3],
                                 "context_sha256": context[-1], "short_answer": answer,
                                 "previous_answers": [{"budget": b, **result} for b, result in mismatches]})
    return {"identical_contexts": len(overlaps), "short_rows_with_identical_context": overlap_rows,
            "mismatched_contexts": len(different), "mismatched_short_rows": mismatch_rows,
            "short_rows_missing_audit_metadata": missing_metadata, "mismatch_examples": examples}


def collect(old_root, short_root):
    old, old_source = frozen.load_rows(old_root / "results.jsonl")
    for key, row in old.items():
        try:
            check_answer(row)
        except ValueError as error:
            raise ValueError(f"Previous result {key}: {error}") from error
    short_file = short_root / "results.jsonl"
    if short_file.exists():
        short, short_source = load_short_rows(short_file)
    elif short_root.is_dir():
        short, short_source = {}, {"path": str(short_file.resolve()), "sha256": None, "pending": True}
    else:
        raise ValueError(f"Missing short study directory: {short_root}")
    protocol, old_manifests = read_protocol(old_root, required=True)
    new_protocol, short_manifests = read_protocol(short_root, required=bool(short))
    if new_protocol is not None and protocol != new_protocol:
        fields = [key for key in IDENTITY_FIELDS if protocol[key] != new_protocol[key]]
        raise ValueError(f"Different previous and short readout protocol fields: {fields}")
    traces, tasks = {}, {}
    for group in (old, short):
        for key, row in group.items():
            if traces.setdefault(key[:3], row["source_trace_sha256"]) != row["source_trace_sha256"]:
                raise ValueError(f"Conflicting source trace for {key[:3]}")
            for field in ("unique_id", "gold"):
                if field in row and tasks.setdefault((key[2], field), row[field]) != row[field]:
                    raise ValueError(f"Conflicting {field} for MATH-500 index {key[2]}")
    selected = {key: row for key, row in old.items() if key[3] in BUDGETS}
    selected.update(short)
    sources = {"previous": {**old_source, "lane_manifests": old_manifests},
               "short": {**short_source, "lane_manifests": short_manifests},
               "protocol_checked_across_runs": new_protocol is not None,
               "plan_identity": compare_plans(old_root, short_root)}
    return selected, sources, protocol, audit_contexts(old, short)


def summarize(rows):
    groups = {(mode, workers, budget): {} for mode, workers in CONDITIONS for budget in BUDGETS}
    for (mode, workers, index, budget), row in rows.items():
        groups[mode, workers, budget][index] = row
    matched = set.intersection(*(set(group) for group in groups.values()))
    n = len(matched)
    cells = []
    for (mode, workers, budget), group in groups.items():
        cohort = [group[index] for index in sorted(matched)]
        correct = sum(row["correct"] for row in cohort)
        cell = {"mode": mode, "workers": workers, "budget": budget, "completed_tasks": len(group),
                "pending_tasks": frozen.EXPECTED_TASKS - len(group), "matched_tasks": n,
                "correct": correct if n else None, "accuracy": correct / n if n else None}
        for field in ("generated_tokens", "answer_tokens", "wall_seconds"):
            total = sum(row[field] for row in cohort)
            cell[f"total_{field}"] = total if n else None
            cell[f"mean_{field}"] = total / n if n else None
        for field, value, name in (("stop_reason", "max_tokens", "answer_cap_reached"),
                                   ("grading_status", "unparsed_prediction", "unparsed_prediction")):
            observed = [row[field] for row in cohort if isinstance(row.get(field), str)]
            cell[f"{field}_tasks"] = len(observed)
            cell[f"{name}_count"] = observed.count(value) if observed else None
            cell[f"{name}_rate"] = observed.count(value) / len(observed) if observed else None
        cells.append(cell)
    return matched, cells




def build_report(old_root, short_root, output, *, native_reference=None):
    old_root, short_root, output = map(Path, (old_root, short_root, output))
    if output.resolve().is_relative_to(old_root.resolve()):
        raise ValueError("Write the extension report outside the frozen previous run")
    if output.resolve() == short_root.resolve():
        raise ValueError("Combined reports require a dedicated output directory")
    rows, sources, protocol, audit = collect(old_root, short_root)
    matched, cells = summarize(rows)
    complete = len(matched) == frozen.EXPECTED_TASKS
    payload = {"status": "complete" if complete else "matched_partial" if matched else "progress_only",
               "complete": complete, "completed_evaluations": len(rows),
               "expected_evaluations": frozen.EXPECTED_TASKS * len(cells), "matched_tasks": len(matched),
               "matched_indices": sorted(matched), "budgets": list(BUDGETS), "cells": cells, "sources": sources,
               "protocol": protocol, "identical_context_audit": audit,
               "native_16k_reference": frozen._reference(Path(native_reference), matched) if native_reference else None,
               "cohort_definition": "Identical MATH-500 indices across all seven arms and all seven cutoffs. Unparsed completed answers count as incorrect."}
    tasks = [{**row, "input_run": "short" if key[3] in SHORT_BUDGETS else "previous",
              "in_matched_cohort": key[2] in matched}
             for key, row in sorted(rows.items(), key=lambda item: (CONDITIONS.index(item[0][:2]), item[0][3], item[0][2]))]
    extras = sorted(set().union(*(row.keys() for row in tasks)) - set(frozen.REQUIRED) - {"input_run", "in_matched_cohort"})
    output.mkdir(parents=True, exist_ok=True)
    frozen._write(output / "summary.csv", frozen._csv(cells, cells[0].keys()))
    frozen._write(output / "tasks.csv", frozen._csv(tasks, (*frozen.REQUIRED, *extras, "input_run", "in_matched_cohort")))
    frozen._write(output / "summary.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("short", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_report(args.previous, args.short, args.output,
                              native_reference=args.native_reference)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Short-budget report failed: {error}\n")
    print(f"{result['completed_evaluations']}/{result['expected_evaluations']} evaluations, "
          f"{result['matched_tasks']} matched tasks, {result['identical_context_audit']['mismatched_contexts']} "
          f"disagreements on identical contexts. {args.output}")


if __name__ == "__main__":
    main()
