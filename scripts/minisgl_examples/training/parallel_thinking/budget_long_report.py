"""Report measured MATH-500 forced readouts at each 1,000-round cutoff through 16,000."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import budget_curve_report as frozen
from . import budget_short_report as checks

LOW = frozen.BUDGETS
HIGH = tuple(range(9000, 16001, 1000))
BUDGETS = LOW + HIGH
CONDITIONS = frozen.CONDITIONS
LOW_CAP = 136192
HIGH_CAP = 262144


def load_rows(path, budgets, cap):
    records, source = frozen._jsonl(Path(path))
    rows, traces = {}, {}
    for number, row in records:
        try:
            key = mode, workers, index, budget = tuple(row[k] for k in ('mode', 'workers', 'index', 'budget'))
            if type(workers) is not int or (mode, workers) not in CONDITIONS or type(index) is not int or not 0 <= index < 500:
                raise ValueError("Invalid MATH-500 condition or index")
            if type(budget) is not int or budget not in budgets or key in rows:
                raise ValueError(f"Unexpected budget or duplicate key {key}")
            for field in ("source_trace_sha256", "context_sha256", "view_sha256"):
                if not checks.SHA256.fullmatch(row[field]):
                    raise ValueError(f"Invalid {field}")
            if traces.setdefault(key[:3], row["source_trace_sha256"]) != row["source_trace_sha256"]:
                raise ValueError(f"Conflicting source trace for {key[:3]}")
            for field in ("answer_tokens", "context_tokens", "prompt_tokens", "reasoning_tokens",
                          "generated_tokens", "effective_rounds", "publication_count"):
                if type(row[field]) is not int or row[field] < 0:
                    raise ValueError(f"Invalid {field}")
            if (type(row["correct"]) is not bool or type(row["reused_readout"]) is not bool
                    or not 1 <= row["answer_tokens"] <= 8 or not row["unique_id"]):
                raise ValueError("Invalid grade, answer token count, or task ID")
            cue = row["cue_token_ids"]
            if not isinstance(cue, list) or not cue or any(type(token) is not int or token < 0 for token in cue):
                raise ValueError("Invalid answer cue")
            if row["context_tokens"] != row["prompt_tokens"] + row["reasoning_tokens"] + len(cue) or row["context_tokens"] + 8 > cap:
                raise ValueError("Context length or engine cap mismatch")
            if type(row["wall_seconds"]) not in (int, float) or not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] < 0:
                raise ValueError("Invalid wall_seconds")
            checks.answer_signature(row)
            checks.check_answer(row)
            rows[key] = row
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return rows, source


def load_long_rows(path):
    return load_rows(path, HIGH, HIGH_CAP)


def collect(previous, extended):
    plans = checks.compare_plans(previous, extended)
    if not plans["verified"]:
        raise ValueError("Both run plans are required for source and runtime identity")
    groups, sources, protocols = [], {}, {}
    for name, root, budgets, cap in (("previous", previous, LOW, LOW_CAP), ("extended", extended, HIGH, HIGH_CAP)):
        config = json.loads((root / "plan.json").read_text())["config"]
        if config["budgets"] != list(budgets):
            raise ValueError(f"{name}: unexpected budgets in plan")
        path = root / "results.jsonl"
        if name == "extended" and not path.exists():
            rows, source = {}, {"path": str(path.resolve()), "pending": True}
        else:
            rows, source = load_rows(path, budgets, cap)
        protocol, lanes = checks.read_protocol(root, required=bool(rows) or name == "previous")
        if protocol is not None and protocol["context_cap"] != cap:
            raise ValueError(f"{name}: unexpected context_cap")
        groups.append(rows)
        sources[name] = {**source, "lane_manifests": lanes}
        protocols[name] = protocol
    old, new = groups
    frozen.verify_completion(previous, old, sources["previous"])
    protocol = protocols["previous"]
    if protocols["extended"] is not None:
        differing = [k for k in protocol if k != "context_cap" and protocol[k] != protocols["extended"][k]]
        if differing:
            raise ValueError(f"Readout protocol differs across studies: {differing}")
    rows = {**old, **new}
    cases, identities, index_by_id = {}, {}, {}
    for (mode, width, index, _), row in rows.items():
        if cases.setdefault((mode, width, index), row["source_trace_sha256"]) != row["source_trace_sha256"]:
            raise ValueError(f"Conflicting source trace for {(mode, width, index)}")
        identity = (row["unique_id"], row["gold"])
        if identities.setdefault(index, identity) != identity or index_by_id.setdefault(identity[0], index) != index:
            raise ValueError(f"Conflicting MATH-500 identity or gold for {index}")
    sources["plan_identity"] = plans
    audit = checks.audit_contexts(old, new)
    audit = {k.replace("short", "extended"): v for k, v in audit.items()}
    for example in audit["mismatch_examples"]:
        example["extended_budget"] = example.pop("short_budget")
        example["extended_answer"] = example.pop("short_answer")
    return rows, sources, protocols, audit


def summarize(rows):
    cells = {(mode, workers, budget): {} for mode, workers in CONDITIONS for budget in BUDGETS}
    for (mode, workers, index, budget), row in rows.items():
        cells[mode, workers, budget][index] = row
    matched = sorted(set.intersection(*(set(group) for group in cells.values())))
    summaries = []
    for (mode, workers, budget), group in cells.items():
        cohort = [group[i] for i in matched]
        correct = sum(r["correct"] for r in cohort)
        summaries.append({"mode": mode, "workers": workers, "budget": budget,
                          "completed_tasks": len(group), "pending_tasks": 500 - len(group),
                          "matched_tasks": len(matched), "correct": correct if matched else None,
                          "accuracy": correct / len(matched) if matched else None,
                          "answer_cap_reached_count": sum(r["stop_reason"] == "max_tokens" for r in cohort),
                          "unparsed_prediction_count": sum(r["grading_status"] == "unparsed_prediction" for r in cohort),
                          "fresh_readout_count": sum(not r["reused_readout"] for r in cohort),
                          "reused_readout_count": sum(r["reused_readout"] for r in cohort)})
    return matched, summaries




def build_report(previous, extended, output):
    previous, extended, output = map(Path, (previous, extended, output))
    if output.resolve().is_relative_to(previous.resolve()) or output.resolve() == extended.resolve():
        raise ValueError("Choose a dedicated report directory outside the previous study")
    rows, sources, protocols, audit = collect(previous, extended)
    matched, cells = summarize(rows)
    complete = len(matched) == frozen.EXPECTED_TASKS
    result = {"status": "complete" if complete else "matched_partial" if matched else "progress_only",
              "complete": complete, "matched_tasks": len(matched), "matched_indices": matched,
              "completed_evaluations": len(rows), "expected_evaluations": len(cells) * frozen.EXPECTED_TASKS,
              "budgets": list(BUDGETS), "cells": cells, "sources": sources, "protocols": protocols,
              "identical_context_audit": audit,
              "cohort_definition": "Intersection of MATH-500 indices across all seven setups and sixteen decimal cutoffs. "
                                   "Unparsed predictions count as incorrect."}
    matched_set = set(matched)
    tasks = [{**row, "input_run": "previous" if key[3] in LOW else "extended",
              "in_matched_cohort": key[2] in matched_set}
             for key, row in sorted(rows.items(), key=lambda pair: (CONDITIONS.index(pair[0][:2]), pair[0][3], pair[0][2]))]
    extras = sorted(set().union(*(r.keys() for r in tasks)) - set(frozen.REQUIRED) - {"input_run", "in_matched_cohort"})
    output.mkdir(parents=True, exist_ok=True)
    frozen._write(output / "summary.csv", frozen._csv(cells, cells[0].keys()))
    frozen._write(output / "tasks.csv", frozen._csv(tasks, (*frozen.REQUIRED, *extras, "input_run", "in_matched_cohort")))
    frozen._write(output / "summary.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("extended", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_report(args.previous, args.extended, args.output)
    except (OSError, KeyError, ValueError, TypeError, ImportError) as error:
        parser.exit(1, f"Long-budget report failed: {error}\n")
    mismatches = result["identical_context_audit"]["mismatched_contexts"]
    print(f"{result['completed_evaluations']}/{result['expected_evaluations']} rows, "
          f"{result['matched_tasks']} matched problems, {mismatches} identical-context disagreements. {args.output}")
    if mismatches:
        parser.exit(2, "Review identical-context disagreements recorded in summary.json.\n")


if __name__ == "__main__":
    main()
