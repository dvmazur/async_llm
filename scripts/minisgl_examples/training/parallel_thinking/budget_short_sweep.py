"""Run the three short cutoffs on the completed publication traces using the original workers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from . import budget_curve_sweep as fixed_lanes
from . import budget_short_report as short_report

BUDGETS = (256, 512, 1024)
CONTEXT_CAP = 136192
FORMAT = "budget-readout-short-fixed-lanes-v1"
RUNTIME_FIELDS = {
    "mode", "workers", "index", "budget", "unique_id", "gold", "prediction", "answer_token_ids",
    "answer_text", "stop_reason", "correct", "grading_status", "generated_tokens", "answer_tokens",
    "reasoning_tokens", "effective_rounds", "prompt_tokens", "context_tokens", "context_sha256",
    "view_sha256", "cue_token_ids", "publication_count", "source_trace_sha256", "source_trace_path",
    "wall_seconds", "reused_readout", "reused_from_budget",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_catalog(source):
    tasks = source / "report/tasks.csv"
    complete = json.loads((source / "complete.json").read_text())
    if sha256(tasks) != complete["tasks_sha256"] or complete["cases"] != 3500:
        raise ValueError("Completed source task catalog differs from its completion record")
    catalog, identities = {}, {}
    with tasks.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["condition"], int(row["workers"]), int(row["index"])
            identity = row["unique_id"], row["gold"]
            path = (source / row["trace_path"]).resolve()
            if (key in catalog or key[:2] not in fixed_lanes.CONDITIONS.values() or not 0 <= key[2] < 500
                    or not row["unique_id"] or not short_report.SHA256.fullmatch(row["trace_sha256"])
                    or identities.setdefault(key[2], identity) != identity or not path.is_relative_to(source.resolve())):
                raise ValueError(f"Invalid source task identity: {key}")
            catalog[key] = (*identity, row["trace_sha256"], str(path))
    if len(catalog) != 3500 or len({value[0] for value in identities.values()}) != 500:
        raise ValueError("The source must contain seven complete conditions on the same 500 problems")
    return catalog


def configuration(args):
    if (args.budgets != list(BUDGETS) or args.indices != list(range(500))
            or args.conditions != list(fixed_lanes.CONDITIONS)):
        raise ValueError("This sweep requires all 500 indices, seven conditions and budgets 256, 512 and 1024")
    if args.memory_ratio != 0.75 or args.max_prefill_rows != 256 or args.wait_for_source:
        raise ValueError("Use memory ratio 0.75, max prefill rows 256 and the completed publication source")
    config = fixed_lanes.configuration(args)
    code = config["code_sha256"]
    for file in (Path(__file__), Path(short_report.__file__)):
        code[str(file.resolve().relative_to(fixed_lanes.PROJECT))] = sha256(file)
    return {**config, "format": FORMAT, "max_context": CONTEXT_CAP}


def worker_command(args, lane):
    return [*fixed_lanes.worker_command(args, lane), "--max-context", str(CONTEXT_CAP)]


def read_rows(path, catalog):
    rows, info = short_report.load_short_rows(path)
    first_budget = {}
    for key, row in sorted(rows.items()):
        if row.keys() != RUNTIME_FIELDS:
            raise ValueError(f"Incomplete or unexpected readout fields for {key} in {path}")
        first_budget.setdefault((*key[:3], row["context_sha256"]), key[3])
    for key, row in rows.items():
        identity = (row["unique_id"], row["gold"], row["source_trace_sha256"], row["source_trace_path"])
        if identity != catalog.get(key[:3]):
            raise ValueError(f"Readout differs from frozen source case {key[:3]} in {path}")
        cue = row["cue_token_ids"]
        if (not isinstance(cue, list) or not cue or any(type(t) is not int or t < 0 for t in cue)
                or type(row["prompt_tokens"]) is not int or row["prompt_tokens"] < 1
                or type(row["context_tokens"]) is not int
                or row["context_tokens"] != row["prompt_tokens"] + row["reasoning_tokens"] + len(cue)
                or row["context_tokens"] + 8 > CONTEXT_CAP or type(row["publication_count"]) is not int
                or row["publication_count"] < 0 or not isinstance(row["answer_text"], str)
                or (row["prediction"] is not None and not isinstance(row["prediction"], str))
                or row["answer_tokens"] < 1 or type(row["reused_readout"]) is not bool):
            raise ValueError(f"Invalid readout metrics for {key} in {path}")
        earlier = row["reused_from_budget"]
        first = first_budget[(*key[:3], row["context_sha256"])]
        if row["reused_readout"] != (first != key[3]) or (row["reused_readout"] and earlier != first):
            raise ValueError(f"Readout reuse disagrees with the first identical context for {key} in {path}")
        if row["reused_readout"]:
            prior = rows.get((*key[:3], earlier)) if type(earlier) is int and earlier < key[3] else None
            if prior is None or any(row[field] != prior[field] for field in (
                    "context_sha256", "answer_token_ids", "answer_text", "prediction", "stop_reason")):
                raise ValueError(f"Invalid short-budget readout reuse for {key} in {path}")
        elif earlier is not None:
            raise ValueError(f"Unexpected reused budget on fresh readout {key} in {path}")
    return rows, info


def finish_report(path, output, catalog):
    rows, source = read_rows(path, catalog)
    cells = []
    for mode, workers in fixed_lanes.CONDITIONS.values():
        for budget in BUDGETS:
            group = [rows[(mode, workers, index, budget)] for index in range(500)]
            correct = sum(row["correct"] for row in group)
            cells.append({"mode": mode, "workers": workers, "budget": budget, "tasks": len(group),
                          "correct": correct, "accuracy": correct / len(group)})
    output.mkdir(parents=True, exist_ok=True)
    fixed_lanes.write_json(output / "summary.json", {"complete": True, "completed_evaluations": len(rows),
                           "expected_evaluations": 10500, "matched_tasks": 500, "budgets": list(BUDGETS),
                           "cells": cells, "source": source})


def coordinator(catalog):
    # Give this invocation its own module namespace. The frozen module and any
    # ordinary 1k–8k coordinator imported in the same interpreter stay intact.
    spec = importlib.util.spec_from_file_location(__package__ + "._short_fixed_lanes", fixed_lanes.__file__)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.configuration = configuration
    runner.worker_command = worker_command
    runner.report = SimpleNamespace(BUDGETS=BUDGETS, _write=fixed_lanes.report._write,
                                    load_rows=lambda path: read_rows(path, catalog),
                                    build_report=lambda path, output: finish_report(path, output, catalog))
    return runner


def main(argv=None):
    paths = argparse.ArgumentParser(add_help=False)
    paths.add_argument("--source-root", type=Path, required=True)
    paths.add_argument("--output", type=Path, required=True)
    # Let the frozen CLI print its normal help without opening a source run.
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        return coordinator({}).main(argv)
    args, _ = paths.parse_known_args(argv)
    source, output = args.source_root.resolve(), args.output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        paths.error("Short-run output must be separate from the source run")
    for directory in (output, *output.parents):
        plan = directory / "plan.json"
        if plan.is_file() and (directory != output or json.loads(plan.read_text())["config"]["format"] != FORMAT):
            paths.error(f"Choose a separate output directory outside the existing study at {directory}")
    return coordinator(read_catalog(source)).main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
