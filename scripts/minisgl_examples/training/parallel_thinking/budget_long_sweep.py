"""Run 9,000–16,000-round greedy readouts over the pinned MATH-500 traces."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from . import budget_curve_sweep as lanes
from . import budget_long_report as long_report
from . import budget_short_report as short_report
from . import budget_short_sweep as short

BUDGETS = tuple(range(9000, 16001, 1000))
CONTEXT_CAP = 262144
FORMAT = "budget-readout-long-fixed-lanes-v1"


def configuration(args):
    if (args.budgets != list(BUDGETS) or args.indices != list(range(500))
            or args.conditions != list(lanes.CONDITIONS)):
        raise ValueError("Require all MATH-500 indices, all seven setups and every 1k cutoff from 9k to 16k")
    if args.memory_ratio != 0.75 or args.max_prefill_rows != 256 or args.wait_for_source:
        raise ValueError("Require the completed source, memory ratio 0.75 and 256-row prefills")
    config = lanes.configuration(args)
    code = config["code_sha256"]
    for file in map(Path, (__file__, long_report.__file__, short.__file__, short_report.__file__)):
        value = short.sha256(file)
        code[str(file.resolve().relative_to(lanes.PROJECT))] = value
    return {**config, "format": FORMAT, "max_context": CONTEXT_CAP}


def worker_command(args, lane):
    # The frozen evaluator clamps this limit to the model's native maximum.
    # At W16 the flattened input can exceed 256,000 tokens, before the answer.
    return [*lanes.worker_command(args, lane), "--max-context", str(CONTEXT_CAP)]


def read_rows(path, catalog):
    rows, metadata = long_report.load_long_rows(path)
    first = {}
    for key, row in sorted(rows.items()):
        if row.keys() != short.RUNTIME_FIELDS:
            raise ValueError(f"Unexpected readout fields for {key} in {path}")
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
            raise ValueError(f"Invalid long readout metrics for {key} in {path}")
        context = (*key[:3], row["context_sha256"])
        previous = first.get(context)
        if previous is None:
            if row["reused_readout"] or row["reused_from_budget"] is not None:
                raise ValueError(f"Missing original high-budget answer for {key} in {path}")
            first[context] = row
        elif (not row["reused_readout"] or type(row["reused_from_budget"]) is not int
                or row["reused_from_budget"] != previous["budget"]
                or any(previous[field] != row[field] for field in (
                    "answer_token_ids", "answer_text", "stop_reason", "prediction", "correct", "grading_status",
                    "cue_token_ids", "view_sha256", "context_tokens"))):
            raise ValueError(f"Identical-context reuse disagrees with original high-budget answer at {key} in {path}")
    return rows, metadata


def finish_report(path, output, catalog):
    rows, source = read_rows(path, catalog)
    cells = []
    for mode, workers in lanes.CONDITIONS.values():
        for budget in BUDGETS:
            correct = sum(rows[(mode, workers, index, budget)]["correct"] for index in range(500))
            cells.append({"mode": mode, "workers": workers, "budget": budget, "tasks": 500,
                          "correct": correct, "accuracy": correct / 500})
    output.mkdir(parents=True, exist_ok=True)
    lanes.write_json(output / "summary.json", {"complete": True, "completed_evaluations": len(rows),
                     "expected_evaluations": len(cells) * 500, "matched_tasks": 500, "budgets": list(BUDGETS),
                     "cells": cells, "source": source})


def coordinator(catalog):
    # Isolate the interval-specific policy from other campaigns using the same runtime.
    spec = importlib.util.spec_from_file_location(__package__ + "._long_fixed_lanes", lanes.__file__)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.configuration = configuration
    runner.worker_command = worker_command
    runner.report = SimpleNamespace(BUDGETS=BUDGETS, _write=lanes.report._write,
                                    load_rows=lambda path: read_rows(path, catalog),
                                    build_report=lambda path, output: finish_report(path, output, catalog))
    return runner


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        return coordinator({}).main(argv)
    paths = argparse.ArgumentParser(add_help=False)
    paths.add_argument("--source-root", type=Path, required=True)
    paths.add_argument("--output", type=Path, required=True)
    args, _ = paths.parse_known_args(argv)
    source, output = args.source_root.resolve(), args.output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        paths.error("Long-run output must be separate from the immutable source")
    for directory in (output, *output.parents):
        plan = directory / "plan.json"
        if plan.is_file() and (directory != output or json.loads(plan.read_text())["config"]["format"] != FORMAT):
            paths.error(f"Choose a separate output directory outside the existing study at {directory}")
    return coordinator(short.read_catalog(source)).main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
