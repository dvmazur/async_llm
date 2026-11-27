"""CPU verification of the complete 1k–16k forced-answer study and numerical report."""

from __future__ import annotations

import argparse
import csv
import json
import time
from argparse import Namespace
from collections import Counter
from functools import lru_cache
from pathlib import Path

from . import budget_curve_eval as evaluator
from . import budget_long_report as report
from . import budget_long_sweep as sweep
from . import math500


def require(condition, evidence):
    if not condition:
        raise ValueError(str(evidence))


def read_json(path):
    return json.loads(path.read_text())


def verify_source_eos(rollout, stop_reason, pinned_eos, key):
    declared = set(rollout.get("eos_token_ids", ()))
    token = rollout.get("eos_token_id")
    worker = rollout.get("eos_worker")
    if token is not None:
        declared.add(token)
    require(declared <= pinned_eos, f"Unrecognized source EOS: {key}")
    streams = rollout["worker_token_ids"]
    require(all(stream and pinned_eos.isdisjoint(stream[:-1]) for stream in streams.values()),
            f"Source worker continued after EOS: {key}")
    terminal = {name: stream[-1] for name, stream in streams.items() if stream[-1] in pinned_eos}
    if stop_reason == "eos":
        require(worker in terminal and terminal[worker] == token and set(terminal.values()) <= declared,
                f"Source EOS differs from sampled tokens: {key}")
    else:
        require(stop_reason in ("max_steps", "context_limit") and worker is None and token is None and not terminal,
                f"Source budget/context stop differs from sampled tokens: {key}")
    return pinned_eos


def reconstruct(rollout, cutoff, eos, opening, closing, cues):
    """Rebuild the view independently, splitting sampled prefixes at publication boundaries."""
    workers = rollout["workers"]
    prompt = rollout["prompt_ids"]
    rounds = min(cutoff, rollout["decode_rounds"])
    streams = rollout["worker_token_ids"]
    require(len(set(workers)) == len(workers) and set(workers) == set(streams), "Worker identities")
    require(all(len(stream) == rollout["decode_rounds"] for stream in streams.values()), "Stream length")
    ends = {name: rounds - (streams[name][rounds - 1] in eos) for name in workers}
    events = [event for event in rollout["publications"] if event["round"] <= rounds]
    view = list(prompt)
    if rollout["layout"] == "native_workers":
        require(rollout["publication_policy"] == "encoded_separator_v1", "Publication policy")
        offsets = dict.fromkeys(workers, 0)
        previous_round = 0
        for event in events:
            name, at = event["worker"], event["round"]
            end = offsets[name] + event["committed_token_count"]
            require(previous_round <= at and offsets[name] < end == at - 1 <= ends[name],
                    f"Publication at round {at} by {name}")
            view.extend(streams[name][offsets[name]:end])
            offsets[name] = end
            previous_round = at
        for name in [*workers[1:], workers[0]]:
            view.extend(streams[name][offsets[name]:ends[name]])
    else:
        require(len(workers) == 1 and rollout["layout"] in ("native","causal"), "Single-worker layout")
        view.extend(streams[workers[0]][:ends[workers[0]]])
    require(len(view) == len(prompt) + sum(ends.values()), "Sample accounting at cutoff")
    last_tag = next((token for token in reversed(view) if token in (opening, closing)), None)
    cue = cues[last_tag == opening]
    return {"view_sha256": evaluator.token_sha256(view), "context_sha256": evaluator.token_sha256(view + cue),
            "cue_token_ids": cue, "context_tokens": len(view) + len(cue), "prompt_tokens": len(prompt),
            "reasoning_tokens": len(view) - len(prompt), "effective_rounds": rounds, "publication_count": len(events)}


def answer_from_tokens(tokens, cue, tokenizer, eos):
    require(1 <= len(tokens) <= 8, "Eight-token answer cap")
    prefix = tokenizer.decode(cue)
    for position, token in enumerate(tokens, 1):
        if token in eos:
            require(position == len(tokens), "Sampled answer continued after EOS")
            return "eos", None
        prediction = evaluator.extract_boxed_answer(prefix + tokenizer.decode(tokens[:position]))
        if prediction is not None:
            require(position == len(tokens), "Sampled answer continued after its first complete box")
            return "answer", prediction
    require(len(tokens) == 8, "Incomplete answer stopped before its cap")
    return "max_tokens", None


def verify_lanes(root, plan, catalog, combined):
    require(set(p.parent.name for p in root.glob("lanes/*/config.json")) ==
            {sweep.lanes.lane_path(root, lane).name for lane in plan["lanes"]}, "Unexpected lane manifests")
    seen = set()
    evidence = {}
    for lane in plan["lanes"]:
        directory = sweep.lanes.lane_path(root, lane)
        config = read_json(directory / "config.json")
        require((lane["modulo"], lane["remainder"]) == (config["index_modulo"], config["index_remainder"]),
                f"Lane assignment: {directory}")
        require(all(config[field] == plan["config"][field] for field in (
                "budgets", "indices", "conditions", "answer_tokens", "seed", "max_prefill_rows", "memory_ratio")),
                f"Lane settings: {directory}")
        found, meta = sweep.read_rows(directory / "results.jsonl", catalog)
        require(found.keys() == sweep.lanes.expected_keys(plan, lane) and not seen.intersection(found),
                f"Lane coverage: {directory}")
        require(all(row == combined[key] for key, row in found.items()), f"Lane row changed: {directory}")
        require(not meta["ignored_uncommitted_trailing_bytes"], f"Incomplete final lane: {directory}")
        seen.update(found)
        evidence[directory.name] = {"rows": len(found), "config_sha256": evaluator.file_sha256(directory / "config.json"),
                                    "results_sha256": meta["sha256"]}
    require(len(seen) == 28000, "High-budget aggregate lane count")
    return evidence


def verify_report(root, rows, matched, cells):
    directory = root / "report-1k-to-16k"
    payload = read_json(directory / "summary.json")
    require(payload["complete"] and payload["matched_indices"] == matched and payload["cells"] == cells
            and payload["expected_evaluations"] == payload["completed_evaluations"] == len(rows),
            "Final combined summary disagrees with scored rows")
    seen = set()
    with (directory / "tasks.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["mode"], int(row["workers"]), int(row["index"]), int(row["budget"])
            require(key in rows and key not in seen and row["in_matched_cohort"] == "True", f"Report key {key}")
            seen.add(key)
            source = rows[key]
            require(row["input_run"] == ("previous" if key[3] in report.LOW else "extended")
                    and row["correct"] == str(source["correct"]) and row["grading_status"] == source["grading_status"]
                    and row["context_sha256"] == source["context_sha256"] and row["source_trace_sha256"] == source["source_trace_sha256"]
                    and int(row["answer_tokens"]) == source["answer_tokens"], f"Report display/grade mismatch: {key}")
    require(seen == rows.keys(), "Combined task CSV lacks a row")


def verify(source, previous, root):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    start = time.monotonic()
    evaluator.assert_source_active(source)
    plan = read_json(root / "plan.json")
    files = [source / "report/tasks.csv", source / "active-source-files.json", previous / "plan.json",
             source / "complete.json", previous / "results.jsonl", previous / "complete.json", root / "plan.json",
             root / "complete.json",
             root / "results.jsonl", root / "report/summary.json", root / "report-1k-to-16k/summary.json",
             root / "report-1k-to-16k/tasks.csv"]
    if (root / "protocol.json").exists():
        files.append(root / "protocol.json")
    fingerprint = {str(p): evaluator.file_sha256(p) for p in files}
    complete = read_json(root / "complete.json")
    require(complete["plan_sha256"] == fingerprint[str(root / "plan.json")] and
            complete["results_sha256"] == fingerprint[str(root / "results.jsonl")] and
            complete["completed"] == complete["target"] == 28000, "Coordinator completion record")
    args = Namespace(**{**plan["config"], "source_root": source})
    require(sweep.configuration(args) == plan["config"], "Runtime source or protocol differs from launched plan")
    rows, sources, protocols, overlaps = report.collect(previous, root)
    matched, cells = report.summarize(rows)
    require(len(rows) == 56000 and len(cells) == 112 and matched == list(range(500)), "Matched study coverage")
    require(not any(sources[s]["ignored_uncommitted_trailing_bytes"] for s in ("previous","extended")),
            "Incomplete final aggregate")
    standalone = read_json(root / "report/summary.json")
    require(standalone["complete"] and standalone["matched_tasks"] == 500 and
            standalone["completed_evaluations"] == standalone["expected_evaluations"] == 28000, "Standalone summary")
    verify_report(root, rows, matched, cells)
    catalog = sweep.short.read_catalog(source)
    cases = evaluator.read_catalog(source)
    require(set(catalog) == set(cases) and len(catalog) == 3500, "Source task coverage")
    lanes = verify_lanes(root, plan, catalog, rows)
    pinned = evaluator.report.COMMON
    model = snapshot_download(pinned["model"], revision=pinned["model_revision"], local_files_only=True)
    model_config = read_json(Path(model) / "config.json")
    require(model_config.get("text_config", model_config)["max_position_embeddings"] >= sweep.CONTEXT_CAP,
            "Pinned model capacity below configured maximum")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    vocab = tokenizer.get_vocab()
    opening, closing = vocab["<think>"], vocab["</think>"]
    cues = {v: tokenizer.encode(("</think>" if v else "") + evaluator.CUE, add_special_tokens=False) for v in (False, True)}
    eos = set(pinned["eos_token_ids"])
    grade = lru_cache(maxsize=None)(math500.grade_math500)
    observed_answer = lru_cache(maxsize=8192)(lambda tokens,cue: answer_from_tokens(tokens, cue, tokenizer, eos))
    decoded = lru_cache(maxsize=8192)(lambda tokens: tokenizer.decode(tokens, skip_special_tokens=False))
    dataset_path, dataset = math500.load_math500()
    require(evaluator.file_sha256(dataset_path) == pinned["dataset_sha256"], "Frozen dataset fingerprint")
    stops, grades, maximum, reused, clamped = Counter(), Counter(), 0, 0, 0
    for key, case in sorted(cases.items()):
        trace = evaluator.load_source_trace(case)
        rollout = trace["rollout"]
        source_eos = verify_source_eos(rollout, case["row"]["stop_reason"], eos, key)
        identity = catalog[key]
        require((dataset[key[2]]["unique_id"], dataset[key[2]]["answer"]) == identity[:2], f"Dataset case: {key}")
        views = {}
        for budget in report.BUDGETS:
            row = rows[(*key, budget)]
            label = (*key, budget)
            require((row["unique_id"], row["gold"], row["source_trace_sha256"], row["source_trace_path"]) == identity,
                    f"Source identity: {label}")
            effective = min(budget, rollout["decode_rounds"])
            if effective not in views:
                views[effective] = reconstruct(rollout, effective, source_eos, opening, closing, cues)
            for name, value in views[effective].items():
                require(row[name] == value, f"Reconstructed {name}: {label}")
            tokens, cue = tuple(row["answer_token_ids"]), tuple(row["cue_token_ids"])
            require(observed_answer(tokens, cue) == (row["stop_reason"], row["prediction"])
                    and decoded(tokens) == row["answer_text"], f"Answer tokens: {label}")
            require(row["grading_status"] == grade(row["gold"], row["prediction"])
                    and row["correct"] == (row["grading_status"] == "correct"), f"Frozen grade: {label}")
            if budget in report.HIGH:
                maximum = max(maximum, row["context_tokens"])
                reused += row["reused_readout"]
                clamped += effective < budget
                stops[row["stop_reason"]] += 1
                grades[row["grading_status"]] += 1
    require(fingerprint == {str(p): evaluator.file_sha256(p) for p in files}
            and sweep.configuration(args) == plan["config"], "Immutable inputs changed during audit")
    return {"status": "review_required" if overlaps["mismatched_contexts"] else "passed",
            "at": sweep.lanes.now(), "duration_seconds": round(time.monotonic()-start,3),
            "model": pinned["model"], "model_revision": pinned["model_revision"], "cpu_only": True,
            "model_forwards": 0, "high_rows": 28000, "combined_rows": len(rows), "cells": len(cells),
            "matched_tasks": len(matched), "verified_source_traces": len(cases), "max_high_input_tokens": maximum,
            "high_stop_reasons": stops, "high_grades": grades, "fresh_high_readouts": 28000 - reused,
            "reused_high_readouts": reused, "clamped_high_rows": clamped, "cross_run_identical_inputs": overlaps,
            "verified_runtime_files": len(plan["config"]["code_sha256"]), "lanes": lanes,
            "source_result_report_sha256": fingerprint, "answer_tokens_and_frozen_grading_checked": True,
            "cutoff_order_context_cue_and_source_hashes_checked": True, "combined_report_checked": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "previous", "root"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        result = verify(args.source.resolve(), args.previous.resolve(), root)
    except Exception as error:
        sweep.lanes.write_json(root / "completion-verification-failed.json",
                               {"status": "failed", "at": sweep.lanes.now(), "error": f"{type(error).__name__}: {error}"})
        raise
    sweep.lanes.write_json(root / "completion-verification.json", result)
    print(f"Audit {result['status']}: {result['combined_rows']} matched rows; "
          f"{result['cross_run_identical_inputs']['mismatched_contexts']} differing identical inputs.", flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
