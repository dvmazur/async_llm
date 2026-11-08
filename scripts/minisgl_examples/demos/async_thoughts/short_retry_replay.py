"""Reproduce the conditional native retry from two archived, prompt-matched MATH-500 runs.

Each row uses a completed async attempt and the corresponding archived ordinary token
sequence. The computation establishes exact policy output for those paired components;
it does not measure the new demo's scheduling, accuracy on fresh data, or wall time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from minisgl_examples.demos.async_thoughts.math500 import digest, last_box
from minisgl_examples.demos.async_thoughts.short_retry import MAX_SHORT_DRAFT, retry_allowance

WRITER_BUDGET = 16_384
MODEL_REVISION = "666943798adbde0b1aff34626007e26986a3c107"
ORDER_DIGEST = "3e8a27223b3312c69126a31f7d326d0469c5960fcd31ae45907a788b825c3459"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _checked_trace(directory, row, expected_sha):
    path = directory / "traces" / f"{row['index']:04d}.json"
    data = path.read_bytes()
    _require(digest(data) == row["trace_sha256"] == expected_sha, f"Changed archive trace {path}")
    return json.loads(data)["batches"]


def _draft_stop(directory, row, comparison, eos):
    batches = _checked_trace(directory, row, comparison["trace_sha256"])
    prefix = list(
        dict.fromkeys(
            req["write_to"]
            for batch in batches
            if batch["kind"] == "prefill"
            for req in batch["requests"]
        )
    )
    writer = prefix[2]
    consumed = [
        token
        for batch in batches
        if batch["kind"] == "decode"
        for req in batch["requests"]
        if req["write_to"] == writer
        for token in req["input_ids"]
    ]
    ids = row["writer_token_ids"]
    _require(len(ids) == comparison["writer_tokens"] and eos not in ids, "Invalid async content")
    closed = comparison["writer_stop"] == "eos"
    _require(comparison["writer_stop"] in ("eos", "budget"), "Missing writer stop reason")
    _require(len(consumed) == len(ids) + int(closed), "Writer terminal trace length mismatch")
    _require(consumed[1:] == (ids if closed else ids[:-1]), "Writer terminal token chain mismatch")
    stop = eos if closed else None
    if "writer_stop_token_id" in row:
        _require(row["writer_stop_token_id"] == stop, "Explicit writer stop disagrees with trace")
    return stop


def replay(async_full, async_retry, comparison_run, tokenizer):
    paired_path = comparison_run / "comparison/comparison_cases.jsonl"
    ordinary_path = comparison_run / "matched/results.jsonl"
    cases, ordinary = _rows(paired_path), _rows(ordinary_path)
    _require(len(cases) == len(ordinary) == 500, "Expected two complete MATH-500 runs")
    _require(len({r["unique_id"] for r in cases}) == 500, "Duplicated MATH IDs")
    ordered = [
        {"unique_id": r["unique_id"], "problem": r["problem"], "answer": r["gold"]} for r in cases
    ]
    order_hash = digest(json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode())
    _require(order_hash == ORDER_DIGEST, "Canonical MATH-500 order, questions or answers changed")
    standard = json.loads((comparison_run / "matched/manifest.json").read_bytes())
    _require(
        standard["settings"]["model_revision"] == MODEL_REVISION
        and standard["settings"]["sampling"] == "greedy"
        and standard["settings"]["ordered_dataset_sha256"] == ORDER_DIGEST
        and standard["environment"]["eos_token_ids"] == [tokenizer.eos_token_id],
        "Matched ordinary archive setup mismatch",
    )
    paths = {
        "full": async_full,
        "retry": async_retry,
        "thinker_only": comparison_run / "thinker_only",
    }
    indices = {}
    for phase, path in paths.items():
        manifest = json.loads((path / "manifest.json").read_bytes())
        settings = manifest["settings"]
        _require(
            settings["model_revision"] == MODEL_REVISION
            and (phase != "full" or settings["ordered_dataset_sha256"] == ORDER_DIGEST)
            and settings["sampling"] == "greedy",
            "Archive setup mismatch",
        )
        _require(
            manifest["environment"]["eos_token_ids"] == [tokenizer.eos_token_id], "EOS mismatch"
        )
        indices[phase] = {r["unique_id"]: r for r in _rows(path / "results.jsonl")}
    totals, folds, records = Counter(), [Counter() for _ in range(5)], []
    for i, (case, native) in enumerate(zip(cases, ordinary)):
        uid = case["unique_id"]
        a, n = case["async_16k"], case["standard_matched_16k"]
        _require(
            i == case["index"] == native["index"] and uid == native["unique_id"], "Order mismatch"
        )
        _require(
            a["prompt_ids_sha256"] == n["prompt_ids_sha256"] == native["prompt_token_ids_sha256"],
            "Native/async prompt mismatch",
        )
        draft = indices[a["phase"]][uid]
        stop = _draft_stop(paths[a["phase"]], draft, a, tokenizer.eos_token_id)
        native_trace = _checked_trace(comparison_run / "matched", native, n["trace_sha256"])
        draft_ids, native_ids = draft["writer_token_ids"], native["token_ids"]
        consumed = [
            token
            for batch in native_trace
            if batch["kind"] == "decode"
            for request in batch["requests"]
            for token in request["input_ids"]
        ]
        stopped = native["stop_token_id"] is not None
        _require(
            len(consumed) == len(native_ids) + int(stopped)
            and consumed[1:] == (native_ids if stopped else native_ids[:-1]),
            "Native terminal token chain mismatch",
        )
        _require(
            tokenizer.eos_token_id not in native_ids
            and native["stop_token_id"] in (None, tokenizer.eos_token_id),
            "Invalid native stop",
        )
        _require(
            (len(native_ids) == WRITER_BUDGET) == (native["stop_token_id"] is None),
            "Invalid native length",
        )
        remaining = retry_allowance(len(draft_ids), stop, WRITER_BUDGET)
        selected = native_ids[:remaining] if remaining else draft_ids
        retry_stop = native["stop_token_id"] if remaining > len(native_ids) else None
        final_box = last_box(tokenizer.decode(selected, skip_special_tokens=True))
        gold = case["gold"].strip()
        draft_right = last_box(tokenizer.decode(draft_ids, skip_special_tokens=True)) == gold
        native_right = last_box(tokenizer.decode(native_ids, skip_special_tokens=True)) == gold
        _require(
            draft_right == a["strict_exact"] and native_right == n["strict_exact"],
            "Source grade mismatch",
        )
        right = final_box == gold
        fold = int(digest(uid.encode())[:16], 16) % len(folds)
        for count in (totals, folds[fold]):
            count.update(
                problems=1,
                native_retries=int(bool(remaining)),
                async_correct=int(draft_right),
                ordinary_correct=int(native_right),
                selected_correct=int(right),
                gained=int(right and not draft_right),
                lost=int(draft_right and not right),
                added_content_tokens=len(selected) if remaining else 0,
                added_sampled_tokens=(len(selected) + int(retry_stop is not None))
                if remaining
                else 0,
                truncated_native=int(bool(remaining) and remaining < len(native_ids)),
            )
        records.append(
            {
                "index": i,
                "unique_id": uid,
                "source": "native_retry" if remaining else "async_draft",
                "draft_content_tokens": len(draft_ids),
                "draft_stop_id": stop,
                "async_trace_sha256": a["trace_sha256"],
                "ordinary_trace_sha256": n["trace_sha256"],
                "retry_allowance": remaining,
                "retry_stop_id": retry_stop,
                "selected_box": final_box,
                "selected_exact": right,
                "async_exact": draft_right,
                "ordinary_exact": native_right,
                "fold": fold,
            }
        )
    return {
        "kind": "archived_paired_policy_replay",
        "model_revision": MODEL_REVISION,
        "ordered_dataset_sha256": ORDER_DIGEST,
        "comparison_sha256": digest(paired_path.read_bytes()),
        "ordinary_results_sha256": digest(ordinary_path.read_bytes()),
        "writer_budget": WRITER_BUDGET,
        "maximum_short_draft": MAX_SHORT_DRAFT,
        "counts": dict(totals),
        "descriptive_id_hash_folds": folds,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("async-full", "async-retry", "comparison-run", "tokenizer", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = replay(args.async_full, args.async_retry, args.comparison_run, tokenizer)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
