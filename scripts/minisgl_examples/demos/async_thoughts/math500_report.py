"""Verify and summarize the paired AsyncReasoning MATH-500 benchmark, including partial runs."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from .demo import LlamaPrompting
from .math500 import (
    ARMS,
    BOX,
    DATASET_REVISION,
    DATASET_SHA,
    ORDER,
    digest,
    encode_json,
    last_box,
    ordered_rows,
    save,
    schedule,
    verify_smoke,
)
from .short_retry import retry_allowance


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def distribution(values):
    if not values:
        return {"n": 0, "sum": 0, "mean": None, "median": None}
    return {
        "n": len(values),
        "sum": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def validate_batches(trace):
    """Verify pre-step block lengths and causal writes against the physical recorded requests."""
    batches = trace["batches"]
    require(
        trace["format_version"] == 2 and trace["steps"] == len(batches) and batches,
        "Invalid trace header",
    )
    contents = defaultdict(int)
    clears = defaultdict(list)
    for update in trace["updates"]:
        require(
            update["refs"] == [] and 0 <= update["step"] <= len(batches),
            "Unexpected cache operation",
        )
        clears[update["step"]].append(update["block"])
    for i, batch in enumerate(batches):
        require(
            batch["step"] == i and batch["kind"] in ("prefill", "decode") and batch["requests"],
            "Invalid batch",
        )
        for block in clears[i]:
            contents.pop(block, None)
        written = set()
        for req in batch["requests"]:
            block = req["write_to"]
            view = req["cache_view"]
            ids = req["input_ids"]
            require(
                ids and block not in written and req["write_start"] == contents[block],
                "Invalid block write",
            )
            require(
                view and view[-1][0] == block and all(contents[b] == n for b, n in view),
                "Incorrect pre-step view",
            )
            require(len({b for b, n in view}) == len(view), "Repeated block in view")
            require(batch["kind"] != "decode" or len(ids) == 1, "Invalid decode width")
            written.add(block)
        for req in batch["requests"]:
            contents[req["write_to"]] += len(req["input_ids"])
    return batches


def token_inputs(batches, kind, block):
    return [
        t
        for b in batches
        if b["kind"] == kind
        for q in b["requests"]
        if q["write_to"] == block
        for t in q["input_ids"]
    ]


def verify_generation_inputs(inputs, seed, content, stopped):
    expected = [seed] + list(content if stopped else content[:-1])
    require(inputs == expected, "Decode token chain differs from generated output")


def verify_ordinary(trace, prompt, content, stopped):
    batches = validate_batches(trace)
    blocks = {q["write_to"] for b in batches for q in b["requests"]}
    require(len(blocks) == 1, "Ordinary generation must use one native cache")
    block = next(iter(blocks))
    require(
        token_inputs(batches, "prefill", block) == prompt[:-1], "Ordinary native prefill mismatch"
    )
    verify_generation_inputs(token_inputs(batches, "decode", block), prompt[-1], content, stopped)


def verify_ar(trace, arm, prompting, tokenizer, prompt, writer, thinker, stopped):
    batches = validate_batches(trace)
    reads = [(b, q) for b in batches if b["kind"] == "decode" for q in b["requests"]]
    thinker_reads = [q for b, q in reads if len(q["cache_view"]) == 2]
    writer_reads = [q for b, q in reads if len(q["cache_view"]) == 3]
    require(
        thinker_reads and writer_reads, "Async trajectory lacks a real thinker or writer decode"
    )
    p, t = (x[0] for x in thinker_reads[0]["cache_view"])
    w = writer_reads[0]["write_to"]
    require(len({p, t, w}) == 3, "Roles alias one another")
    for _, q in reads:
        view = [x[0] for x in q["cache_view"]]
        require(
            (q["write_to"] == t and view == [p, t]) or (q["write_to"] == w and view == [p, t, w]),
            "Unexpected decode cache view",
        )
    if arm == "native-chat":
        native = prompting.native_turns
        require(native.prompt_ids == prompt, "Native template inputs differ")
        prefix_p, prefix_t, prefix_w = prompt[:-1], [], native.writer_ids[:-1]
        thinker_seed, writer_seed = prompt[-1], native.writer_ids[-1]
    else:
        encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
        prefix_p, prefix_t, prefix_w = (
            prompt,
            encode(prompting.thinker_output_prefix),
            encode(prompting.writer_output_prefix),
        )
        thinker_seed, writer_seed = encode("\n")[0], encode(prompting.writer_seed)[0]
    for block, expected in ((p, prefix_p), (t, prefix_t), (w, prefix_w)):
        require(
            token_inputs(batches, "prefill", block) == expected,
            "Actor prefill differs from protocol",
        )
    consumed = token_inputs(batches, "decode", t)
    require(
        consumed[0] == thinker_seed and consumed[1:] == thinker[: len(consumed) - 1],
        "Thinker seed/content chain mismatch",
    )
    require(
        len(thinker) - 1 <= len(consumed) - 1 <= len(thinker), "Unaccounted sampled thinker content"
    )
    verify_generation_inputs(token_inputs(batches, "decode", w), writer_seed, writer, stopped)
    joint = sum(
        any(len(q["cache_view"]) == 2 for q in b["requests"])
        and any(len(q["cache_view"]) == 3 for q in b["requests"])
        for b in batches
        if b["kind"] == "decode"
    )
    visible = [q["cache_view"][1][1] for q in writer_reads]
    # Views in the trace contain pre-step lengths. A decode can also attend to
    # the thinker's input appended in the same batch, before either output is sampled.
    first_writer_batch = next(b for b, q in reads if q["write_to"] == w)
    joint_first = any(q["write_to"] == t for q in first_writer_batch["requests"])
    return {
        "joint_decode_batches": joint,
        "writer_first_visible_thinker_block_tokens": visible[0],
        "writer_last_visible_thinker_block_tokens": visible[-1],
        "writer_saw_thinker_grow": len(set(visible)) > 1,
        "first_visible_generated_thought_tokens": max(
            0, visible[0] + int(joint_first) - len(prefix_t) - 1
        ),
    }


def verify_row(path, rank, task, arm, tokenizer, eos_ids, budget, directory):
    row = load(path)
    require(
        (row["rank"], row["index"], row["unique_id"], row["arm"], row["gold"])
        == (rank, task["index"], task["unique_id"], arm, task["answer"]),
        "Scheduled row identity mismatch",
    )
    zipped = (directory / "traces" / (path.stem + ".json.gz")).read_bytes()
    require(digest(zipped) == row["trace_sha256"], "Trace checksum mismatch")
    traces = json.loads(gzip.decompress(zipped))
    text = task["problem"] + "\n\n" + BOX
    form = LlamaPrompting(text, tokenizer, protocol=arm if arm != "ordinary" else "working-notes")
    content = (
        text
        + "\n\nWork through the problem in your working notes. In the final answer, answer the original question directly."
    )
    prompt = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
    )
    require(digest(encode_json(prompt)) == row["prompt_ids_sha256"], "Pinned prompt hash differs")
    require(
        tokenizer.encode(form.input_prompt, add_special_tokens=False) == prompt,
        "Actor user input differs",
    )
    writer, thinker, stop = row["writer_ids"], row["thinker_ids"], row["writer_stop_id"]
    require(
        (stop is None or stop in eos_ids)
        and not eos_ids.intersection(writer)
        and not eos_ids.intersection(thinker),
        "Invalid EOS",
    )
    require(
        len(thinker) <= budget
        and (len(writer) < budget if stop is not None else len(writer) == budget),
        "Primary sample budget/stop mismatch",
    )
    if arm == "ordinary":
        require(not thinker, "Ordinary run has a thinker")
        verify_ordinary(traces["primary"], prompt, writer, stop is not None)
        overlap = {
            "joint_decode_batches": 0,
            "writer_first_visible_thinker_block_tokens": None,
            "writer_last_visible_thinker_block_tokens": None,
            "writer_saw_thinker_grow": False,
            "first_visible_generated_thought_tokens": None,
        }
        remaining = 0
    else:
        overlap = verify_ar(
            traces["primary"], arm, form, tokenizer, prompt, writer, thinker, stop is not None
        )
        remaining = retry_allowance(len(writer), stop, budget)
    for key, val in overlap.items():
        if key != "first_visible_generated_thought_tokens":
            require(row[key] == val, f"Trace disagrees with recorded {key}")
    retry, retry_stop = row["retry_ids"], row["retry_stop_id"]
    source = "ordinary" if arm == "ordinary" else "native_retry" if remaining else "async_draft"
    require(row["selected_source"] == source, "Wrong policy source")
    if remaining:
        require(
            retry is not None
            and (retry_stop is None or retry_stop in eos_ids)
            and not eos_ids.intersection(retry),
            "Invalid native retry",
        )
        require(
            (len(retry) < remaining if retry_stop is not None else len(retry) == remaining),
            "Retry failed its residual budget",
        )
        verify_ordinary(traces["retry"], prompt, retry, retry_stop is not None)
        selected = retry
    else:
        require(
            traces["retry"] is None and retry is None and retry_stop is None, "Unexpected retry"
        )
        selected = writer
    charged = len(writer) + int(stop is not None) + len(retry or []) + int(retry_stop is not None)
    require(
        charged == row["writer_samples_charged"] <= budget, "Incorrect total charged writer samples"
    )
    for ids, tk, bk, sk in [
        (writer, "writer_text", "writer_box", "pure_exact"),
        (selected, "selected_text", "selected_box", "selected_exact"),
    ]:
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        box = last_box(decoded)
        require(
            decoded == row[tk] and box == row[bk] and row[sk] == (box == task["answer"].strip()),
            "Output or grade differs from token IDs",
        )
    times = [row["draft_seconds"], row["total_seconds"]]
    require(
        all(math.isfinite(x) for x in times) and 0 <= times[0] <= times[1], "Invalid compute timing"
    )
    ttft = row["draft_first_content_seconds"]
    require(
        (ttft is None and not writer)
        or (bool(writer) and ttft is not None and math.isfinite(ttft) and 0 <= ttft <= times[0]),
        "Invalid draft first-content time",
    )
    row["first_visible_generated_thought_tokens"] = overlap[
        "first_visible_generated_thought_tokens"
    ]
    return row


def summaries(records):
    return {
        "rows": len(records),
        "pure_correct": sum(r["pure_exact"] for r in records),
        "selected_correct": sum(r["selected_exact"] for r in records),
        "pure_accuracy": sum(r["pure_exact"] for r in records) / len(records) if records else None,
        "selected_accuracy": sum(r["selected_exact"] for r in records) / len(records)
        if records
        else None,
        "output_sources": dict(Counter(r["selected_source"] for r in records)),
        "writer_stops": dict(
            Counter("eos" if r["writer_stop_id"] is not None else "budget" for r in records)
        ),
        "retry_stops": dict(
            Counter(
                "eos" if r["retry_stop_id"] is not None else "budget"
                for r in records
                if r["retry_ids"] is not None
            )
        ),
        "thinker_content_tokens": distribution([len(r["thinker_ids"]) for r in records]),
        "writer_content_tokens": distribution([len(r["writer_ids"]) for r in records]),
        "selected_content_tokens": distribution(
            [len(r["writer_ids"] if r["retry_ids"] is None else r["retry_ids"]) for r in records]
        ),
        "writer_samples_charged": distribution([r["writer_samples_charged"] for r in records]),
        "native_retry_content_tokens": distribution(
            [len(r["retry_ids"]) for r in records if r["retry_ids"] is not None]
        ),
        "draft_seconds": distribution([r["draft_seconds"] for r in records]),
        "draft_first_content_seconds": distribution(
            [
                r["draft_first_content_seconds"]
                for r in records
                if r["draft_first_content_seconds"] is not None
            ]
        ),
        "selected_seconds": distribution([r["total_seconds"] for r in records]),
        "retry_seconds": distribution(
            [r["total_seconds"] - r["draft_seconds"] for r in records if r["retry_ids"] is not None]
        ),
        "rows_with_joint_decode": sum(r["joint_decode_batches"] > 0 for r in records),
        "rows_with_visible_thought_growth": sum(r["writer_saw_thinker_grow"] for r in records),
        "rows_with_joint_decode_and_growth": sum(
            r["joint_decode_batches"] > 0 and r["writer_saw_thinker_grow"] for r in records
        ),
        "joint_batches": sum(r["joint_decode_batches"] for r in records),
        "first_writer_visible_generated_thought_tokens": distribution(
            [
                r["first_visible_generated_thought_tokens"]
                for r in records
                if r["first_visible_generated_thought_tokens"] is not None
            ]
        ),
    }


def paired(baseline, candidate):
    require(set(baseline) == set(candidate), "Paired sets differ")
    a, b = baseline, candidate
    gain = [i for i in a if not a[i] and b[i]]
    loss = [i for i in a if a[i] and not b[i]]
    discordant = len(gain) + len(loss)
    sign_p = (
        min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(min(len(gain), len(loss)) + 1))
            / 2**discordant,
        )
        if discordant
        else 1.0
    )
    return {
        "n": len(a),
        "baseline_correct": sum(a.values()),
        "candidate_correct": sum(b.values()),
        "both": sum(a[i] and b[i] for i in a),
        "neither": sum(not a[i] and not b[i] for i in a),
        "gained": gain,
        "lost": loss,
        "gained_count": len(gain),
        "lost_count": len(loss),
        "exact_two_sided_sign_p": sign_p,
        "delta_percentage_points": 100 * (len(gain) - len(loss)) / len(a) if a else None,
    }


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def report(directory, dataset, tokenizer, source_dir=None, historical=None):
    settings = load(directory / "manifest.json")
    require(
        settings["dataset_sha256"] == DATASET_SHA
        and settings["dataset_revision"] == DATASET_REVISION
        and settings["order"] == ORDER
        and settings["sampling"] == "greedy"
        and settings["dtype"] == "bfloat16",
        "Benchmark settings differ",
    )
    rows = ordered_rows(dataset)
    p, l, budget = (settings[k] for k in ("paired", "limit", "budget"))
    require(
        0 <= p <= l <= len(rows)
        and budget > 0
        and settings["context"] > 2 * budget
        and settings["probe_period"] == 20
        and not settings["smoke_only"],
        "Invalid evaluation settings",
    )
    require(settings["ordered_ids"] == [r["unique_id"] for r in rows[:l]], "ID order mismatch")
    require(
        set(settings["source_sha256"]) == {"math500.py", "short_retry.py", "engine.py", "demo.py"},
        "Unexpected evaluated modules",
    )
    if source_dir is not None:
        for file, sha in settings["source_sha256"].items():
            require(digest((source_dir / file).read_bytes()) == sha, f"Source mismatch in {file}")
    tasks = list(schedule(rows, p, l))
    names = [f"{rank:03d}.{arm}.json" for rank, _, arm in tasks]
    present = {f.name for f in (directory / "rows").glob("*.json")}
    environment = load(directory / "environment.json") if present else {}
    if present:
        require(
            bool(environment["gpu"])
            and environment["gpu"] == settings.get("gpu", environment["gpu"])
            and environment["effective_context"] >= settings["context"]
            and environment["eos_token_ids"],
            "Engine identity, context, or EOS metadata mismatch",
        )
        verify_smoke(load(directory / "smoke.json"))
    eos_ids = set(environment.get("eos_token_ids", ()))
    require(
        present == set(names[: len(present)]),
        "Completed rows are not a prefix of the immutable schedule",
    )
    complete = (directory / "completed.json").exists()
    if complete:
        receipt = load(directory / "completed.json")
        require(
            len(present) == receipt["planned_rows"] == len(names)
            and receipt["settings_sha256"] == digest(encode_json(settings)),
            "Terminal receipt/schedule mismatch",
        )
    data = defaultdict(dict)
    for filename, (rank, row, arm) in zip(names[: len(present)], tasks):
        try:
            data[rank][arm] = verify_row(
                directory / "rows" / filename, rank, row, arm, tokenizer, eos_ids, budget, directory
            )
        except (ValueError, KeyError, OSError, IndexError) as exc:
            raise ValueError(f"{filename}: {exc}") from exc
    matched = [r for r in range(p) if len(data.get(r, {})) == 3]
    partial = [r for r, arms in data.items() if r < p and len(arms) < 3]

    def stratum(ranks, arm):
        return [data[r][arm] for r in ranks if arm in data.get(r, {})]

    comparisons = {}
    for arm, base in [(ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])]:
        for field in ("pure_exact", "selected_exact"):
            get = lambda a: {data[r][a]["unique_id"]: data[r][a][field] for r in matched}
            comparisons[f"{arm}_vs_{base}_{field}"] = paired(get(base), get(arm))
    parity = []
    for r in matched:
        ordinary, native = data[r]["ordinary"], data[r]["native-chat"]
        obs, nat = ordinary["writer_ids"], native["thinker_ids"]
        parity.append(
            {
                "rank": r,
                "unique_id": native["unique_id"],
                "first_ordinary_id": obs[0] if obs else None,
                "first_native_thinker_id": nat[0] if nat else None,
                "shared_first_ids": common_prefix(nat, obs),
                "thinker_tokens": len(nat),
                "ordinary_tokens": len(obs),
            }
        )
    result = {
        "status": "complete" if complete else "in_progress",
        "settings_sha256": digest(encode_json(settings)),
        "source_sha256": settings["source_sha256"],
        "model": settings["model"],
        "model_revision": settings["model_revision"],
        "environment": environment,
        "paper_schedule": (p, l, budget, settings["context"], settings["model_revision"])
        == (64, 500, 16384, 40960, "666943798adbde0b1aff34626007e26986a3c107"),
        "dataset_sha256": DATASET_SHA,
        "schedule": {
            "planned_rows": len(tasks),
            "verified_rows": len(present),
            "paired_target": p,
            "complete_paired_ranks": len(matched),
            "paired_target_complete": len(matched) == p,
            "partial_paired_ranks": partial,
            "native_only_extension_target": l - p,
            "native_only_extension_completed": sum(r >= p for r in data),
            "native_total_target": l,
            "native_total_completed": sum("native-chat" in v for v in data.values()),
        },
        "complete_paired_cohort": {arm: summaries(stratum(matched, arm)) for arm in ARMS},
        "paired_effects": comparisons,
        "paired_statistics_note": "Two-sided exact sign tests condition on discordant solves. P-values are unadjusted across contrasts. Partial snapshots are exploratory.",
        "native_thinker_vs_ordinary": {
            "matched_rows": len(parity),
            "shared_prefix_length": distribution([i["shared_first_ids"] for i in parity]),
            "equal_first_8": sum(q["shared_first_ids"] >= 8 for q in parity),
            "equal_first_1": sum(q["shared_first_ids"] >= 1 for q in parity),
            "comparable_first_1": sum(
                min(q["thinker_tokens"], q["ordinary_tokens"]) >= 1 for q in parity
            ),
            "equal_first_32": sum(q["shared_first_ids"] >= 32 for q in parity),
            "comparable_first_32": sum(
                min(q["thinker_tokens"], q["ordinary_tokens"]) >= 32 for q in parity
            ),
            "comparable_first_8": sum(
                min(q["thinker_tokens"], q["ordinary_tokens"]) >= 8 for q in parity
            ),
            "rows": parity,
        },
        "extension_only": summaries(stratum(range(p, l), "native-chat")),
        "all_native": summaries(stratum(range(l), "native-chat")),
        "available_by_arm": {a: summaries(stratum(range(l), a)) for a in ARMS},
        "cases": [
            {
                key: r[key]
                for key in (
                    "rank",
                    "index",
                    "unique_id",
                    "arm",
                    "gold",
                    "writer_box",
                    "selected_box",
                    "pure_exact",
                    "selected_exact",
                    "selected_source",
                    "writer_stop_id",
                    "retry_stop_id",
                    "writer_samples_charged",
                    "first_visible_generated_thought_tokens",
                    "draft_first_content_seconds",
                    "draft_seconds",
                    "total_seconds",
                    "trace_sha256",
                )
            }
            for arms in data.values()
            for r in arms.values()
        ],
        "timing": "Per-row inference seconds. Totals exclude setup, smoke and artifact I/O. First-content time measures the draft. In the buffered retry policy the selected answer is available at completion.",
    }
    if historical:
        hist = {
            x["unique_id"]: x for x in (json.loads(l) for l in historical.read_bytes().splitlines())
        }
        require(
            len(hist) == 500 and set(hist) == {r["unique_id"] for r in rows},
            "Incomplete historical cohort",
        )
        comps = {}
        for field, base in (("ordinary", "standard_matched_16k"), ("async", "async_16k")):
            for grade in ("pure_exact", "selected_exact"):
                fresh = stratum(range(l), "native-chat")
                shares = {item["unique_id"]: item for item in fresh}
                for uid, item in shares.items():
                    prev = hist[uid]
                    require(
                        prev["index"] == item["index"]
                        and prev["gold"].strip() == item["gold"].strip()
                        and prev[base]["strict_exact"]
                        == (prev[base]["boxed"] == prev["gold"].strip())
                        and prev[base]["prompt_ids_sha256"] == item["prompt_ids_sha256"],
                        "Historical cohort/prompt mismatch",
                    )
                comps[f"native_vs_historical_{field}_{grade}"] = paired(
                    {uid: hist[uid][base]["strict_exact"] for uid in shares},
                    {uid: t[grade] for uid, t in shares.items()},
                )
        result["historical"] = {
            "label": "Earlier run on the same IDs and user prompts; independent process and historical async implementation.",
            "source_sha256": digest(historical.read_bytes()),
            "comparisons": comps,
        }
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("eval-dir", "dataset", "tokenizer", "output"):
        cli.add_argument("--" + name, required=True, type=Path)
    cli.add_argument("--source-dir", type=Path)
    cli.add_argument("--historical", type=Path)
    args = cli.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = report(args.eval_dir, args.dataset, tokenizer, args.source_dir, args.historical)
    save(
        args.output,
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n",
    )
    print(json.dumps(result["schedule"]))


if __name__ == "__main__":
    main()
