"""Compare a verified prefix of the three new Tülu arms to their archived matched controls."""

import argparse
import gzip
import json
from hashlib import sha256
from pathlib import Path

from transformers import AutoTokenizer

from .math500 import (
    DATASET_REVISION,
    DATASET_SHA,
    ORDER,
    digest,
    encode_json,
    last_box,
    ordered_rows,
)
from .math500_report import distribution, paired, require, token_inputs, validate_batches
from .math500_tulu_paired import CHECKPOINT, REVISION, SOURCES, make_prompt

ARMS = ("direct", "serial-thinking", "evidence-turns")
REFERENCE_SHA256 = "24411345e713b72088c7a4930258b66ec2144b811ff4781b9ca2bffd85aece83"


def load_json(path):
    return json.loads(path.read_bytes())


def archive_rows(path):
    raw = path.read_bytes()
    require(
        sha256(raw).hexdigest() == REFERENCE_SHA256,
        "Reference ledger differs from published results",
    )
    return {r["unique_id"]: r for r in map(json.loads, raw.splitlines())}


def manifest(directory, tasks):
    m = load_json(directory / "manifest.json")
    pinned = {
        "schema_version": 1,
        "model": CHECKPOINT,
        "model_revision": REVISION,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA,
        "order": ORDER,
        "arms": list(ARMS),
        "budget": 16384,
        "serial_draft_cap": 8192,
        "dtype": "bfloat16",
        "sampling": "greedy",
        "context": 40960,
        "memory_ratio": 0.85,
        "probe_period": 20,
        "llama_gate": "evidence-turns",
        "probe_include_pending": True,
        "smoke_only": False,
    }
    for key, expected in pinned.items():
        require(m.get(key) == expected, f"Unexpected run setting: {key}")
    require(0 <= m["start_rank"] < m["limit"] <= len(tasks), "Invalid rank bounds")
    require(m["ordered_ids"] == [r["unique_id"] for r in tasks[: m["limit"]]], "Order differs")
    here = Path(__file__).parent
    require(
        m["source_sha256"] == {s: digest((here / s).read_bytes()) for s in SOURCES},
        "Run source differs from this evaluator checkout",
    )
    return m, digest(encode_json(m))


def verify_physical(batches, roles, prompt, row):
    arm = row["arm"]
    if arm != "evidence-turns":
        blocks = {req["write_to"] for b in batches for req in b["requests"]}
        require(len(blocks) == 1, "Ordinary baseline did not use one cache")
        block = next(iter(blocks))
        expected = prompt.initial[:-1]
        for b in batches:
            if b["kind"] == "decode":
                require(
                    len(b["requests"]) == 1 and len(roles.get(b["step"], {})) == 1,
                    "Missing serial sample",
                )
        inputs = {k: [] for k in ("thinker", "writer")}
        for step, group in sorted(roles.items()):
            role = next(iter(group))
            inputs[role].extend(batches[step]["requests"][0]["input_ids"])
        if arm == "serial-thinking":
            thoughts = row["thinker_ids"]
            stopped = row["thinker_stop_id"] is not None
            require(
                inputs["thinker"]
                == [prompt.initial[-1]] + (thoughts if stopped else thoughts[:-1]),
                "Serial private decode chain differs",
            )
            expected += ([] if stopped else thoughts[-1:]) + prompt.bridge[:-1]
        require(token_inputs(batches, "prefill", block) == expected, "Native turn prefill differs")
        seed = (prompt.bridge or prompt.initial)[-1]
        writer = row["writer_ids"]
        require(
            inputs["writer"]
            == [seed] + (writer if row["writer_stop_id"] is not None else writer[:-1]),
            "Serial answer decode chain differs",
        )
        return
    blocks = {}
    views = {}
    for b in batches:
        if b["kind"] != "decode":
            continue
        group = roles.get(b["step"], {})
        seen = set()
        for req in b["requests"]:
            view = [x[0] for x in req["cache_view"]]
            require(len(view) in (2, 3), "Unexpected async actor")
            role = "thinker" if len(view) == 2 else "writer"
            require(role in group and role not in seen, "Wrong actor charged for physical decode")
            require(views.setdefault(role, view) == view, "Actor cache changed")
            blocks[role] = req["write_to"]
            seen.add(role)
        require(seen == set(group), "Sample did not produce a decode")
    require(
        "writer" in views and "thinker" in views and views["writer"][:-1] == views["thinker"],
        "Writer does not share the live thinker cache",
    )
    base, thinker, writer = views["writer"]
    require(
        token_inputs(batches, "prefill", base) == prompt.initial[:-1]
        and token_inputs(batches, "prefill", writer) == prompt.bridge[:-1],
        "Async prompt/bridge differs",
    )
    for role, block, seed in (
        ("thinker", thinker, prompt.initial[-1]),
        ("writer", writer, prompt.bridge[-1]),
    ):
        sampled = [group[role] for _, group in sorted(roles.items()) if role in group]
        require(
            token_inputs(batches, "decode", block) == [seed] + sampled[:-1],
            "Async decode inputs differ",
        )
    flush = token_inputs(batches, "prefill", thinker)
    require(
        not flush
        or (
            len(flush) == 1 and row["thinker_samples"] == 16384 and flush == row["thinker_ids"][-1:]
        ),
        "Unexpected thinker flush",
    )


def verify_row(path, tokenizer, tasks, archive, settings, settings_sha):
    row = load_json(path)
    rank, arm = row["rank"], row["arm"]
    require(0 <= rank < len(tasks) and arm in ARMS, "Unexpected rank or arm")
    require(path.stem == f"{rank:03}.{arm}", "Filename and row identity differ")
    task = tasks[rank]
    uid = task["unique_id"]
    require(
        (row["unique_id"], row["index"], row["gold"]) == (uid, task["index"], task["answer"]),
        "Task or gold differs",
    )
    require(
        archive[uid]["study_rank"] == rank and row["gold"] == archive[uid]["gold"],
        "Reference identity differs",
    )
    require(
        digest(encode_json(row["initial_prompt_ids"])) == row["prompt_ids_sha256"],
        "Prompt digest differs",
    )
    require(
        settings["start_rank"] <= rank < settings["limit"]
        and row["settings_sha256"] == settings_sha,
        "Row does not belong to this session manifest",
    )
    prompt = make_prompt(tokenizer, task["problem"], arm)
    require(
        row["initial_prompt_ids"] == prompt.initial
        and row["bridge_ids"] == prompt.bridge
        and row["user_message"] == prompt.user
        and row["user_message_sha256"] == digest(prompt.user.encode())
        and row["system_message"] == prompt.system
        and row["followup_message"] == prompt.followup,
        "Executed chat differs from the pinned protocol",
    )
    if arm == "evidence-turns":
        require(
            row["prompt_ids_sha256"] == archive[uid]["prompt_ids_sha256"],
            "Gate changed the original task prompt",
        )
    zipped = (path.parent.parent / "traces" / (path.stem + ".json.gz")).read_bytes()
    require(digest(zipped) == row["trace_sha256"], "Trace hash differs")
    trace = json.loads(gzip.decompress(zipped))
    batches = validate_batches(trace["trace"])
    decodes = [b for b in batches if b["kind"] == "decode"]
    require(
        len(decodes) == row["decode_batches"] and len(batches) == row["forward_batches"],
        "Inference step counts differ",
    )
    groups = {}
    role_ids = {"thinker": [], "writer": []}
    eos = tokenizer.eos_token_id
    for sample in trace["samples"]:
        role, token, step = sample["role"], sample["token"], sample["batch_step"]
        require(
            role in role_ids and 0 <= step < len(batches) and batches[step]["kind"] == "decode",
            "Invalid sampled role or step",
        )
        role_ids[role].append(token)
        group = groups.setdefault(step, {})
        require(role not in group, "Role sampled twice in one batch")
        group[role] = token
    verify_physical(batches, groups, prompt, row)
    writer_prefix = []
    spent = 0
    for step in sorted(groups):
        group = groups[step]
        if spent + len(group) > 16384:
            break
        spent += len(group)
        if group.get("writer") not in (None, eos):
            writer_prefix.append(group["writer"])
    for role, ids in role_ids.items():
        content, stop = row[f"{role}_ids"], row[f"{role}_stop_id"]
        require(stop in (None, eos) and eos not in content, "EOS in content")
        tail = ids[len(content) :]
        cancelled = (
            arm == "evidence-turns"
            and role == "thinker"
            and stop is None
            and len(tail) == 1
            and tail[0] != eos
            and groups[max(groups)].get("thinker") == tail[0]
        )
        require(
            ids[: len(content)] == content
            and (tail == ([stop] if stop is not None else []) or cancelled),
            "Samples differ from accepted tokens/EOS",
        )

        require(len(ids) == row[f"{role}_samples"], "Role charge differs")
    votes = trace["probe_events"]
    require(not votes or arm == "evidence-turns", "A serial baseline issued a writer gate probe")
    admitted_votes = [v for v in votes if v["applied"] and v["vote_write"]]
    row["probe_count"] = len(votes)
    row["positive_probe_count"] = len(admitted_votes)
    for phase in ("admission", "continuation"):
        applied = [v for v in votes if v["applied"] and v["phase"] == phase]
        row[phase + "_applied"] = len(applied)
        row[phase + "_yes"] = sum(v["vote_write"] for v in applied)
    if arm == "evidence-turns":
        last_thought = max(step for step, group in groups.items() if "thinker" in group)
        row["writer_tokens_after_last_thinker_decode"] = sum(
            step > last_thought and g.get("writer") not in (None, eos) for step, g in groups.items()
        )
        committed = [v["thinker_block_tokens"] for v in votes]
        require(
            all(earlier < later for earlier, later in zip(committed, committed[1:])),
            "Probe scored twice without new committed thought",
        )
        require(all(not v["writer_allowed_before"] for v in votes), "Active writer was re-probed")
        require(
            all(
                v["source"] != "live_cache"
                or v["query_pending_answer_token_id"] == v["pending_answer_token_id"]
                for v in votes
            ),
            "A live writer probe missed the pending token",
        )
    else:
        row["writer_tokens_after_last_thinker_decode"] = None
    row["joint_decode_verified"] = sum(len(groups[step]) == 2 for step in groups)
    require(row["joint_decode_verified"] == row["joint_decode_batches"], "Joint step count differs")
    nt, nw = len(role_ids["thinker"]), len(role_ids["writer"])
    require(
        max(nt, nw) <= 16384 and (arm == "evidence-turns" or nt + nw <= 16384),
        "Token budget exceeded",
    )
    require(arm != "direct" or nt == 0, "Direct arm had private samples")
    require(arm != "serial-thinking" or nt <= 8192, "Private phase exceeded its allowance")
    answer = tokenizer.decode(row["writer_ids"], skip_special_tokens=True)
    box = last_box(answer)
    exact = box == row["gold"].strip()
    shared = (
        last_box(tokenizer.decode(writer_prefix, skip_special_tokens=True)) == row["gold"].strip()
    )
    require(
        (answer, box, exact) == (row["writer_text"], row["writer_box"], row["pure_exact"]),
        "Final grade differs",
    )
    require(
        (len(writer_prefix), shared)
        == (row["shared_prefix_writer_tokens"], row["shared_prefix_exact"]),
        "Shared-cap grade differs",
    )
    return row


def summarize(rows):
    fields = {}
    for field in (
        "writer_tokens",
        "thinker_tokens",
        "content_tokens",
        "decode_batches",
        "forward_batches",
        "elapsed_seconds",
        "joint_decode_batches",
        "probe_count",
        "positive_probe_count",
    ):
        fields[field] = distribution([r[field] for r in rows])
    if rows[0]["arm"] == "evidence-turns":
        fields["writer_tokens_after_last_thinker_decode"] = distribution(
            [r["writer_tokens_after_last_thinker_decode"] for r in rows]
        )
        for phase in ("admission", "continuation"):
            yes = sum(r[phase + "_yes"] for r in rows)
            n = sum(r[phase + "_applied"] for r in rows)
            fields[phase + "_votes"] = {
                "yes": yes,
                "applied": n,
                "yes_fraction": yes / n if n else None,
            }
    return {
        "coverage": len(rows),
        "correct": sum(r["pure_exact"] for r in rows),
        "shared_16k_correct": sum(r["shared_prefix_exact"] for r in rows),
        **fields,
    }


def comparison(eval_dirs, tasks, archive, tokenizer, expected, partial=False):
    observed = {arm: {} for arm in ARMS}
    for directory in eval_dirs:
        settings, settings_sha = manifest(directory, tasks)
        for path in sorted((directory / "rows").glob("*.json")):
            row = verify_row(path, tokenizer, tasks, archive, settings, settings_sha)
            rank, arm = row["rank"], row["arm"]
            if rank >= expected:
                continue
            row.update(
                writer_tokens=len(row["writer_ids"]),
                thinker_tokens=len(row["thinker_ids"]),
                content_tokens=len(row["writer_ids"]) + len(row["thinker_ids"]),
            )
            previous = observed[arm].get(rank)
            require(previous is None or previous == row, "Duplicate predictions differ")
            observed[arm][rank] = row
    ranks = sorted(set.intersection(*(set(by_rank) for by_rank in observed.values())))
    require(ranks == list(range(len(ranks))), "Comparison cohort is not a prefix")
    require(ranks and (partial or len(ranks) == expected), "Requested paired cohort is incomplete")
    summaries, outcomes = {}, {}
    cases = []
    controls = ("archived-ordinary", "archived-native-chat")
    for arm in controls:
        outcomes[arm] = {}
    for arm in ARMS:
        rows = [observed[arm][i] for i in ranks]
        summaries[arm] = summarize(rows)
        outcomes[arm] = {r["rank"]: r["pure_exact"] for r in rows}
    for rank in ranks:
        uid = tasks[rank]["unique_id"]
        ref = archive[uid]
        outcomes[controls[0]][rank] = ref["ordinary_exact"]
        outcomes[controls[1]][rank] = ref["native_exact"]
        cases.append(
            {
                "rank": rank,
                "unique_id": uid,
                "gold": ref["gold"],
                "outcomes": {arm: outcomes[arm][rank] for arm in (*controls, *ARMS)},
                "traces": {arm: observed[arm][rank]["trace_sha256"] for arm in ARMS},
            }
        )
    for arm, prefix in zip(controls, ("ordinary", "native")):
        ref = [archive[tasks[i]["unique_id"]] for i in ranks]
        wt = "ordinary_tokens" if prefix == "ordinary" else "writer_tokens"
        secs = "ordinary_seconds" if prefix == "ordinary" else "pure_seconds"
        summaries[arm] = {
            "coverage": len(ref),
            "correct": sum(outcomes[arm].values()),
            "writer_tokens": distribution([r[wt] for r in ref]),
            "thinker_tokens": distribution(
                [0 if prefix == "ordinary" else r["thinker_tokens"] for r in ref]
            ),
            "elapsed_seconds": distribution([r[secs] for r in ref]),
        }
    candidate = outcomes["evidence-turns"]
    return {
        "coverage": len(ranks),
        "target": expected,
        "arm_counts": {k: len(v) for k, v in observed.items()},
        "summary": summaries,
        "paired_against_evidence_turns": {
            k: paired(v, candidate) for k, v in outcomes.items() if k != "evidence-turns"
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, action="append", required=True)
    for name in ("dataset", "tokenizer", "reference"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected", type=int, choices=(64, 500), default=64)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = comparison(
        args.eval_dir,
        ordered_rows(args.dataset),
        archive_rows(args.reference),
        tok,
        args.expected,
        args.partial,
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
