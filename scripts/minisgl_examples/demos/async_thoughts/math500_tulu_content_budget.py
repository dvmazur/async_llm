"""Compare five Tülu arms at shared generated-content budgets using recorded trajectories."""

import argparse
import gzip
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from .math500 import digest, encode_json, last_box
from .math500_report import require
from .math500_token_budget import (
    Case,
    checked_bytes,
    match_spend,
    read_cases,
    replay_schedule,
    score,
)

NEW_ARMS = ("direct", "serial-thinking", "evidence-turns")
ORDINARY = "archived-ordinary"
NATIVE = "archived-native-chat"
DEFAULT_CAPS = (256, 512, 1024, 2048, 4096, 8192, 16384)


def content_schedule(samples, thinker, writer):
    """Charge accepted role outputs together at the end of their recorded decode batch."""
    expected = {"thinker": thinker, "writer": writer}
    by_step = defaultdict(dict)
    for event in samples:
        role = event["role"]
        step = by_step[event["batch_step"]]
        require(
            role in expected and role not in step, "Duplicate or unknown role in a sample batch"
        )
        step[role] = event["token"]
    seen = {"thinker": 0, "writer": 0}
    ends, writer_at = [0], []
    for step in sorted(by_step):
        charge = 0
        wrote = False
        for role, token in by_step[step].items():
            i = seen[role]
            if i < len(expected[role]):
                require(token == expected[role][i], "Sample differs from accepted content")
                charge += 1
                wrote |= role == "writer"
            seen[role] += 1
        if charge:
            ends.append(ends[-1] + charge)
            if wrote:
                writer_at.append(ends[-1])
    require(ends[-1] == len(thinker) + len(writer), "Sample log omitted accepted content")
    return ends, writer_at


def as_case(reference, writer, ends, writer_at, decodes, forwards):
    return Case(
        reference.gold,
        reference.ordinary,
        writer,
        ends,
        writer_at,
        reference.ordinary_decode,
        reference.ordinary_forwards,
        decodes,
        forwards,
        {},
    )


def ordinary_case(reference):
    n = len(reference.ordinary)
    return as_case(
        reference,
        reference.ordinary,
        list(range(n + 1)),
        list(range(1, n + 1)),
        reference.ordinary_decode,
        reference.ordinary_forwards,
    )


def cap_stats(cases, tokenizer, cap, *, full_ordinary_grades=None):
    stats = score(cases, tokenizer, cap)
    writer_lengths = [bisect_right(c.writer_at, c.native_spend(cap)) for c in cases]
    writers = sum(writer_lengths)
    result = {
        "correct": stats["native_correct"],
        "content": stats["native_content_tokens"],
        "writer_content": writers,
        "thinker_content": stats["native_content_tokens"] - writers,
    }
    if full_ordinary_grades is None:
        result.update(
            only_this_arm_vs_capped_ordinary=stats["native_only"],
            only_capped_ordinary=stats["ordinary_only"],
        )
    else:
        gained = lost = 0
        for c, end, baseline in zip(cases, writer_lengths, full_ordinary_grades, strict=True):
            correct = last_box(tokenizer.decode(c.writer[:end], skip_special_tokens=True)) == c.gold
            gained += correct and not baseline
            lost += baseline and not correct
        result.update(only_this_arm_vs_full_ordinary=gained, only_full_ordinary=lost)
    return result


def summarize(arms, tokenizer, caps=DEFAULT_CAPS):
    ordinary = arms[ORDINARY]
    require(bool(ordinary) and all(cap > 0 for cap in caps), "Need cases and positive caps")
    target = sum(len(c.ordinary) for c in ordinary)
    ordinary_exact = [
        last_box(tokenizer.decode(c.ordinary, skip_special_tokens=True)) == c.gold for c in ordinary
    ]
    curve = {cap: {} for cap in caps}
    equal, full = {}, {}
    for arm, cases in arms.items():
        require(
            len(cases) == len(ordinary)
            and all(
                c.gold == base.gold and c.ordinary == base.ordinary
                for c, base in zip(cases, ordinary, strict=True)
            ),
            "Arms do not contain the same ordered reference cases",
        )
        natural = sum(c.batch_ends[-1] for c in cases)
        full_cap = max(c.batch_ends[-1] for c in cases)
        full[arm] = {
            **cap_stats(cases, tokenizer, full_cap, full_ordinary_grades=ordinary_exact),
            "natural_full_cap": full_cap,
            "recorded_decode_batches": sum(c.native_decode for c in cases),
            "recorded_forward_batches": sum(c.native_forwards for c in cases),
        }
        for cap in caps:
            curve[cap][arm] = cap_stats(cases, tokenizer, cap)
        cap = full_cap if natural < target else match_spend(cases)
        stats = cap_stats(cases, tokenizer, cap, full_ordinary_grades=ordinary_exact)
        residual = stats["content"] - target
        equal[arm] = {
            **stats,
            "uniform_cap": cap,
            "residual_from_target": residual,
            "status": "below_target_at_natural_completion"
            if natural < target
            else ("exact" if residual == 0 else "closest_uniform_cap"),
        }
    return {
        "coverage": len(ordinary),
        "ordinary_content_target": target,
        "full_ordinary_correct": sum(ordinary_exact),
        "full_observed": full,
        "common_caps": curve,
        "equal_aggregate_content": equal,
        "units": "Accepted thinker and writer content only. EOS, unaccepted terminal samples, prompts, "
        "probe inputs and flushes are excluded. A joint batch is indivisible. Prefixes are read "
        "from observed trajectories with no new decoding or extrapolation. Recorded forward "
        "batches may contain internally chunked prefills and are distinct from FLOPs and latency.",
    }


def compare(
    eval_dirs,
    native_eval,
    ordinary_results,
    archive_comparison,
    tokenizer,
    *,
    expected=64,
    partial=False,
    caps=DEFAULT_CAPS,
):
    from .math500_tulu_paired import verify_saved

    summary = json.loads((archive_comparison / "full_comparison.json").read_bytes())
    raw_pairs = checked_bytes(
        archive_comparison / "paired_cases.jsonl", summary["paired_cases_sha256"]
    )
    pairs = [json.loads(line) for line in raw_pairs.splitlines()]
    by_rank = {p["rank"]: p for p in pairs}
    require(len(by_rank) == len(pairs) == 500 and 0 < expected <= 500, "Invalid frozen rank cohort")
    paths = {arm: {} for arm in NEW_ARMS}
    source = {}
    manifest_hashes = []
    for directory in eval_dirs:
        manifest = json.loads((directory / "manifest.json").read_bytes())
        meta_hash = digest(encode_json(manifest))
        require(
            not manifest["smoke_only"]
            and manifest["model_revision"] == summary["revision"]
            and manifest["dataset_sha256"] == summary["dataset_sha256"]
            and manifest["budget"] == 16384
            and manifest["serial_draft_cap"] == 8192
            and manifest["sampling"] == "greedy"
            and manifest["dtype"] == "bfloat16"
            and manifest["llama_gate"] == "evidence-turns"
            and manifest["probe_include_pending"],
            "New-arm manifest differs from the matched measurement",
        )
        if source:
            require(source == manifest["source_sha256"], "New-arm source hashes differ")
        source = manifest["source_sha256"]
        manifest_hashes.append(meta_hash)
        for path in sorted((directory / "rows").glob("*.json")):
            rank_text, arm = path.stem.split(".", 1)
            rank = int(rank_text)
            if rank >= expected:
                continue
            require(
                arm in paths
                and path.stem == f"{rank:03}.{arm}"
                and 0 <= rank
                and manifest["ordered_ids"][rank] == by_rank[rank]["unique_id"],
                "Unexpected row or rank",
            )
            entry = (directory, manifest, meta_hash)
            if rank in paths[arm]:
                prior = paths[arm][rank][0] / "rows" / path.name
                require(path.read_bytes() == prior.read_bytes(), "Duplicate row differs")
            paths[arm][rank] = entry
    ranks = sorted(set.intersection(*(set(found) for found in paths.values())))
    require(
        ranks and ranks == list(range(len(ranks))) and (partial or len(ranks) == expected),
        "Expected a completed common prefix of ranks for all new arms",
    )
    chosen = [p for p in pairs if p["rank"] in ranks]
    verified = read_cases(
        native_eval, ordinary_results, archive_comparison, tokenizer, ranks=set(ranks)
    )
    controls = {p["rank"]: c for p, c in zip(chosen, verified, strict=True)}
    arms = {arm: [] for arm in (*NEW_ARMS, ORDINARY, NATIVE)}
    for rank in ranks:
        ref = controls[rank]
        pair = by_rank[rank]
        arms[NATIVE].append(ref)
        arms[ORDINARY].append(ordinary_case(ref))
        for arm in NEW_ARMS:
            directory, manifest, meta_hash = paths[arm][rank]
            stem = f"{rank:03}.{arm}"
            row = verify_saved(
                directory,
                stem,
                (pair["unique_id"], arm, rank),
                meta_hash,
                settings=manifest,
                tokenizer=tokenizer,
            )
            require(
                row["gold"].strip() == ref.gold
                and row["index"] == pair["index"]
                and (
                    arm != "evidence-turns" or row["prompt_ids_sha256"] == pair["prompt_ids_sha256"]
                ),
                "Arm differs from reference identity or native prompt",
            )
            zipped = checked_bytes(directory / "traces" / (stem + ".json.gz"), row["trace_sha256"])
            payload = json.loads(gzip.decompress(zipped))
            ends, writer_at = content_schedule(
                payload["samples"], row["thinker_ids"], row["writer_ids"]
            )
            if arm == "evidence-turns":
                require(
                    (ends, writer_at)
                    == replay_schedule(
                        payload["trace"], len(row["thinker_ids"]), len(row["writer_ids"])
                    ),
                    "Sample-based schedule differs from the native block witness",
                )
            arms[arm].append(
                as_case(
                    ref,
                    row["writer_ids"],
                    ends,
                    writer_at,
                    row["decode_batches"],
                    row["forward_batches"],
                )
            )
    return {
        "rank_ids": [by_rank[r]["unique_id"] for r in ranks],
        "expected": expected,
        "partial": len(ranks) != expected,
        "new_manifest_sha256": manifest_hashes,
        "archived_cases_sha256": summary["paired_cases_sha256"],
        **summarize(arms, tokenizer, caps),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--eval-dir", action="append", required=True, type=Path)
    for key in ("native-eval", "ordinary-results", "archive-comparison", "tokenizer"):
        cli.add_argument("--" + key, type=Path, required=True)
    cli.add_argument("--caps", type=int, nargs="+", default=DEFAULT_CAPS)
    cli.add_argument("--expected", type=int, default=64)
    cli.add_argument("--partial", action="store_true")
    args = cli.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = compare(
        args.eval_dir,
        args.native_eval,
        args.ordinary_results,
        args.archive_comparison,
        tok,
        expected=args.expected,
        partial=args.partial,
        caps=args.caps,
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
