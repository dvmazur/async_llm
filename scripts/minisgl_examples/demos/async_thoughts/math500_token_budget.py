"""Replay MATH-500 traces at a shared content-token budget without generating new answers."""

import argparse
import gzip
import json
from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer

from .math500 import last_box
from .math500_probe_audit import audit_probe_schedule, summarize_probe_schedules


@dataclass
class Case:
    gold: str
    ordinary: list[int]
    writer: list[int]
    batch_ends: list[int]
    writer_at: list[int]
    ordinary_decode: int
    ordinary_forwards: int
    native_decode: int
    native_forwards: int
    probe_schedule: dict

    def native_spend(self, cap):
        return self.batch_ends[bisect_right(self.batch_ends, cap) - 1]


def replay_schedule(trace, thinker_length, writer_length):
    """Charge recorded content in whole decode batches, including joint steps."""
    batch_ends = [0]
    writer_at = []
    thinker_used = writer_used = 0
    for batch in trace["batches"]:
        if batch["kind"] != "decode":
            continue
        roles = [len(request["cache_view"]) for request in batch["requests"]]
        if len(set(roles)) != len(roles) or not set(roles) <= {2, 3}:
            raise ValueError("Unexpected native decode roles")
        thinker_used += int(2 in roles and thinker_used < thinker_length)
        wrote = 3 in roles and writer_used < writer_length
        writer_used += int(wrote)
        spent = thinker_used + writer_used
        if spent > batch_ends[-1]:
            batch_ends.append(spent)
        if wrote:
            writer_at.append(spent)
    if (thinker_used, writer_used) != (thinker_length, writer_length):
        raise ValueError("The trace does not cover all recorded content")
    return batch_ends, writer_at


def checked_bytes(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Digest mismatch for {path}")
    return raw


def read_cases(eval_dir, ordinary_results, comparison, tokenizer, *, ranks=None):
    summary = json.loads((comparison / "full_comparison.json").read_bytes())
    if summary["status"] != "verified_500" or summary["matched_prompt_hashes"] != 500:
        raise ValueError("Expected a verified full matched comparison")
    records = checked_bytes(ordinary_results, summary["ordinary_full_reference"]["results_sha256"])
    ordinary = {row["unique_id"]: row for row in map(json.loads, records.splitlines())}
    pairs = checked_bytes(comparison / "paired_cases.jsonl", summary["paired_cases_sha256"])
    for pair in map(json.loads, pairs.splitlines()):
        if ranks is not None and pair["rank"] not in ranks:
            continue
        stem = f"{pair['rank']:03}.native-chat"
        native = json.loads((eval_dir / "rows" / f"{stem}.json").read_bytes())
        other = ordinary[pair["unique_id"]]
        if not (
            native["unique_id"] == pair["unique_id"]
            and native["prompt_ids_sha256"]
            == pair["prompt_ids_sha256"]
            == other["prompt_token_ids_sha256"]
            and native["gold"] == pair["gold"] == other["expected_answer"]
            and len(native["thinker_ids"]) == pair["native_thinker_tokens"]
            and len(native["writer_ids"]) == pair["native_writer_tokens"]
            and native["trace_sha256"] == pair["native_trace_sha256"]
            and native["retry_ids"] is None
        ):
            raise ValueError(f"Unmatched case {stem}")
        data = checked_bytes(eval_dir / "traces" / f"{stem}.json.gz", pair["native_trace_sha256"])
        trace = json.loads(gzip.decompress(data))["primary"]
        batch_ends, writer_at = replay_schedule(
            trace, len(native["thinker_ids"]), len(native["writer_ids"])
        )
        case = Case(
            pair["gold"].strip(),
            other["token_ids"],
            native["writer_ids"],
            batch_ends,
            writer_at,
            other["work"]["decode_requests"],
            other["forward_batches"],
            sum(batch["kind"] == "decode" for batch in trace["batches"]),
            len(trace["batches"]),
            audit_probe_schedule(trace),
        )
        if other["work"]["max_batch_size"] != 1:
            raise ValueError("Expected the single-stream ordinary baseline")
        for ids, text, correct in (
            (case.writer, native["writer_text"], pair["native_exact"]),
            (case.ordinary, other["text"], pair["ordinary_exact"]),
        ):
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            if decoded != text or (last_box(decoded) == case.gold) != correct:
                raise ValueError(f"Full text or grade mismatch for {stem}")
        yield case


def match_spend(cases):
    """Select a uniform cap by token spend alone, before looking at any prefix grade."""
    target = sum(len(c.ordinary) for c in cases)

    def spend(cap):
        return sum(c.native_spend(cap) for c in cases)

    lo, hi = 0, max(c.batch_ends[-1] for c in cases)
    while lo < hi:
        mid = (lo + hi) // 2
        if spend(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return min({lo, max(0, lo - 1)}, key=lambda cap: (abs(spend(cap) - target), cap))


def score(cases, tokenizer, cap):
    native_correct = ordinary_correct = native_tokens = ordinary_tokens = 0
    gains = losses = 0
    for c in cases:
        spent = c.native_spend(cap)
        end = bisect_right(c.writer_at, spent)
        native = last_box(tokenizer.decode(c.writer[:end], skip_special_tokens=True)) == c.gold
        ordinary = last_box(tokenizer.decode(c.ordinary[:cap], skip_special_tokens=True)) == c.gold
        native_correct += native
        ordinary_correct += ordinary
        native_tokens += spent
        ordinary_tokens += min(cap, len(c.ordinary))
        gains += native and not ordinary
        losses += ordinary and not native
    return {
        "cap": cap,
        "native_correct": native_correct,
        "ordinary_correct": ordinary_correct,
        "native_content_tokens": native_tokens,
        "ordinary_content_tokens": ordinary_tokens,
        "native_only": gains,
        "ordinary_only": losses,
    }


def inference_table(cases, full):
    table = {}
    for label, streams, decodes, forwards, correct in (
        (
            "ordinary",
            [c.ordinary for c in cases],
            [c.ordinary_decode for c in cases],
            [c.ordinary_forwards for c in cases],
            full["ordinary_correct"],
        ),
        (
            "native_chat",
            [c.writer for c in cases],
            [c.native_decode for c in cases],
            [c.native_forwards for c in cases],
            full["native_correct"],
        ),
    ):
        lengths = [len(ids) for ids in streams]
        table[label] = {
            "correct": correct,
            "problems": len(cases),
            "mean_writer_tokens": mean(lengths),
            "median_writer_tokens": median(lengths),
            "mean_decode_steps": mean(decodes),
            "mean_forward_steps": mean(forwards),
        }
    return table


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("eval-dir", "ordinary-results", "comparison", "tokenizer"):
        cli.add_argument("--" + arg, required=True, type=Path)
    cli.add_argument(
        "--caps", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096, 8192, 16384]
    )
    args = cli.parse_args()
    if any(cap <= 0 for cap in args.caps):
        cli.error("Caps must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    cases = list(read_cases(args.eval_dir, args.ordinary_results, args.comparison, tokenizer))
    if len(cases) != 500:
        raise ValueError("Expected 500 cases")
    equal_spend_cap = match_spend(cases)
    full_cap = max(max(c.batch_ends[-1], len(c.ordinary)) for c in cases)
    caps = sorted({*args.caps, equal_spend_cap, full_cap})
    curve = [score(cases, tokenizer, cap) for cap in caps]
    print(
        json.dumps(
            {
                "problems": len(cases),
                "equal_spend_native_cap": equal_spend_cap,
                "full_observed_cap": full_cap,
                "inference": inference_table(
                    cases, next(row for row in curve if row["cap"] == full_cap)
                ),
                "curve": curve,
                "probe_schedule": summarize_probe_schedules(c.probe_schedule for c in cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
