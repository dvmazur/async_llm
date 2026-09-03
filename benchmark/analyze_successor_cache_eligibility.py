#!/usr/bin/env python3
"""Measure realized post-recurrent successor-cache opportunities in a topology trace."""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path


def _matches_successor(before: list[list[int]], write_id: int, after: list[list[int]]) -> bool:
    """Whether *after* differs only by applying one token to terminal *write_id*."""

    if before and before[-1][0] == write_id:
        # Existing mutable tail: upstream is byte-for-byte the same topology/version,
        # while the tail keeps its identity and receives a new affine version.
        return (
            len(after) == len(before)
            and after[:-1] == before[:-1]
            and after[-1][0] == write_id
            and after[-1][1] != before[-1][1]
        )

    # Fresh empty write block was absent from the effective affine chain.  Its first
    # captured token appends exactly one terminal node to the unchanged old chain.
    return (
        len(after) == len(before) + 1
        and after[:-1] == before
        and after[-1][0] == write_id
    )


def analyze(trace_path: Path) -> dict[str, object]:
    trace = json.loads(trace_path.read_text())
    events = trace["events"]

    # All later occurrences where a block is the effective terminal of a worker.
    occurrences: dict[int, list[tuple[int, list[list[int]]]]] = defaultdict(list)
    for event_index, event in enumerate(events):
        for chain in event["chains"]:
            if chain:
                occurrences[int(chain[-1][0])].append((event_index, chain))
    occurrence_indices = {
        block_id: [event_index for event_index, _ in rows]
        for block_id, rows in occurrences.items()
    }

    counters = defaultdict(int)
    by_phase: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for event_index, event in enumerate(events):
        next_event = events[event_index + 1] if event_index + 1 < len(events) else None
        next_chains = [] if next_event is None else next_event["chains"]
        for worker, (chain, write_id) in enumerate(
            zip(event["chains"], event["write_block_ids"])
        ):
            phase = str(event["phase"])
            counters["workers"] += 1
            by_phase[phase]["workers"] += 1
            if chain and chain[-1][0] == write_id:
                kind = "existing_terminal_tail"
            else:
                kind = "fresh_appended_tail"
            counters[kind] += 1
            by_phase[phase][kind] += 1

            adjacent_match = any(
                _matches_successor(chain, int(write_id), candidate)
                for candidate in next_chains
            )
            if adjacent_match:
                counters["adjacent_matches"] += 1
                by_phase[phase]["adjacent_matches"] += 1

            rows = occurrences.get(int(write_id), [])
            indices = occurrence_indices.get(int(write_id), [])
            position = bisect.bisect_right(indices, event_index)
            if position >= len(rows):
                counters["no_later_terminal_use"] += 1
                by_phase[phase]["no_later_terminal_use"] += 1
                continue
            counters["has_later_terminal_use"] += 1
            by_phase[phase]["has_later_terminal_use"] += 1
            _, next_chain = rows[position]
            if _matches_successor(chain, int(write_id), next_chain):
                counters["next_use_matches"] += 1
                by_phase[phase]["next_use_matches"] += 1

    def summarize(values: dict[str, int]) -> dict[str, object]:
        workers = values["workers"]
        later = values["has_later_terminal_use"]
        return {
            **dict(values),
            "adjacent_match_percent_of_workers": 100.0 * values["adjacent_matches"] / workers,
            "next_use_match_percent_of_workers": 100.0 * values["next_use_matches"] / workers,
            "next_use_match_percent_when_reused": (
                100.0 * values["next_use_matches"] / later if later else None
            ),
        }

    return {
        "schema_version": 1,
        "trace": str(trace_path),
        "definition": (
            "A match means the next effective chain is either old_chain + first(write_to), "
            "or the same chain with only terminal write_to's affine version changed; all "
            "upstream block identities and versions are unchanged."
        ),
        "overall": summarize(counters),
        "by_phase": {phase: summarize(values) for phase, values in by_phase.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.trace.with_name("successor_cache_eligibility.json")
    result = analyze(args.trace)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    for label, row in [("overall", result["overall"]), *result["by_phase"].items()]:
        print(
            f"{label:8} workers={row['workers']:5d} "
            f"adjacent={row['adjacent_match_percent_of_workers']:6.2f}% "
            f"next-use/all={row['next_use_match_percent_of_workers']:6.2f}% "
            f"next-use/reused={row['next_use_match_percent_when_reused']:6.2f}%"
        )


if __name__ == "__main__":
    main()
