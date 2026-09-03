#!/usr/bin/env python3
"""Offline cache-policy simulation for a representative-layer compose trace."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--budget-mib",
        type=float,
        action="append",
        default=[],
        help="additional global byte budget to simulate (repeatable)",
    )
    return parser.parse_args()


def _prefixes(chain):
    chain = tuple(tuple(node) for node in chain)
    return [chain[:depth] for depth in range(1, len(chain) + 1)]


def simulate(events, *, capacity, policy, skip_current_write):
    cache = OrderedDict()
    ghost = OrderedDict()
    frequency = defaultdict(int)
    clock = 0
    baseline_nodes = computed_nodes = 0
    baseline_kernel_rounds = cached_kernel_rounds = 0
    worker_chains = terminal_hits = 0
    skipped_worker_blocks = total_worker_blocks = 0
    admissions = evictions = lookups = hits = 0
    phases = defaultdict(lambda: {"baseline_nodes": 0, "computed_nodes": 0})

    for event in events:
        write_ids = set(event["write_block_ids"])
        event_baseline = set()
        event_computed = set()
        event_admission = set()
        event_baseline_rounds = 0
        event_cached_rounds = 0

        for raw_chain in event["chains"]:
            prefixes = _prefixes(raw_chain)
            event_baseline.update(prefixes)
            worker_chains += 1
            total_worker_blocks += len(prefixes)
            deepest_hit = 0

            # Production lookup starts from the most valuable (deepest) prefix
            # and stops at the first hit.
            for key in reversed(prefixes):
                if len(key) < 2:
                    continue
                lookups += 1
                frequency[key] += 1
                if key in cache:
                    hits += 1
                    deepest_hit = len(key)
                    cache.move_to_end(key)
                    break

            if deepest_hit == len(prefixes) and prefixes:
                terminal_hits += 1
            skipped_worker_blocks += deepest_hit
            event_baseline_rounds = max(event_baseline_rounds, max(0, len(prefixes) - 1))
            if deepest_hit == 0:
                event_cached_rounds = max(event_cached_rounds, max(0, len(prefixes) - 1))
            else:
                event_cached_rounds = max(event_cached_rounds, len(prefixes) - deepest_hit)

            for key in prefixes[deepest_hit:]:
                event_computed.add(key)
                event_admission.add(key)

        baseline_nodes += len(event_baseline)
        computed_nodes += len(event_computed)
        baseline_kernel_rounds += event_baseline_rounds
        cached_kernel_rounds += event_cached_rounds
        phases[event["phase"]]["baseline_nodes"] += len(event_baseline)
        phases[event["phase"]]["computed_nodes"] += len(event_computed)

        # Results become visible only after the whole batched compose call.
        # Shallow-to-deep admission matches their dependency order.
        for key in sorted(event_admission, key=lambda item: (len(item), item)):
            if len(key) < 2:
                continue
            if skip_current_write and key[-1][0] in write_ids:
                continue
            admit = True
            if policy == "two_hit_lru":
                if key not in ghost:
                    ghost[key] = None
                    admit = False
                    if len(ghost) > max(16, 4 * capacity):
                        ghost.popitem(last=False)
                else:
                    ghost.pop(key, None)
            if not admit or capacity == 0:
                continue
            if key in cache:
                cache.move_to_end(key)
                continue
            if len(cache) >= capacity:
                if policy == "lfu":
                    victim = min(cache, key=lambda item: (frequency[item], cache[item]))
                    del cache[victim]
                else:
                    cache.popitem(last=False)
                evictions += 1
            clock += 1
            cache[key] = clock
            admissions += 1

    saved_nodes = baseline_nodes - computed_nodes
    saved_rounds = baseline_kernel_rounds - cached_kernel_rounds
    return {
        "policy": policy,
        "skip_current_write": skip_current_write,
        "capacity_prefixes_per_layer": capacity,
        "baseline_unique_prefix_nodes": baseline_nodes,
        "computed_unique_prefix_nodes": computed_nodes,
        "saved_unique_prefix_nodes_percent": 100.0 * saved_nodes / baseline_nodes,
        "baseline_estimated_kernel_rounds": baseline_kernel_rounds,
        "cached_estimated_kernel_rounds": cached_kernel_rounds,
        "saved_estimated_kernel_rounds_percent": 100.0 * saved_rounds / baseline_kernel_rounds,
        "worker_chains": worker_chains,
        "terminal_hits": terminal_hits,
        "terminal_hit_percent": 100.0 * terminal_hits / worker_chains,
        "mean_skipped_prefix_blocks_per_worker": skipped_worker_blocks / worker_chains,
        "mean_effective_blocks_per_worker": total_worker_blocks / worker_chains,
        "lookups": lookups,
        "lookup_hits": hits,
        "admissions": admissions,
        "evictions": evictions,
        "final_entries": len(cache),
        "phase_saved_nodes_percent": {
            phase: 100.0
            * (values["baseline_nodes"] - values["computed_nodes"])
            / values["baseline_nodes"]
            for phase, values in phases.items()
        },
    }


def _expand_across_layers(events, layers):
    """Replay representative topology in actual model layer-access order."""

    expanded = []
    for event in events:
        for layer in range(layers):
            copy = dict(event)
            # The layer tag makes otherwise identical prefix keys independent,
            # while block_id remains in slot zero for the write-target filter.
            copy["chains"] = [
                [[node[0], node[1], layer] for node in chain] for chain in event["chains"]
            ]
            expanded.append(copy)
    return expanded


def main():
    args = _parse_args()
    trace = json.loads(args.trace.read_text())
    events = trace["events"]
    state_bytes = trace["state_bytes_per_cached_prefix"]
    layers = trace["num_gdn_layers"]
    capacities = (1, 2, 4, 8, 16, 32, 64)
    rows = []
    for skip_write in (False, True):
        for policy in ("lru", "two_hit_lru", "lfu"):
            for capacity in capacities:
                row = simulate(
                    events,
                    capacity=capacity,
                    policy=policy,
                    skip_current_write=skip_write,
                )
                row["total_cache_mib"] = capacity * state_bytes * layers / 2**20
                rows.append(row)

    expanded = _expand_across_layers(events, layers)
    unlimited_capacity = sum(sum(len(chain) for chain in event["chains"]) for event in expanded)
    unlimited_first_touch = simulate(
        expanded,
        capacity=unlimited_capacity,
        policy="lru",
        skip_current_write=True,
    )
    unlimited_two_hit = simulate(
        expanded,
        capacity=unlimited_capacity,
        policy="two_hit_lru",
        skip_current_write=True,
    )
    global_rows = []
    budgets_mib = sorted({64.0, 128.0, 256.0, 512.0, 1024.0, *args.budget_mib})
    for budget_mib in budgets_mib:
        capacity = max(1, budget_mib * 2**20 // state_bytes)
        for policy in ("lru", "two_hit_lru"):
            row = simulate(
                expanded,
                capacity=capacity,
                policy=policy,
                skip_current_write=True,
            )
            row["total_cache_mib"] = budget_mib
            same_policy_unlimited = (
                unlimited_two_hit if policy == "two_hit_lru" else unlimited_first_touch
            )
            row["fraction_of_same_policy_unlimited_round_savings_percent"] = (
                100.0
                * row["saved_estimated_kernel_rounds_percent"]
                / same_policy_unlimited["saved_estimated_kernel_rounds_percent"]
            )
            row["fraction_of_first_touch_unlimited_round_savings_percent"] = (
                100.0
                * row["saved_estimated_kernel_rounds_percent"]
                / unlimited_first_touch["saved_estimated_kernel_rounds_percent"]
            )
            global_rows.append(row)

    output = {
        "schema_version": 1,
        "trace": str(args.trace.resolve()),
        "events": len(events),
        "state_mib_per_prefix_per_layer": state_bytes / 2**20,
        "num_gdn_layers": layers,
        "per_layer_capacity_simulations": rows,
        "global_budget_simulations": global_rows,
        "unlimited_first_touch_oracle": unlimited_first_touch,
        "unlimited_two_hit": unlimited_two_hit,
    }
    output_path = args.output or args.trace.with_name("compose_cache_policy_simulation.json")
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(output_path)
    for row in rows:
        if row["skip_current_write"] and row["capacity_prefixes_per_layer"] in (1, 2, 4, 8):
            print(
                f"{row['policy']:12s} {row['total_cache_mib']:5.0f} MiB "
                f"nodes={row['saved_unique_prefix_nodes_percent']:5.1f}% "
                f"rounds={row['saved_estimated_kernel_rounds_percent']:5.1f}% "
                f"terminal={row['terminal_hit_percent']:5.1f}%"
            )
    print("GLOBAL BUDGETS")
    for row in global_rows:
        print(
            f"{row['policy']:12s} {row['total_cache_mib']:7.1f} MiB "
            f"rounds={row['saved_estimated_kernel_rounds_percent']:5.1f}% "
            f"of_unlimited={row['fraction_of_same_policy_unlimited_round_savings_percent']:5.1f}% "
            f"of_oracle={row['fraction_of_first_touch_unlimited_round_savings_percent']:5.1f}%"
        )


if __name__ == "__main__":
    main()
