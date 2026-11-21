"""Paired, evaluator-only analysis of candidate generation experiments."""

import json
import math
from collections import Counter

import numpy as np

from .geometry import Grid, reachable


def reference_distance(scenario):
    """Approximate offline shortest path to the evaluator's 1.15 m success zone.

    Full world geometry is used ONLY by this evaluator/report, never by either
    navigation policy. This reference is a 10 cm 2D grid, not an exact optimum.
    """
    xmin, xmax, ymin, ymax = scenario["bounds"]
    resolution = 0.1
    width, height = math.ceil((xmax - xmin) / resolution), math.ceil((ymax - ymin) / resolution)
    values = np.zeros((height, width), np.int16)
    yy, xx = np.indices(values.shape)
    x, y = xmin + (xx + 0.5) * resolution, ymin + (yy + 0.5) * resolution
    values[(x < xmin + 0.075) | (x > xmax - 0.075) | (y < ymin + 0.075) | (y > ymax - 0.075)] = 100
    objects = [(ox, oy, 0.4, 0.4) for _, ox, oy in scenario["distractors"]]
    if not scenario.get("absent"):
        objects.append((*scenario["target_xy"], 0.4, 0.4))
    for ox, oy, sx, sy in list(scenario["partitions"]) + objects:
        values[(abs(x - ox) <= sx / 2) & (abs(y - oy) <= sy / 2)] = 100
    reach = reachable(Grid(values, resolution, (xmin, ymin)), scenario["spawn"], max_distance=1000)
    tx, ty = scenario["target_xy"]
    goal = np.hypot(x - tx, y - ty) < 1.15
    return float(reach.distance[goal].min())


def build_comparison(root):
    manifest = json.loads((root / "experiment.json").read_text())
    rows = []
    for trial in manifest["trials"]:
        directory = root / trial["episode"]
        if not (directory / "summary.json").exists():
            continue
        summary = json.loads((directory / "summary.json").read_text())
        config = json.loads((directory / "run_config.json").read_text())
        audit_file = directory / "artifact_audit.json"
        audit = json.loads(audit_file.read_text()) if audit_file.exists() else {"passed": False}
        agent = summary.get("agent", {})
        validations = []
        for file in (directory / "generations").glob("*/validation.json"):
            validations += json.loads(file.read_text())
        row = {
            **trial,
            **summary,
            "audit_passed": audit["passed"],
            "config": config,
            "proposal_rejections": dict(Counter(v.get("reason", "accepted") for v in validations)),
            "mean_completed_model_seconds_per_decision": (
                agent.get("generation_seconds", 0) + agent.get("selection_seconds", 0)
            )
            / max(1, agent.get("decisions", 0)),
        }
        if not summary.get("target_absent"):
            from .scenes import SCENARIOS

            reference = reference_distance(SCENARIOS[trial["task"]])
            row["reference_shortest_m"] = reference
            row["approximate_spl"] = (
                float(summary["passed"]) * reference / max(reference, summary["truth_path_m"])
            )
        rows.append(row)
    # Verify the actual pair inputs, not just the runner's declared arguments.
    pair_errors = []
    for task in manifest["tasks"]:
        for seed in manifest["seeds"]:
            pair = [r for r in rows if r["task"] == task and r["seed"] == seed]
            if len(pair) == 2:
                for key in (
                    "world_sha256",
                    "task_sha256",
                    "sim_limit",
                    "wall_limit",
                    "sensor_seed",
                ):
                    if pair[0]["config"][key] != pair[1]["config"][key]:
                        pair_errors.append(f"{task}/{seed}: unequal {key}")
    aggregates = {}
    for mode in ("geometric", "qwen"):
        selected = [r for r in rows if r["mode"] == mode]
        spl = [r["approximate_spl"] for r in selected if "approximate_spl" in r]
        aggregates[mode] = {
            "completed": len(selected),
            "passed": sum(r["passed"] for r in selected),
            "contacts": sum(r["contacts"] for r in selected),
            "audit_passed": sum(r["audit_passed"] for r in selected),
            "mean_approximate_spl_present_targets": float(np.mean(spl)) if spl else None,
            "total_wall_seconds": sum(r["wall_seconds"] for r in selected),
            "proposal_raw": sum(r.get("agent", {}).get("raw_proposals", 0) for r in selected),
            "proposal_accepted": sum(
                r.get("agent", {}).get("accepted_proposals", 0) for r in selected
            ),
        }
    result = {
        "completed": len(rows),
        "planned": len(manifest["trials"]),
        "pair_errors": pair_errors,
        "aggregates": aggregates,
        "episodes": rows,
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Candidate generation: paired experiment",
        "",
        f"Completed {len(rows)}/{len(manifest['trials'])}; pair configuration errors: {len(pair_errors)}.",
        "",
        "Both arms use the same Qwen selector, sensors, online mapping and Nav2. Qwen generation adds inference cost. No geometric candidate fallback is used in the Qwen arm.",
        "",
        "| Task | Seed | Mode | Pass | Reason | Path m | Sim s | Wall s | Decisions | Proposal accepted/raw | Model s/decision | Audit |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["task"], r["seed"], r["mode"])):
        a = r.get("agent", {})
        lines.append(
            f"| [{r['task']}]({r['episode']}/replay.mp4) | {r['seed']} | {r['mode']} | {r['passed']} | {r['reason']} | {r['truth_path_m']:.2f} | {a.get('sim_seconds', 0):.1f} | {r['wall_seconds']:.1f} | {a.get('decisions', 0)} | {a.get('accepted_proposals', 0)}/{a.get('raw_proposals', 0)} | {r['mean_completed_model_seconds_per_decision']:.2f} | {r['audit_passed']} |"
        )
    lines += [
        "",
        "Approximate SPL uses an evaluator-only full-world 10 cm grid and the 1.15 m success zone; it is not an exact continuous optimum. Absent-target tests are excluded from SPL. All outcomes, including failures, remain in the table. A fixed seed does not guarantee bitwise deterministic asynchronous Gazebo/ROS execution.",
        "",
        "Aggregate results:",
        "",
        "```json",
        json.dumps(aggregates, indent=2),
        "```",
        "",
        "Two sensor seeds per scene are a development comparison, not evidence of broad statistical reliability.",
    ]
    (root / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    return result
