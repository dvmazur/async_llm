"""Offline plots and descriptive paired analysis; never imported by a policy."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from robot_sim.candidate_nav.geometry import Grid, visible_unknown

    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    result = json.loads((root / "comparison.json").read_text())
    manifest = json.loads((root / "experiment.json").read_text())
    rows = result["episodes"]
    modes = ("geometric", "qwen")
    paired = Counter()
    paired_without_startup_failures = Counter()
    infrastructure = [
        r["episode"]
        for r in rows
        if r["reason"] == "nav2_unavailable" and r.get("agent", {}).get("executed_goals", 0) == 0
    ]
    ratios = []
    source_errors = []
    for row in rows:
        hashes = json.loads((root / row["episode"] / "source_hashes.json").read_text())
        for file, digest in manifest["source_hashes"].items():
            if file in hashes and hashes[file] != digest:
                source_errors.append(f"{row['episode']}: {file}")
    for task in manifest["tasks"]:
        for seed in manifest["seeds"]:
            pair = {r["mode"]: r for r in rows if r["task"] == task and r["seed"] == seed}
            if len(pair) != 2:
                continue
            g, q = pair["geometric"], pair["qwen"]
            outcome = (
                "both_pass"
                if g["passed"] and q["passed"]
                else "geometric_only"
                if g["passed"]
                else "qwen_only"
                if q["passed"]
                else "both_fail"
            )
            paired[outcome] += 1
            if not any(r["episode"] in infrastructure for r in pair.values()):
                paired_without_startup_failures[outcome] += 1
            if g["passed"] and q["passed"] and not g["target_absent"]:
                ratios.append(q["truth_path_m"] / g["truth_path_m"])
    proposal_quality = {}
    for mode in modes:
        counts = Counter()
        reasons = Counter()
        for row in rows:
            if row["mode"] != mode:
                continue
            folder = root / row["episode"]
            if mode == "qwen":
                candidates = []
                for file in (folder / "generations").glob("*/validation.json"):
                    with np.load(file.parent / "grid.npz") as data:
                        source_grid = Grid(
                            data["occupancy"],
                            float(data["resolution"]),
                            tuple(data["origin"]),
                            yaw=float(data["yaw"]),
                        )
                    for check in json.loads(file.read_text()):
                        reasons[check.get("reason", "accepted")] += 1
                        if check["accepted"]:
                            candidate = check["candidate"]
                            candidates.append(candidate)
                            cell = source_grid.cell((candidate["x"], candidate["y"]))
                            if candidate["kind"] == "frontier" and source_grid.contains(cell):
                                counts["frontier_evaluable_at_source"] += 1
                                before = visible_unknown(source_grid, cell)
                                after = candidate.get("information_m2", 0)
                                counts["frontier_below_threshold_at_source"] += int(before < 0.015)
                                counts["frontier_gain_lost_during_generation"] += int(
                                    before >= 0.015 and after < 0.015
                                )
            else:
                candidates = []
                for file in (folder / "observations").glob("*/observation.json"):
                    candidates += json.loads(file.read_text())["candidates"]
            for candidate in candidates:
                counts[candidate["kind"]] += 1
                if candidate["kind"] == "frontier" and candidate.get("information_m2", 0) < 0.015:
                    counts["frontier_below_information_threshold"] += 1
        proposal_quality[mode] = {
            "candidate_counts": dict(counts),
            "validation_reasons": dict(reasons),
        }
    analysis = {
        "completed": result["completed"],
        "planned": result["planned"],
        "paired_outcomes": dict(paired),
        "paired_outcomes_excluding_whole_pairs_with_pre_motion_nav2_failure": dict(
            paired_without_startup_failures
        ),
        "pre_motion_nav2_failures": infrastructure,
        "median_path_ratio_qwen_over_geometric_both_successful_present": float(np.median(ratios))
        if ratios
        else None,
        "source_hash_errors": source_errors,
        "pair_configuration_errors": result["pair_errors"],
        "proposal_quality": proposal_quality,
    }
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2))
    lines = [
        "# Descriptive analysis of candidate generation",
        "",
        f"Completed {result['completed']}/{result['planned']} episodes.",
        "",
        "Paired outcomes (each scene and sensor seed is one pair):",
        "",
        "```json",
        json.dumps(dict(paired), indent=2),
        "```",
        "",
        "Sensitivity check excluding BOTH original arms of pairs with a pre-motion Nav2 startup failure:",
        "```json",
        json.dumps(dict(paired_without_startup_failures), indent=2),
        "```",
        "",
        "Path ratios are computed only when BOTH methods find a present target; absent-target trials are excluded. Failed trials are retained in success and approximate SPL statistics.",
        "",
        f"Median Qwen/geometric path ratio for those pairs: {analysis['median_path_ratio_qwen_over_geometric_both_successful_present']}.",
        "",
        "Candidate diagnostic counts:",
        "",
        "```json",
        json.dumps(proposal_quality, indent=2),
        "```",
        "",
        "The information diagnostic uses the same optimistic 2.5 m ray proxy as the baseline. It is not a measure of true future visibility or semantic relevance. A Qwen point labelled frontier can still be a useful intermediate waypoint even with zero proxy gain. Counts are taken from accepted generation batches for Qwen and selector snapshots for geometric; repeated points are not unique locations.",
        "Source-time gain is recomputed at the accepted coordinate on the saved input grid. Validation-time gain uses the newer grid when proposals return. This separates points that were already low-gain in the input map from points whose gain disappeared while generation and rover movement were in progress. Small accepted map-frame corrections and bounded snapping also affect this diagnostic; it is not an independent task oracle.",
        "",
        "Model s/decision in COMPARISON.md is the sum of backend-reported latencies for completed generation and selection requests divided by decisions, not pure GPU compute time or complete client round-trip time. It excludes waiting for the backend request lock. Early selection responses omit remaining background analysis; incomplete requests at a budget boundary are not in this latency sum. Whole-episode wall/simulation time includes their effect.",
        "",
        f"Source hash discrepancies: {len(source_errors)}; pair configuration discrepancies: {len(result['pair_errors'])}.",
        "",
        "This experiment tests direct constrained pixel-coordinate proposals from Qwen, followed by bounded geometric validation. It does not test every possible language-guided semantic proposal architecture. Two seeds per scene are descriptive development evidence, not a general reliability estimate.",
        "",
        "![Comparison plots](comparison.png)",
    ]
    (root / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = manifest["tasks"]
    colors = {"geometric": "#2276aa", "qwen": "#e18235"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    x = np.arange(len(tasks))
    for mode, offset in zip(modes, (-0.18, 0.18)):
        success, spl, wall, paths = [], [], [], []
        for task in tasks:
            subset = [r for r in rows if r["task"] == task and r["mode"] == mode]
            success.append(np.mean([r["passed"] for r in subset]) if subset else np.nan)
            values = [r["approximate_spl"] for r in subset if "approximate_spl" in r]
            spl.append(np.mean(values) if values else np.nan)
            wall.append(np.mean([r["wall_seconds"] for r in subset]) if subset else np.nan)
            values = [r["truth_path_m"] for r in subset if r["passed"] and not r["target_absent"]]
            paths.append(np.mean(values) if values else np.nan)
            if not values:
                label = (
                    "pending"
                    if not subset
                    else "absent"
                    if subset[0]["target_absent"]
                    else "no success"
                )
                axes[1, 0].annotate(
                    label,
                    (len(paths) - 1 + offset, 0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=colors[mode],
                )
        for ax, values in zip(axes.flat, (success, spl, paths, wall)):
            ax.bar(x + offset, values, 0.35, color=colors[mode], label=mode)
    for ax, title in zip(
        axes.flat,
        (
            "Success fraction (all completed trials)",
            "Approximate SPL (present targets)",
            "Mean path, successful present targets ONLY (m)",
            "Mean wall time, including failures (s)",
        ),
    ):
        ax.set_title(title)
        ax.set_xticks(x, tasks, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend()
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 1].set_ylim(0, 1.05)
    fig.suptitle(
        f"Direct Qwen coordinate proposals vs geometric candidates — {result['completed']}/{result['planned']} trials"
    )
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
