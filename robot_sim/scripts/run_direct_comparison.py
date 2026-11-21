"""Predeclared direct-command series against the preserved candidate baseline."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def hashes():
    files = list((ROOT / "robot_sim/candidate_nav").glob("*.py"))
    files += list((ROOT / "robot_sim/mapping").glob("*.py"))
    files += list((ROOT / "robot_sim/candidate_tasks").glob("*.json"))
    files += [
        Path(__file__),
        ROOT / "robot_sim/scripts/queue_direct_experiment.py",
        ROOT / "robot_sim/DIRECT_COMMAND_EXPERIMENT.md",
        ROOT / "robot_sim/control.py",
        ROOT / "robot_sim/scripts/ros_nodes.py",
        ROOT / "robot_sim/scripts/rover_drive.py",
        ROOT / "robot_sim/config/nav2.yaml",
        ROOT / "robot_sim/scripts/run_candidate_navigation.sh",
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)
    }


def report(root, baseline):
    """Read existing outcomes, never rerun or rewrite baseline artifacts."""
    from robot_sim.candidate_nav.audit import records
    from robot_sim.candidate_nav.comparison import reference_distance
    from robot_sim.candidate_nav.report import build
    from robot_sim.candidate_nav.scenes import SCENARIOS

    manifest = json.loads((root / "experiment.json").read_text())
    previous = json.loads((baseline / "comparison.json").read_text())
    rows, errors = [], []
    for trial in manifest["trials"]:
        directory = root / trial["episode"]
        if not (directory / "summary.json").exists():
            continue
        summary = json.loads((directory / "summary.json").read_text())
        config = json.loads((directory / "run_config.json").read_text())
        source = json.loads((directory / "source_hashes.json").read_text())
        for file, digest in source.items():
            if file in manifest["source_hashes"] and digest != manifest["source_hashes"][file]:
                errors.append(f"Source mismatch: {trial['episode']} {file}")
        for other in previous["episodes"]:
            if other["task"] == trial["task"] and other["seed"] == trial["seed"]:
                for key in (
                    "world_sha256",
                    "task_sha256",
                    "sensor_seed",
                    "sim_limit",
                    "wall_limit",
                ):
                    if config[key] != other["config"][key]:
                        errors.append(f"Unequal {key}: {trial['episode']} / {other['episode']}")
        events = records(directory / "events.jsonl")
        actions = [e for e in events if e["event"] == "command_finished"]
        from collections import Counter

        row = {
            **trial,
            **summary,
            "config": config,
            "audit_passed": json.loads((directory / "artifact_audit.json").read_text()).get(
                "passed", False
            ),
            "actions": dict(Counter(a["action"] for a in actions)),
            "action_outcomes": dict(Counter(a["outcome"] for a in actions)),
        }
        if not summary["target_absent"]:
            distance = reference_distance(SCENARIOS[trial["task"]])
            row["approximate_spl"] = (
                float(summary["passed"]) * distance / max(distance, summary["truth_path_m"])
            )
        rows.append(row)
    aggregates = {}
    for mode in ("geometric", "qwen", "direct"):
        group = rows if mode == "direct" else [r for r in previous["episodes"] if r["mode"] == mode]
        spl = [r["approximate_spl"] for r in group if "approximate_spl" in r]
        aggregates[mode] = {
            "completed": len(group),
            "passed": sum(r["passed"] for r in group),
            "contacts": sum(r["contacts"] for r in group),
            "audit_passed": sum(r["audit_passed"] for r in group),
            "mean_approximate_spl_present_targets": sum(spl) / len(spl) if spl else None,
            "wall_seconds": sum(r["wall_seconds"] for r in group),
        }
    result = {
        "completed": len(rows),
        "planned": len(manifest["trials"]),
        "configuration_errors": errors,
        "aggregates": aggregates,
        "episodes": rows,
        "baseline_sha256": manifest["baseline_comparison_sha256"],
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2))
    build(root)
    lines = [
        "# Direct commands versus preserved waypoint baseline",
        "",
        f"Direct episodes completed {len(rows)}/{len(manifest['trials'])}. Configuration errors: {len(errors)}.",
        "",
        "Baseline is the original 28 trials; the supplementary infrastructure pair remains separate. These methods were run at different times, not interleaved. Same task/world hashes, seeds, simulation and wall budgets are checked.",
        "",
        "| Scene | Seed | Geometric | Qwen coordinates | Direct | Direct reason | Path m | Decisions | Audit |",
        "|---|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["task"], r["seed"])):
        pair = {
            r["mode"]: r
            for r in previous["episodes"]
            if r["task"] == row["task"] and r["seed"] == row["seed"]
        }
        lines.append(
            f"| [{row['task']}]({row['episode']}/replay.mp4) | {row['seed']} | {pair['geometric']['passed']} | {pair['qwen']['passed']} | {row['passed']} | {row['reason']} | {row['truth_path_m']:.2f} | {row.get('agent', {}).get('decisions', 0)} | {row['audit_passed']} |"
        )
    lines += [
        "",
        "```json",
        json.dumps(aggregates, indent=2),
        "```",
        "",
        "Direct mode uses bounded stop-think-act primitives; waypoint modes can keep moving during inference. Equal budgets measure the full systems, not reasoning quality at equal action counts. SPL uses the same offline approximate 10 cm grid and excludes absent targets. All failures are retained; two seeds do not establish general reliability.",
        "",
    ]
    (root / "COMPARISON.md").write_text("\n".join(lines))
    return result


def main():
    from robot_sim.candidate_nav.audit import audit_episode

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    root, baseline = args.output.resolve(), args.baseline.resolve()
    root.mkdir(parents=True, exist_ok=True)
    old_plan = json.loads((baseline / "experiment.json").read_text())
    trials = [
        {"task": task, "seed": seed, "mode": "direct", "episode": f"{task}__direct__s{seed}"}
        for seed in old_plan["seeds"]
        for task in old_plan["tasks"]
    ]
    manifest = {
        "tasks": old_plan["tasks"],
        "seeds": old_plan["seeds"],
        "trials": trials,
        "wall_limit": old_plan["wall_limit"],
        "source_hashes": hashes(),
        "baseline_comparison_sha256": hashlib.sha256(
            (baseline / "comparison.json").read_bytes()
        ).hexdigest(),
        "design": "All original task/seed pairs, same budgets; frozen direct policy; baseline retained; no outcome-driven retries.",
    }
    file = root / "experiment.json"
    if file.exists():
        if json.loads(file.read_text()) != manifest:
            raise ValueError("Frozen sources/plan/baseline changed; use a new output directory")
    elif args.report_only:
        raise ValueError("No frozen experiment to report")
    else:
        file.write_text(json.dumps(manifest, indent=2))
    if not args.report_only:
        for trial in trials:
            if hashes() != manifest["source_hashes"]:
                raise ValueError("Sources changed during frozen experiment")
            destination = root / trial["episode"]
            if destination.exists() and not (destination / "summary.json").exists():
                raise ValueError(f"Partial episode preserved; inspect before resume: {destination}")
            if not destination.exists():
                print("START", trial["episode"], flush=True)
                with (root / (trial["episode"] + "_launcher.log")).open("w") as log:
                    process = subprocess.run(
                        [
                            "bash",
                            "robot_sim/scripts/run_candidate_navigation.sh",
                            "--control-mode",
                            "direct",
                            "--task",
                            str(ROOT / "robot_sim/candidate_tasks" / (trial["task"] + ".json")),
                            "--output",
                            str(destination),
                            "--seed",
                            str(trial["seed"]),
                            "--wall-limit",
                            str(old_plan["wall_limit"]),
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if not (destination / "summary.json").exists():
                    raise RuntimeError(
                        f"Infrastructure failure {process.returncode}: {destination}"
                    )
            if not (destination / "replay.mp4").exists():
                from robot_sim.candidate_nav.render import video

                video(destination)
            audit = audit_episode(destination, args.server_log)
            result = report(root, baseline)
            row = next(r for r in result["episodes"] if r["episode"] == trial["episode"])
            print(
                "RESULT",
                trial["episode"],
                row["passed"],
                row["reason"],
                f"{row['truth_path_m']:.2f}m",
                "audit",
                audit["passed"],
                flush=True,
            )
            if not audit["passed"] or result["configuration_errors"]:
                raise RuntimeError("Artifact audit/configuration errors; preserve and inspect")
    result = report(root, baseline)
    print(json.dumps(result["aggregates"], indent=2), flush=True)
    return int(bool(result["configuration_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
