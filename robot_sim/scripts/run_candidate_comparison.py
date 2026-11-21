"""Predeclared paired trials; resumable without overwriting or selecting outcomes."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from robot_sim.candidate_nav.audit import audit_episode
    from robot_sim.candidate_nav.comparison import build_comparison
    from robot_sim.candidate_nav.report import build

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[29, 43])
    p.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "red_near",
            "blue_occluded",
            "green_long",
            "absent_yellow",
            "red_deadends",
            "blue_clutter",
            "yellow_zigzag",
        ],
    )
    p.add_argument("--wall-limit", type=float, default=3600)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    trials = []
    for repeat, seed in enumerate(args.seeds):
        for index, task in enumerate(args.tasks):
            modes = ("qwen", "geometric") if (repeat + index) % 2 == 0 else ("geometric", "qwen")
            for mode in modes:
                trials.append(
                    {
                        "task": task,
                        "mode": mode,
                        "seed": seed,
                        "episode": f"{task}__{mode}__s{seed}",
                    }
                )
    sources = list((ROOT / "robot_sim/candidate_nav").glob("*.py")) + [Path(__file__)]
    manifest = {
        "tasks": args.tasks,
        "seeds": args.seeds,
        "wall_limit": args.wall_limit,
        "trials": trials,
        "source_hashes": {
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources
        },
        "design": "paired sensor seeds, alternating execution order, equal simulation/wall budgets, all outcomes retained",
    }
    file = root / "experiment.json"
    if file.exists():
        if json.loads(file.read_text()) != manifest:
            raise ValueError("Frozen experiment differs; choose a NEW output directory")
    else:
        file.write_text(json.dumps(manifest, indent=2))
    for trial in trials:
        destination = root / trial["episode"]
        if destination.exists():
            if (destination / "summary.json").exists():
                print("EXISTING", trial["episode"], flush=True)
                continue
            raise ValueError(
                "Partial episode retained; inspect before resuming: " + str(destination)
            )
        print("START", trial["episode"], flush=True)
        with (root / (trial["episode"] + "_launcher.log")).open("w") as log:
            process = subprocess.run(
                [
                    "bash",
                    "robot_sim/scripts/run_candidate_navigation.sh",
                    "--task",
                    str(ROOT / "robot_sim/candidate_tasks" / (trial["task"] + ".json")),
                    "--output",
                    str(destination),
                    "--candidate-mode",
                    trial["mode"],
                    "--seed",
                    str(trial["seed"]),
                    "--wall-limit",
                    str(args.wall_limit),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if not (destination / "summary.json").exists():
            raise RuntimeError(f"Infrastructure failed, exit {process.returncode}: {destination}")
        if (destination / "replay.mp4").exists():
            audit_episode(destination, args.server_log)
        summary = json.loads((destination / "summary.json").read_text())
        print(
            "RESULT",
            trial["episode"],
            summary["passed"],
            summary["reason"],
            f"{summary['truth_path_m']:.2f}m",
            flush=True,
        )
        build(root)
        build_comparison(root)
    result = build_comparison(root)
    print(json.dumps(result["aggregates"], indent=2), flush=True)
    return int(bool(result["pair_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
