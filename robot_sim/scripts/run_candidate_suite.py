"""Run all fixed search scenarios without selecting only successful episodes."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    p.add_argument("--wall-limit", type=float, default=3600)
    p.add_argument("--server-log", type=Path, help="Optional complete AsyncLLM inference audit log")
    p.add_argument(
        "--tasks", nargs="+", default=["red_near", "blue_occluded", "green_long", "absent_yellow"]
    )
    args = p.parse_args()
    output = (
        args.output
        or ROOT / "robot_sim/results/candidate_navigation" / f"suite_{datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name in args.tasks:
        task = ROOT / f"robot_sim/candidate_tasks/{name}.json"
        if not task.exists():
            raise ValueError(f"Unknown task {name}")
        destination = output / name
        if destination.exists():
            raise ValueError("Refusing to overwrite an episode: " + str(destination))
        with (output / (name + "_launcher.log")).open("w") as log:
            result = subprocess.run(
                [
                    "bash",
                    "robot_sim/scripts/run_candidate_navigation.sh",
                    "--task",
                    str(task),
                    "--output",
                    str(destination),
                    "--wall-limit",
                    str(args.wall_limit),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        summary_file = destination / "summary.json"
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        passed = result.returncode == 0 and summary.get("passed", False)
        if (destination / "video_provenance.json").exists():
            from robot_sim.candidate_nav.audit import audit_episode

            audit = audit_episode(destination, args.server_log)
            passed = passed and audit["passed"]
        failed += int(not passed)
        print(
            name,
            "PASS" if passed else "NOT COMPLETED",
            "exit",
            result.returncode,
            "output",
            destination,
            flush=True,
        )
        from robot_sim.candidate_nav.report import build

        build(output)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
