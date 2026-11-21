"""Read-only live progress, including audit failures and a dead supervisor."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "robot_sim/results/candidate_navigation/direct_comparison_20261106_v2"


def read(file):
    try:
        return json.loads(file.read_text())
    except (OSError, ValueError):
        return {}


def last(file):
    try:
        with file.open("rb") as stream:
            stream.seek(max(0, file.stat().st_size - 65536))
            for line in reversed(stream.read().splitlines()):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
    except OSError:
        pass
    return {}


def show(root):
    state = read(root / "runtime_state.json")
    pid = state.get("supervisor_pid")
    try:
        alive = "queue_direct_experiment.py" in Path(f"/proc/{pid}/cmdline").read_text()
    except OSError:
        alive = False
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "|", root.name)
    print(
        f"Phase: {state.get('phase', 'not started')} | supervisor PID {pid}: {'RUNNING' if alive else 'NOT RUNNING'}"
    )
    if state.get("error"):
        print("ERROR:", state["error"])
    plan = read(root / "experiment.json")
    if not plan:
        print("Primary plan not frozen yet; waiting for GPUs or native pilot.")
        print("Pilot:", state.get("pilot", "not started"))
        return
    completed = verified = passed = active = 0
    lines = []
    for trial in plan["trials"]:
        directory = root / trial["episode"]
        result = read(directory / "summary.json")
        audit = read(directory / "artifact_audit.json")
        if result:
            completed += 1
            verified += int(bool(audit.get("passed")))
            passed += int(bool(result["passed"] and audit.get("passed")))
            outcome = "PASS" if result["passed"] else "FAIL"
            check = "OK" if audit.get("passed") else "FAILED" if audit else "pending"
            lines.append(
                f"{outcome:5} {trial['episode']}: {result['reason']}, {result['truth_path_m']:.2f} m, audit {check}"
            )
            if audit.get("failures"):
                lines.append("      " + "; ".join(audit["failures"]))
        elif directory.exists():
            active += 1
            frame = last(directory / "frames.jsonl")
            event = last(directory / "events.jsonl")
            label = "ACTIVE" if alive else "PARTIAL"
            lines.append(
                f"{label} {trial['episode']}: {frame.get('state', 'startup')}, decisions {frame.get('decisions', 0)}, path {frame.get('path_m', 0):.2f} m"
            )
            lines.append(f"      last event: {event.get('event', '-')}")
    total = len(plan["trials"])
    print(
        f"Completed: {completed}/{total} | remaining: {total - completed} (active/partial: {active}, not started: {total - completed - active})"
    )
    print(
        f"Audits passed: {verified}/{completed} | navigation success with passed audit: {passed}/{completed}"
    )
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", type=Path, default=DEFAULT)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=5)
    args = p.parse_args()
    try:
        while True:
            if args.watch and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            show(args.root.resolve())
            if not args.watch:
                break
            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
