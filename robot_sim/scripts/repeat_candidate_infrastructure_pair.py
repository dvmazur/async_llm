"""One explicitly labelled matched repeat after a documented Nav2 startup failure."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from robot_sim.candidate_nav.audit import audit_episode
    from robot_sim.candidate_nav.comparison import build_comparison
    from robot_sim.candidate_nav.report import build

    p = argparse.ArgumentParser()
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--final-output", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--wait-primary", action="store_true")
    args = p.parse_args()
    primary, output, final = (f.resolve() for f in (args.primary, args.output, args.final_output))
    original = json.loads((primary / "experiment.json").read_text())
    failed = json.loads((primary / "red_deadends__qwen__s43/summary.json").read_text())
    if failed["reason"] != "nav2_unavailable" or failed["agent"]["executed_goals"]:
        raise ValueError("Repeat requires the recorded pre-motion Nav2 startup failure")
    for file, digest in original["source_hashes"].items():
        if hashlib.sha256((ROOT / file).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen navigation sources changed: " + file)
    trials = [
        {**trial, "episode": trial["episode"] + "__infra_repeat1"}
        for trial in original["trials"]
        if trial["task"] == "red_deadends" and trial["seed"] == 43
    ]
    manifest = {
        "tasks": ["red_deadends"],
        "seeds": [43],
        "wall_limit": original["wall_limit"],
        "trials": trials,
        "source_hashes": original["source_hashes"],
        "design": "One supplementary repeat of BOTH arms in original order after an objective Nav2 startup failure; original 28 trials remain primary and are never replaced",
        "original_pair": str(primary),
        "trigger": "nav2_unavailable, zero executed goals",
        "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    file = output / "experiment.json"
    if file.exists() and json.loads(file.read_text()) != manifest:
        raise ValueError("Supplement manifest differs")
    file.write_text(json.dumps(manifest, indent=2))
    if args.wait_primary:
        print("WAITING for all primary trials, videos and audits", flush=True)
        while True:
            try:
                result = json.loads((primary / "comparison.json").read_text())
                if result["completed"] == result["planned"]:
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(30)
    for trial in trials:
        directory = output / trial["episode"]
        if directory.exists() and not (directory / "summary.json").exists():
            raise ValueError("Partial episode preserved: " + str(directory))
        if not directory.exists():
            print("START", trial["episode"], flush=True)
            with (output / (trial["episode"] + "_launcher.log")).open("w") as log:
                subprocess.run(
                    [
                        "bash",
                        "robot_sim/scripts/run_candidate_navigation.sh",
                        "--task",
                        str(ROOT / "robot_sim/candidate_tasks/red_deadends.json"),
                        "--output",
                        str(directory),
                        "--candidate-mode",
                        trial["mode"],
                        "--seed",
                        "43",
                        "--wall-limit",
                        str(original["wall_limit"]),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        audit = audit_episode(directory, args.server_log)
        summary = json.loads((directory / "summary.json").read_text())
        print(
            "RESULT",
            trial["episode"],
            summary["passed"],
            summary["reason"],
            "audit",
            audit["passed"],
            flush=True,
        )
        build(output)
        build_comparison(output)
    primary_result = json.loads((primary / "comparison.json").read_text())
    supplement = build_comparison(output)
    build(final, [primary, output])
    for root in (primary, output):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "robot_sim/scripts/analyze_candidate_comparison.py"),
                str(root),
            ],
            cwd=ROOT,
            check=True,
        )
    lines = [
        "# Candidate generation experiment",
        "",
        "## Primary series (all originally scheduled trials)",
        "",
        f"{primary_result['completed']}/{primary_result['planned']} completed. Original outcomes, including the Nav2 startup failure, are retained.",
        "",
        "```json",
        json.dumps(primary_result["aggregates"], indent=2),
        "```",
        "",
        f"[Primary comparison]({primary}/COMPARISON.md) · [Primary analysis]({primary}/ANALYSIS.md)",
        "",
        "## Supplementary infrastructure repeat",
        "",
        "One original Qwen run on red_deadends seed 43 had a lost planner_server/change_state service response; Nav2 never activated and zero goals executed. After observing this, ONE additional matched pair was declared and run with unchanged navigation sources, scene, seed, order and budgets. It is a supplementary check, not a replacement or an extension chosen by success. No further retries are scheduled based on the outcomes.",
        "",
        "```json",
        json.dumps(supplement["aggregates"], indent=2),
        "```",
        "",
        f"[Supplement results]({output}/COMPARISON.md)",
        "",
        "The HTML index contains all 30 recorded episodes. Use primary statistics for the predeclared 28-trial comparison; do not pool the two extra runs as if they were independently scheduled scenes. Model s/decision is response latency, not pure GPU compute time. Absent-target tests measure no false completion within budget, not proof of absence. See the primary analysis for candidate diagnostics and approximate SPL limitations.",
    ]
    (final / "FINAL.md").write_text("\n".join(lines) + "\n")
    print("COMPLETE", final, flush=True)


if __name__ == "__main__":
    main()
