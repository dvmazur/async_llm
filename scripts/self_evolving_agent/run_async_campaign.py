"""Real-time campaign: minimal evolution first, then boxed/logit/random baselines."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import threading

from run_budget_sweep import atomic_json, report, seed_for, file_lock

HERE = Path(__file__).resolve().parent
STOP = threading.Event()


def evolution_report(root):
    from scipy.stats import t
    rows = []
    for task in ("doom", "health_gathering"):
        for rnd in range(1, 6):
            values = []
            for run in range(10):
                path = root / "minimal" / task / f"run{run:02}" / "round_metrics.csv"
                if path.exists():
                    with path.open() as f:
                        scores = [float(r["score"]) for r in csv.DictReader(f)
                                  if r.get("valid_round") == str(rnd)]
                    if len(scores) == 1:
                        values.extend(scores)
            n = len(values)
            mean = statistics.mean(values) if n else None
            half = float(t.ppf(.975, n-1)) * statistics.stdev(values) / math.sqrt(n) if n > 1 else None
            rows.append(dict(task=task, valid_round=rnd, completed_runs=n,
                             mean_reward=mean, ci95_half_width=half))
    with (root / "evolution_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ["# Asynchronous minimal self-evolution", "", "Mean of five-episode run means ± 95% Student-t CI across independent runs.", "",
             "| Environment | Valid round | Runs | Reward ± 95% CI |", "|---|---:|---:|---:|"]
    for r in rows:
        value = "—" if r["mean_reward"] is None else f"{r['mean_reward']:.3f}"
        half = "—" if r["ci95_half_width"] is None else f"{r['ci95_half_width']:.3f}"
        lines.append(f"| {r['task']} | {r['valid_round']} | {r['completed_runs']}/10 | {value} ± {half} |")
    (root / "evolution_report.md").write_text("\n".join(lines)+"\n")
    return rows


def command(cmd, env, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        process = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if STOP.wait(1):
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise RuntimeError("Campaign stopped after another worker failed")
        if process.returncode:
            STOP.set()
            raise subprocess.CalledProcessError(process.returncode, cmd)


def evolution_worker(root, gpu, task):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HOME="/mnt/LLM",
               SEA_LLM_PORT=str(2490+int(gpu)), SEA_GAME_TICRATE="35", SEA_INFERENCE_ACTION="wait")
    for run in range(10):
        if STOP.is_set(): return
        out = root / "minimal" / task / f"run{run:02}"
        complete = out / "completion.json"
        if complete.exists() and json.loads(complete.read_text())["complete"]:
            continue
        if out.exists() and (out / "round_metrics.csv").exists():
            raise RuntimeError(f"Incomplete evolution requires inspection, not overwriting: {out}")
        mutable = out / "mutable"
        mutable.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / "seeds/engine_seed.py", mutable / "engine.py")
        shutil.copyfile(HERE / "seeds/prompt_seed_minimal.py", mutable / "prompt.py")
        env.update(SEA_MUTABLE_DIR=str(mutable), SEA_LOG_DIR=str(out), SEA_RUN_INDEX=str(run),
                   SEA_RUN_SEED=str(seed_for(20260915, task+":evolution", run, 0)))
        print(f"START minimal {task} run={run+1} GPU={gpu}", flush=True)
        command([sys.executable, str(HERE / "run_persistent.py"), "5", "32000", task, "minimal", "20"], env, out / "process.log")
        if not complete.exists() or not json.loads(complete.read_text())["complete"]:
            STOP.set()
            raise RuntimeError(f"Evolution did not reach five valid rounds: {out}")
        print(f"DONE minimal {task} run={run+1}", flush=True)


def baseline_worker(root, gpu, plans):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HOME="/mnt/LLM",
               SEA_GAME_TICRATE="35", SEA_INFERENCE_ACTION="wait")
    for mode, budgets in plans:
        out = root / mode
        command([sys.executable, str(HERE / "run_budget_sweep.py"), "--output", str(out),
                 "--mode", "no_think" if mode == "no_think" else "reasoning",
                 "--budgets", *map(str, budgets), "--allowed-gpus", "5", "6", "--port", str(2490+int(gpu))],
                env, out / "process.log")
        if any(r["completed_runs"] != 10 or r["errors"] for r in report(out)):
            raise RuntimeError(f"Incomplete baseline: {out}")


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / "campaign.lock", blocking=False) as owner:
        if not owner:
            raise RuntimeError("Campaign already running")
        state = dict(pid=os.getpid(), status="running", phase="minimal", gpus=[5,6],
                     runs=10, episodes=5, valid_rounds=5, ticrate=35, action_policy="wait")
        atomic_json(root / "status.json", state)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(evolution_worker, root, gpu, task)
                           for gpu, task in [("5", "doom"), ("6", "health_gathering")]]
                while not all(f.done() for f in futures):
                    for f in futures:
                        if f.done(): f.result()
                    evolution_report(root)
                    state["updated"] = time.time(); atomic_json(root / "status.json", state)
                    time.sleep(30)
                for f in futures: f.result()
            evolution_report(root)
            state.update(phase="baselines", updated=time.time()); atomic_json(root / "status.json", state)
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(baseline_worker, root, "5", [("reasoning", [16384]), ("logit", [0])]),
                           pool.submit(baseline_worker, root, "6", [("no_think", [64,128,512,1024])]),
                           pool.submit(command, [sys.executable, str(HERE / "run_random_baseline.py"),
                                       "--output", str(root / "random")],
                                       dict(os.environ, CUDA_VISIBLE_DEVICES="", SEA_GAME_TICRATE="35", SEA_INFERENCE_ACTION="wait"),
                                       root / "random/process.log")]
                while not all(f.done() for f in futures):
                    for f in futures:
                        if f.done(): f.result()
                    state["updated"] = time.time(); atomic_json(root / "status.json", state)
                    time.sleep(30)
                for f in futures: f.result()
            rows = [r for name in ("reasoning","logit","no_think","random") for r in report(root / name)]
            with (root / "baseline_summary.csv").open("w", newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            state.update(status="complete", phase="review_before_detailed")
        except BaseException as error:
            state.update(status="failed", error=repr(error)); raise
        finally:
            state["updated"] = time.time(); atomic_json(root / "status.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    main(args.output.resolve())
