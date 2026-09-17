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


def process_running(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def evolution_worker(root, gpu, adoptions):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HOME="/mnt/LLM",
               SEA_LLM_PORT=str(2490+int(gpu)), SEA_GAME_TICRATE="35", SEA_INFERENCE_ACTION="wait")
    jobs = [(task, run) for run in range(10) for task in ("doom", "health_gathering")]
    # Finish the already-running process on this GPU before claiming new work.
    jobs.sort(key=lambda job: 0 if adoptions.get(f"{job[0]}/{job[1]}", {}).get("gpu") == gpu else 1)
    for task, run in jobs:
        if STOP.is_set(): return
        adoption = adoptions.get(f"{task}/{run}")
        if adoption and adoption["gpu"] != gpu:
            continue
        out = root / "minimal" / task / f"run{run:02}"
        out.mkdir(parents=True, exist_ok=True)
        with file_lock(out / "worker.lock", blocking=False) as claimed:
            if not claimed:
                continue
            complete = out / "completion.json"
            if complete.exists() and json.loads(complete.read_text())["complete"]:
                continue
            if adoption:
                print(f"ADOPT minimal {task} run={run+1} GPU={gpu} PID={adoption['pid']}", flush=True)
                while process_running(adoption["pid"]):
                    if STOP.wait(5):
                        return
            else:
                if (out / "round_metrics.csv").exists():
                    STOP.set()
                    raise RuntimeError(f"Incomplete evolution requires inspection: {out}")
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
                 "--budgets", *map(str, budgets), "--allowed-gpus", gpu, "--port", str(2490+int(gpu))],
                env, out / "process.log")
        if any(r["completed_runs"] != 10 or r["errors"] for r in report(out)):
            raise RuntimeError(f"Incomplete baseline: {out}")


def main(root, gpus, adoptions, baselines_only=False):
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / "campaign.lock", blocking=False) as owner:
        if not owner:
            raise RuntimeError("Campaign already running")
        state = dict(pid=os.getpid(), status="running", phase="baselines" if baselines_only else "minimal", gpus=[int(g) for g in gpus],
                     runs=10, episodes=5, valid_rounds=5, ticrate=35, action_policy="wait")
        atomic_json(root / "status.json", state)
        try:
            if not baselines_only:
                with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                    futures = [pool.submit(evolution_worker, root, gpu, adoptions) for gpu in gpus]
                    while not all(f.done() for f in futures):
                        for f in futures:
                            if f.done(): f.result()
                        evolution_report(root)
                        state["updated"] = time.time(); atomic_json(root / "status.json", state)
                        time.sleep(30)
                    for f in futures: f.result()
                evolution_report(root)
            state.update(phase="baselines", updated=time.time()); atomic_json(root / "status.json", state)
            from action_efficiency import report as efficiency_report
            plans = [[] for _ in gpus]
            for index, plan in enumerate([("reasoning", [16384]), ("no_think", [64,128,512,1024]), ("logit", [0])]):
                plans[index % len(gpus)].append(plan)
            with ThreadPoolExecutor(max_workers=len(gpus)+1) as pool:
                futures = [pool.submit(baseline_worker, root, gpu, plan) for gpu, plan in zip(gpus, plans) if plan]
                futures.append(pool.submit(command, [sys.executable, str(HERE / "run_random_baseline.py"),
                                       "--output", str(root / "random")],
                                       dict(os.environ, CUDA_VISIBLE_DEVICES="", SEA_GAME_TICRATE="35", SEA_INFERENCE_ACTION="wait"),
                                       root / "random/process.log"))
                while not all(f.done() for f in futures):
                    for f in futures:
                        if f.done(): f.result()
                    efficiency_report(root)
                    state["updated"] = time.time(); atomic_json(root / "status.json", state)
                    time.sleep(30)
                for f in futures: f.result()
            efficiency_report(root)
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
    p.add_argument("--gpus", nargs="+", default=["5", "6"])
    p.add_argument("--adopt", type=Path, help="Explicit task/run -> live PID and GPU mapping")
    p.add_argument("--baselines-only", action="store_true", help="Skip evolution and run the agreed seven baseline conditions")
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        p.error("GPU IDs must be unique")
    adoptions = json.loads(args.adopt.read_text()) if args.adopt else {}
    if any(a["gpu"] not in args.gpus for a in adoptions.values()):
        p.error("Every adopted worker must use an authorized GPU")
    if args.baselines_only and adoptions:
        p.error("Cannot adopt evolution workers in baselines-only mode")
    main(args.output.resolve(), args.gpus, adoptions, args.baselines_only)
