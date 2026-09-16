"""Run the baseline comparison on explicitly selected GPUs with a live report."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_budget_sweep import BUDGETS, TASKS, atomic_json, file_lock, report, retry_disk_full


@retry_disk_full
def combined_report(root):
    rows = []
    episodes = 0
    for mode in ["reasoning", "no_think"]:
        out = root / mode
        if (out / "config.json").exists():
            rows.extend(report(out))
            episodes += len(list(out.glob("episode_*.json")))
    if not rows:
        return False
    complete = len(rows) == 24 and all(r["completed_runs"] == 10 and not r["errors"] for r in rows)
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"
    lines = ["# Thinking-mode and token-budget baselines", "",
             f"Status: **{'COMPLETE' if complete else 'IN PROGRESS / INCOMPLETE'}**. "
             f"{episodes}/1200 episode records saved.", "",
             "Qwen/Qwen3.8-27B BF16; one model per selected GPU. "
             "Each condition requires 10 independent runs × 5 episodes. "
             "Reward is mean of run means ± two-sided 95% Student-t CI. "
             "Only complete, error-free runs contribute to reward estimates.", "",
             "- **Direct:** closed `<think></think>`; no generated reasoning or answer; legal-action logit argmax.",
             "- **Thinking disabled:** closed `<think></think>`; ordinary answer generation up to its budget, "
             "then legal-action logit argmax. Ordinary text may still explain the choice.",
             "- **Reasoning:** open `<think>`; generate until `</think>`, EOS, or budget; "
             "then legal-action logit argmax. 16k is a maximum, not a forced minimum.", "",
             "All conditions use only the current image (plus health in Health Gathering). "
             "Step caps: Defend the Line 100, Health Gathering 2500. "
             "Seeds are paired across conditions; previous baseline results are excluded.", "",
             "| Environment | Mode | Budget | Runs | Reward ± 95% CI | Tokens/action | Seconds/action | Cap-hit fraction | Errors |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        mode = "direct" if r["budget"] == 0 else r["mode"]
        lines.append(f"| {r['task']} | {mode} | {r['budget']} | {r['completed_runs']}/10 | "
                     f"{fmt(r['mean_reward'])} ± {fmt(r['ci95_half_width'])} | "
                     f"{fmt(r['mean_generated_tokens'])} | {fmt(r['mean_act_latency_s'])} | "
                     f"{fmt(r['budget_exhaustion_rate'])} | {r['errors']} |")
    if complete:
        lines += ["", "## Best observed mean reward", ""]
        for task in TASKS:
            best = max((r for r in rows if r["task"] == task), key=lambda r: r["mean_reward"])
            lines.append(f"- {task}: {best['mode']} / {best['budget']} tokens, "
                         f"{fmt(best['mean_reward'])} ± {fmt(best['ci95_half_width'])}.")
        lines += ["", "These are rankings by observed means; selecting the highest mean does not "
                  "establish statistically significant superiority. Intervals are per-condition "
                  "intervals, without adjustment for selecting among multiple conditions."]
    tmp = root / "report.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(root / "report.md")
    if any(r["completed_runs"] > 1 for r in rows):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, task in zip(axes, TASKS):
            for mode, label in [("reasoning", "Genuine reasoning"), ("no_think", "Thinking disabled")]:
                cell = [r for r in rows if r["task"] == task and r["mode"] == mode
                        and r["budget"] > 0 and r["completed_runs"] > 1]
                if cell:
                    ax.errorbar([r["budget"] for r in cell], [r["mean_reward"] for r in cell],
                                yerr=[r["ci95_half_width"] for r in cell], marker="o", capsize=3, label=label)
            direct = next((r for r in rows if r["task"] == task and r["budget"] == 0 and r["completed_runs"] > 1), None)
            if direct:
                ax.axhline(direct["mean_reward"], color="gray", label="Direct (0 generated tokens)")
                ax.axhspan(direct["ci95_low"], direct["ci95_high"], color="gray", alpha=.15)
            ax.set_xscale("log", base=2)
            ax.set(title=task, xlabel="Maximum generated tokens", ylabel="Mean episode reward (95% CI)")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle("Complete results" if complete else "Partial results — see completed-run counts")
        fig.tight_layout()
        fig.savefig(root / "reward_ci.png", dpi=180)
        plt.close(fig)
    return complete


def worker_plans(gpus):
    if not gpus or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError("Specify distinct nonnegative GPU indices")
    return {gpu: (["reasoning", "no_think"] if i % 2 == 0 else ["no_think", "reasoning"])
            for i, gpu in enumerate(gpus)}


def main(root, gpus):
    root.mkdir(parents=True, exist_ok=True)
    # A single supervisor may own this output directory at a time.
    with file_lock(root / "campaign.lock", blocking=False) as owner:
        if not owner:
            raise RuntimeError("Campaign supervisor already running")
        script = Path(__file__).with_name("run_budget_sweep.py").resolve()
        plans = worker_plans(gpus)
        active, logs = {}, {}
        state = {"supervisor_pid": os.getpid(), "started": time.time(), "status": "running",
                 "gpus": gpus, "workers": {}}
        atomic_json(root / "status.json", state)
        try:
            while plans or active:
                for gpu in list(plans):
                    if gpu in active:
                        continue
                    if not plans[gpu]:
                        del plans[gpu]
                        continue
                    mode = plans[gpu].pop(0)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HOME="/mnt/LLM")
                    command = [sys.executable, "-u", str(script), "--output", str(root / mode),
                               "--mode", mode, "--port", str(2390 + int(gpu)), "--budgets"]
                    command += [str(b) for b in (BUDGETS if mode == "reasoning" else [64, 128, 512, 1024])]
                    command += ["--allowed-gpus", *gpus]
                    logs[gpu] = (root / f"gpu{gpu}.log").open("a")
                    process = subprocess.Popen(command, env=env, stdout=logs[gpu], stderr=subprocess.STDOUT)
                    active[gpu] = process
                    state["workers"][gpu] = {"pid": process.pid, "mode": mode, "status": "running"}
                    atomic_json(root / "status.json", state)
                for gpu, process in list(active.items()):
                    code = process.poll()
                    if code is not None:
                        logs.pop(gpu).close()
                        del active[gpu]
                        state["workers"][gpu]["status"] = f"exit {code}"
                        if code:
                            raise RuntimeError(f"GPU {gpu} worker failed with exit {code}; see gpu{gpu}.log")
                complete = combined_report(root)
                state["updated"] = time.time()
                atomic_json(root / "status.json", state)
                if plans or active:
                    time.sleep(30)
            if not complete:
                raise RuntimeError("Workers exited but not all 24 conditions have 10 valid runs")
            state["status"] = "complete"
        except BaseException as error:
            state["status"] = "failed"
            state["error"] = str(error)
            for process in active.values():
                process.terminate()
            for process in active.values():
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            state["updated"] = time.time()
            atomic_json(root / "status.json", state)
            for handle in logs.values():
                handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--gpus", nargs="+", default=["1", "2"], help="One worker per selected physical GPU")
    args = parser.parse_args()
    if args.report_only:
        combined_report(args.output.resolve())
    else:
        main(args.output.resolve(), args.gpus)
