"""Primary path: evolve engine.py for many steps within a single persistent
process. The live Engine instance (cache blocks, memory, anything else it
holds) survives every self-rewrite via reload_engine_methods()'s in-place
method patching -- see self_edit_env.py. Optionally scores the agent's
act() against a plug-and-play task env every `task_every` steps (see
tasks/)."""
import os
os.environ["HF_HOME"] = "/mnt/LLM"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import asyncio
import csv
import faulthandler
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

faulthandler.enable()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import transformers

import minisgl.llm
from agent import SelfEvolvingAgent
from self_edit_env import SelfEditEnv
from tasks.doom_env import DoomEnv

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
MUTABLE = HERE / "mutable"

# One directory per run, timestamped, holding several purpose-specific log
# files instead of one giant interleaved stream -- makes it possible to
# `tail -f task_results.log` for scores or `grep ERROR main.log` for
# problems without wading through thousands of streamed thought tokens.
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
LOGS_DIR = HERE / "logs" / RUN_ID
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOGS_DIR / "main.log"), logging.StreamHandler(sys.stdout)],
)

# Plain append-only files for the three hook streams below (not routed
# through `logging` -- these are structured, one-record-per-line dumps for
# grepping/tailing, not diagnostic messages). Line-buffered + flush on every
# write so `tail -f` sees each record immediately, same spirit as the
# existing `print(..., flush=True)` calls.
_thoughts_f = open(LOGS_DIR / "thoughts.log", "a", buffering=1)
_tool_results_f = open(LOGS_DIR / "tool_results.log", "a", buffering=1)
_task_results_f = open(LOGS_DIR / "task_results.log", "a", buffering=1)

print(f"[run {RUN_ID}] logging to {LOGS_DIR}", flush=True)

# Whole-process watchdog: if nothing "progresses" (a thought token, a tool
# result, a task result, an episode start) for HANG_EXIT_TIMEOUT seconds, we
# assume a low-level stall (e.g. a stuck CUDA kernel inside the async cache
# engine) with no Python exception to catch, and hard-exit the process after
# dumping every thread's stack. We deliberately do NOT try to cancel the
# hung await/future instead: AsyncCacheEngine's queue design assumes a
# queued request's future is never cancelled out from under it (free_block()
# refuses to free a block "referenced by a queued request", and the
# success-path result-setting loops don't guard against an already-cancelled
# future) -- so a partial, in-process cancellation could corrupt the engine's
# internal state or leave other in-flight requests hanging forever instead.
# A full process exit is safe regardless of that internal state.
HANG_EXIT_TIMEOUT = 600.0
HANG_CHECK_EVERY = 30.0
_last_progress = time.monotonic()


def _touch_progress() -> None:
    global _last_progress
    _last_progress = time.monotonic()


async def _hang_watchdog() -> None:
    while True:
        await asyncio.sleep(HANG_CHECK_EVERY)
        silent_for = time.monotonic() - _last_progress
        if silent_for > HANG_EXIT_TIMEOUT:
            logging.critical(
                "no progress for %.0fs (limit %.0fs) -- assuming a stuck low-level "
                "call with no Python exception to catch; dumping stacks and exiting",
                silent_for, HANG_EXIT_TIMEOUT)
            faulthandler.dump_traceback()
            os._exit(1)


_step = 0
_metrics: list[dict] = []
_current_score: Optional[float] = None


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def on_thought_start() -> None:
    global _step
    _step += 1
    _touch_progress()
    print(f"[thought] ---- start step {_step} ----", flush=True)


def on_thought_token(token: str) -> None:
    _touch_progress()
    print(token, end="", flush=True)


def on_completion(completion: str) -> None:
    _thoughts_f.write(f"===== step {_step} @ {_ts()} ({len(completion)} chars) =====\n")
    _thoughts_f.write(completion)
    _thoughts_f.write("\n\n")


def on_tool_result(r: dict) -> None:
    _touch_progress()
    status = "OK" if r.get("ok") else "ERROR"
    print(f"\n[tool result] {r.get('tool')}: {status}", flush=True)
    if not r.get("ok"):
        print(r.get("error"), flush=True)
    elif r.get("changed"):
        print(f"  changed: {', '.join(r['changed'])}", flush=True)
    record = {"step": _step, "ts": _ts(), "tool": r.get("tool"), "ok": r.get("ok")}
    if not r.get("ok"):
        record["error"] = r.get("error")
    if r.get("changed"):
        record["changed"] = r["changed"]
    _tool_results_f.write(json.dumps(record) + "\n")


def on_task_result(result: dict) -> None:
    _touch_progress()
    print(f"\n[task result] {result['env']}: avg_reward={result['avg_reward']:.3f} "
          f"over {len(result['episodes'])} episode(s)", flush=True)
    record = {
        "step": _step, "ts": _ts(), "env": result["env"], "avg_reward": result["avg_reward"],
        "episodes": [{"reward": e["reward"], "info": e.get("info")} for e in result["episodes"]],
    }
    _task_results_f.write(json.dumps(record) + "\n")


def on_episode_start() -> None:
    _touch_progress()


def _write_metrics_csv() -> None:
    with open(LOGS_DIR / "round_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "score", "delay_s"])
        for m in _metrics:
            w.writerow([m["round"], m["score"], m["delay_s"]])


def on_step_result(result: dict) -> None:
    global _current_score
    _touch_progress()
    if "task" in result:
        _current_score = result["task"]["avg_reward"]
    _metrics.append({"round": _step, "score": _current_score, "delay_s": result.get("round_delay_s")})
    _write_metrics_csv()


def plot_metrics(metrics: list[dict], out_dir: Path) -> None:
    """Two per-run plots: env score (avg_reward, carried forward between task
    calls) and round delay (full round wall-clock -- generation plus any
    start_task/restart_task episodes run that round -- the interactivity
    metric a person watching the run would actually feel)."""
    if not metrics:
        return
    rounds = [m["round"] for m in metrics]
    scores = [m["score"] if m["score"] is not None else float("nan") for m in metrics]
    delays_min = [m["delay_s"] / 60 if m["delay_s"] is not None else float("nan") for m in metrics]

    fig, ax = plt.subplots()
    ax.plot(rounds, scores, marker="o")
    ax.set_xlabel("Evolution round")
    ax.set_ylabel("avg_reward")
    ax.set_title("Env score per evolution round")
    fig.savefig(out_dir / "score_vs_round.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(rounds, delays_min, marker="o", color="tab:orange")
    ax.set_xlabel("Evolution round")
    ax.set_ylabel("Round delay (minutes)")
    ax.set_title("Evolution round delay (wall-clock: generation + any task episodes)")
    fig.savefig(out_dir / "delay_vs_round.png")
    plt.close(fig)


async def main(n_steps: int, max_new_tokens: int) -> None:
    llm = minisgl.llm.AsyncLLM(
        "Qwen/Qwen3.8-27B", dtype=torch.bfloat16, max_running_req=4, memory_ratio=0.9,
        # repetition_penalty addresses the root cause of a failure mode seen
        # live: with no penalty at all, the model can get stuck emitting the
        # same short <use_tool> tag dozens/hundreds of times in one
        # completion. agent.py's MAX_TOOL_CALLS_PER_ROUND/
        # MAX_CONSECUTIVE_IDENTICAL_CALLS remain as a hard backstop the agent
        # can't edit away, but this fixes the actual cause instead of just
        # capping its damage.
        generation_config=transformers.GenerationConfig(
            do_sample=True, temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.15),
        distributed_addr="tcp://127.0.0.1:2370")

    engine_path = MUTABLE / "engine.py"
    prompt_path = MUTABLE / "prompt.py"
    env = SelfEditEnv(llm, engine_path=str(engine_path), prompt_path=str(prompt_path))
    env.task_env = DoomEnv()
    agent = SelfEvolvingAgent(env, hooks=dict(
        on_thought_start=on_thought_start,
        on_thought_token=on_thought_token,
        on_completion=on_completion,
        on_tool_result=on_tool_result,
        on_task_result=on_task_result,
        on_episode_start=on_episode_start,
        on_step_result=on_step_result,
    ))

    watchdog = asyncio.create_task(_hang_watchdog())
    try:
        await agent.run(max_steps=n_steps, max_new_tokens=max_new_tokens)
    finally:
        watchdog.cancel()
        if agent._history.block is not None:
            await llm.free_block(agent._history.block)
        for f in (_thoughts_f, _tool_results_f, _task_results_f):
            f.close()
        try:
            plot_metrics(_metrics, LOGS_DIR)
        except Exception:
            logging.exception("failed to plot round metrics")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 32_000
    asyncio.run(main(n, max_new_tokens))
