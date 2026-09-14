"""Primary path: evolve engine.py for many steps within a single persistent
process. The live Engine instance (cache blocks, memory, anything else it
holds) survives every self-rewrite via reload_engine_methods()'s in-place
method patching -- see self_edit_env.py. Optionally scores the agent's
act() against a plug-and-play task env every `task_every` steps (see
tasks/)."""
import os
# setdefault so a caller can override from outside without editing this file,
# e.g. `CUDA_VISIBLE_DEVICES=2 uv run python run_persistent.py ...`.
os.environ.setdefault("HF_HOME", "/mnt/LLM")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import asyncio
import csv
import faulthandler
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

faulthandler.enable()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from PIL import Image

import minisgl.llm
from agent import SelfEvolvingAgent
from self_edit_env import SelfEditEnv
from tasks.doom_env import DoomEnv
from tasks.health_gathering_env import HealthGatheringEnv
from tasks.my_way_home_env import MyWayHomeEnv

# Plug-and-play task envs selectable via argv[3] -- each must follow the
# shared reset()/step()/restart() shape (see tasks/doom_env.py etc).
TASKS = {
    "doom": DoomEnv,
    "health_gathering": HealthGatheringEnv,
    "my_way_home": MyWayHomeEnv,
}

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
# Points at whatever reset_engine.py --mutable-dir was pointed at for this
# same slot -- lets several concurrent runs each own a private engine.py/
# prompt.py instead of racing over one shared mutable/ dir.
MUTABLE = HERE / os.environ.get("SEA_MUTABLE_DIR", "mutable")

TASK_NAME = sys.argv[3] if len(sys.argv) > 3 else "doom"
if TASK_NAME not in TASKS:
    raise SystemExit(f"unknown task {TASK_NAME!r}; expected one of {list(TASKS)}")
# Purely a label for logging/naming -- reset_engine.py is what actually
# installs a prompt variant into mutable/prompt.py before this process starts.
PROMPT_VARIANT = sys.argv[4] if len(sys.argv) > 4 else "detailed"

# One directory per run, timestamped and task/variant-prefixed, with separate
# log files per stream so e.g. `tail -f task_results.log` isn't buried in
# thought tokens. SEA_MUTABLE_DIR is included so two concurrent grid slots
# starting the same (task, variant) cell in the same second don't collide.
RUN_ID = (f"{TASK_NAME}_{PROMPT_VARIANT}_{time.strftime('%Y%m%d_%H%M%S')}"
          f"_{os.environ.get('SEA_MUTABLE_DIR', 'mutable')}")
LOGS_DIR = HERE / "logs" / RUN_ID
LOGS_DIR.mkdir(parents=True, exist_ok=True)
# Structured run metadata -- avoids fragile parsing of RUN_ID later (task
# names like "health_gathering" already contain an underscore).
with open(LOGS_DIR / "meta.json", "w") as f:
    json.dump({"task": TASK_NAME, "prompt_variant": PROMPT_VARIANT, "run_id": RUN_ID}, f)
GIFS_DIR = LOGS_DIR / "gifs"
GIFS_DIR.mkdir(parents=True, exist_ok=True)
# Snapshots of the agent's own engine.py/prompt.py, taken on each new best
# score (plus a final one at exit) -- reset_engine.py wipes mutable/ before
# the next run, so this is the only surviving record of a good self-edit.
SNAPSHOTS_DIR = LOGS_DIR / "engine_snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
ENGINE_PATH = MUTABLE / "engine.py"
PROMPT_PATH = MUTABLE / "prompt.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOGS_DIR / "main.log"), logging.StreamHandler(sys.stdout)],
)

# Plain append-only, line-buffered dumps (not routed through `logging` --
# these are structured records for grepping/tailing, not diagnostics).
_thoughts_f = open(LOGS_DIR / "thoughts.log", "a", buffering=1)
_tool_results_f = open(LOGS_DIR / "tool_results.log", "a", buffering=1)
_task_results_f = open(LOGS_DIR / "task_results.log", "a", buffering=1)

print(f"[run {RUN_ID}] logging to {LOGS_DIR}", flush=True)

# Whole-process watchdog: if nothing progresses (thought token, tool result,
# task result, episode start) for HANG_EXIT_TIMEOUT seconds, assume a stuck
# low-level call (e.g. a wedged CUDA kernel) with no Python exception to
# catch, dump stacks, and hard-exit. We don't try to cancel the hung
# await/future instead: AsyncCacheEngine assumes a queued request's future is
# never cancelled out from under it, so a partial cancellation could corrupt
# engine state -- a full process exit is safe regardless.
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
_current_interactivity_ms: Optional[float] = None
_best_score: Optional[float] = None
_frame_buffer: list = []
_episode_in_round = 0


def _snapshot_engine(tag: str) -> None:
    try:
        shutil.copyfile(ENGINE_PATH, SNAPSHOTS_DIR / f"{tag}_engine.py")
        shutil.copyfile(PROMPT_PATH, SNAPSHOTS_DIR / f"{tag}_prompt.py")
    except Exception:
        logging.exception("failed to snapshot engine/prompt for %s", tag)


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def on_thought_start() -> None:
    global _step, _episode_in_round
    _step += 1
    _episode_in_round = 0
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


def on_frame(frame) -> None:
    # Only image-observation envs give us anything worth recording.
    # health_gathering's observation is a dict; unwrap before the ndarray check.
    if isinstance(frame, dict):
        frame = frame.get("screen")
    if isinstance(frame, np.ndarray):
        _frame_buffer.append(frame)


def on_episode_end() -> None:
    global _episode_in_round
    _touch_progress()
    _episode_in_round += 1
    if not _frame_buffer:
        return
    try:
        frames = [Image.fromarray(f) for f in _frame_buffer]
        out_path = GIFS_DIR / f"round{_step}_ep{_episode_in_round}.gif"
        frames[0].save(out_path, format="GIF", save_all=True, append_images=frames[1:],
                        duration=100, loop=0)
    except Exception:
        logging.exception("failed to save episode replay gif")
    finally:
        _frame_buffer.clear()


def _task_avg_act_latency_ms(task: Optional[dict]) -> Optional[float]:
    """Env-agnostic interactivity metric: mean of each episode's own
    avg_act_latency_s (wall-clock time between an observation arriving and
    engine.act() returning, carried in the step() info dict). Envs that don't
    populate this key yield None, same convention as `score` before the
    first task call."""
    if not task:
        return None
    latencies = [
        e["info"]["avg_act_latency_s"] for e in task.get("episodes", [])
        if isinstance(e.get("info"), dict) and e["info"].get("avg_act_latency_s") is not None
    ]
    if not latencies:
        return None
    return sum(latencies) / len(latencies) * 1000


def _write_metrics_csv() -> None:
    with open(LOGS_DIR / "round_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "score", "delay_s", "avg_act_latency_ms", "round_crashed",
                    "round_compile_failed"])
        for m in _metrics:
            w.writerow([m["round"], m["score"], m["delay_s"], m.get("avg_act_latency_ms"),
                        m.get("round_crashed", False), m.get("round_compile_failed", False)])


def on_step_result(result: dict) -> None:
    global _current_score, _current_interactivity_ms, _best_score
    _touch_progress()
    if "task" in result:
        _current_score = result["task"]["avg_reward"]
        latency_ms = _task_avg_act_latency_ms(result["task"])
        if latency_ms is not None:
            _current_interactivity_ms = latency_ms
        if _best_score is None or _current_score > _best_score:
            _best_score = _current_score
            _snapshot_engine(f"round{_step}_score{_current_score:.3f}")
    # A failed reload_engine_methods call (compile/construct error in the
    # agent's own engine.py) -- distinct from round_crashed (a whole-round
    # generation crash, zero tool calls attempted). Recorded so offline
    # reports can apply the same skip criterion agent.py's run() uses live.
    round_compile_failed = any(
        r.get("tool") == "reload_engine_methods" and not r.get("ok")
        for r in result.get("tool_results", []))
    _metrics.append({
        "round": _step, "score": _current_score, "delay_s": result.get("round_delay_s"),
        "avg_act_latency_ms": _current_interactivity_ms,
        # A whole-round generation crash (agent.py's outer except, zero tool
        # calls attempted). A task episode whose act() itself raised is NOT
        # this -- runner.py already scores that 0 as real data.
        "round_crashed": bool(result.get("crash")),
        "round_compile_failed": round_compile_failed,
    })
    _write_metrics_csv()


def plot_metrics(metrics: list[dict], out_dir: Path) -> None:
    """Three per-run plots: env score, round delay (full wall-clock per
    round, generation plus any task episodes), and in-episode interactivity
    (avg_act_latency_ms) -- independent of how long the round's self-edit took."""
    if not metrics:
        return
    rounds = [m["round"] for m in metrics]
    scores = [m["score"] if m["score"] is not None else float("nan") for m in metrics]
    delays_min = [m["delay_s"] / 60 if m["delay_s"] is not None else float("nan") for m in metrics]
    act_latency_ms = [
        m.get("avg_act_latency_ms") if m.get("avg_act_latency_ms") is not None else float("nan")
        for m in metrics
    ]

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

    if any(v == v for v in act_latency_ms):  # v == v is False only for NaN
        fig, ax = plt.subplots()
        ax.plot(rounds, act_latency_ms, marker="o", color="tab:green")
        ax.set_xlabel("Evolution round")
        ax.set_ylabel("Avg act() latency (ms)")
        ax.set_title("Env interactivity per evolution round (in-episode act() latency)")
        fig.savefig(out_dir / "interactivity_vs_round.png")
        plt.close(fig)


async def main(target_valid_steps: int, max_new_tokens: int, max_attempts: int) -> None:
    llm = minisgl.llm.AsyncLLM(
        "Qwen/Qwen3.8-27B", dtype=torch.bfloat16, max_running_req=4, memory_ratio=0.9,
        # repetition_penalty fixes the root cause of a failure mode seen live:
        # with no penalty, the model can get stuck emitting the same short
        # <use_tool> tag dozens/hundreds of times. agent.py's
        # MAX_TOOL_CALLS_PER_ROUND remains as a backstop, but this addresses
        # the cause rather than just capping the damage.
        generation_config=transformers.GenerationConfig(
            do_sample=True, temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.15),
        distributed_addr=f"tcp://127.0.0.1:{os.environ.get('SEA_LLM_PORT', '2370')}")

    env = SelfEditEnv(llm, engine_path=str(ENGINE_PATH), prompt_path=str(PROMPT_PATH))
    env.task_env = TASKS[TASK_NAME]()
    logging.info("task env %s seed=%s", TASK_NAME, getattr(env.task_env, "seed", None))
    agent = SelfEvolvingAgent(env, hooks=dict(
        on_thought_start=on_thought_start,
        on_thought_token=on_thought_token,
        on_completion=on_completion,
        on_tool_result=on_tool_result,
        on_task_result=on_task_result,
        on_episode_start=on_episode_start,
        on_step_result=on_step_result,
        on_frame=on_frame,
        on_episode_end=on_episode_end,
    ))

    watchdog = asyncio.create_task(_hang_watchdog())
    try:
        await agent.run(max_new_tokens=max_new_tokens, target_valid_steps=target_valid_steps,
                        max_attempts=max_attempts)
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
        _snapshot_engine("final")


if __name__ == "__main__":
    # argv[1]: target number of *valid* (non-crashed) rounds, not a raw round
    # count -- a crashed round doesn't count toward this (up to argv[5]'s cap).
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 32_000
    # argv[3]/argv[4] (task/prompt variant) are consumed above at import time.
    max_attempts = int(sys.argv[5]) if len(sys.argv) > 5 else 2 * k
    asyncio.run(main(k, max_new_tokens, max_attempts))
