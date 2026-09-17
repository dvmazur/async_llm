"""Resumable, single-GPU baseline sweep; see reports/budget_sweep_protocol.md."""
from __future__ import annotations

import argparse
import asyncio
import csv
from contextlib import contextmanager
import errno
import fcntl
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/LLM")

BUDGETS = [0, 64, 128, 512, 1024, 4096, 8192, 16384]
TASKS = ["doom", "health_gathering"]
CAPS = {"doom": 100, "health_gathering": 2500}
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PROMPT = (
    "Choose the best next action using the current screenshot. "
    "Consider the visible geometry, hazards, and targets. "
    "You may reason before deciding; finish thinking once you have a decision. "
    "Return the chosen action in \\boxed{{action}}, using exactly one of: {actions}. "
    "The game continues at 35 tics/sec while you decide."
)


def retry_disk_full(func):
    """Keep computed results in memory until the shared filesystem has space."""
    @wraps(func)
    def retry(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except OSError as error:
                if error.errno != errno.ENOSPC:
                    raise
                try:
                    print(f"DISK FULL: pausing {func.__name__}; retrying in 30 seconds", flush=True)
                except OSError:
                    pass  # The log may live on the same full filesystem.
                time.sleep(30)
    return retry


@retry_disk_full
def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def seed_for(base, task, run, episode):
    key = f"{base}:{task}:{run}:{episode}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % (2**31 - 2) + 1


@contextmanager
def file_lock(path, blocking=True):
    """OS releases locks on crashes; another GPU can safely resume unfinished work."""
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class BudgetEngine:
    def __init__(self, llm, task, budget, mode="reasoning"):
        from run_baseline import TASK_CONTEXT
        from tasks.doom_env import ACTION_NAMES as DA
        from tasks.health_gathering_env import ACTION_NAMES as HA

        self.llm, self.task, self.budget = llm, task, budget
        self.mode = mode
        self.actions = DA if task == "doom" else HA
        self.context = TASK_CONTEXT[task]
        self.trace = []
        tok = llm.tokenizer
        self.action_ids = []
        for action in self.actions:
            ids = tok.encode(action, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Action must be a single token: {action}: {ids}")
            self.action_ids.append(ids[0])
        self.end_think = tok.convert_tokens_to_ids("</think>")
        if self.end_think is None or tok.decode([self.end_think]) != "</think>":
            raise ValueError("Checkpoint must support </think> token")
        self.eos = {tok.eos_token_id}
        eos = llm.config.generation_config.eos_token_id
        self.eos.update(eos if isinstance(eos, list) else [eos])

    async def act(self, observation, on_token=None):
        from run_baseline import _obs_image_and_extra

        start = time.monotonic()
        image, extra = _obs_image_and_extra(observation, self.task)
        from tasks.realtime_vizdoom import action_policy_doc
        text = action_policy_doc() + "\n" + self.context + "\n" + extra + PROMPT.format(actions=", ".join(self.actions))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": text},
        ]}]
        enc = self.llm.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            enable_thinking=self.budget > 0 and self.mode == "reasoning", reasoning_effort="medium",
            return_dict=True, return_tensors="pt",
        )
        block = await self.llm.create_block()
        count, stop = 0, "no_reasoning"
        generated = []
        action = None
        cancelled = False
        try:
            output = await self.llm(**enc, cache_view=[block])
            if self.budget:
                stop = "budget"
                for _ in range(self.budget):
                    token = await self.llm.sample(output)
                    token_id = int(token)
                    count += 1
                    generated.append(token_id)
                    decoded = self.llm.tokenizer.decode(generated, skip_special_tokens=False)
                    answer = decoded.rsplit("</think>", 1)[-1] if self.mode == "reasoning" else decoded
                    match = re.search(r"\\boxed\{\s*([a-z_]+)\s*\}", answer)
                    if match and match.group(1) in self.actions and (self.mode != "reasoning" or self.end_think in generated):
                        action, stop = match.group(1), "boxed_action"
                        break
                    if token_id in self.eos:
                        stop = "eos"
                        break
                    output = await self.llm(token.view(1), cache_view=[block])
                # Text baselines act only on a completed legal boxed answer.
                # An incomplete or malformed answer consumes its time and waits.
                if action is None:
                    action = "wait"
            else:
                output = await self.llm(
                    self.llm.tokenizer.encode("Action:", add_special_tokens=False), cache_view=[block])
                action = self.actions[int(output.logits[self.action_ids].argmax())]
            return action
        except asyncio.CancelledError:
            cancelled, stop = True, "episode_ended"
            raise
        finally:
            self.trace.append({"attempt": len(self.trace) + 1, "action": action,
                               "reasoning_tokens": (generated.index(self.end_think) + 1 if self.end_think in generated else count) if self.mode == "reasoning" else 0,
                               "generated_tokens": count, "stop": stop, "cancelled": cancelled,
                               "text": self.llm.tokenizer.decode(generated, skip_special_tokens=False),
                               "latency_s": time.monotonic() - start})
            await self.llm.free_block(block)


@retry_disk_full
def report(out):
    with file_lock(out / "report.lock"):
        return _report(out)


def _report(out):
    from scipy.stats import t

    config = json.loads((out / "config.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted(out.glob("episode_*.json"))]
    rows = []
    for task in config["tasks"]:
        for budget in config["budgets"]:
            cell = [r for r in records if r["task"] == task and r["budget"] == budget]
            means = []
            for run in range(config["runs"]):
                eps = [r for r in cell if r["run"] == run]
                if len(eps) == config["episodes"] and not any(r["error"] for r in eps):
                    means.append(statistics.mean(r["reward"] for r in eps))
            n = len(means)
            mean = statistics.mean(means) if n else None
            sd = statistics.stdev(means) if n > 1 else None
            half = float(t.ppf(.975, n - 1)) * sd / math.sqrt(n) if n > 1 else None
            steps = sum(r["steps"] for r in cell)
            attempts = sum(r.get("decision_attempts", r["steps"]) for r in cell)
            rows.append(dict(task=task, mode=config.get("mode", "reasoning"), budget=budget, completed_runs=n,
                             expected_runs=config["runs"], mean_reward=mean,
                             run_sd=sd, ci95_half_width=half,
                             ci95_low=mean-half if half is not None else None,
                             ci95_high=mean+half if half is not None else None,
                             decision_attempts=attempts, completed_actions=steps,
                             cancelled_decisions=sum(s.get("cancelled", False) for r in cell for s in r.get("trace", [])),
                             mean_reasoning_tokens=sum(r["reasoning_tokens"] for r in cell)/attempts if attempts else None,
                             mean_generated_tokens=sum(r.get("generated_tokens", r["reasoning_tokens"]) for r in cell)/attempts if attempts else None,
                             mean_act_latency_s=sum(r["act_seconds"] for r in cell)/attempts if attempts else None,
                             budget_exhaustion_rate=sum(r["budget_hits"] for r in cell)/attempts if attempts else None,
                             capped_episodes=sum(r["hit_step_cap"] for r in cell),
                             errors=sum(bool(r["error"]) for r in cell)))
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Fresh baseline budget sweep", "", f"Generation mode: **{config.get('mode', 'reasoning')}**.", "",
             "Reward: mean of run means ± two-sided 95% Student-t CI. Each complete run has "
             f"{config['episodes']} episodes; target {config['runs']} runs per condition. "
             "Incomplete runs are omitted from CIs and errors are reported explicitly. "
             "Token and latency averages include terminally cancelled decision attempts.", "",
             "| Environment | Generation budget | Complete runs | Mean reward ± 95% CI | Tokens/attempt | Seconds/attempt | Errors |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return f"{x:.3f}" if x is not None else "—"
    for r in rows:
        lines.append(f"| {r['task']} | {r['budget'] or 'none'} | {r['completed_runs']}/{config['runs']} | "
                     f"{fmt(r['mean_reward'])} ± {fmt(r['ci95_half_width'])} | "
                     f"{fmt(r['mean_generated_tokens'])} | {fmt(r['mean_act_latency_s'])} | {r['errors']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    if any(r["completed_runs"] > 1 for r in rows):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(config["tasks"]), figsize=(13, 5), squeeze=False)
        for ax, task in zip(axes[0], config["tasks"]):
            cell = [r for r in rows if r["task"] == task and r["completed_runs"] > 1]
            ax.errorbar(range(len(cell)), [r["mean_reward"] for r in cell],
                        yerr=[r["ci95_half_width"] for r in cell], fmt="o", capsize=4)
            ax.set_xticks(range(len(cell)), [str(r["budget"] or "none") for r in cell], rotation=45)
            ax.set(title=task, xlabel="Maximum generated tokens", ylabel="Mean episode reward (95% CI)")
            ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(out / "reward_ci.png", dpi=180)
        plt.close(fig)
    return rows


async def run(args):
    import torch
    import transformers
    from minisgl.llm import AsyncLLM
    from run_baseline import TASKS as ENVS
    from tasks.runner import run_episodes

    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    if gpu not in args.allowed_gpus or os.environ["HF_HOME"] != "/mnt/LLM":
        raise ValueError("Use one explicitly allowed GPU and HF_HOME=/mnt/LLM")
    if torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible GPU required")
    args.output.mkdir(parents=True, exist_ok=True)
    from tasks.realtime_vizdoom import realtime_options
    config = dict(model=args.model, revision=args.revision, mode=args.mode, runs=args.runs, episodes=args.episodes,
                  budgets=args.budgets, tasks=args.tasks, seed=args.seed, game_tic_limits={"doom":1000,"health_gathering":10000},
                  realtime=realtime_options(),
                  prompt=PROMPT, protocol_version=3, max_seq_len=32768,
                  sampling=dict(temperature=.7, top_k=20, top_p=.9, repetition_penalty=1.0))
    path = args.output / "config.json"
    with file_lock(args.output / "config.lock"):
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Resume configuration differs; use a new output directory")
        atomic_json(path, config)
    from huggingface_hub import snapshot_download
    model_path = args.model if Path(args.model).is_dir() else snapshot_download(
        args.model, revision=args.revision,
    )
    llm = AsyncLLM(model_path, dtype=torch.bfloat16, max_running_req=1, memory_ratio=.9,
                   max_seq_len_override=config["max_seq_len"],
                   generation_config=transformers.GenerationConfig(
                       do_sample=True, **config["sampling"]),
                   distributed_addr=f"tcp://127.0.0.1:{args.port}")
    metadata = args.output / f"environment_gpu{gpu}.json"
    if not metadata.exists() or not any(args.output.glob("episode_*.json")):
        sources = [Path(__file__), Path(__file__).with_name("run_baseline.py")]
        sources.extend(Path(__file__).parent / "tasks" / name
                       for name in ["runner.py", "doom_env.py", "health_gathering_env.py"])
        atomic_json(metadata, dict(torch=torch.__version__, transformers=transformers.__version__,
                                  gpu=torch.cuda.get_device_name(),
                                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                  cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                                  hf_home=os.environ["HF_HOME"]))
        with (args.output / f"packages_gpu{gpu}.txt").open("w") as f:
            subprocess.run(["uv", "pip", "freeze", "--python", os.sys.executable], stdout=f, check=True)
    from warmup import warmup
    await warmup(llm)
    from action_efficiency import ForwardCounter, ratio
    forward_counter = ForwardCounter(llm)
    try:
        for run_index in range(args.runs):
            conditions = [(task, budget) for task in args.tasks for budget in args.budgets]
            random.Random(args.seed + run_index).shuffle(conditions)
            for task, budget in conditions:
                for episode in range(args.episodes):
                    path = args.output / f"episode_{task}_b{budget}_r{run_index:02}_e{episode:02}.json"
                    with file_lock(path.with_suffix(".lock"), blocking=False) as claimed:
                        if not claimed:
                            continue
                        if path.exists():
                            if json.loads(path.read_text())["error"]:
                                raise RuntimeError(f"Previously failed episode requires investigation: {path}")
                            continue
                        seed = seed_for(args.seed, task, run_index, episode)
                        random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        env = ENVS[task](seed=seed)
                        env.max_episodes, env.max_steps_per_episode = 1, CAPS[task]
                        engine = BudgetEngine(llm, task, budget, mode=args.mode)
                        forwards_before = forward_counter.calls
                        start = time.monotonic()
                        print(f"START {task} budget={budget} run={run_index+1} episode={episode+1} seed={seed}", flush=True)
                        def progress():
                            if len(engine.trace) and len(engine.trace) % 10 == 0:
                                print(f"STEP {task} b={budget} r={run_index+1} e={episode+1} "
                                      f"steps={len(engine.trace)} elapsed={time.monotonic()-start:.1f}s", flush=True)
                        try:
                            result = await run_episodes(env, engine, on_step=progress)
                        finally:
                            env.env.close()
                        ep = result["episodes"][0]
                        error = ep["info"].get("error", "")
                        row = dict(task=task, budget=budget, run=run_index, episode=episode, gpu=gpu,
                                   seed=seed, reward=ep["reward"], steps=ep["steps"], error=error,
                                   llm_forward_calls=forward_counter.calls - forwards_before,
                                   actions_per_forward=ratio(ep["steps"], forward_counter.calls - forwards_before),
                                   hit_step_cap=False, info=ep["info"], decision_attempts=len(engine.trace),
                                   reasoning_tokens=sum(s["reasoning_tokens"] for s in engine.trace),
                                   generated_tokens=sum(s["generated_tokens"] for s in engine.trace),
                                   budget_hits=sum(s["stop"] == "budget" for s in engine.trace),
                                   act_seconds=sum(s["latency_s"] for s in engine.trace),
                                   elapsed_seconds=time.monotonic()-start, trace=engine.trace)
                        atomic_json(path, row)
                        print(f"DONE {path.name} reward={ep['reward']} steps={ep['steps']} "
                              f"seconds={row['elapsed_seconds']:.1f} error={bool(error)}", flush=True)
                        report(args.output)
                        if error:
                            raise RuntimeError(error)
    finally:
        await llm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--mode", choices=["reasoning", "no_think"], default="reasoning")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=TASKS)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--port", type=int, default=2391)
    parser.add_argument("--allowed-gpus", nargs="+", default=["5", "6"])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.episodes < 1 or any(b < 0 for b in args.budgets):
        parser.error("Runs/episodes must be positive and budgets nonnegative")
    if args.report_only:
        report(args.output)
    else:
        asyncio.run(run(args))
