"""Fixed, non-evolving baselines for `doom` and `health_gathering`, run
directly against tasks/runner.py's own run_episodes() -- the same episode-
scoring protocol every self-evolving round is scored with. Deliberately
bypasses agent.py/self_edit_env.py entirely: one hardcoded Engine per
condition, evaluated for many episodes back to back, no round loop.

Two conditions per task (`--baseline` below), both restricted to a single
current frame (no history, no temporal stacking):
  no_reasoning -- a pure logit probe over the action-name tokens. Structurally
    cannot "reason": one forward pass, argmax over 4 logits, done.
  reasoning -- same single frame, but bounded free-text chain-of-thought
    ending in "ACTION: <word>". The only axis that differs from no_reasoning
    is whether reasoning happens before the action.
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/LLM")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import asyncio
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import transformers

import minisgl.llm
from tasks.doom_env import DoomEnv
from tasks.health_gathering_env import HealthGatheringEnv
from tasks.runner import run_episodes

TASKS = {"doom": DoomEnv, "health_gathering": HealthGatheringEnv}

# Bumped above the round-scoring default (2 episodes) since baselines are
# cheap to run and per-episode reward is high-variance. Trimmed down from an
# original 10 after measuring real per-step cost (see MAX_STEPS_OVERRIDE) --
# 10 reasoning episodes on health_gathering alone would run ~65+ hours.
DEFAULT_EPISODES = 5

# health_gathering's native max_steps_per_episode (2500) assumes a fast
# policy; the `reasoning` baseline's ~11.76s/step would make a full episode
# take ~8.17 hours. Applied symmetrically to both baselines for fairness.
# doom's native cap (100 steps) is already cheap enough and left alone.
MAX_STEPS_OVERRIDE = {"health_gathering": 400}

# Short, static task framing -- NOT the full agent.py-facing DOC strings
# (tasks/doom_env.py's DOC etc), which carry self-editing meta-guidance that
# has nothing to do with a fixed baseline policy.
TASK_CONTEXT = {
    "doom": (
        "You are playing a first-person shooter mini-game (ViZDoom 'Defend the Line'). "
        "Enemies approach from the front and must be shot before they reach you. "
        "Available actions: wait, fire, right (turn right), left (turn left)."
    ),
    "health_gathering": (
        "You are navigating a first-person 3D environment. The floor is toxic and "
        "continuously drains your health. Small glowing GREEN canisters standing upright on "
        "the floor (usually near walls) are medkits -- walking over one restores health. "
        "Available actions: wait, forward (move forward), right (turn right), left (turn left)."
    ),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_baseline")


def _obs_image_and_extra(observation: Any, task: str) -> tuple[Any, str]:
    if isinstance(observation, dict) and "screen" in observation:
        health = observation.get("health", 0.0)
        return observation["screen"], f"Health: {health:.0f}/100.\n"
    return observation, ""


class NoReasoningEngine:
    """One current frame, one forward pass, argmax over the 4 action-name
    logits -- no free-generation step exists in this path at all, so there
    is no room for chain-of-thought reasoning to sneak in."""

    def __init__(self, llm: "minisgl.llm.AsyncLLM", task: str) -> None:
        self.llm = llm
        self.task = task

    async def act(self, observation: Any, on_token=None) -> str:
        from tasks.doom_env import ACTION_NAMES as DOOM_ACTIONS
        from tasks.health_gathering_env import ACTION_NAMES as HG_ACTIONS
        actions = DOOM_ACTIONS if self.task == "doom" else HG_ACTIONS

        image, extra = _obs_image_and_extra(observation, self.task)
        proc = self.llm.processor
        prompt_text = (
            f"{TASK_CONTEXT[self.task]}\n{extra}"
            f"Look at the screenshot. Choose exactly one action: {', '.join(actions)}. Action:"
        )
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt")
        block = await self.llm.create_block()
        try:
            output = await self.llm(**enc, cache_view=[block])
            # output.logits is a flat (vocab_size,) vector for the single new
            # token position (no batch/seq dims) -- indexing [0] first grabs
            # one scalar logit instead and throws IndexError.
            ids = [proc.tokenizer.convert_tokens_to_ids(a) for a in actions]
            return actions[int(output.logits[ids].argmax())]
        finally:
            await self.llm.free_block(block)


class ReasoningEngine:
    """Same single current frame as NoReasoningEngine, but a bounded
    free-text generation (chain-of-thought allowed) ending in
    'ACTION: <word>' before an action is parsed out -- the only axis that
    differs from NoReasoningEngine."""

    # Bounded to a *brief* chain-of-thought: at 300 tokens/step this took
    # 30+ min to not even finish one health_gathering episode and had to be
    # killed. 64 tokens is enough for a short clause + "ACTION: word".
    MAX_REASONING_TOKENS = 64

    def __init__(self, llm: "minisgl.llm.AsyncLLM", task: str) -> None:
        self.llm = llm
        self.task = task

    def _eos_ids(self) -> set:
        llm = self.llm
        eos_ids = {llm.tokenizer.eos_token_id}
        gen_cfg = getattr(getattr(getattr(llm, "engine", None), "config", None),
                          "generation_config", None)
        gen_eos = getattr(gen_cfg, "eos_token_id", None)
        if isinstance(gen_eos, int):
            eos_ids.add(gen_eos)
        elif isinstance(gen_eos, (list, tuple, set)):
            eos_ids.update(gen_eos)
        return eos_ids

    def _parse_action(self, text: str, actions: list[str]) -> str:
        text_lower = text.lower()
        m = re.search(r"action:\s*(\w+)", text_lower)
        if m and m.group(1) in actions:
            return m.group(1)
        for a in actions:
            if a in text_lower:
                return a
        return "wait"

    async def act(self, observation: Any, on_token=None) -> str:
        from tasks.doom_env import ACTION_NAMES as DOOM_ACTIONS
        from tasks.health_gathering_env import ACTION_NAMES as HG_ACTIONS
        actions = DOOM_ACTIONS if self.task == "doom" else HG_ACTIONS

        image, extra = _obs_image_and_extra(observation, self.task)
        proc = self.llm.processor
        prompt_text = (
            f"{TASK_CONTEXT[self.task]}\n{extra}"
            f"Look at the screenshot. Think briefly about what you see, then choose exactly one "
            f"action from: {', '.join(actions)}. End your reply with a final line "
            f"'ACTION: <action>'."
        )
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt")
        eos_ids = self._eos_ids()
        block = await self.llm.create_block()
        try:
            output = await self.llm(**enc, cache_view=[block])
            tokens: list[int] = []
            for _ in range(self.MAX_REASONING_TOKENS):
                new_token_id = await self.llm.sample(output)
                if on_token is not None:
                    on_token(self.llm.tokenizer.decode(new_token_id))
                tokens.append(int(new_token_id))
                if int(new_token_id) in eos_ids:
                    break
                output = await self.llm(new_token_id.view(1), cache_view=[block])
            text = self.llm.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            return self._parse_action(text, actions)
        finally:
            await self.llm.free_block(block)


ENGINES = {"no_reasoning": NoReasoningEngine, "reasoning": ReasoningEngine}


async def run_one(llm, task: str, baseline: str, episodes: int, out_dir: Path) -> dict:
    env = TASKS[task]()
    env.max_episodes = episodes
    if task in MAX_STEPS_OVERRIDE:
        env.max_steps_per_episode = MAX_STEPS_OVERRIDE[task]
    engine = ENGINES[baseline](llm, task)

    out_dir.mkdir(parents=True, exist_ok=True)
    rewards: list[float] = []

    step_count = 0
    t0 = time.monotonic()

    def on_step() -> None:
        nonlocal step_count
        step_count += 1
        if step_count % 50 == 0:
            elapsed = time.monotonic() - t0
            logger.info("%s/%s: %d steps in %.0fs (%.2f steps/s)",
                        task, baseline, step_count, elapsed, step_count / elapsed)

    def on_episode_start() -> None:
        logger.info("%s/%s: episode start (t=%.0fs)", task, baseline, time.monotonic() - t0)

    result = await run_episodes(env, engine, on_step=on_step, on_episode_start=on_episode_start)
    errors = []
    latencies_ms: list[float] = []
    with open(out_dir / "episode_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "avg_act_latency_ms", "error"])
        for i, ep in enumerate(result["episodes"], 1):
            rewards.append(ep["reward"])
            info = ep.get("info", {}) or {}
            error = info.get("error", "")
            # Same env-side timer self-evolved engines are scored with --
            # lets a fixed baseline's interactivity be compared against an
            # evolving run's.
            latency_s = info.get("avg_act_latency_s")
            latency_ms = latency_s * 1000 if latency_s is not None else None
            if latency_ms is not None:
                latencies_ms.append(latency_ms)
            if error:
                errors.append((i, error))
            w.writerow([i, ep["reward"], latency_ms, error])
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else None
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"task": task, "baseline": baseline, "episodes": episodes,
                    "avg_reward": result["avg_reward"], "rewards": rewards,
                    "avg_act_latency_ms": avg_latency_ms}, f, indent=2)
    for i, error in errors:
        logger.warning("%s/%s: episode %d errored -- scored 0, see episode_metrics.csv:\n%s",
                        task, baseline, i, error)
    logger.info("%s/%s: avg_reward=%.3f avg_act_latency_ms=%s over %d episodes (%d errored)",
                task, baseline, result["avg_reward"],
                f"{avg_latency_ms:.1f}" if avg_latency_ms is not None else "n/a",
                episodes, len(errors))
    return {"task": task, "baseline": baseline, "avg_reward": result["avg_reward"],
            "episodes": episodes, "avg_act_latency_ms": avg_latency_ms}


async def main(episodes: int, only_task: Optional[str], only_baseline: Optional[str]) -> None:
    llm = minisgl.llm.AsyncLLM(
        "Qwen/Qwen3.8-27B", dtype=torch.bfloat16, max_running_req=4, memory_ratio=0.9,
        generation_config=transformers.GenerationConfig(
            do_sample=True, temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.15),
        distributed_addr=f"tcp://127.0.0.1:{os.environ.get('SEA_LLM_PORT', '2390')}")

    run_id = f"baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    logs_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "logs" / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)

    tasks = [only_task] if only_task else list(TASKS)
    baselines = [only_baseline] if only_baseline else list(ENGINES)

    summary = []
    for task in tasks:
        for baseline in baselines:
            logger.info("running %s/%s (%d episodes)", task, baseline, episodes)
            row = await run_one(llm, task, baseline, episodes, logs_dir / f"{task}_{baseline}")
            summary.append(row)

    with open(logs_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "baseline", "avg_reward", "episodes", "avg_act_latency_ms"])
        for row in summary:
            w.writerow([row["task"], row["baseline"], row["avg_reward"], row["episodes"],
                        row.get("avg_act_latency_ms")])
    logger.info("done -- summary at %s", logs_dir / "summary.csv")
    for row in summary:
        lat = row.get("avg_act_latency_ms")
        print(f"{row['task']:>16s} / {row['baseline']:<12s} avg_reward={row['avg_reward']:.3f} "
              f"avg_act_latency_ms={f'{lat:.1f}' if lat is not None else 'n/a'} "
              f"({row['episodes']} episodes)")


if __name__ == "__main__":
    # argv[1]: episode count (default DEFAULT_EPISODES). argv[2]: restrict to
    # one task (doom|health_gathering), or "all" (default). argv[3]: restrict
    # to one baseline (no_reasoning|reasoning), or "all" (default).
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EPISODES
    only_task = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "all" else None
    only_baseline = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "all" else None
    asyncio.run(main(episodes, only_task, only_baseline))
