"""Uniform random actions, optionally queued after a complete LLM campaign."""
import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
import random
import time

from run_budget_sweep import CAPS, TASKS, atomic_json, file_lock, report, seed_for


class RandomEngine:
    def __init__(self, actions, seed):
        self.actions = list(actions)
        self.rng = random.Random(seed)
        self.trace = []

    async def act(self, observation, on_token=None):
        start = time.monotonic()
        action = self.rng.choice(self.actions)
        self.trace.append({"action": action, "latency_s": time.monotonic() - start})
        return action


def campaign_complete(root):
    """Do not start on partial results, a stopped worker, or a failed campaign."""
    try:
        status = json.loads((root / "status.json").read_text())
        if status.get("status") != "complete":
            return False
        with (root / "summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        return (len(rows) == 24 and all(int(r["completed_runs"]) == 10
                                       and int(r["errors"]) == 0 for r in rows))
    except (OSError, ValueError, KeyError):
        return False


async def evaluate(out):
    from tasks.doom_env import ACTION_NAMES as DA, DoomEnv
    from tasks.health_gathering_env import ACTION_NAMES as HA, HealthGatheringEnv
    from tasks.runner import run_episodes

    from tasks.realtime_vizdoom import realtime_options
    config = dict(mode="uniform_random", tasks=TASKS, budgets=[0], runs=10, episodes=5,
                  seed=20260915, game_tic_limits={"doom":1000,"health_gathering":10000},
                  realtime=realtime_options(), protocol_version=3,
                  description="Independent uniform choice among all four legal actions each step; no LLM.")
    path = out / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Existing random-baseline configuration differs")
    atomic_json(path, config)
    for task, env_class, actions in [("doom", DoomEnv, DA), ("health_gathering", HealthGatheringEnv, HA)]:
        for run in range(10):
            for episode in range(5):
                path = out / f"episode_{task}_b0_r{run:02}_e{episode:02}.json"
                if path.exists():
                    if json.loads(path.read_text())["error"]:
                        raise RuntimeError(f"Investigate failed episode before resuming: {path}")
                    continue
                env_seed = seed_for(config["seed"], task, run, episode)
                # Separate reproducible stream for policy randomness; env seeds
                # exactly match the corresponding episodes in all LLM conditions.
                action_seed = seed_for(config["seed"], task + ":random_actions", run, episode)
                engine = RandomEngine(actions, action_seed)
                env = env_class(seed=env_seed)
                env.max_episodes, env.max_steps_per_episode = 1, CAPS[task]
                start = time.monotonic()
                try:
                    result = await run_episodes(env, engine)
                finally:
                    env.env.close()
                ep = result["episodes"][0]
                error = ep["info"].get("error", "")
                row = dict(task=task, budget=0, run=run, episode=episode, seed=env_seed,
                           action_seed=action_seed, reward=ep["reward"], steps=ep["steps"], error=error,
                           llm_forward_calls=0, actions_per_forward=None,
                           hit_step_cap=False, info=ep["info"], decision_attempts=len(engine.trace), reasoning_tokens=0,
                           generated_tokens=0, budget_hits=0,
                           act_seconds=sum(s["latency_s"] for s in engine.trace),
                           elapsed_seconds=time.monotonic() - start, trace=engine.trace)
                atomic_json(path, row)
                print(f"DONE {path.name} reward={ep['reward']} steps={ep['steps']} error={bool(error)}", flush=True)
                report(out)
                if error:
                    raise RuntimeError(error)
    return report(out)


def main(out, after):
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out / "random_baseline.lock", blocking=False) as owner:
        if not owner:
            raise RuntimeError("Random baseline already queued or running")
        status = dict(pid=os.getpid(), status="queued" if after else "running",
                      after_campaign=str(after) if after else None, episodes_expected=100)
        try:
            if after:
                print(f"Queued: waiting for all 1200 LLM episodes in {after}", flush=True)
                while not campaign_complete(after):
                    status["updated"] = time.time()
                    atomic_json(out / "status.json", status)
                    time.sleep(30)
            status.update(status="running", updated=time.time())
            atomic_json(out / "status.json", status)
            rows = asyncio.run(evaluate(out))
            if len(rows) != 2 or any(r["completed_runs"] != 10 or r["errors"] for r in rows):
                raise RuntimeError("Random baseline is incomplete")
            status["status"] = "complete"
        except BaseException as error:
            status.update(status="failed", error=str(error))
            raise
        finally:
            status["updated"] = time.time()
            atomic_json(out / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after-campaign", type=Path)
    args = parser.parse_args()
    main(args.output.resolve(), args.after_campaign.resolve() if args.after_campaign else None)
