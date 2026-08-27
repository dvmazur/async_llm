"""Standalone eval: play the *current* mutable/engine.py's Engine against the
real VizdoomDefendLine-v1 env and record each episode as a GIF, instead of a
live notebook dashboard.

Loads Engine straight from disk by file path -- never through
self_edit_env.py's compile/patch machinery, and no self-rewriting, curriculum,
or tool-call loop. Isolated from agent.py/self_edit_env.py/run_persistent.py:
safe to run standalone or alongside a live run_persistent.py process (distinct
distributed_addr port, and it only *reads* mutable/engine.py).

Usage:
    .venv/bin/python scripts/self_evolving_agent/eval_doom_engine.py [n_episodes] [max_steps]
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/LLM")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import asyncio
import importlib.util
import sys
import time
from pathlib import Path

import torch
import transformers
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minisgl.llm
from tasks.doom_env import DoomEnv

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ENGINE_PATH = HERE / "mutable" / "engine.py"

OUT_DIR = HERE / "eval_runs" / time.strftime("%Y%m%d_%H%M%S")


def load_engine_class(path: Path):
    """Load the current on-disk Engine class by file path -- a plain one-shot
    import, not self_edit_env.py's compile/patch-onto-live-instance dance."""
    spec = importlib.util.spec_from_file_location("mutable_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Engine


def annotate_frame(frame, text_lines: list[str]) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    pad = 3
    for i, line in enumerate(text_lines):
        y = pad + i * 12
        draw.rectangle([0, y, img.width, y + 12], fill=(0, 0, 0))
        draw.text((pad, y), line, fill=(255, 255, 0))
    return img


async def run_episode(engine, env, ep_idx: int, max_steps: int, log_f) -> float:
    obs = env.reset()
    total_reward = 0.0
    frames: list[Image.Image] = []

    for step in range(max_steps):
        tokens: list[str] = []
        action = await engine.act(obs, on_token=tokens.append)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        line = (f"[ep {ep_idx} step {step}] action={action!r} reward={reward:+.2f} "
                f"total={total_reward:+.2f}")
        print(line, flush=True)
        log_f.write(line + "\n")
        raw_text = "".join(tokens).strip().replace("\n", " ")
        if raw_text:
            log_f.write(f"    raw model output: {raw_text!r}\n")
        log_f.flush()

        frames.append(annotate_frame(obs, [
            f"ep {ep_idx} step {step}",
            f"action={action} reward={reward:+.2f} total={total_reward:+.2f}",
        ]))

        if done:
            break

    gif_path = OUT_DIR / f"episode_{ep_idx}.gif"
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                        duration=150, loop=0)
        print(f"[ep {ep_idx}] wrote {gif_path} ({len(frames)} frames)")

    return total_reward


async def main(n_episodes: int, max_steps: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine_source = ENGINE_PATH.read_text()
    print(f"loaded Engine from {ENGINE_PATH} ({len(engine_source)} chars)")

    llm = minisgl.llm.AsyncLLM(
        "Qwen/Qwen3.8-27B", dtype=torch.bfloat16, max_running_req=4, memory_ratio=0.9,
        generation_config=transformers.GenerationConfig(
            do_sample=True, temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.15),
        # Distinct port from doom_basic (2369) and run_persistent.py (2370) --
        # safe to run this eval alongside either.
        distributed_addr="tcp://127.0.0.1:2372")

    Engine = load_engine_class(ENGINE_PATH)
    engine = Engine(llm)
    env = DoomEnv(episode_timeout=1000)

    log_path = OUT_DIR / "episodes.log"
    with open(log_path, "a", buffering=1) as log_f:
        rewards = []
        for ep in range(n_episodes):
            rewards.append(await run_episode(engine, env, ep, max_steps, log_f))

    print(f"\nepisode rewards: {rewards}")
    print(f"avg reward: {sum(rewards) / len(rewards):.3f}")
    print(f"logs/gifs in {OUT_DIR}")


if __name__ == "__main__":
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else DoomEnv.max_steps_per_episode
    asyncio.run(main(n_episodes, max_steps))
