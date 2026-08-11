#!/usr/bin/env python3
"""Capture one complete Doom decision with ``torch.profiler``.

The model and complete unrecorded Doom decisions are warmed up before the
profiler is created. The first pass compiles JIT kernels and the second exercises
the compiled paths. The profiler context then contains exactly one decision: image
preprocessing, prefill, autoregressive reasoning, the action probe, and
``env.step``.  The resulting Chrome trace can be opened directly in Perfetto.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import random
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = Path("/tmp/minisgl-cache")
DEFAULT_TRACE_DIR = REPO_ROOT / "wip" / "profiles"

PROMPT = """
This is a first-person video game called Doom. Choose the next action: `wait`, `fire`, `right`, or `left`.
A shot hits only when the enemy is centered precisely above the gun barrel. Aim by moving left or right toward the enemy.
Analyze the image in one short paragraph. Do not write the action yet.
""".strip()
ACTION_NAMES = ("wait", "fire", "right", "left")

CUDA_TRACE_CATEGORIES = (
    "kernel",
    "cuda_runtime",
    "cuda_driver",
    "gpu_memcpy",
    "gpu_memset",
    "gpu_user_annotation",
)


@dataclass
class Decision:
    observation: dict
    reasoning: str
    reasoning_token_ids: list[int]
    action_name: str
    action_index: int
    action_scores: list[float]
    reward: float
    terminated: bool
    truncated: bool
    wall_seconds: float
    model_forwards: int


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Warm up Mini-SGLang, then torch-profile exactly one full Doom step."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MINISGL_DOOM_MODEL", "Qwen/Qwen3.5-0.8B"),
    )
    parser.add_argument("--gpu", default=os.environ.get("MINISGL_DOOM_GPU", "0"))
    parser.add_argument("--gdn-backend", choices=("fla", "torch"), default="fla")
    parser.add_argument("--memory-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--max-reasoning-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_TRACE_DIR / f"doom_full_step_{timestamp}.json",
    )
    return parser.parse_args()


def configure_process(args: argparse.Namespace) -> int:
    """Configure CUDA and writable caches before importing torch or Mini-SGLang."""
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("HF_HOME", str(args.cache_root / "huggingface"))
    os.environ.setdefault("TVM_FFI_CACHE_DIR", str(args.cache_root / "tvm-ffi"))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(args.cache_root / "flashinfer"))
    os.environ.setdefault("MPLCONFIGDIR", str(args.cache_root / "matplotlib"))
    for variable in (
        "HF_HOME",
        "TVM_FFI_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "MPLCONFIGDIR",
    ):
        Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        return int(port_socket.getsockname()[1])


def count_trace_categories(trace_path: Path) -> dict[str, int]:
    """Count CUDA categories without materializing a potentially multi-GB JSON trace."""
    needles = {
        category: f'"cat": "{category}"'.encode() for category in CUDA_TRACE_CATEGORIES
    }
    counts = dict.fromkeys(CUDA_TRACE_CATEGORIES, 0)
    overlap = max(map(len, needles.values())) - 1
    tail = b""
    with trace_path.open("rb") as trace_file:
        while chunk := trace_file.read(8 * 1024 * 1024):
            searchable = tail + chunk
            for category, needle in needles.items():
                counts[category] += searchable.count(needle) - tail.count(needle)
            tail = searchable[-overlap:]
    return counts


async def async_main(args: argparse.Namespace, distributed_port: int) -> None:
    # These imports must follow configure_process(): Engine requires CUDA to be
    # untouched when it is constructed.
    import gymnasium as gym
    import numpy as np
    import torch
    import vizdoom  # noqa: F401
    import vizdoom.gymnasium_wrapper  # noqa: F401 -- registers VizdoomBasic-v0
    from minisgl.engine import EngineConfig
    from minisgl.llm import AsyncLLM
    from minisgl.shared_cache import AsyncContext

    EngineConfig.distributed_addr = property(
        lambda self: f"tcp://127.0.0.1:{distributed_port}"
    )

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before Mini-SGLang Engine construction")
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("This PyTorch build has no CUDA profiler activity support")

    print(
        f"Model: {args.model} | physical GPU: {args.gpu} | "
        f"torch: {torch.__version__} | CUDA build: {torch.version.cuda} | "
        f"memory ratio: {args.memory_ratio:.2f}"
    )
    print(f"Requested Gated DeltaNet backend: {args.gdn_backend}")

    llm = AsyncLLM(
        args.model,
        dtype=torch.bfloat16,
        max_running_req=1,
        memory_ratio=args.memory_ratio,
    )
    from minisgl.models import qwen3_5_delta

    if args.gdn_backend == "fla" and (
        qwen3_5_delta._fla_chunk is None or qwen3_5_delta._fla_recurrent is None
    ):
        await llm.close()
        raise RuntimeError("FLA Gated DeltaNet kernels are unavailable in this environment")
    if args.gdn_backend == "torch":
        qwen3_5_delta._fla_chunk = None
        qwen3_5_delta._fla_recurrent = None
    env = None
    try:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Gated DeltaNet backend: {args.gdn_backend}")
        if llm.processor is None:
            raise ValueError(f"{args.model} is not a supported multimodal model")

        def seed_everything(seed: int) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        cache = await llm.create_block()
        decode_context = AsyncContext(cache_view=[cache])
        env = gym.make("VizdoomBasic-v0", render_mode="rgb_array", frame_skip=8)
        seed_everything(args.seed)
        observation, _ = env.reset(seed=args.seed)

        def single_token_id(text: str) -> int:
            token_ids = llm.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Action {text!r} is not one token: {token_ids}")
            return int(token_ids[0])

        action_token_ids = [single_token_id(name) for name in ACTION_NAMES]
        eos_token_id = llm.config.generation_config.eos_token_id or llm.tokenizer.eos_token_id
        eos_token_ids = set(torch.as_tensor(eos_token_id).view(-1).tolist())
        async def run_decision(current_observation: dict) -> Decision:
            decision_started = time.perf_counter()
            with torch.profiler.record_function("doom.image_preprocessing"):
                model_inputs = llm.processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": current_observation["screen"]},
                                {"type": "text", "text": PROMPT},
                            ],
                        }
                    ],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                )

            with torch.profiler.record_function("doom.cache_reset"):
                cache.clear()
                decode_context.next_input_id = None

            with torch.profiler.record_function("doom.initial_prefill"):
                output = await llm(**model_inputs, cache_view=[cache])
            model_forwards = 1

            reasoning_tokens: list[str] = []
            reasoning_token_ids: list[int] = []
            with torch.profiler.record_function("doom.reasoning"):
                for token_index in range(args.max_reasoning_tokens):
                    with torch.profiler.record_function("doom.sample"):
                        new_token_id = int(
                            (
                                await llm.sample(
                                    output,
                                    temperature=0.7,
                                    top_k=-1,
                                    top_p=1.0,
                                )
                            ).item()
                        )
                        new_text = llm.tokenizer.decode([new_token_id])
                    reasoning_token_ids.append(new_token_id)
                    reasoning_tokens.append(new_text)
                    if new_token_id in eos_token_ids or "\n" in new_text:
                        break
                    decode_context.next_input_id = new_token_id
                    with torch.profiler.record_function(f"doom.decode.{token_index:03d}"):
                        output = await llm(cache_view=decode_context)
                    model_forwards += 1

            with torch.profiler.record_function("doom.action_probe"):
                finisher = llm.tokenizer(
                    "\nAction: `", return_tensors="pt", add_special_tokens=False
                )
                scores = (await llm(**finisher, cache_view=[cache])).logits.softmax(-1).flatten()
                model_forwards += 1
                action_scores_tensor = scores[action_token_ids]
                action_index = int(action_scores_tensor.argmax().item())
                action_scores = [float(score) for score in action_scores_tensor.tolist()]

            # Keep the complete GPU tail inside doom.full_step before stepping the CPU game.
            with torch.profiler.record_function("doom.cuda_synchronize"):
                torch.cuda.synchronize()
            with torch.profiler.record_function("doom.environment_step"):
                next_observation, reward, terminated, truncated, _ = env.step(action_index)

            return Decision(
                observation=next_observation,
                reasoning="".join(reasoning_tokens).strip(),
                reasoning_token_ids=reasoning_token_ids,
                action_name=ACTION_NAMES[action_index],
                action_index=action_index,
                action_scores=action_scores,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                wall_seconds=time.perf_counter() - decision_started,
                model_forwards=model_forwards,
            )

        if args.warmup_steps < 1:
            raise ValueError("--warmup-steps must be at least 1")
        print(f"Warmup: {args.warmup_steps} complete unrecorded Doom steps...")
        for warmup_index in range(args.warmup_steps):
            warmup = await run_decision(observation)
            observation = warmup.observation
            if warmup.terminated or warmup.truncated:
                observation, _ = env.reset(seed=args.seed + warmup_index + 1)
            torch.cuda.synchronize()
            print(
                f"  warmup {warmup_index + 1}/{args.warmup_steps}: "
                f"forwards={warmup.model_forwards}, wall={warmup.wall_seconds * 1e3:.2f} ms"
            )

        # Warmup may consume both environment and RNG state differently across
        # backends. Reset all of it so the profiled A/B input is identical.
        seed_everything(args.seed)
        observation, _ = env.reset(seed=args.seed)
        observation_sha256 = hashlib.sha256(
            memoryview(observation["screen"]).cast("B")
        ).hexdigest()
        torch.cuda.synchronize()
        print(f"Profile input: seed={args.seed}, screen_sha256={observation_sha256}")

        print("Profile: exactly one complete Doom step...")
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
        )
        profiler.start()
        try:
            with torch.profiler.record_function("doom.full_step"):
                decision = await run_decision(observation)
            torch.cuda.synchronize()
        finally:
            profiler.stop()

        args.trace = args.trace.expanduser().resolve()
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.trace))

        cuda_categories = count_trace_categories(args.trace)
        print(f"Reasoning: {decision.reasoning}")
        token_sha256 = hashlib.sha256(
            ",".join(map(str, decision.reasoning_token_ids)).encode()
        ).hexdigest()
        print(
            f"Reasoning tokens: count={len(decision.reasoning_token_ids)}, "
            f"sha256={token_sha256}"
        )
        print(
            "Scores: "
            + ", ".join(
                f"{name}={score:.4f}"
                for name, score in zip(ACTION_NAMES, decision.action_scores)
            )
        )
        print(
            f"Action: {decision.action_name} ({decision.action_index}) | "
            f"reward={decision.reward} | forwards={decision.model_forwards} | "
            f"wall={decision.wall_seconds * 1e3:.2f} ms"
        )
        print(
            "CUDA trace events: "
            + ", ".join(f"{category}={count}" for category, count in cuda_categories.items())
        )
        print(
            f"Chrome trace: {args.trace} "
            f"({args.trace.stat().st_size / (1024**2):.1f} MiB)"
        )
        if cuda_categories["kernel"] == 0:
            raise RuntimeError(
                "The trace has no CUDA kernel events. On B300 use a PyTorch CUDA 13.2 build; "
                "the cu128/cu130 Kineto builds on this host only recorded CPU activity."
            )
    finally:
        if env is not None:
            env.close()
        await llm.close()


def main() -> None:
    args = parse_args()
    distributed_port = configure_process(args)
    asyncio.run(async_main(args, distributed_port))


if __name__ == "__main__":
    main()
