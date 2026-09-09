#!/usr/bin/env python3
"""Run five real async-v48 Speleo pipelines concurrently on one AsyncLLM.

The canonical prompt and async decision routine are loaded from
``minecraft_async_v48_dashboard_gif_eval100.ipynb``.  Each client owns an
independent environment, history, image pair and per-round cache graph.  Clients
advance hogwild (there is no cross-client action barrier), while one scheduler
continuously batches descriptor, thinker and meta-probe decode streams.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import numpy as np


WORKSPACE = Path("/workspace")
REPO = Path(__file__).resolve().parents[1]
# The canonical async notebook predates the self-contained path injection used
# by newer generated notebooks. A standalone runner must provide both trees.
for _local_python in (
    REPO / "python",
    WORKSPACE / "deps" / "craftium",
):
    if str(_local_python) not in sys.path:
        sys.path.insert(0, str(_local_python))
DEFAULT_NOTEBOOK = (
    WORKSPACE
    / "async_reasoning_minecraft_handoff_2026-08-31"
    / "notebooks"
    / "minecraft_async_v48_dashboard_gif_eval100.ipynb"
)
DEFAULT_RESULTS = WORKSPACE / "results" / "speleo_async_parallel_5seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--decisions", type=int, default=100)
    parser.add_argument(
        "--environment-max-steps",
        type=int,
        default=100,
        help="Gym TimeLimit, independent from a shorter runner smoke budget",
    )
    parser.add_argument("--sampling-seed", type=int, default=20260903)
    parser.add_argument("--mixed", action="store_true", help="partially mixed prefill/decode")
    parser.add_argument("--model", default="/workspace/models/Qwen3.6-35B-A3B-FP8")
    parser.add_argument(
        "--legacy-prefill",
        action="store_true",
        help="Ablate only the ragged GDN prefill batching while keeping this repo otherwise fixed",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="retain compact CPU+CUDA torch.profiler aggregates around the real action loops",
    )
    parser.add_argument(
        "--torch-profile-trace",
        action="store_true",
        help="also export the very large Chrome trace (off by default)",
    )
    parser.add_argument("--max-running-req", type=int, default=16)
    parser.add_argument(
        "--num-pages",
        type=int,
        default=65_536,
        help="shared KV token slots for five long histories and transient branches",
    )
    parser.add_argument(
        "--evict-checkpoint-page-cache-after-load",
        action="store_true",
        help=(
            "drop clean safetensors file pages after model load so unified-memory "
            "reclaim does not contaminate the first measured action"
        ),
    )
    parser.add_argument(
        "--batching-yield-rounds",
        type=int,
        default=3,
        help=(
            "event-loop turns allowed for newly-ready pipeline stages to enqueue "
            "before the next model batch"
        ),
    )
    args = parser.parse_args()
    if args.mixed and args.legacy_prefill:
        parser.error("legacy prefill ablation does not implement projected mixed inputs")
    args.seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique integers")
    # v48 has at most three simultaneous decode streams per client.
    if args.max_running_req < 3 * len(args.seeds):
        parser.error("--max-running-req must be at least 3 × number of clients")
    if (
        args.decisions <= 0
        or args.environment_max_steps <= 0
        or args.num_pages <= 0
        or args.batching_yield_rounds <= 0
    ):
        parser.error("decision/environment/page budgets must be positive")
    if args.environment_max_steps < args.decisions:
        parser.error("--environment-max-steps cannot be smaller than --decisions")
    return args


def evict_checkpoint_page_cache(model_path: str) -> None:
    """Drop only clean checkpoint pages; loaded CUDA tensors remain untouched."""

    model_dir = Path(model_path)
    shards = sorted(model_dir.glob("*.safetensors")) if model_dir.is_dir() else []
    for shard in shards:
        fd = os.open(shard, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    if shards:
        print(f"Evicted clean page cache for {len(shards)} checkpoint shards", flush=True)


def strip_ipython_magics(source: str) -> str:
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("%")
    )


def install_partial_mixed_profile_scopes():
    """Attribute real-run host/CUDA work without changing ordinary timings."""
    from functools import wraps
    import torch
    from minisgl.models.qwen3_5_delta import Qwen3_5GatedDeltaNet
    from minisgl.models.qwen3_5_moe import Qwen3_5MoeMLP
    from minisgl.models.qwen3_5_attn import Qwen3_5Attention
    from minisgl.shared_cache.gdn import SharedCacheGDN
    from minisgl.shared_cache.session import SharedCacheSession
    classes = {
        Qwen3_5GatedDeltaNet: ("_project_ar", "_ar_output", "_forward_ar_prefill", "_forward_ar_decode"),
        Qwen3_5MoeMLP: ("forward",),
        Qwen3_5Attention: ("forward",),
        SharedCacheGDN: ("compose_initial_recurrent_state", "capture_token_affines", "prior_conv_states",
                         "set_conv_states", "affine_scan_initial_state", "store_affine_scan_state"),
        SharedCacheSession: ("_prepare_prefill_batch", "_prepare_decode"),
    }
    for cls, methods in classes.items():
        for name in methods:
            original = getattr(cls, name)
            def scoped(self, *a, __fn=original, __label=f"partial::{cls.__name__}.{name}", **kw):
                with torch.profiler.record_function(__label):
                    return __fn(self, *a, **kw)
            setattr(cls, name, wraps(original)(scoped))


def load_runtime(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, list[tuple[int, Any, np.ndarray]]]:
    os.environ["ASYNC_MODEL"] = args.model
    notebook = json.loads(args.notebook.read_text())
    cells = notebook["cells"]
    setup = strip_ipython_magics("".join(cells[0]["source"]))
    running_anchor = "dtype=torch.bfloat16, max_running_req=4, memory_ratio=0.9,"
    if setup.count(running_anchor) != 1:
        raise RuntimeError("canonical max_running_req anchor changed")
    setup = setup.replace(
        running_anchor,
        "dtype=torch.bfloat16, "
        f"max_running_req={args.max_running_req}, memory_ratio=0.9,",
    )
    page_anchor = "num_page_override=32_768"
    if setup.count(page_anchor) != 1:
        raise RuntimeError("canonical num_page_override anchor changed")
    setup = setup.replace(page_anchor, f"num_page_override={args.num_pages}")

    # Minetest/OpenGL needs a small amount of unified GPU memory during launch.
    # Start every environment before AsyncLLM reserves almost all remaining
    # memory; otherwise the fifth process is intermittently unable to open its
    # Craftium socket on this GB10 machine.
    model_anchor = "llm = minisgl.llm.AsyncLLM("
    if setup.count(model_anchor) != 1:
        raise RuntimeError("canonical AsyncLLM construction anchor changed")
    model_offset = setup.index(model_anchor)
    namespace: dict[str, Any] = {"__name__": __name__, "__file__": str(args.notebook)}
    exec(compile(setup[:model_offset], f"{args.notebook}#environment_setup", "exec"), namespace)
    prepared: list[tuple[int, Any, np.ndarray]] = []
    try:
        for seed in args.seeds:
            env = namespace["make_env"](
                "speleo",
                max_steps=args.environment_max_steps,
                **namespace["SPELEO_ENV_CONFIG"],
            )
            obs = env.reset(seed=seed)
            prepared.append((seed, env, obs))
        exec(compile(setup[model_offset:], f"{args.notebook}#model_setup", "exec"), namespace)
        llm = namespace["llm"]
        llm.batching_yield_rounds = args.batching_yield_rounds
        llm.async_engine.enable_mixed_batch = args.mixed
        # The stable notebook counts separate session calls. Extend its counter
        # locally; do not change the notebook or count a mixed pass twice.
        counter = namespace["batched_forward_counter"]
        type(counter)._DELTA_KEYS += ("mixed",)
        counter.counts["mixed"] = 0
        original_mixed = llm.async_engine.session.mixed_step
        def counted_mixed(jobs, group, input_ids):
            result = original_mixed(jobs, group, input_ids)
            counter.counts["mixed"] += 1
            counter.counts["prefill"] += 1
            counter.counts["decode"] += 1
            counter.counts["total"] += 1
            counter.counts["prefill_jobs"] += len(jobs)
            counter.counts["prefill_input_tokens"] += sum(int(j.input_ids.numel()) for j in jobs)
            counter.counts["decode_tokens"] += int(input_ids.numel())
            return result
        llm.async_engine.session.mixed_step = counted_mixed
        if args.torch_profile:
            install_partial_mixed_profile_scopes()
        exec(
            compile("".join(cells[1]["source"]), f"{args.notebook}#prompt", "exec"),
            namespace,
        )
    except BaseException:
        for _, env, _ in reversed(prepared):
            try:
                env.close()
            except Exception:
                pass
        raise

    rollout = "".join(cells[2]["source"])
    helper_start = rollout.index("READINESS_INTERVAL =")
    helper_end = rollout.index("try:\n    for i in range(MAX_DECISIONS):", helper_start)
    return namespace, rollout[helper_start:helper_end], prepared


class Slot:
    def __init__(self):
        self.value = ""


class HeadlessDashboard:
    def __init__(self):
        self.descriptions = Slot()
        self.actions = Slot()
        self.probe = Slot()


class NoopGif:
    def capture(self, event: str, hold: int = 1) -> None:
        return None


class BatchObserver:
    def __init__(self, llm: Any):
        self.decode: Counter[int] = Counter()
        self.prefill: Counter[int] = Counter()
        self.prefill_shapes: Counter[tuple[int, ...]] = Counter()
        self.prefill_lengths: Counter[int] = Counter()
        self.prefill_job_records: list[list[dict[str, Any]]] = []
        self.forward_timeline: list[dict[str, Any]] = []
        self.scheduler_ticks: list[dict[str, Any]] = []
        self.timeline_origin = None
        session = llm.async_engine.session
        async_engine = llm.async_engine
        original_decode = session.decode_step
        original_prefill = session.prefill_batch
        original_mixed = session.mixed_step
        original_tick = async_engine.tick

        def compatible_prefill_prefix_len() -> int:
            count = 0
            write_ids = set()
            context_ids = set()
            for req in async_engine._prefill_queue:
                write_id = id(req.write_to)
                ctx_ids = {id(block) for block in req.context}
                if count and (
                    write_id in write_ids
                    or write_id in context_ids
                    or ctx_ids & write_ids
                ):
                    break
                count += 1
                write_ids.add(write_id)
                context_ids |= ctx_ids
            return count

        def tick():
            prefill_queue = len(async_engine._prefill_queue)
            decode_queue = len(async_engine._decode_queue)
            record = {
                    "prefill_queue": prefill_queue,
                    "compatible_prefill_prefix": compatible_prefill_prefix_len(),
                    "prefill_input_tokens": sum(
                        int(req.input_ids.numel())
                        for req in async_engine._prefill_queue
                    ),
                    "decode_queue": decode_queue,
                    "selected": None,
                }
            result = original_tick()
            record["selected"] = result
            self.scheduler_ticks.append(record)
            return result

        async_engine.tick = tick

        def decode(group, input_ids):
            self.decode[int(input_ids.numel())] += 1
            return self._timed_forward(
                "decode", int(input_ids.numel()), lambda: original_decode(group, input_ids)
            )

        def record_prefill(jobs):
            self.prefill[len(jobs)] += 1
            shape = tuple(int(job.input_ids.numel()) for job in jobs)
            self.prefill_shapes[shape] += 1
            self.prefill_lengths.update(shape)
            self.prefill_job_records.append(
                [
                    {
                        "new_tokens": int(job.input_ids.numel()),
                        "write_prefix_tokens": int(job.block.num_tokens),
                        "context_block_tokens": [
                            int(block.num_tokens)
                            for block in job.context
                            if block.num_tokens > 0
                        ],
                        "has_pixels": job.pixel_values is not None,
                        "has_image_embeds": job.image_embeds is not None,
                    }
                    for job in jobs
                ]
            )
            return shape

        def prefill(jobs):
            shape = record_prefill(jobs)
            return self._timed_forward(
                "prefill", sum(shape), lambda: original_prefill(jobs)
            )

        def mixed(jobs, group, input_ids):
            shape = record_prefill(jobs)
            self.decode[int(input_ids.numel())] += 1
            return self._timed_forward(
                "mixed", sum(shape) + int(input_ids.numel()),
                lambda: original_mixed(jobs, group, input_ids),
            )

        session.decode_step = decode
        session.prefill_batch = prefill
        session.mixed_step = mixed

    def _timed_forward(self, kind: str, rows: int, call):
        import torch

        started = time.perf_counter()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # This scope is inert unless torch.profiler is active, and lets the
        # exported Chrome trace attribute launches to real prefill vs decode.
        with torch.profiler.record_function(f"minisgl::{kind}_forward"):
            result = call()
        end_event.record()
        ended = time.perf_counter()
        self.forward_timeline.append(
            {
                "kind": kind,
                "rows": rows,
                "host_start": started,
                "host_end": ended,
                "start_event": start_event,
                "end_event": end_event,
                "allocated_before": allocated_before,
                "allocated_after": torch.cuda.memory_allocated(),
                "reserved_before": reserved_before,
                "reserved_after": torch.cuda.memory_reserved(),
            }
        )
        return result

    def clear(self) -> None:
        import torch

        self.decode.clear()
        self.prefill.clear()
        self.prefill_shapes.clear()
        self.prefill_lengths.clear()
        self.prefill_job_records.clear()
        self.forward_timeline.clear()
        self.scheduler_ticks.clear()
        self.timeline_origin = torch.cuda.Event(enable_timing=True)
        self.timeline_origin.record()

    def timeline_summary(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        import torch

        torch.cuda.synchronize()
        if not self.forward_timeline:
            return {}, []
        assert self.timeline_origin is not None
        rows = []
        previous_host_end = None
        previous_cuda_end_ms = None
        for index, item in enumerate(self.forward_timeline):
            cuda_start_ms = self.timeline_origin.elapsed_time(item["start_event"])
            cuda_end_ms = self.timeline_origin.elapsed_time(item["end_event"])
            host_gap = (
                0.0 if previous_host_end is None else item["host_start"] - previous_host_end
            )
            cuda_gap_ms = (
                0.0
                if previous_cuda_end_ms is None
                else max(0.0, cuda_start_ms - previous_cuda_end_ms)
            )
            row = {
                "index": index,
                "kind": item["kind"],
                "rows": item["rows"],
                "host_call_seconds": item["host_end"] - item["host_start"],
                "host_gap_before_seconds": host_gap,
                "cuda_start_ms": cuda_start_ms,
                "cuda_end_ms": cuda_end_ms,
                "cuda_forward_span_ms": cuda_end_ms - cuda_start_ms,
                "cuda_gap_before_ms": cuda_gap_ms,
                "allocated_before_bytes": item["allocated_before"],
                "allocated_after_bytes": item["allocated_after"],
                "reserved_before_bytes": item["reserved_before"],
                "reserved_after_bytes": item["reserved_after"],
            }
            rows.append(row)
            previous_host_end = item["host_end"]
            previous_cuda_end_ms = cuda_end_ms

        by_kind = {}
        for kind in ("prefill", "decode", "mixed"):
            selected = [row for row in rows if row["kind"] == kind]
            by_kind[kind] = {
                "calls": len(selected),
                "rows": sum(row["rows"] for row in selected),
                "host_call_seconds": sum(row["host_call_seconds"] for row in selected),
                "cuda_forward_span_ms": sum(row["cuda_forward_span_ms"] for row in selected),
            }
        summary = {
            "calls": len(rows),
            "host_observed_span_seconds": (
                self.forward_timeline[-1]["host_end"]
                - self.forward_timeline[0]["host_start"]
            ),
            "host_inside_forward_calls_seconds": sum(
                row["host_call_seconds"] for row in rows
            ),
            "host_between_forward_gaps_seconds": sum(
                row["host_gap_before_seconds"] for row in rows
            ),
            "cuda_observed_span_ms": rows[-1]["cuda_end_ms"] - rows[0]["cuda_start_ms"],
            "cuda_inside_forward_spans_ms": sum(
                row["cuda_forward_span_ms"] for row in rows
            ),
            "cuda_between_forward_gaps_ms": sum(row["cuda_gap_before_ms"] for row in rows),
            "by_kind": by_kind,
        }
        return summary, rows


def legacy_ar_prefill(self, x, ar):
    """Exact per-request GDN prefill dispatch used before ragged batching."""
    import torch

    lin = self._lin_idx
    prior_conv = ar.prior_conv_states(lin)
    initial_state = ar.compose_initial_recurrent_state(
        lin, dtype=torch.float32, state_v_first=True
    )
    segments = ar.prefill_segments
    if segments is None or len(segments) == 1:
        return self._forward_ar_prefill_one(x, ar, 0, prior_conv, initial_state)
    outputs = []
    offset = 0
    for worker, length in enumerate(segments):
        outputs.append(
            self._forward_ar_prefill_one(
                x[offset : offset + length], ar, worker, prior_conv, initial_state
            )
        )
        offset += length
    assert offset == x.shape[0]
    return torch.cat(outputs)


@dataclass
class Client:
    seed: int
    env: Any
    obs: np.ndarray
    image_pair: list[np.ndarray]
    history: Any
    image_block: Any
    namespace: dict[str, Any]
    log_path: Path
    initial_frame_sha1: str
    initial_player_pos: list[float] | None
    trajectory: list[dict[str, Any]] = field(default_factory=list)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


async def run(
    args: argparse.Namespace,
    base: dict[str, Any],
    helpers: str,
    prepared: list[tuple[int, Any, np.ndarray]],
    *, close_llm: bool = True,
) -> Path:
    torch = base["torch"]
    llm = base["llm"]
    prompting = base["prompting"]
    tokenizer_kwargs = base["tokenizer_kwargs"]
    forward_counter = base["batched_forward_counter"]

    random.seed(args.sampling_seed)
    np.random.seed(args.sampling_seed)
    torch.manual_seed(args.sampling_seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.results_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    observer = BatchObserver(llm)
    clients: list[Client] = []
    cache_prompt = await llm.create_block()

    async def one_action(client: Client, round_index: int) -> dict[str, Any]:
        started = time.perf_counter()
        stage_started = started
        await llm.free_block(client.image_block)
        await llm(
            **prompting.get_image_dict(
                *client.image_pair, without_eot=True, end_previous_turn=True
            ),
            cache_view=[cache_prompt, client.history, client.image_block],
            return_logits=False,
        )
        image_prefill_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        decide = client.namespace["think_and_choose_action"]
        action_index, description, thought, round_blocks = await decide(
            client.history, round_index
        )
        decision_generation_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        obs, reward, done, info = await asyncio.to_thread(client.env.step, action_index)
        env_step_seconds = time.perf_counter() - stage_started
        action_name = prompting.action_names[action_index]
        player_y = -float(reward)
        outcome = f"reward={reward:+.3f}, player_y={player_y:+.3f}, done={done}"
        sanitize = client.namespace["sanitize_thought_for_history"]
        clean_thought = sanitize(thought)
        stage_started = time.perf_counter()
        await client.namespace["commit_round"](
            round_index, description, clean_thought, action_name, outcome
        )
        history_commit_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        await client.namespace["free_round_blocks"](round_blocks)
        round_block_free_seconds = time.perf_counter() - stage_started
        client.image_pair = [client.image_pair[-1], obs]
        client.obs = obs
        row = {
            "seed": client.seed,
            "step": round_index + 1,
            "action": action_name,
            "action_index": int(action_index),
            "reward": float(reward),
            "player_y": player_y,
            "done": bool(done),
            "description": description.strip(),
            "thought": clean_thought.strip(),
            "client_seconds": time.perf_counter() - started,
            "stage_seconds": {
                "image_prefill": image_prefill_seconds,
                "decision_generation": decision_generation_seconds,
                "env_step": env_step_seconds,
                "history_commit": history_commit_seconds,
                "round_block_free": round_block_free_seconds,
            },
        }
        client.trajectory.append(row)
        with client.log_path.open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"seed={client.seed} step={round_index + 1}/{args.decisions} "
            f"action={action_name} y={player_y:+.3f}",
            flush=True,
        )
        return row

    async def client_loop(client: Client) -> None:
        # No barrier: a fast client immediately begins its next observation/action.
        for round_index in range(args.decisions):
            row = await one_action(client, round_index)
            if row["done"]:
                break

    try:
        for seed, env, obs in prepared:
            history, image_block = await asyncio.gather(
                llm.create_block(), llm.create_block()
            )
            client_namespace = dict(base)
            client_namespace.update(
                {
                    "cache_prompt": cache_prompt,
                    "cache_history": history,
                    "cache_image_pair": image_block,
                    "dashboard": HeadlessDashboard(),
                    "dashboard_gif": NoopGif(),
                }
            )
            exec(
                compile(helpers, f"{args.notebook}#async_helpers_seed{seed}", "exec"),
                client_namespace,
            )
            log_path = run_dir / f"seed{seed}.jsonl"
            log_path.write_text("")
            clients.append(
                Client(
                    seed=seed,
                    env=env,
                    obs=obs,
                    image_pair=[obs, obs],
                    history=history,
                    image_block=image_block,
                    namespace=client_namespace,
                    log_path=log_path,
                    initial_frame_sha1=hashlib.sha1(np.asarray(obs).tobytes()).hexdigest(),
                    initial_player_pos=(
                        np.asarray(env.info["player_pos"], dtype=float).tolist()
                        if "player_pos" in env.info
                        else None
                    ),
                )
            )

        await llm(
            **llm.tokenizer(prompting.common_prompt, **tokenizer_kwargs),
            cache_view=[cache_prompt],
            return_logits=False,
        )
        await asyncio.gather(
            *(
                llm(
                    **llm.tokenizer(prompting.history_init, **tokenizer_kwargs),
                    cache_view=[cache_prompt, client.history],
                    return_logits=False,
                )
                for client in clients
            )
        )

        observer.clear()
        before = forward_counter.snapshot()
        started = time.perf_counter()
        profile_path = None
        profile_trace = None
        if args.torch_profile:
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            ) as profiler:
                await asyncio.gather(*(client_loop(client) for client in clients))
                torch.cuda.synchronize()
            profile_path = run_dir / "real_pipeline_torch_profile.json"
            profile_rows = []
            for event in profiler.key_averages():
                profile_rows.append(
                    {
                        "name": event.key,
                        "count": event.count,
                        "self_cpu_time_us": event.self_cpu_time_total,
                        "cpu_time_us": event.cpu_time_total,
                        "self_cuda_time_us": getattr(event, "self_device_time_total", 0.0),
                        "cuda_time_us": getattr(event, "device_time_total", 0.0),
                    }
                )
            profile_path.write_text(
                json.dumps(profile_rows, indent=2, ensure_ascii=False) + "\n"
            )
            if args.torch_profile_trace:
                profile_trace = run_dir / "real_pipeline_cpu_cuda_trace.json"
                profiler.export_chrome_trace(str(profile_trace))
        else:
            await asyncio.gather(*(client_loop(client) for client in clients))
        wall_seconds = time.perf_counter() - started
        after = forward_counter.snapshot()
        forwards = forward_counter.delta(before, after)
        total_actions = sum(len(client.trajectory) for client in clients)
        decode_hist = dict(sorted(observer.decode.items()))
        prefill_hist = dict(sorted(observer.prefill.items()))
        eligible_shapes = {
            shape: calls
            for shape, calls in observer.prefill_shapes.items()
            if len(shape) > 1 and all(length == 1 for length in shape)
        }
        eligible_calls = sum(eligible_shapes.values())
        eligible_jobs = sum(len(shape) * calls for shape, calls in eligible_shapes.items())
        observed_prefill_rows = sum(
            sum(shape) * calls for shape, calls in observer.prefill_shapes.items()
        )
        if sum(observer.decode.values()) != forwards["decode"]:
            raise RuntimeError("decode histogram call count disagrees with forward counter")
        if sum(size * calls for size, calls in observer.decode.items()) != forwards["decode_tokens"]:
            raise RuntimeError("decode histogram token count disagrees with forward counter")
        timeline_summary, timeline_rows = observer.timeline_summary()
        memory_stats = torch.cuda.memory_stats()
        compose_cache = llm.async_engine.session.sc_gdn.compose_state_cache
        compose_cache_stats = None
        if compose_cache is not None:
            compose_cache_stats = {
                **compose_cache.stats,
                "resident_entries": compose_cache.resident_entries,
                "resident_bytes": compose_cache.resident_bytes,
                "max_bytes": compose_cache.max_bytes,
            }
        timeline_path = run_dir / "model_forward_timeline.json"
        timeline_path.write_text(
            json.dumps(timeline_rows, indent=2, ensure_ascii=False) + "\n"
        )

        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "policy": "async_v48_five_client_hogwild",
            "mixed_prefill_decode": args.mixed,
            "counter_semantics": "prefill/decode include mixed; total counts each physical forward once",
            "notebook_source": str(args.notebook),
            "model": os.environ["ASYNC_MODEL"],
            "environment_seeds": list(args.seeds),
            "sampling_seed": args.sampling_seed,
            "gdn_prefill_impl": "legacy_per_request" if args.legacy_prefill else "ragged",
            "pipelines": len(clients),
            "max_simultaneous_decode_streams_per_pipeline": 3,
            "max_expected_decode_batch": 3 * len(clients),
            "decision_budget_per_pipeline": args.decisions,
            "environment_max_steps": args.environment_max_steps,
            "max_running_req": args.max_running_req,
            "scheduler_policy": {
                "batching_yield_rounds": args.batching_yield_rounds,
            },
            "num_pages": args.num_pages,
            "total_actions": total_actions,
            "wall_seconds": wall_seconds,
            "aggregate_actions_per_second": safe_ratio(total_actions, wall_seconds),
            "batched_forwards": forwards,
            "decode_batch_histogram": decode_hist,
            "prefill_jobs_per_batch_histogram": prefill_hist,
            "prefill_segment_shape_histogram": {
                ",".join(map(str, shape)): calls
                for shape, calls in sorted(observer.prefill_shapes.items())
            },
            "prefill_segment_length_histogram": {
                str(length): calls for length, calls in sorted(observer.prefill_lengths.items())
            },
            "prefill_job_records": observer.prefill_job_records,
            "batched_one_token_prefill": {
                "eligible_forward_count": eligible_calls,
                "eligible_job_count": eligible_jobs,
                "eligible_forward_fraction": safe_ratio(
                    eligible_calls, sum(observer.prefill_shapes.values())
                ),
                "eligible_job_fraction": safe_ratio(eligible_jobs, forwards["prefill_jobs"]),
                "eligible_input_row_fraction": safe_ratio(
                    eligible_jobs, observed_prefill_rows
                ),
            },
            "mean_decode_batch_size": safe_ratio(
                forwards["decode_tokens"], forwards["decode"]
            ),
            "mean_prefill_jobs_per_batch": safe_ratio(
                forwards["prefill_jobs"], forwards["prefill"]
            ),
            "decode_forwards_per_environment_action": safe_ratio(
                forwards["decode"], total_actions
            ),
            "decode_tokens_per_environment_action": safe_ratio(
                forwards["decode_tokens"], total_actions
            ),
            "model_forward_timeline": timeline_summary,
            "model_forward_timeline_path": str(timeline_path),
            "scheduler_ticks": observer.scheduler_ticks,
            "scheduler_contention": {
                "ticks": len(observer.scheduler_ticks),
                "both_queues_nonempty": sum(
                    tick["prefill_queue"] > 0 and tick["decode_queue"] > 0
                    for tick in observer.scheduler_ticks
                ),
                "singleton_prefill_selected_over_decode": sum(
                    tick["compatible_prefill_prefix"] == 1
                    and tick["decode_queue"] > 0
                    and tick["selected"] == "prefill"
                    for tick in observer.scheduler_ticks
                ),
            },
            "cuda_allocator": {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "num_alloc_retries": memory_stats.get("num_alloc_retries", 0),
                "num_ooms": memory_stats.get("num_ooms", 0),
            },
            "compose_cache": compose_cache_stats,
            "client_seconds": distribution(
                [row["client_seconds"] for client in clients for row in client.trajectory]
            ),
            "stage_seconds": {
                stage: distribution(
                    [
                        row["stage_seconds"][stage]
                        for client in clients
                        for row in client.trajectory
                    ]
                )
                for stage in (
                    "image_prefill",
                    "decision_generation",
                    "env_step",
                    "history_commit",
                    "round_block_free",
                )
            },
            "torch_profile": None if profile_path is None else str(profile_path),
            "torch_profile_trace": None if profile_trace is None else str(profile_trace),
            "clients": [
                {
                    "seed": client.seed,
                    "initial_frame_sha1": client.initial_frame_sha1,
                    "initial_player_pos": client.initial_player_pos,
                    "actions": len(client.trajectory),
                    "final_player_y": (
                        client.trajectory[-1]["player_y"] if client.trajectory else None
                    ),
                    "lowest_player_y": min(
                        (row["player_y"] for row in client.trajectory), default=None
                    ),
                    "trajectory_jsonl": str(client.log_path),
                    "trajectory": client.trajectory,
                }
                for client in clients
            ],
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

        plt = base["plt"]
        fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
        for client in clients:
            ax.plot(
                [row["step"] for row in client.trajectory],
                [row["player_y"] for row in client.trajectory],
                label=f"seed {client.seed}",
                linewidth=1.8,
            )
        ax.set_title("Five concurrent async-v48 Speleo pipelines")
        ax.set_xlabel("Environment action per pipeline")
        ax.set_ylabel("player_y (lower is deeper)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "height_by_seed.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
        sizes = sorted(observer.decode)
        ax.bar([str(size) for size in sizes], [observer.decode[size] for size in sizes])
        ax.set_title("Actual async-v48 decode batch sizes")
        ax.set_xlabel("Decode tokens in one model forward")
        ax.set_ylabel("Forward count")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(run_dir / "decode_batch_histogram.png", bbox_inches="tight")
        plt.close(fig)

        print(
            "SUMMARY",
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "total_actions",
                        "wall_seconds",
                        "batched_forwards",
                        "decode_batch_histogram",
                        "mean_decode_batch_size",
                        "decode_forwards_per_environment_action",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        print(f"SUMMARY_PATH {summary_path}", flush=True)
        return summary_path
    finally:
        for _, env, _ in reversed(prepared):
            try:
                env.close()
            except Exception as exc:
                print(f"warning: failed to close environment: {exc}", file=sys.stderr)
        for client in clients:
            for block in (client.image_block, client.history):
                try:
                    await llm.free_block(block)
                except Exception:
                    pass
        try:
            await llm.free_block(cache_prompt)
        except Exception:
            pass
        if close_llm:
            await llm.close()


def main() -> None:
    args = parse_args()
    args.notebook = args.notebook.resolve()
    base, helpers, prepared = load_runtime(args)
    if args.evict_checkpoint_page_cache_after_load:
        evict_checkpoint_page_cache(os.environ["ASYNC_MODEL"])
    if not args.legacy_prefill:
        asyncio.run(run(args, base, helpers, prepared))
        return
    from minisgl.models.qwen3_5_delta import Qwen3_5GatedDeltaNet

    optimized = Qwen3_5GatedDeltaNet._forward_ar_prefill
    Qwen3_5GatedDeltaNet._forward_ar_prefill = legacy_ar_prefill
    try:
        asyncio.run(run(args, base, helpers, prepared))
    finally:
        Qwen3_5GatedDeltaNet._forward_ar_prefill = optimized


if __name__ == "__main__":
    main()
