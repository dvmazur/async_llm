#!/usr/bin/env python3
"""Profile mini-sglang's batch-64 cached-prefix extend and decode paths."""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")
os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/.triton-cache-cu130")

import torch
import transformers
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.shared_cache import PrefillJob, SharedCacheSession, WorkerGroup
from minisgl.utils.hf import load_processor
from torch.profiler import ProfilerActivity, profile

MODEL = "/workspace/models/Qwen3.6-35B-A3B"
RESULTS = Path("/workspace/results")
BATCH_SIZE = 64
COMMON_TOKENS = 256


def merge_duration_us(events: list[tuple[float, float]]) -> float:
    if not events:
        return 0.0
    intervals = sorted((start, start + duration) for start, duration in events)
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def kernel_category(name: str) -> str:
    lower = name.lower()
    if "fused_moe" in lower or "moe" in lower:
        return "moe"
    if "cutlass" in lower or "gemm" in lower or "cublas" in lower:
        return "gemm"
    if "gdn" in lower or "gated_delta" in lower or "recurrent" in lower:
        return "gdn"
    if "attention" in lower or "flash" in lower:
        return "attention"
    # PyTorch names many copies ``elementwise_kernel<...direct_copy...>``.  Test
    # copy markers first or those memory movements are silently misclassified.
    if "copy" in lower or "memcpy" in lower or "memset" in lower:
        return "copy_or_set"
    if "elementwise" in lower or "vectorized" in lower:
        return "elementwise"
    if "reduce" in lower or "softmax" in lower or "topk" in lower or "sort" in lower:
        return "routing_or_reduction"
    return "other"


def parse_trace(path: Path) -> dict[str, object]:
    trace = json.loads(path.read_text())
    device_events: list[tuple[float, float]] = []
    kernels: list[tuple[float, float]] = []
    by_name: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)
    category_calls: dict[str, int] = defaultdict(int)
    for event in trace["traceEvents"]:
        if event.get("ph") != "X" or event.get("cat") not in {
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        }:
            continue
        duration = float(event.get("dur", 0.0))
        start = float(event.get("ts", 0.0))
        device_events.append((start, duration))
        if event.get("cat") == "kernel":
            name = event.get("name", "unknown")
            kernels.append((start, duration))
            by_name[name].append(duration)
            category = kernel_category(name)
            by_category[category].append(duration)
            category_calls[category] += 1
        else:
            by_category["copy_or_set"].append(duration)
            category_calls["copy_or_set"] += 1
    if not device_events:
        raise RuntimeError("No GPU activities in trace")
    first = min(start for start, _ in device_events)
    last = max(start + duration for start, duration in device_events)
    busy = merge_duration_us(device_events)
    categories = sorted(
        (
            {
                "category": category,
                "calls": category_calls[category],
                "total_ms": sum(durations) / 1000,
            }
            for category, durations in by_category.items()
        ),
        key=lambda row: row["total_ms"],
        reverse=True,
    )
    top = sorted(
        (
            {"name": name, "calls": len(ds), "total_ms": sum(ds) / 1000}
            for name, ds in by_name.items()
        ),
        key=lambda row: row["total_ms"],
        reverse=True,
    )[:20]
    return {
        "kernel_launches": len(kernels),
        "device_activity_events": len(device_events),
        "gpu_span_ms": (last - first) / 1000,
        "gpu_busy_union_ms": busy / 1000,
        "gpu_idle_gaps_inside_span_ms": (last - first - busy) / 1000,
        "gpu_busy_fraction": busy / (last - first),
        "kernel_duration_sum_ms": sum(duration for _, duration in kernels) / 1000,
        "categories": categories,
        "top_kernels": top,
    }


def make_common(tokenizer) -> torch.Tensor:
    text = (
        "Inspect the embodied observation history, reason about geometry and progress, "
        "then choose the next action. "
    ) * 64
    ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").view(-1)
    return ids[:COMMON_TOKENS].to(dtype=torch.int32, device="cpu")


def create_branch(session, common, seed_offset: int):
    prefixes = [session.create_block() for _ in range(BATCH_SIZE)]
    tails = [session.create_block() for _ in range(BATCH_SIZE)]
    jobs = [
        PrefillJob(
            block=prefix,
            input_ids=torch.tensor([10_000 + seed_offset + i], dtype=torch.int32),
            context=[common],
        )
        for i, prefix in enumerate(prefixes)
    ]
    return prefixes, tails, jobs


def prefill_and_group(session, common, prefixes, tails, jobs):
    first_logits = session.prefill_batch(jobs)
    current = torch.stack([rows[0].argmax() for rows in first_logits]).to(torch.int32)
    group = WorkerGroup(
        cache_structure=[
            [common, prefix, tail] for prefix, tail in zip(prefixes, tails)
        ],
        write_to=tails,
    )
    return current, group


def free_branch(session, prefixes, tails):
    for block in [*tails, *prefixes]:
        session.free_block(block)
    torch.cuda.empty_cache()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = RESULTS / f"minisgl_batch64_bottlenecks_{stamp}.json"
    config = EngineConfig(
        model_path=MODEL,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=4,
        memory_ratio=0.9,
        max_seq_len_override=4096,
        num_page_override=8192,
        attention_backend="fi",
        generation_config=transformers.GenerationConfig(
            do_sample=False, temperature=None, top_k=None, top_p=None
        ),
        distributed_addr="tcp://127.0.0.1:2393",
    )
    print("Loading mini-sglang", flush=True)
    load_started = time.perf_counter()
    engine = Engine(config)
    session = SharedCacheSession(engine)
    tokenizer = load_processor(MODEL).tokenizer
    common = session.create_block()
    report: dict[str, object] = {
        "timestamp_utc": stamp,
        "model": MODEL,
        "batch_size": BATCH_SIZE,
        "common_tokens": COMMON_TOKENS,
        "load_seconds": time.perf_counter() - load_started,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    try:
        session.prefill_block(common, make_common(tokenizer))

        # Shape warm-up, excluded.
        prefixes, tails, jobs = create_branch(session, common, 1_000)
        current, group = prefill_and_group(session, common, prefixes, tails, jobs)
        for _ in range(4):
            current = session.decode_step(group, current).argmax(dim=-1).to(torch.int32)
        torch.cuda.synchronize()
        free_branch(session, prefixes, tails)

        extend_samples = []
        decode_samples = []
        for sample in range(2):
            prefixes, tails, jobs = create_branch(session, common, 2_000 + sample * 100)
            torch.cuda.synchronize()
            started = time.perf_counter()
            current, group = prefill_and_group(session, common, prefixes, tails, jobs)
            torch.cuda.synchronize()
            extend_samples.append(time.perf_counter() - started)

            started = time.perf_counter()
            for _ in range(8):
                current = (
                    session.decode_step(group, current).argmax(dim=-1).to(torch.int32)
                )
            torch.cuda.synchronize()
            decode_samples.append((time.perf_counter() - started) / 8)
            free_branch(session, prefixes, tails)

        report["normal_timing"] = {
            "extend_sample_seconds": extend_samples,
            "mean_extend_seconds": statistics.fmean(extend_samples),
            "decode_seconds_per_forward_samples": decode_samples,
            "mean_decode_seconds_per_forward": statistics.fmean(decode_samples),
        }
        print("TIMING", json.dumps(report["normal_timing"]), flush=True)

        # Prime CUPTI outside the retained traces.
        prefixes, tails, jobs = create_branch(session, common, 3_000)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
            current, group = prefill_and_group(session, common, prefixes, tails, jobs)
            torch.cuda.synchronize()
        free_branch(session, prefixes, tails)

        with tempfile.TemporaryDirectory(prefix="minisgl-b64-profile-") as tmp:
            tmp = Path(tmp)
            prefixes, tails, jobs = create_branch(session, common, 4_000)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=False,
                record_shapes=False,
            ) as prof:
                current, group = prefill_and_group(
                    session, common, prefixes, tails, jobs
                )
                torch.cuda.synchronize()
            extend_trace = tmp / "extend.json"
            prof.export_chrome_trace(str(extend_trace))
            report["extend_profile"] = parse_trace(extend_trace)

            # The previous operation gives us a valid branch for a decode trace.
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=False,
                record_shapes=False,
            ) as prof:
                current = (
                    session.decode_step(group, current).argmax(dim=-1).to(torch.int32)
                )
                torch.cuda.synchronize()
            decode_trace = tmp / "decode.json"
            prof.export_chrome_trace(str(decode_trace))
            report["decode_profile"] = parse_trace(decode_trace)
            free_branch(session, prefixes, tails)

        result_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            "EXTEND_PROFILE",
            json.dumps(
                {
                    k: v
                    for k, v in report["extend_profile"].items()
                    if k != "top_kernels"
                }
            ),
            flush=True,
        )
        print(
            "DECODE_PROFILE",
            json.dumps(
                {
                    k: v
                    for k, v in report["decode_profile"].items()
                    if k != "top_kernels"
                }
            ),
            flush=True,
        )
        print(f"RESULT {result_path}", flush=True)
    finally:
        session.free_block(common)
        engine.shutdown()


if __name__ == "__main__":
    main()
