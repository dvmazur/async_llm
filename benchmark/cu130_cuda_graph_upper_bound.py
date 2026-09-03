#!/usr/bin/env python3
"""Estimate an optimistic CUDA-graph ceiling from one eager decode trace."""

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
from torch.profiler import ProfilerActivity, profile

from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.shared_cache import SharedCacheSession, WorkerGroup
from minisgl.utils.hf import load_processor


MODEL = "/workspace/models/Qwen3.6-35B-A3B"
RESULTS = Path("/workspace/results")
BATCHES = (2, 8, 16)


def make_prompt(tokenizer, length: int = 256) -> torch.Tensor:
    text = (
        "Inspect the embodied observation history, reason about geometry and progress, "
        "then choose the next action. "
    ) * 64
    ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").view(-1)
    return ids[:length].to(dtype=torch.int32, device="cpu")


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


def parse_trace(path: Path) -> dict[str, object]:
    trace = json.loads(path.read_text())
    device_events = []
    kernel_events = []
    by_name: dict[str, list[float]] = defaultdict(list)
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
            kernel_events.append((start, duration))
            by_name[event.get("name", "unknown")].append(duration)

    if not device_events:
        raise RuntimeError("Profiler trace contains no GPU device activities")
    first = min(start for start, _ in device_events)
    last = max(start + duration for start, duration in device_events)
    span_us = last - first
    busy_union_us = merge_duration_us(device_events)
    kernel_sum_us = sum(duration for _, duration in kernel_events)
    top = sorted(
        (
            {
                "name": name,
                "calls": len(durations),
                "total_ms": sum(durations) / 1000,
            }
            for name, durations in by_name.items()
        ),
        key=lambda row: row["total_ms"],
        reverse=True,
    )[:12]
    return {
        "kernel_launches": len(kernel_events),
        "device_activity_events": len(device_events),
        "gpu_span_ms": span_us / 1000,
        "gpu_busy_union_ms": busy_union_us / 1000,
        "kernel_duration_sum_ms": kernel_sum_us / 1000,
        "gaps_inside_gpu_span_ms": (span_us - busy_union_us) / 1000,
        "gpu_busy_fraction_of_span": busy_union_us / span_us,
        "top_kernels": top,
    }


def profile_batch(session, common, tokenizer, batch_size: int) -> dict[str, object]:
    tails = [session.create_block() for _ in range(batch_size)]
    group = WorkerGroup(
        cache_structure=[[common, tail] for tail in tails],
        write_to=tails,
    )
    seeds = tokenizer.encode(
        " forward left right wait descend inspect turn continue choose safely",
        add_special_tokens=False,
    )
    current = torch.tensor([seeds[i % len(seeds)] for i in range(batch_size)], dtype=torch.int32)
    try:
        for _ in range(6):
            logits = session.decode_step(group, current)
            current = logits.argmax(dim=-1).to(torch.int32)
        torch.cuda.synchronize()

        started = time.perf_counter()
        for _ in range(16):
            logits = session.decode_step(group, current)
            current = logits.argmax(dim=-1).to(torch.int32)
        torch.cuda.synchronize()
        normal_wall_ms = (time.perf_counter() - started) / 16 * 1000

        # Prime CUPTI/profiler setup separately so its first-use cost is excluded.
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
            logits = session.decode_step(group, current)
            current = logits.argmax(dim=-1).to(torch.int32)
            torch.cuda.synchronize()

        with tempfile.TemporaryDirectory(prefix="minisgl-graph-trace-") as directory:
            trace_path = Path(directory) / f"batch_{batch_size}.json"
            profiled_started = time.perf_counter()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                logits = session.decode_step(group, current)
                current = logits.argmax(dim=-1).to(torch.int32)
                torch.cuda.synchronize()
            profiled_wall_ms = (time.perf_counter() - profiled_started) * 1000
            prof.export_chrome_trace(str(trace_path))
            parsed = parse_trace(trace_path)

        busy_ms = float(parsed["gpu_busy_union_ms"])
        gaps_ms = float(parsed["gaps_inside_gpu_span_ms"])
        # Two intentionally optimistic ceilings:
        # 1) graph-only: remove every gap observed between GPU activities but retain
        #    the normal wall time outside those gaps;
        # 2) absolute: pretend all non-device work disappears as well.
        graph_only_floor_ms = max(busy_ms, normal_wall_ms - gaps_ms)
        parsed.update(
            {
                "batch_size": batch_size,
                "normal_wall_ms_per_forward": normal_wall_ms,
                "profiled_wall_ms": profiled_wall_ms,
                "optimistic_graph_only_floor_ms": graph_only_floor_ms,
                "optimistic_graph_only_speedup": normal_wall_ms / graph_only_floor_ms,
                "absolute_device_only_floor_ms": busy_ms,
                "absolute_device_only_speedup": normal_wall_ms / busy_ms,
                "normal_aggregate_tokens_per_second": batch_size * 1000 / normal_wall_ms,
                "optimistic_graph_only_tokens_per_second": (
                    batch_size * 1000 / graph_only_floor_ms
                ),
                "absolute_device_only_tokens_per_second": batch_size * 1000 / busy_ms,
            }
        )
        return parsed
    finally:
        for tail in tails:
            session.free_block(tail)
        torch.cuda.empty_cache()


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = RESULTS / f"minisgl_cuda_graph_upper_bound_{stamp}.json"
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
        distributed_addr="tcp://127.0.0.1:2387",
    )
    print("Loading mini-sglang", flush=True)
    engine = Engine(config)
    session = SharedCacheSession(engine)
    tokenizer = load_processor(MODEL).tokenizer
    common = session.create_block()
    report: dict[str, object] = {
        "timestamp_utc": stamp,
        "model": MODEL,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "interpretation": {
            "graph_only": "removes all measured gaps inside GPU activity span",
            "device_only": "impossible best case removing all host and non-device time",
        },
    }
    try:
        session.prefill_block(common, make_prompt(tokenizer))
        rows = []
        for batch_size in BATCHES:
            print(f"PROFILE BATCH {batch_size}", flush=True)
            row = profile_batch(session, common, tokenizer, batch_size)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "top_kernels"}), flush=True)
        report["rows"] = rows
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"RESULT {result_path}", flush=True)
    finally:
        session.free_block(common)
        engine.shutdown()


if __name__ == "__main__":
    main()
