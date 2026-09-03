#!/usr/bin/env python3
"""Measure large decode batches through mini-sglang's shared-cache path."""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")
os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/.triton-cache-cu130")

import torch
import transformers

from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.shared_cache import SharedCacheSession, WorkerGroup
from minisgl.utils.hf import load_processor


MODEL = "/workspace/models/Qwen3.6-35B-A3B"
RESULTS = Path("/workspace/results")
BATCHES = (8, 16, 32, 64)
PROMPT_TOKENS = 256
WARMUP_STEPS = 4
MEASURED_STEPS = 32


def memory_gib() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {"free": free / 2**30, "total": total / 2**30}


def make_prompt(tokenizer) -> torch.Tensor:
    text = (
        "Inspect the embodied observation history, reason about geometry and progress, "
        "then choose the next action. "
    ) * 64
    ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").view(-1)
    if ids.numel() < PROMPT_TOKENS:
        raise RuntimeError(f"Only produced {ids.numel()} prompt tokens")
    return ids[:PROMPT_TOKENS].to(dtype=torch.int32, device="cpu")


def run_batch(session, common, tokenizer, batch_size: int) -> dict[str, object]:
    tails = [session.create_block() for _ in range(batch_size)]
    group = WorkerGroup(
        cache_structure=[[common, tail] for tail in tails],
        write_to=tails,
    )
    # Distinct first tokens avoid routing every worker identically.
    seed_ids = tokenizer.encode(
        " forward left right wait descend inspect turn continue choose safely",
        add_special_tokens=False,
    )
    current = torch.tensor(
        [seed_ids[i % len(seed_ids)] for i in range(batch_size)], dtype=torch.int32
    )
    before = memory_gib()
    try:
        for _ in range(WARMUP_STEPS):
            logits = session.decode_step(group, current)
            current = logits.argmax(dim=-1).to(torch.int32)
        torch.cuda.synchronize()
        warm = memory_gib()
        samples = []
        for _ in range(2):
            started = time.perf_counter()
            for _ in range(MEASURED_STEPS):
                logits = session.decode_step(group, current)
                current = logits.argmax(dim=-1).to(torch.int32)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
        elapsed = statistics.fmean(samples)
        row: dict[str, object] = {
            "batch_size": batch_size,
            "status": "ok",
            "measured_steps": MEASURED_STEPS,
            "sample_seconds": samples,
            "mean_seconds": elapsed,
            "milliseconds_per_batched_forward": elapsed / MEASURED_STEPS * 1000,
            "aggregate_decode_tokens_per_second": batch_size * MEASURED_STEPS / elapsed,
            "memory_gib_before": before,
            "memory_gib_after_warmup": warm,
            "memory_gib_after_measurement": memory_gib(),
        }
        return row
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        # Preserve non-OOM runtime errors too; large shape support is part of the sweep.
        return {
            "batch_size": batch_size,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "memory_gib_before": before,
            "memory_gib_at_error": memory_gib(),
        }
    finally:
        for tail in tails:
            session.free_block(tail)
        torch.cuda.empty_cache()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS / f"minisgl_cu130_large_batch_{stamp}.json"
    config = EngineConfig(
        model_path=MODEL,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        # SharedCacheSession does not consume the ordinary request-state rows.
        max_running_req=4,
        memory_ratio=0.9,
        max_seq_len_override=4096,
        num_page_override=8192,
        attention_backend="fi",
        generation_config=transformers.GenerationConfig(
            do_sample=False, temperature=None, top_k=None, top_p=None
        ),
        distributed_addr="tcp://127.0.0.1:2385",
    )
    report: dict[str, object] = {
        "timestamp_utc": stamp,
        "model": MODEL,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "path": "mini-sglang SharedCacheSession; one shared prompt block",
        "prompt_tokens": PROMPT_TOKENS,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps_per_sample": MEASURED_STEPS,
    }
    print("Loading mini-sglang", flush=True)
    started = time.perf_counter()
    engine = Engine(config)
    report["load_seconds"] = time.perf_counter() - started
    report["gpu"] = torch.cuda.get_device_name()
    session = SharedCacheSession(engine)
    tokenizer = load_processor(MODEL).tokenizer
    common = session.create_block()
    try:
        session.prefill_block(common, make_prompt(tokenizer))
        rows = []
        for batch_size in BATCHES:
            print(f"BATCH {batch_size}", flush=True)
            row = run_batch(session, common, tokenizer, batch_size)
            rows.append(row)
            print(json.dumps(row), flush=True)
        report["rows"] = rows
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"RESULT {path}", flush=True)
    finally:
        session.free_block(common)
        engine.shutdown()


if __name__ == "__main__":
    main()
