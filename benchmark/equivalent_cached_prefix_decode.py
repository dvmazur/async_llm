#!/usr/bin/env python3
"""Equivalent cached-prefix generation benchmark: mini-sglang side."""

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
from minisgl.shared_cache import PrefillJob, SharedCacheSession, WorkerGroup
from minisgl.utils.hf import load_processor


MODEL = "/workspace/models/Qwen3.6-35B-A3B"
RESULTS = Path("/workspace/results")
BATCHES = (8, 16, 32, 64)
COMMON_TOKENS = 256
OUTPUT_TOKENS = 32
MEASURED_SAMPLES = 2


def make_common(tokenizer) -> torch.Tensor:
    text = (
        "Inspect the embodied observation history, reason about geometry and progress, "
        "then choose the next action. "
    ) * 64
    ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").view(-1)
    if ids.numel() < COMMON_TOKENS:
        raise RuntimeError(f"Only produced {ids.numel()} common tokens")
    return ids[:COMMON_TOKENS].to(dtype=torch.int32, device="cpu")


def seed_ids(batch_size: int, sample_index: int) -> list[int]:
    # Same deterministic, non-special token IDs are used by the SGLang side.
    base = 10_000 + batch_size * 1_000 + sample_index * 100
    return [base + i for i in range(batch_size)]


def run_sample(
    session: SharedCacheSession,
    common,
    batch_size: int,
    sample_index: int,
    output_tokens: int,
) -> tuple[float, int]:
    branch_prefixes = [session.create_block() for _ in range(batch_size)]
    decode_tails = [session.create_block() for _ in range(batch_size)]
    seeds = seed_ids(batch_size, sample_index)
    try:
        started = time.perf_counter()
        first_logits = session.prefill_batch(
            [
                PrefillJob(
                    block=branch,
                    input_ids=torch.tensor([seed], dtype=torch.int32),
                    context=[common],
                )
                for branch, seed in zip(branch_prefixes, seeds)
            ]
        )
        current = torch.stack([rows[0].argmax() for rows in first_logits]).to(torch.int32)
        group = WorkerGroup(
            cache_structure=[
                [common, branch, tail]
                for branch, tail in zip(branch_prefixes, decode_tails)
            ],
            write_to=decode_tails,
        )
        for _ in range(output_tokens - 1):
            logits = session.decode_step(group, current)
            current = logits.argmax(dim=-1).to(torch.int32)
        torch.cuda.synchronize()
        return time.perf_counter() - started, batch_size * output_tokens
    finally:
        for block in [*decode_tails, *branch_prefixes]:
            session.free_block(block)
        torch.cuda.empty_cache()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS / f"equivalent_cached_prefix_minisgl_{stamp}.json"
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
        distributed_addr="tcp://127.0.0.1:2391",
    )
    report: dict[str, object] = {
        "timestamp_utc": stamp,
        "framework": "mini-sglang SharedCacheSession",
        "model": MODEL,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "protocol": {
            "common_prefix_tokens_prefilled_outside_timing": COMMON_TOKENS,
            "new_input_tokens_per_request_inside_timing": 1,
            "output_tokens_per_request": OUTPUT_TOKENS,
            "timed_model_calls": "one batched 1-token prefill plus 31 batched decode calls",
            "sampling": "greedy",
            "samples": MEASURED_SAMPLES,
        },
    }
    print("Loading mini-sglang", flush=True)
    load_started = time.perf_counter()
    engine = Engine(config)
    report["load_seconds"] = time.perf_counter() - load_started
    session = SharedCacheSession(engine)
    tokenizer = load_processor(MODEL).tokenizer
    common = session.create_block()
    try:
        session.prefill_block(common, make_common(tokenizer))
        rows = []
        for batch_size in BATCHES:
            print(f"WARM BATCH {batch_size}", flush=True)
            run_sample(session, common, batch_size, -1, 4)
            samples = []
            for sample_index in range(MEASURED_SAMPLES):
                elapsed, total_tokens = run_sample(
                    session, common, batch_size, sample_index, OUTPUT_TOKENS
                )
                samples.append(elapsed)
                print(
                    "SAMPLE",
                    json.dumps(
                        {
                            "batch_size": batch_size,
                            "sample": sample_index + 1,
                            "seconds": elapsed,
                            "total_output_tokens": total_tokens,
                        }
                    ),
                    flush=True,
                )
            mean_seconds = statistics.fmean(samples)
            row = {
                "batch_size": batch_size,
                "sample_seconds": samples,
                "mean_seconds": mean_seconds,
                "aggregate_output_tokens_per_second": (
                    batch_size * OUTPUT_TOKENS / mean_seconds
                ),
                "effective_milliseconds_per_model_step": (
                    mean_seconds / OUTPUT_TOKENS * 1000
                ),
            }
            rows.append(row)
            report["rows"] = rows
            path.write_text(json.dumps(report, indent=2) + "\n")
            print("ROW", json.dumps(row), flush=True)
        print(f"RESULT {path}", flush=True)
    finally:
        session.free_block(common)
        engine.shutdown()


if __name__ == "__main__":
    main()
