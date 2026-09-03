#!/usr/bin/env python3
"""Pointer-compose cache on/off ablation on the established batch-64 workload."""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.shared_cache import SharedCacheSession
from minisgl.utils.hf import load_processor

from profile_batch64_bottlenecks import (
    BATCH_SIZE,
    COMMON_TOKENS,
    MODEL,
    create_branch,
    free_branch,
    make_common,
    parse_trace,
    prefill_and_group,
)
from compose_cache_trace import ComposeCacheTrace
from torch.profiler import ProfilerActivity, profile

RESULTS = Path("/workspace/results")
COMPOSE_CACHE_RATIO = 1.6
TIMING_FORWARDS = 10
WARMUP_FORWARDS = 4
PROFILE_FORWARDS = 3
ENABLE_CUPTI = os.environ.get("MINISGL_ENABLE_CUPTI_PROFILE", "0") == "1"


def _cache_stats(gdn):
    cache = gdn.compose_state_cache
    if cache is None:
        return None
    return {
        **cache.stats,
        "resident_entries": cache.resident_entries,
        "resident_bytes": cache.resident_bytes,
        "max_bytes": cache.max_bytes,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = RESULTS / f"compose_cache_batch64_ablation_{stamp}.json"
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
        distributed_addr="tcp://127.0.0.1:2397",
    )
    print("Loading mini-sglang", flush=True)
    engine = Engine(config)
    session = SharedCacheSession(engine)
    tokenizer = load_processor(MODEL).tokenizer
    common = session.create_block()
    report = {
        "schema_version": 1,
        "timestamp_utc": stamp,
        "model": MODEL,
        "batch_size": BATCH_SIZE,
        "common_tokens": COMMON_TOKENS,
        "timing_forwards": TIMING_FORWARDS,
        "warmup_forwards": WARMUP_FORWARDS,
        "gdn_storage_bytes": session.sc_gdn.gdn_storage_bytes,
        "cache_ratio": COMPOSE_CACHE_RATIO,
        "variants": [],
    }
    try:
        session.sc_gdn.configure_compose_cache(0)
        session.prefill_block(common, make_common(tokenizer))

        variants = (
            ("pointer_no_cache", 0.0),
            ("pointer_cache", COMPOSE_CACHE_RATIO),
        )
        for variant_index, (name, ratio) in enumerate(variants):
            gdn = session.sc_gdn
            gdn.configure_compose_cache(ratio)
            prefixes, tails, jobs = create_branch(session, common, 10_000 + variant_index * 1_000)
            current, group = prefill_and_group(session, common, prefixes, tails, jobs)
            for _ in range(WARMUP_FORWARDS):
                current = session.decode_step(group, current).argmax(dim=-1).to(torch.int32)
            torch.cuda.synchronize()

            samples = []
            for _ in range(TIMING_FORWARDS):
                torch.cuda.synchronize()
                started = time.perf_counter()
                current = session.decode_step(group, current).argmax(dim=-1).to(torch.int32)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - started)

            event_trace_path = (
                RESULTS / f"compose_cache_batch64_{name}_{stamp}_cuda_events.json"
            )
            event_trace = ComposeCacheTrace(event_trace_path)
            event_trace.install(gdn)
            for _ in range(PROFILE_FORWARDS):
                current = session.decode_step(group, current).argmax(dim=-1).to(torch.int32)
            event_trace.finalize()
            event_trace.uninstall(gdn)

            trace_path = RESULTS / f"compose_cache_batch64_{name}_{stamp}_trace.json"
            if ENABLE_CUPTI:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_stack=False,
                    record_shapes=False,
                ) as prof:
                    current = session.decode_step(group, current).argmax(dim=-1).to(torch.int32)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(trace_path))
                try:
                    profile_summary = parse_trace(trace_path)
                except RuntimeError as exc:
                    profile_summary = {"unavailable": str(exc)}
            else:
                trace_path = None
                profile_summary = {
                    "unavailable": "CUPTI disabled after CUPTI_ERROR_INVALID_DEVICE on GB10"
                }
            mean_seconds = statistics.fmean(samples)
            row = {
                "name": name,
                "cache_ratio": ratio,
                "cache_budget_bytes": (
                    None if gdn.compose_state_cache is None else gdn.compose_state_cache.max_bytes
                ),
                "decode_seconds_samples": samples,
                "mean_decode_seconds": mean_seconds,
                "median_decode_seconds": statistics.median(samples),
                "aggregate_decode_tokens_per_second": BATCH_SIZE / mean_seconds,
                "cache_stats": _cache_stats(gdn),
                "cuda_event_trace": str(event_trace_path),
                "trace": None if trace_path is None else str(trace_path),
                "profile": profile_summary,
            }
            report["variants"].append(row)
            result_path.write_text(json.dumps(report, indent=2) + "\n")
            print("VARIANT", json.dumps(row), flush=True)
            free_branch(session, prefixes, tails)
        print(f"RESULT {result_path}", flush=True)
    finally:
        session.free_block(common)
        engine.shutdown()


if __name__ == "__main__":
    main()
