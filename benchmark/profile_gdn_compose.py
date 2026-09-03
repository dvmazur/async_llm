#!/usr/bin/env python3
"""Profile old-reference vs batched GDN initial-state composition on CUDA."""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_affine import compose_gdn_affines, init_gdn_affine
from torch.profiler import ProfilerActivity, profile

RESULTS = Path("/workspace/results")
WORKERS = 64
HEADS = 32
D_K = 128
D_V = 128
TIMING_REPEATS = 20
OUTPUT_DTYPE = torch.float32  # actual model call site


@dataclass(eq=False)
class _Block:
    name: str
    linear_affine: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


def _pair(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    eye = torch.eye(D_K, dtype=torch.float32, device=device).view(1, 1, D_K, D_K)
    A = eye.expand(1, HEADS, D_K, D_K).clone()
    A.add_(0.01 * torch.randn(1, HEADS, D_K, D_K, device=device, generator=generator))
    B = 0.02 * torch.randn(1, HEADS, D_V, D_K, device=device, generator=generator)
    return A, B


def _block(name: str, seed: int, device: torch.device) -> _Block:
    return _Block(name=name, linear_affine={0: _pair(seed, device)})


def _packed_blocks(prefix: str, seed: int, device: torch.device) -> list[_Block]:
    generator = torch.Generator(device=device).manual_seed(seed)
    eye = torch.eye(D_K, dtype=torch.float32, device=device).view(1, 1, D_K, D_K)
    A = eye.expand(WORKERS, HEADS, D_K, D_K).clone()
    A.add_(
        0.01
        * torch.randn(
            WORKERS, HEADS, D_K, D_K, dtype=torch.float32, device=device, generator=generator
        )
    )
    B = 0.02 * torch.randn(
        WORKERS, HEADS, D_V, D_K, dtype=torch.float32, device=device, generator=generator
    )
    return [
        _Block(prefix + str(row), {0: (A[row : row + 1], B[row : row + 1])})
        for row in range(WORKERS)
    ]


def _make_case(depth: int, device: torch.device) -> SharedCacheGDN:
    common = _block("common", 1, device)
    branches = [_block(f"branch-{worker}", 100 + worker, device) for worker in range(WORKERS)]
    # Decode capture writes all worker rows as slices of one batched allocation.
    # Branch prefills are currently serialized and therefore remain fragmented.
    tails = _packed_blocks("tail-", 1_000, device) if depth == 3 else None
    chains = [
        [common, branches[worker]] if tails is None else [common, branches[worker], tails[worker]]
        for worker in range(WORKERS)
    ]
    gdn = SharedCacheGDN(
        num_heads=HEADS,
        head_k_dim=D_K,
        head_v_dim=D_V,
        conv_dim=1,
        conv_kernel=1,
        device=device,
    )
    gdn.set_context(chains, [chain[-1] for chain in chains])
    return gdn


def _reference_compose(
    gdn: SharedCacheGDN, lin_idx: int = 0, dtype: torch.dtype = OUTPUT_DTYPE
) -> torch.Tensor | None:
    """Pre-optimization implementation retained locally for profiling."""
    if not gdn.has_previous_affine(lin_idx):
        return None
    prefix_memo: dict = {}

    def compose_chain(chain):
        acc = None
        key: tuple = ()
        for block in chain:
            key = key + (id(block),)
            pair = block.linear_affine.get(lin_idx)
            if pair is None:
                continue
            if key in prefix_memo:
                acc = prefix_memo[key]
                continue
            A_b = pair[0].to(dtype=torch.float32, device=gdn.device)
            B_b = pair[1].to(dtype=torch.float32, device=gdn.device)
            if acc is None:
                acc = (A_b, B_b)
            else:
                acc = compose_gdn_affines(
                    A_first=acc[0], B_first=acc[1], A_second=A_b, B_second=B_b
                )
            prefix_memo[key] = acc
        if acc is None:
            acc = init_gdn_affine(
                batch_size=1,
                num_heads=gdn.num_heads,
                d_k=gdn.head_k_dim,
                d_v=gdn.head_v_dim,
                dtype=torch.float32,
                device=gdn.device,
            )
        return acc

    per_worker = [compose_chain(chain)[1] for chain in gdn.cache_structure]
    state = torch.cat(per_worker, dim=0)
    return state.transpose(-1, -2).contiguous().to(dtype=dtype)


def _merge_duration_us(events: list[tuple[float, float]]) -> float:
    intervals = sorted((start, start + duration) for start, duration in events)
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _category(name: str) -> str:
    lower = name.lower()
    if "gemm" in lower or "cublas" in lower or "cutlass" in lower:
        return "gemm"
    if "copy" in lower or "memcpy" in lower or "memset" in lower:
        return "copy_or_set"
    if "add" in lower or "elementwise" in lower or "vectorized" in lower:
        return "add_or_elementwise"
    if "index" in lower or "select" in lower or "gather" in lower:
        return "gather_or_index"
    return "other"


def _parse_trace(path: Path) -> dict[str, object]:
    trace = json.loads(path.read_text())
    device_events = []
    kernels = []
    by_name: dict[str, list[float]] = defaultdict(list)
    categories: dict[str, list[float]] = defaultdict(list)
    for event in trace["traceEvents"]:
        if event.get("ph") != "X" or event.get("cat") not in {
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        }:
            continue
        start = float(event.get("ts", 0.0))
        duration = float(event.get("dur", 0.0))
        name = event.get("name", "unknown")
        device_events.append((start, duration))
        by_name[name].append(duration)
        categories[_category(name)].append(duration)
        if event.get("cat") == "kernel":
            kernels.append((start, duration))
    first = min(start for start, _ in device_events)
    last = max(start + duration for start, duration in device_events)
    busy = _merge_duration_us(device_events)
    return {
        "kernel_launches": len(kernels),
        "device_activity_events": len(device_events),
        "gpu_span_ms": (last - first) / 1_000,
        "gpu_busy_union_ms": busy / 1_000,
        "gpu_idle_gaps_inside_span_ms": (last - first - busy) / 1_000,
        "categories": [
            {
                "category": category,
                "calls": len(durations),
                "total_ms": sum(durations) / 1_000,
            }
            for category, durations in sorted(
                categories.items(), key=lambda item: sum(item[1]), reverse=True
            )
        ],
        "top_kernels": [
            {"name": name, "calls": len(ds), "total_ms": sum(ds) / 1_000}
            for name, ds in sorted(by_name.items(), key=lambda item: sum(item[1]), reverse=True)[
                :15
            ]
        ],
    }


def _time_call(fn) -> tuple[list[float], float]:
    samples = []
    for _ in range(TIMING_REPEATS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1_000)
    return samples, statistics.median(samples)


def _profile_call(fn, path: Path) -> dict[str, object]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    return _parse_trace(path)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = RESULTS / f"gdn_compose_profile_{stamp}.json"
    report: dict[str, object] = {
        "timestamp_utc": stamp,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "shape": {"workers": WORKERS, "heads": HEADS, "d_k": D_K, "d_v": D_V},
        "output_dtype": str(OUTPUT_DTYPE),
        "timing_repeats": TIMING_REPEATS,
        "cases": [],
    }

    for depth in (2, 3):
        gdn = _make_case(depth, device)
        old_fn = partial(_reference_compose, gdn)
        new_fn = partial(gdn.compose_initial_recurrent_state, 0, OUTPUT_DTYPE, state_v_first=True)

        for _ in range(5):
            old_fn()
        torch.cuda.synchronize()

        expected = old_fn()
        old_samples, old_median = _time_call(old_fn)
        reference_trace = RESULTS / f"gdn_compose_reference_depth{depth}_{stamp}_trace.json"
        reference_profile = _profile_call(old_fn, reference_trace)

        actual = new_fn()
        assert expected is not None and actual is not None
        actual_hf = actual.transpose(-1, -2)
        max_abs = (actual_hf.float() - expected.float()).abs().max().item()
        relative_l2 = (
            (actual_hf.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)
        ).item()

        for _ in range(5):
            new_fn()
        torch.cuda.synchronize()
        new_samples, new_median = _time_call(new_fn)
        batched_trace = RESULTS / f"gdn_compose_batched_depth{depth}_{stamp}_trace.json"
        batched_profile = _profile_call(new_fn, batched_trace)

        case = {
            "affine_depth": depth,
            "parity": {"max_abs": max_abs, "relative_l2": relative_l2},
            "reference_timing_ms": {
                "samples": old_samples,
                "median": old_median,
            },
            "batched_timing_ms": {
                "samples": new_samples,
                "median": new_median,
            },
            "speedup": old_median / new_median,
            "reference_trace": str(reference_trace),
            "reference_profile": reference_profile,
            "batched_trace": str(batched_trace),
            "batched_profile": batched_profile,
        }
        report["cases"].append(case)
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        print("CASE", json.dumps(case), flush=True)
        del gdn, expected, actual
        torch.cuda.empty_cache()

    print(f"RESULT {result_path}", flush=True)


if __name__ == "__main__":
    main()
