#!/usr/bin/env python3
"""Run the canonical Speleo async pipeline without notebook UI/GIF overhead.

The pipeline source is loaded from minecraft_async_v48_dashboard_gif_eval100.ipynb
rather than maintained as a second copy.  Only the dashboard/GIF implementation,
the decision budget, deterministic sampling seed, and timing instrumentation are
changed.  Select the mini-sglang implementation by launching this script with the
desired repository's ``python`` directory first on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import inspect
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WORKSPACE = Path("/workspace")
DEFAULT_NOTEBOOK = (
    WORKSPACE
    / "async_reasoning_minecraft_handoff_2026-08-31"
    / "notebooks"
    / "minecraft_async_v48_dashboard_gif_eval100.ipynb"
)
RESULTS_ROOT = WORKSPACE / "results" / "speleo_eval25_comparison"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--decisions", type=int, default=25)
    parser.add_argument("--episode-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=20260903)
    parser.add_argument("--compose-trace", action="store_true")
    return parser.parse_args()


def _compile_and_eval(source: str, filename: str, namespace: dict[str, object]):
    code = compile(
        source,
        filename,
        "exec",
        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        dont_inherit=True,
    )
    return eval(code, namespace)


def _strip_ipython_magics(source: str) -> str:
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one timing anchor, found {count}: {old!r}")
    return source.replace(old, new)


def _prepare_rollout_source(source: str, *, compose_trace: bool) -> str:
    # Everything before this anchor only implements the interactive dashboard and
    # asynchronous GIF writer.  The environment and inference pipeline below it
    # are retained verbatim.
    anchor = "env = make_env(task_name, max_steps=MAX_DECISIONS, **SPELEO_ENV_CONFIG)"
    if source.count(anchor) != 1:
        raise RuntimeError("canonical rollout anchor changed")
    source = source[source.index(anchor) :]

    # Existing decision_seconds stops immediately after env.step.  Record another
    # timestamp after commit_round so model-side history maintenance is visible.
    history_anchor = "            round_row = {\n"
    source = _replace_once(
        source,
        history_anchor,
        "            model_pipeline_seconds = time.perf_counter() - decision_started\n"
        + history_anchor,
    )
    source = _replace_once(
        source,
        '                "decision_seconds": float(decision_latency),\n',
        '                "decision_seconds": float(decision_latency),\n'
        '                "model_pipeline_seconds": float(model_pipeline_seconds),\n',
    )

    # This point is after round-block cleanup and image/dashboard bookkeeping and
    # immediately before the next action.  The dashboard itself is a no-op below.
    cycle_anchor = '        dashboard_gif.capture("action", hold=DASHBOARD_GIF_ACTION_HOLD)\n'
    source = _replace_once(
        source,
        cycle_anchor,
        cycle_anchor
        + '        trajectory[-1]["cycle_seconds"] = (\n'
        + "            time.perf_counter() - decision_started\n"
        + "        )\n",
    )
    if compose_trace:
        close_anchor = "                await llm.close()\n"
        source = _replace_once(
            source,
            close_anchor,
            "                compose_cache_trace.finalize()\n" + close_anchor,
        )
    close_anchor = "                await llm.close()\n"
    cache_stats_source = (
        "                _sc_gdn = llm.async_engine.session.sc_gdn\n"
        "                _compose_cache = (\n"
        "                    _sc_gdn.compose_state_cache\n"
        "                )\n"
        "                compose_cache_runtime_stats = (\n"
        "                    {\n"
        "                        'gdn_storage_bytes': _sc_gdn.gdn_storage_bytes,\n"
        "                        'cache_ratio': _sc_gdn.compose_cache_ratio,\n"
        "                        **({} if _compose_cache is None else _compose_cache.stats),\n"
        "                        'resident_entries': _compose_cache.resident_entries,\n"
        "                        'resident_bytes': _compose_cache.resident_bytes,\n"
        "                        'max_bytes': _compose_cache.max_bytes,\n"
        "                        'enabled': _compose_cache is not None,\n"
        "                    }\n"
        "                    if _compose_cache is not None else {\n"
        "                        'gdn_storage_bytes': _sc_gdn.gdn_storage_bytes,\n"
        "                        'cache_ratio': _sc_gdn.compose_cache_ratio,\n"
        "                        'enabled': False,\n"
        "                    }\n"
        "                )\n"
    )
    source = _replace_once(source, close_anchor, cache_stats_source + close_anchor)
    return source


class _Slot:
    def __init__(self, value=b""):
        self.value = value


class _Dashboard:
    image_A = _Slot()
    image_B = _Slot()
    descriptions = _Slot("")
    actions = _Slot("")
    probe = _Slot("")


class _NoopGifRecorder:
    """API-compatible recorder which deliberately performs no rendering or I/O."""

    def __init__(self, widget_state, output_path, fps=10, queue_size=32):
        self.output_path = Path(output_path)

    def capture(self, event, hold=1):
        return None

    def finish(self):
        return self.output_path


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "stdev": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    notebook = args.notebook.resolve()
    expected_python = (repo / "python").resolve()
    if not expected_python.is_dir():
        raise RuntimeError(f"missing mini-sglang Python tree: {expected_python}")
    if Path(sys.path[0]).resolve() == expected_python:
        pass
    elif str(expected_python) not in [str(Path(p).resolve()) for p in sys.path if p]:
        raise RuntimeError(
            f"{expected_python} is not on sys.path; launch with " f"PYTHONPATH={expected_python}"
        )

    llm_spec = importlib.util.find_spec("minisgl.llm")
    if llm_spec is None or llm_spec.origin is None:
        raise RuntimeError("cannot resolve minisgl.llm")
    resolved_llm = Path(llm_spec.origin).resolve()
    if not resolved_llm.is_relative_to(expected_python):
        raise RuntimeError(f"wrong mini-sglang selected: {resolved_llm}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = RESULTS_ROOT / f"raw_{args.variant}_{stamp}"
    raw_dir.mkdir(parents=True, exist_ok=False)
    os.environ["SPELEO_MAX_DECISIONS"] = str(args.decisions)
    os.environ["SPELEO_EPISODE_SEED"] = str(args.episode_seed)
    os.environ["SPELEO_RESULTS_DIR"] = str(raw_dir)
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")
    os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/.triton-cache-cu130")

    notebook_data = json.loads(notebook.read_text())
    cells = notebook_data["cells"]
    setup_source = _strip_ipython_magics("".join(cells[0]["source"]))
    prompt_source = "".join(cells[1]["source"])
    rollout_source = _prepare_rollout_source(
        "".join(cells[2]["source"]), compose_trace=args.compose_trace
    )

    namespace: dict[str, object] = {
        "__name__": "__speleo_eval25__",
        "__file__": str(notebook),
        "datetime": datetime,
        "timezone": timezone,
        "dashboard": _Dashboard,
        "DashboardGifRecorder": _NoopGifRecorder,
    }
    print(
        "BENCHMARK_CONFIG",
        json.dumps(
            {
                "variant": args.variant,
                "repo": str(repo),
                "minisgl_llm": str(resolved_llm),
                "decisions": args.decisions,
                "episode_seed": args.episode_seed,
                "sampling_seed": args.sampling_seed,
                "raw_dir": str(raw_dir),
            }
        ),
        flush=True,
    )

    original_cwd = Path.cwd()
    os.chdir(notebook.parent)
    wall_started = time.perf_counter()
    try:
        _compile_and_eval(setup_source, f"{notebook}#setup", namespace)
        _compile_and_eval(prompt_source, f"{notebook}#prompt", namespace)
        compose_trace_path = raw_dir / "compose_cache_trace.json"
        if args.compose_trace:
            from compose_cache_trace import ComposeCacheTrace

            compose_cache_trace = ComposeCacheTrace(compose_trace_path)
            compose_cache_trace.install(namespace["llm"].async_engine.session.sc_gdn)
            namespace["compose_cache_trace"] = compose_cache_trace
        random.seed(args.sampling_seed)
        np.random.seed(args.sampling_seed)
        namespace["torch"].manual_seed(args.sampling_seed)
        result = _compile_and_eval(rollout_source, f"{notebook}#rollout", namespace)
        if not inspect.isawaitable(result):
            raise RuntimeError("rollout cell unexpectedly stopped being asynchronous")
        asyncio.run(result)
    finally:
        os.chdir(original_cwd)
    wall_seconds = time.perf_counter() - wall_started

    summaries = sorted(raw_dir.glob("speleo_eval100_async_seed*.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"expected one canonical summary, found {summaries}")
    canonical = json.loads(summaries[0].read_text())
    trajectory = canonical["trajectory"]
    if len(trajectory) != args.decisions:
        raise RuntimeError(
            f"episode ended after {len(trajectory)} actions, expected {args.decisions}"
        )

    decision = [float(row["decision_seconds"]) for row in trajectory]
    model_pipeline = [float(row["model_pipeline_seconds"]) for row in trajectory]
    cycle = [float(row["cycle_seconds"]) for row in trajectory]
    report = {
        "schema_version": 1,
        "timestamp_utc": stamp,
        "variant": args.variant,
        "repo": str(repo),
        "minisgl_llm": str(resolved_llm),
        "python": sys.executable,
        "notebook_source": str(notebook),
        "decisions": args.decisions,
        "episode_seed": args.episode_seed,
        "sampling_seed": args.sampling_seed,
        "one_time_setup_excluded": True,
        "timing_definitions": {
            "decision_seconds": "image prefill + async decision generation + env.step",
            "model_pipeline_seconds": "decision_seconds + history commit prefill",
            "cycle_seconds": "full action loop through cleanup and image bookkeeping; GIF disabled",
        },
        "decision_seconds": _distribution(decision),
        "model_pipeline_seconds": _distribution(model_pipeline),
        "cycle_seconds": _distribution(cycle),
        "decision_seconds_excluding_action_1": _distribution(decision[1:]),
        "model_pipeline_seconds_excluding_action_1": _distribution(model_pipeline[1:]),
        "cycle_seconds_excluding_action_1": _distribution(cycle[1:]),
        "mean_decode_batch_size": canonical["mean_decode_batch_size"],
        "mean_decision_batched_forwards_per_action": canonical[
            "mean_decision_batched_forwards_per_action"
        ],
        "mean_pipeline_batched_forwards_per_action": canonical[
            "mean_pipeline_batched_forwards_per_action"
        ],
        "one_time_setup_batched_forwards": canonical["one_time_setup_batched_forwards"],
        "process_wall_seconds_including_model_load_and_setup": wall_seconds,
        "canonical_summary": str(summaries[0]),
        "compose_cache_trace": (str(compose_trace_path) if args.compose_trace else None),
        "compose_cache_runtime_stats": namespace.get("compose_cache_runtime_stats"),
        "per_action": [
            {
                "step": row["step"],
                "action": row["action"],
                "decision_seconds": row["decision_seconds"],
                "model_pipeline_seconds": row["model_pipeline_seconds"],
                "cycle_seconds": row["cycle_seconds"],
                "decision_batched_forwards": row["decision_batched_forwards"],
                "pipeline_batched_forwards": row["pipeline_batched_forwards"],
                "decision_mean_decode_batch_size": row["decision_mean_decode_batch_size"],
            }
            for row in trajectory
        ],
    }
    report_path = RESULTS_ROOT / f"{args.variant}_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("BENCHMARK_RESULT", json.dumps(report, ensure_ascii=False), flush=True)
    print(f"BENCHMARK_RESULT_PATH {report_path}", flush=True)


if __name__ == "__main__":
    main()
