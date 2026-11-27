"""Run independent budget readouts on immediately free GPUs with fixed index lanes.

An invocation never waits to acquire a GPU. A resumed plan keeps its original
GPU UUIDs and index-modulo lanes; busy lanes remain pending for an explicit resume.
Only this coordinator's process groups are stopped on failure.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import budget_curve_report as report

CONDITIONS = {f"{mode}-w{workers}": (mode, workers) for mode, workers in report.CONDITIONS}
STOP_MARKERS = ("quality-hold.json", "superseded.json", "multiworker-stopped.json")
PROJECT = Path(__file__).resolve().parents[4]
LAUNCHER = PROJECT / "run_hogwild_budget.sh"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    report._write(path, json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def gpu_state(gpu):
    fields = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=20).strip().split(",")
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=20).strip()
    index, gpu_uuid, memory, utilization = (value.strip() for value in fields)
    if int(index) != gpu or not gpu_uuid.startswith("GPU-"):
        raise RuntimeError(f"GPU identity mismatch: requested {gpu}, got {fields}")
    return {"gpu": gpu, "uuid": gpu_uuid, "memory_mib": int(memory), "utilization": int(utilization),
            "compute_pids": processes.splitlines(),
            "free": not processes and 0 <= int(memory) <= 8 and int(utilization) == 0}


def identity(args):
    directory = Path(__file__).parent
    files = [directory / name for name in (
        "budget_curve_sweep.py", "budget_curve_eval.py", "budget_curve_report.py", "budget_views.py",
        "report.py", "trace_io.py", "math500.py", "rollout.py", "eval.py",
    )] + [LAUNCHER, PROJECT / "run_hogwild.sh", PROJECT / "pyproject.toml", PROJECT / "uv.lock"]
    files += [path for path in (PROJECT / "python/minisgl").rglob("*")
              if path.is_file() and path.suffix in {".py", ".cu", ".cuh", ".h", ".cpp"}]
    snapshot = args.source_root / "source-files.json"
    active = args.source_root / "active-source-files.json"
    active_sources = json.loads(active.read_text())
    if not isinstance(active_sources, dict) or not active_sources:
        raise ValueError("The source run requires a nonempty source identity manifest")
    for relative, expected in active_sources.items():
        path = (PROJECT / relative).resolve()
        if not path.is_relative_to(PROJECT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Immutable source code differs from active source snapshot: {relative}")
    source_identity = {"source_root": str(args.source_root.resolve()),
            "source_manifest_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "active_source_manifest_sha256": hashlib.sha256(active.read_bytes()).hexdigest(),
            "code_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in files}}
    if not args.wait_for_source:
        tasks_digest = hashlib.sha256((args.source_root / "report/tasks.csv").read_bytes()).hexdigest()
        complete = json.loads((args.source_root / "complete.json").read_text())
        if (complete["cases"] != 3500 or complete["tasks_sha256"] != tasks_digest
                or complete["source_manifest_sha256"] != source_identity["source_manifest_sha256"]
                or complete["active_source_manifest_sha256"] != source_identity["active_source_manifest_sha256"]):
            raise ValueError("Completed source results or manifest identity changed")
        source_identity["source_tasks_sha256"] = tasks_digest
    return source_identity


def configuration(args):
    return {"format": "budget-readout-fixed-lanes-v1", **identity(args),
            "indices": args.indices, "conditions": args.conditions, "budgets": args.budgets,
            "wait_for_source": args.wait_for_source, "max_prefill_rows": args.max_prefill_rows,
            "memory_ratio": args.memory_ratio, "seed": 42, "answer_tokens": 8,
            "gpu_acquisition": "immediate_only_no_reacquisition"}


def load_plan(output, config, requested, states):
    path = output / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["config"] != config:
            raise ValueError("Existing plan source/code/config identity differs; do not reuse this output")
        if not set(requested) <= set(plan["requested_gpus"]):
            raise ValueError("Resume cannot add GPUs or redistribute the fixed index lanes")
        return plan
    if (output / "results.jsonl").exists() or (output / "lanes").exists():
        raise ValueError("Results exist without their immutable lane plan")
    free = [state for state in states if state["free"]]
    if not free:
        return None
    plan = {"created_at": now(), "config": config, "requested_gpus": requested,
            "lanes": [{"gpu": state["gpu"], "uuid": state["uuid"],
                       "modulo": len(free), "remainder": index}
                      for index, state in enumerate(free)]}
    write_json(path, plan)
    return plan


def expected_keys(plan, lane=None):
    config = plan["config"]
    indices = config["indices"]
    if lane is not None:
        indices = [index for index in indices if index % lane["modulo"] == lane["remainder"]]
    return {(*CONDITIONS[name], index, budget) for name in config["conditions"]
            for index in indices for budget in config["budgets"]}


def lane_path(output, lane):
    return output / "lanes" / f"lane-{lane['remainder']:03d}"


def consolidate(output, plan):
    rows = {}
    for lane in plan["lanes"]:
        path = lane_path(output, lane) / "results.jsonl"
        if not path.exists():
            continue
        found, _ = report.load_rows(path)
        if not found.keys() <= expected_keys(plan, lane):
            raise ValueError(f"Results outside the configured lane in {path}")
        if rows.keys() & found.keys():
            raise ValueError(f"Duplicate or conflicting lane results in {path}")
        rows.update(found)
    target = output / "results.jsonl"
    if target.exists():
        previous, _ = report.load_rows(target)
        if any(key not in rows or rows[key] != value for key, value in previous.items()):
            raise ValueError("Committed consolidated results disappeared or changed")
    report._write(target, "".join(json.dumps(rows[key], sort_keys=True, allow_nan=False) + "\n"
                                 for key in sorted(rows)))
    return rows


def check_open(source, output):
    for directory in (source, output):
        for name in STOP_MARKERS:
            if (directory / name).exists():
                raise RuntimeError(f"Stop marker present: {directory / name}")


def worker_command(args, lane):
    command = [str(LAUNCHER), "--worker", "--source-root", str(args.source_root.resolve()),
               "--output", str(lane_path(args.output, lane).resolve()),
               "--index-modulo", str(lane["modulo"]), "--index-remainder", str(lane["remainder"]),
               "--indices", *map(str, args.indices), "--budgets", *map(str, args.budgets),
               "--conditions", *args.conditions, "--max-prefill-rows", str(args.max_prefill_rows),
               "--memory-ratio", str(args.memory_ratio), "--answer-tokens", "8", "--seed", "42"]
    if args.wait_for_source:
        command.append("--wait-for-source")
    return command


def occupied_before_load(worker):
    text = Path(worker["log"]).read_text(errors="replace")
    return (f"RuntimeError: GPU {worker['gpu']} is occupied:" in text
            and "GPU_PREFLIGHT " not in text and "START " not in text)


def stop_owned(children):
    running = [child for child in children if child.poll() is None]
    # A failed process-group leader may leave engine children alive.
    for child in children:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for child in running:
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
    # Reaped leaders cannot be waited on for their remaining descendants.
    for child in children:
        try:
            os.killpg(child.pid, 0)
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def progress(output, plan, rows, active, skipped):
    data = {"updated_at": now(), "completed": len(rows), "target": len(expected_keys(plan)),
            "active_gpus": active, "skipped_lanes": skipped,
            "gpu_acquisition": "immediate_only_no_reacquisition",
            "conditions": {name: sum(key[:2] == condition for key in rows)
                           for name, condition in CONDITIONS.items()
                           if name in plan["config"]["conditions"]},
            "cells": [{"condition": name, "budget": budget,
                       "completed": sum(key[:2] == CONDITIONS[name] and key[3] == budget for key in rows),
                       "target": len(plan["config"]["indices"])}
                      for name in plan["config"]["conditions"] for budget in plan["config"]["budgets"]]}
    write_json(output / "progress.json", data)


def run(args):
    check_open(args.source_root, args.output)
    config = configuration(args)
    states = [gpu_state(gpu) for gpu in args.gpus]
    initial = {state["gpu"]: state for state in states}
    plan = load_plan(args.output, config, args.gpus, states)
    if plan is None:
        write_json(args.output / "gpu-unavailable.json", {"at": now(), "devices": states,
                   "reason": "No immediately free requested GPU; no lane plan created"})
        return 3
    rows = consolidate(args.output, plan)
    wanted = expected_keys(plan)
    attempt = {"started_at": now(), "admission": states, "workers": [], "skipped_lanes": []}
    attempts = args.output / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt_path = attempts / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.json"
    children, handles, skipped = [], [], attempt["skipped_lanes"]
    occupied = set()
    try:
        for lane in plan["lanes"]:
            if expected_keys(plan, lane) <= rows.keys():
                continue
            check_open(args.source_root, args.output)
            if configuration(args) != config:
                raise ValueError("Source/code/config identity changed before worker launch")
            state = initial.get(lane["gpu"])
            if state is not None and state["free"]:
                state = gpu_state(lane["gpu"])
            if state is not None and state["uuid"] != lane["uuid"]:
                raise RuntimeError(f"GPU UUID changed for fixed lane {lane}")
            if state is None or not state["free"]:
                skipped.append({**lane, "state": state, "reason": "not requested or immediately occupied"})
                continue
            path = lane_path(args.output, lane)
            path.mkdir(parents=True, exist_ok=True)
            log = attempts / f"{attempt_path.stem}-lane-{lane['remainder']:03d}.log"
            handle = log.open("ab")
            handles.append(handle)
            command = worker_command(args, lane)
            env = {**os.environ, "HOGWILD_GPU": str(lane["gpu"]), "CUDA_VISIBLE_DEVICES": lane["uuid"]}
            child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            attempt["workers"].append({**lane, "pid": child.pid, "command": command, "log": str(log)})
            write_json(attempt_path, attempt)
        while True:
            check_open(args.source_root, args.output)
            if configuration(args) != config:
                raise ValueError("Source/code/config identity changed during evaluation")
            rows = consolidate(args.output, plan)
            active = [worker["gpu"] for worker, child in zip(attempt["workers"], children)
                      if child.poll() is None]
            progress(args.output, plan, rows, active, skipped)
            for worker, child in zip(attempt["workers"], children):
                code = child.poll()
                if code is not None and code != 0:
                    if child.pid in occupied:
                        continue
                    if occupied_before_load(worker):
                        occupied.add(child.pid)
                        skipped.append({**worker, "reason": "occupied at evaluator preflight; not retried"})
                        continue
                    raise RuntimeError(f"Worker GPU {worker['gpu']} exited {code}; see {worker['log']}")
            if not active:
                break
            time.sleep(10)
        # Workers may have committed their last row between consolidation and poll.
        rows = consolidate(args.output, plan)
        progress(args.output, plan, rows, [], skipped)
        code = 0 if rows.keys() == wanted else 3
        if code == 0:
            report.build_report(args.output / "results.jsonl", args.output / "report")
            write_json(args.output / "complete.json", {"at": now(), "completed": len(rows),
                       "target": len(wanted), "plan_sha256": hashlib.sha256((args.output / "plan.json").read_bytes()).hexdigest(),
                       "results_sha256": hashlib.sha256((args.output / "results.jsonl").read_bytes()).hexdigest()})
            (args.output / "gpu-unavailable.json").unlink(missing_ok=True)
        else:
            write_json(args.output / "gpu-unavailable.json", {"at": now(), "completed": len(rows),
                       "target": len(wanted), "skipped_lanes": skipped,
                       "reason": "Pending fixed lanes or unavailable source; explicit resume required"})
        attempt["exit_code"] = code
        return code
    except BaseException as error:
        attempt["error"] = f"{type(error).__name__}: {error}"
        attempt["exit_code"] = 1
        raise
    finally:
        stop_owned(children)
        for handle in handles:
            handle.close()
        attempt["finished_at"] = now()
        write_json(attempt_path, attempt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(500)))
    parser.add_argument("--budgets", type=int, nargs="+", default=list(report.BUDGETS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--wait-for-source", action="store_true")
    parser.add_argument("--max-prefill-rows", type=int, default=256)
    parser.add_argument("--memory-ratio", type=float, default=0.75)
    args = parser.parse_args(argv)
    for name in ("gpus", "indices", "budgets", "conditions"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"Repeated {name}")
    if (any(gpu < 0 for gpu in args.gpus) or any(not 0 <= index < 500 for index in args.indices)
            or not set(args.budgets) <= set(report.BUDGETS) or args.max_prefill_rows < 1
            or not 0 < args.memory_ratio < 1):
        parser.error("Invalid GPU, index, budget, prefill limit or memory ratio")
    args.indices.sort()
    args.budgets.sort()
    if not args.source_root.is_dir() or args.output.resolve().is_relative_to(args.source_root.resolve()):
        parser.error("Output must be separate from the existing immutable source directory")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".coordinator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another coordinator holds this output root")
        old_handlers = {}
        def interrupted(signum, _frame):
            raise RuntimeError(f"Coordinator interrupted by signal {signum}")
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signum] = signal.signal(signum, interrupted)
            return run(args)
        except Exception as error:
            write_json(args.output / "coordinator-error.json", {"at": now(), "error": str(error)})
            print(f"ERROR: {error}", flush=True)
            return 1
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
