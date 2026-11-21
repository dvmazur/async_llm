"""Wait for idle GPUs, own one backend, run pilot and all direct trials.

Never signals existing processes. Cleanup only targets the new backend process
group created by this supervisor. State is persisted while waiting or running.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def gpu_available():
    compute = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free = {int(line.split(",")[0]): int(line.split(",")[1]) for line in memory.splitlines()}
    return all(free.get(gpu, 0) >= 21000 for gpu in (0, 1)), free, compute


def health(port):
    with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
        connection.settimeout(3)
        connection.sendall(b'{"health":true}\n')
        return json.loads(connection.makefile("rb").readline())


def main():
    from robot_sim.candidate_nav.audit import audit_episode
    from robot_sim.candidate_nav.io import write_json_atomic

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = {"supervisor_pid": os.getpid(), "backend_pid": None}

    def status(phase, **extra):
        state.update(phase=phase, updated_unix=time.time(), **extra)
        write_json_atomic(root / "runtime_state.json", state)
        print(json.dumps(state), flush=True)

    def interrupt(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    server = None
    stage = None
    environment = {**os.environ, "ROBOT_SIM_CPU": "1"}

    def run_stage(command, **kwargs):
        nonlocal stage
        stage = subprocess.Popen(
            command, cwd=ROOT, env=environment, start_new_session=True, **kwargs
        )
        code = stage.wait()
        stage = None
        if code:
            raise RuntimeError(f"Experiment stage exited {code}")

    try:
        while True:
            ready, memory, processes = gpu_available()
            if ready:
                break
            status("waiting_for_idle_gpus", free_mib=memory, existing_compute_pids=processes)
            time.sleep(30)
        # Refuse to take over a backend/port owned by another task.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 18766))
        if args.server_log.exists():
            raise ValueError("Server log already exists; preserve it and use a new path")
        with args.server_log.open("w") as log:
            server = subprocess.Popen(
                [
                    "bash",
                    "robot_sim/scripts/run_candidate_llm.sh",
                    "--analysis-tokens",
                    "24",
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status(
                "loading_qwen27",
                backend_pid=server.pid,
                existing_compute_pids=processes,
                free_mib=memory,
            )
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Own backend exited {server.returncode}")
                try:
                    if health(18766).get("ready"):
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Own backend startup timeout")
                time.sleep(2)
            pilot = root.parent / (root.name + "_pilot")
            if pilot.exists():
                raise ValueError("Pilot directory already exists; preserve and inspect")
            status("native_qwen_pilot", pilot=str(pilot))
            with (root / "pilot_launcher.log").open("w") as pilot_log:
                run_stage(
                    [
                        "bash",
                        "robot_sim/scripts/run_candidate_navigation.sh",
                        "--control-mode",
                        "direct",
                        "--task",
                        str(ROOT / "robot_sim/candidate_tasks/red_near.json"),
                        "--output",
                        str(pilot),
                        "--seed",
                        "29",
                        "--wall-limit",
                        "3600",
                    ],
                    stdout=pilot_log,
                    stderr=subprocess.STDOUT,
                )
            audit = audit_episode(pilot, args.server_log.resolve())
            if not audit["passed"]:
                raise RuntimeError("Native pilot artifact audit failed: " + str(audit["failures"]))
            status("running_frozen_14_trials")
            run_stage(
                [
                    sys.executable,
                    "robot_sim/scripts/run_direct_comparison.py",
                    "--output",
                    str(root),
                    "--baseline",
                    str(args.baseline.resolve()),
                    "--server-log",
                    str(args.server_log.resolve()),
                ],
            )
            status("complete")
    except BaseException as error:
        status("stopped_with_error", error=repr(error))
        raise
    finally:
        if stage is not None and stage.poll() is None:
            try:
                os.killpg(stage.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stage.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(stage.pid, signal.SIGKILL)
                stage.wait()
        if server is not None:
            # The group was created above; no existing job can belong to it.
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
            # Torchrun can exit before its children; signal ONLY its own group.
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            status(state["phase"], backend_stopped=True)


if __name__ == "__main__":
    main()
