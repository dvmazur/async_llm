import argparse
import hashlib
import json

import pytest
from minisgl_examples.training.parallel_thinking import budget_curve_sweep as sweep


def test_source_identity_uses_the_completed_fresh_run_and_detects_changes(tmp_path, monkeypatch):
    project = tmp_path / "checkout"
    directory = project / "scripts/minisgl_examples/training/parallel_thinking"
    directory.mkdir(parents=True)
    names = ("budget_curve_sweep.py", "budget_curve_eval.py", "budget_curve_report.py", "budget_views.py",
             "report.py", "trace_io.py", "math500.py", "rollout.py", "eval.py")
    for name in names:
        (directory / name).write_text("source")
    for name in ("run_hogwild_budget.sh", "run_hogwild.sh", "pyproject.toml", "uv.lock"):
        (project / name).write_text("runtime")
    monkeypatch.setattr(sweep, "PROJECT", project)
    monkeypatch.setattr(sweep, "LAUNCHER", project / "run_hogwild_budget.sh")
    monkeypatch.setattr(sweep, "__file__", str(directory / "budget_curve_sweep.py"))
    source = tmp_path / "source"
    (source / "report").mkdir(parents=True)
    tasks = source / "report/tasks.csv"
    tasks.write_text("catalog")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {str((directory / "rollout.py").relative_to(project)): digest(directory / "rollout.py")}
    for name in ("source-files.json", "active-source-files.json"):
        (source / name).write_text(json.dumps(manifest))
    (source / "complete.json").write_text(json.dumps({
        "cases": 3500, "tasks_sha256": digest(tasks),
        "source_manifest_sha256": digest(source / "source-files.json"),
        "active_source_manifest_sha256": digest(source / "active-source-files.json")}))
    options = argparse.Namespace(source_root=source, wait_for_source=False)
    observed = sweep.identity(options)
    assert observed["source_tasks_sha256"] == digest(tasks)
    assert observed["code_sha256"]["run_hogwild_budget.sh"] == digest(sweep.LAUNCHER)
    tasks.write_text("changed catalog")
    with pytest.raises(ValueError, match="Completed source"):
        sweep.identity(options)
    tasks.write_text("catalog")
    (directory / "rollout.py").write_text("changed source")
    with pytest.raises(ValueError, match="Immutable source code"):
        sweep.identity(options)


def row(index=0, budget=1000, **extra):
    return {"mode": "native", "workers": 1, "index": index, "budget": budget, "correct": True,
            "generated_tokens": 12, "answer_tokens": 4, "source_trace_sha256": "a" * 64,
            "wall_seconds": 0.5, **extra}


def state(gpu, free=True, **extra):
    return {"gpu": gpu, "uuid": f"GPU-{gpu}", "memory_mib": 0 if free else 100,
            "utilization": 0, "compute_pids": [] if free else ["42"], "free": free, **extra}


@pytest.fixture
def args(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    monkeypatch.setattr(sweep, "identity", lambda args: {"source_root": str(args.source_root), "code": "fixed"})
    return argparse.Namespace(source_root=source, output=output, gpus=[2, 5], indices=[0, 1, 2, 3],
                              budgets=[1000], conditions=["native-w1"], wait_for_source=False,
                              max_prefill_rows=256, memory_ratio=0.75)


def plan(args, free=(2, 5)):
    return sweep.load_plan(args.output, sweep.configuration(args), args.gpus,
                           [state(gpu, gpu in free) for gpu in args.gpus])


def save_lane(args, lane, rows, tail=""):
    directory = sweep.lane_path(args.output, lane)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + tail)


@pytest.mark.parametrize("memory,utilization,pids,free", [
    (0, 0, "", True), (8, 0, "", True), (9, 0, "", False),
    (0, 1, "", False), (0, 0, "999", False), (-1, 0, "", False),
])
def test_gpu_admission_requires_memory_utilization_and_processes(monkeypatch, memory, utilization, pids, free):
    values = iter([f"2, GPU-two, {memory}, {utilization}\n", pids])
    monkeypatch.setattr(sweep.subprocess, "check_output", lambda *a, **kw: next(values))
    assert sweep.gpu_state(2)["free"] is free


def test_gpu_identity_mismatch_rejected(monkeypatch):
    values = iter(["1, GPU-one, 0, 0", ""])
    monkeypatch.setattr(sweep.subprocess, "check_output", lambda *a, **kw: next(values))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        sweep.gpu_state(2)


def test_plan_uses_only_initially_free_devices_and_fixed_modulo(args):
    args.gpus = [0, 2, 3, 5]
    current = plan(args, free=(2, 5))
    assert [(lane["gpu"], lane["modulo"], lane["remainder"]) for lane in current["lanes"]] == [(2, 2, 0), (5, 2, 1)]
    assert {key[2] for key in sweep.expected_keys(current, current["lanes"][0])} == {0, 2}
    assert {key[2] for key in sweep.expected_keys(current, current["lanes"][1])} == {1, 3}
    # Newly free GPUs never alter the partition of a saved run.
    assert sweep.load_plan(args.output, sweep.configuration(args), args.gpus,
                           [state(gpu) for gpu in args.gpus]) == current


def test_resume_can_narrow_devices_but_not_repartition_or_change_config(args):
    current = plan(args)
    assert sweep.load_plan(args.output, sweep.configuration(args), [5], [state(5)]) == current
    with pytest.raises(ValueError, match="cannot add"):
        sweep.load_plan(args.output, sweep.configuration(args), [6], [state(6)])
    args.memory_ratio = 0.5
    with pytest.raises(ValueError, match="identity differs"):
        sweep.load_plan(args.output, sweep.configuration(args), [2, 5], [state(2), state(5)])


def test_no_free_gpu_does_not_create_a_plan_or_launch(args, monkeypatch):
    monkeypatch.setattr(sweep, "gpu_state", lambda gpu: state(gpu, False))
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **kw: pytest.fail("No GPU launch allowed"))
    assert sweep.run(args) == 3
    assert not (args.output / "plan.json").exists()
    assert (args.output / "gpu-unavailable.json").exists()


def test_consolidation_ignores_only_partial_last_row_and_preserves_false(args):
    current = plan(args)
    save_lane(args, current["lanes"][0], [row(0), row(2, correct=False)], '{"index":')
    save_lane(args, current["lanes"][1], [row(1)])
    records = sweep.consolidate(args.output, current)
    assert len(records) == 3 and records[("native", 1, 2, 1000)]["correct"] is False
    assert len((args.output / "results.jsonl").read_text().splitlines()) == 3
    # A later complete append safely extends the frozen consolidated prefix.
    save_lane(args, current["lanes"][1], [row(1), row(3)])
    assert len(sweep.consolidate(args.output, current)) == 4


@pytest.mark.parametrize("replacement", [[], [row(0, correct=False)]])
def test_consolidation_rejects_removed_or_changed_committed_rows(args, replacement):
    current = plan(args)
    save_lane(args, current["lanes"][0], [row(0)])
    sweep.consolidate(args.output, current)
    before = (args.output / "results.jsonl").read_bytes()
    save_lane(args, current["lanes"][0], replacement)
    with pytest.raises(ValueError, match="disappeared or changed"):
        sweep.consolidate(args.output, current)
    assert (args.output / "results.jsonl").read_bytes() == before


def test_consolidation_rejects_duplicate_rows_and_wrong_lane(args):
    current = plan(args)
    save_lane(args, current["lanes"][0], [row(0), row(0)])
    with pytest.raises(ValueError, match="Duplicate result"):
        sweep.consolidate(args.output, current)
    save_lane(args, current["lanes"][0], [row(1)])
    with pytest.raises(ValueError, match="outside the configured lane"):
        sweep.consolidate(args.output, current)


class Process:
    def __init__(self, code=0):
        self.pid = 987654
        self.code = code

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code


def simulate_workers(args, monkeypatch, *, completed=True, code=0):
    calls, stopped = [], []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        index = int(command[command.index("--index-remainder") + 1])
        current = json.loads((args.output / "plan.json").read_text())
        lane = current["lanes"][index]
        if completed:
            save_lane(args, lane, [row(key[2], key[3]) for key in sorted(sweep.expected_keys(current, lane))])
        return Process(code)
    monkeypatch.setattr(sweep.subprocess, "Popen", popen)
    monkeypatch.setattr(sweep, "stop_owned", lambda children: stopped.extend(children))
    return calls, stopped


def test_success_requires_every_requested_tuple_and_reports_without_plots(args, monkeypatch):
    monkeypatch.setattr(sweep, "gpu_state", state)
    calls, stopped = simulate_workers(args, monkeypatch)
    assert sweep.run(args) == 0
    assert len(calls) == len(stopped) == 2
    for command, kwargs in calls:
        assert command[1] == "--worker" and kwargs["start_new_session"] is True
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == f"GPU-{kwargs['env']['HOGWILD_GPU']}"
    complete = json.loads((args.output / "complete.json").read_text())
    assert complete["completed"] == complete["target"] == 4
    assert (args.output / "report/summary.json").exists()
    assert not list((args.output / "report").glob("*.png"))
    assert json.loads((args.output / "progress.json").read_text())["cells"] == [
        {"condition": "native-w1", "budget": 1000, "completed": 4, "target": 4}]


def test_worker_zero_exit_with_missing_rows_is_not_complete(args, monkeypatch):
    monkeypatch.setattr(sweep, "gpu_state", state)
    simulate_workers(args, monkeypatch, completed=False)
    assert sweep.run(args) == 3
    assert not (args.output / "complete.json").exists()
    assert (args.output / "gpu-unavailable.json").exists()


def test_busy_resume_lane_is_never_rechecked_or_reacquired(args, monkeypatch):
    plan(args)
    queries = []
    def query(gpu):
        queries.append(gpu)
        # GPU2 would be free on a forbidden later acquisition attempt.
        return state(gpu, gpu != 2 or queries.count(gpu) > 1)
    monkeypatch.setattr(sweep, "gpu_state", query)
    calls, _ = simulate_workers(args, monkeypatch)
    assert sweep.run(args) == 3
    assert queries.count(2) == 1
    assert len(calls) == 1 and calls[0][1]["env"]["HOGWILD_GPU"] == "5"
    assert len((args.output / "results.jsonl").read_text().splitlines()) == 2


def test_gpu_becoming_busy_between_admission_and_launch_is_skipped(args, monkeypatch):
    counts = {}
    def query(gpu):
        counts[gpu] = counts.get(gpu, 0) + 1
        return state(gpu, gpu != 2 or counts[gpu] == 1)
    monkeypatch.setattr(sweep, "gpu_state", query)
    calls, _ = simulate_workers(args, monkeypatch)
    assert sweep.run(args) == 3
    assert len(calls) == 1 and calls[0][1]["env"]["HOGWILD_GPU"] == "5"


def test_worker_failure_stops_only_started_groups_and_preserves_attempt(args, monkeypatch):
    monkeypatch.setattr(sweep, "gpu_state", state)
    calls, stopped = simulate_workers(args, monkeypatch, completed=False, code=1)
    with pytest.raises(RuntimeError, match="Worker GPU .* exited 1"):
        sweep.run(args)
    assert len(stopped) == len(calls) == 2
    attempt = json.loads(next((args.output / "attempts").glob("*.json")).read_text())
    assert attempt["exit_code"] == 1 and "error" in attempt


@pytest.mark.parametrize("marker", sweep.STOP_MARKERS)
def test_source_hold_or_superseded_stops_before_admission(args, monkeypatch, marker):
    (args.source_root / marker).write_text("{}")
    monkeypatch.setattr(sweep, "gpu_state", lambda gpu: pytest.fail("Source is stopped"))
    with pytest.raises(RuntimeError, match="Stop marker"):
        sweep.run(args)


def test_full_completion_domain_is_all_28000_unique_cells(args):
    args.indices = list(range(500))
    args.budgets = list(sweep.report.BUDGETS)
    args.conditions = list(sweep.CONDITIONS)
    current = plan(args)
    assert len(sweep.expected_keys(current)) == 28000
    pieces = [sweep.expected_keys(current, lane) for lane in current["lanes"]]
    assert not pieces[0] & pieces[1]
    assert pieces[0] | pieces[1] == sweep.expected_keys(current)


def test_completed_resume_never_launches(args, monkeypatch):
    current = plan(args)
    for lane in current["lanes"]:
        save_lane(args, lane, [row(key[2], key[3]) for key in sweep.expected_keys(current, lane)])
    monkeypatch.setattr(sweep, "gpu_state", lambda gpu: state(gpu, False))
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **kw: pytest.fail("Already complete"))
    assert sweep.run(args) == 0


def test_worker_command_passes_protocol_and_explicit_lane(args):
    args.wait_for_source = True
    command = sweep.worker_command(args, plan(args)["lanes"][1])
    assert "--wait-for-source" in command
    assert command[command.index("--index-modulo") + 1] == "2"
    assert command[command.index("--index-remainder") + 1] == "1"
    assert command[command.index("--answer-tokens") + 1] == "8"
    assert command[command.index("--seed") + 1] == "42"


def test_source_identity_drift_stops_before_launch(args, monkeypatch):
    original = sweep.configuration
    calls = []
    def changed_configuration(args):
        calls.append(None)
        return {**original(args), "drift": len(calls) > 1}
    monkeypatch.setattr(sweep, "configuration", changed_configuration)
    monkeypatch.setattr(sweep, "gpu_state", state)
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **kw: pytest.fail("Code drift"))
    with pytest.raises(ValueError, match="changed before worker launch"):
        sweep.run(args)


def test_stop_owned_signals_group_even_when_leader_already_failed(monkeypatch):
    signals = []
    monkeypatch.setattr(sweep.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    sweep.stop_owned([Process(1)])
    assert signals == [(987654, sweep.signal.SIGTERM), (987654, 0), (987654, sweep.signal.SIGKILL)]


def test_evaluator_preflight_occupancy_race_skips_lane_without_stopping_other_results(args, monkeypatch):
    monkeypatch.setattr(sweep, "gpu_state", state)
    calls, _ = simulate_workers(args, monkeypatch)
    normal_popen = sweep.subprocess.Popen
    def raced_popen(command, **kwargs):
        if kwargs["env"]["HOGWILD_GPU"] == "2":
            kwargs["stdout"].write(b"RuntimeError: GPU 2 is occupied: memory=100 MiB, processes='555'\n")
            kwargs["stdout"].flush()
            child = Process(1)
            child.pid = 987653
            return child
        return normal_popen(command, **kwargs)
    monkeypatch.setattr(sweep.subprocess, "Popen", raced_popen)
    assert sweep.run(args) == 3
    assert len(calls) == 1 and calls[0][1]["env"]["HOGWILD_GPU"] == "5"
    progress = json.loads((args.output / "progress.json").read_text())
    assert progress["completed"] == 2
    assert progress["skipped_lanes"][0]["reason"] == "occupied at evaluator preflight; not retried"


@pytest.mark.parametrize("prefix", ["GPU_PREFLIGHT {}\n", "START native-w1 i0\n"])
def test_occupied_text_after_gpu_load_is_not_a_harmless_admission_failure(tmp_path, prefix):
    log = tmp_path / "worker.log"
    log.write_text(prefix + "RuntimeError: GPU 2 is occupied: memory=100 MiB\n")
    assert not sweep.occupied_before_load({"gpu": 2, "log": str(log)})


def test_source_stop_after_first_launch_stops_only_that_owned_worker(args, monkeypatch):
    monkeypatch.setattr(sweep, "gpu_state", state)
    stopped = []
    def launch(command, **kwargs):
        (args.source_root / "quality-hold.json").write_text('{"reason":"source diagnostic"}')
        return Process(None)
    monkeypatch.setattr(sweep.subprocess, "Popen", launch)
    monkeypatch.setattr(sweep, "stop_owned", lambda children: stopped.extend(children))
    with pytest.raises(RuntimeError, match="quality-hold.json"):
        sweep.run(args)
    assert len(stopped) == 1
    assert json.loads((args.source_root / "quality-hold.json").read_text()) == {"reason": "source diagnostic"}


def test_resume_rejects_replaced_gpu_uuid(args, monkeypatch):
    plan(args)
    monkeypatch.setattr(sweep, "gpu_state", lambda gpu: state(gpu, uuid="GPU-replaced"))
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **kw: pytest.fail("Wrong physical device"))
    with pytest.raises(RuntimeError, match="UUID changed"):
        sweep.run(args)


def test_root_lock_blocks_duplicate_coordinator(args, monkeypatch):
    monkeypatch.setattr(sweep, "run", lambda args: pytest.fail("Duplicate coordinator"))
    with (args.output / ".coordinator.lock").open("a") as lock:
        sweep.fcntl.flock(lock, sweep.fcntl.LOCK_EX | sweep.fcntl.LOCK_NB)
        with pytest.raises(SystemExit):
            sweep.main(["--source-root", str(args.source_root), "--output", str(args.output), "--gpus", "2", "5"])
