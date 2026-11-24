"""Accept meaningful short-game success; never mistake CPU smoke checks for teacher data."""

import pytest
from minisgl_examples.training.doom_health.environment import Setup
from minisgl_examples.training.doom_health.experiment import assert_real, assess


def data():
    return {
        "initial_h_tokens": 5,
        "steps": [
            {
                "health_before": 100,
                "health_after": 96,
                "consolidated": True,
                "h_before": 5,
                "dead": False,
                "return": 4,
            },
            {
                "health_before": 96,
                "health_after": 100,
                "consolidated": False,
                "h_before": 20,
                "dead": False,
                "return": 8,
            },
        ],
    }


def test_healing_alive_and_reading_consolidated_h():
    assert assess(data())["goal_achieved"]


@pytest.mark.parametrize("change", ["no_healing", "dead", "no_append", "unused_history"])
def test_living_reward_alone_does_not_pass(change):
    d = data()
    if change == "no_healing":
        d["steps"][1]["health_after"] = 92
    if change == "dead":
        d["steps"][1]["dead"] = True
    if change == "no_append":
        d["steps"][0]["consolidated"] = False
    if change == "unused_history":
        d["steps"][1]["h_before"] = 5
    assert not assess(d)["goal_achieved"]


def test_cpu_preflight_is_never_a_real_teacher():
    for kind in ("cpu_environment_preflight", "cpu_scheduler_stub", None):
        with pytest.raises(ValueError):
            assert_real({"runtime": {"kind": kind}})


def test_config_validation():
    Setup().validate()
    for options in ({"frame_skip": 0}, {"thought_length": 1}, {"min_actions": 20}):
        with pytest.raises(ValueError):
            Setup(**options).validate()


def test_gpu_guard_rejects_other_process_even_with_free_memory(monkeypatch):
    from minisgl_examples.training.doom_health import gpu

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("DOOM_ALLOW_SHARED_GPU", raising=False)
    monkeypatch.setattr(
        gpu.subprocess,
        "check_output",
        lambda args, **kwargs: (
            "GPU-test, 123456789\n"
            if "--query-compute-apps=gpu_uuid,pid" in args
            else "0, GPU-test, 10, 80000, 81559, 0\n"
        ),
    )
    assert not gpu.gpu_state()[0]["available"]
    with pytest.raises(RuntimeError, match="No free GPU"):
        gpu.require_free_gpu()


def test_shared_gpu_requires_allocator_budget_plus_reserve(monkeypatch):
    from minisgl_examples.training.doom_health import gpu

    monkeypatch.setenv("DOOM_ALLOW_SHARED_GPU", "1")
    monkeypatch.setenv("DOOM_GPU_LIMIT_GIB", "32")
    remaining = [67237]
    monkeypatch.setattr(
        gpu.subprocess,
        "check_output",
        lambda args, **kwargs: (
            "GPU-test, 123456789\n"
            if "--query-compute-apps=gpu_uuid,pid" in args
            else f"1, GPU-test, 13843, {remaining[0]}, 81559, 50\n"
        ),
    )
    assert gpu.gpu_state()[0]["available"]
    remaining[0] = 47 * 1024
    assert not gpu.gpu_state()[0]["available"]


def test_gpu_guard_respects_explicit_device(monkeypatch):
    from minisgl_examples.training.doom_health import gpu

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        gpu,
        "gpu_state",
        lambda: [
            {"index": 0, "uuid": "free", "available": True},
            {"index": 1, "uuid": "occupied", "available": False},
        ],
    )
    with pytest.raises(RuntimeError, match="No free GPU"):
        gpu.require_free_gpu()


def test_explicit_replay_settings_and_cached_report_guard():
    from minisgl_examples.training.doom_health import experiment as exp

    assert exp.replay_options({}) == {"attention_backend": "torch", "query_chunk_size": 64}
    setup = {"replay_attention_backend": "auto", "replay_query_chunk_size": 1024}
    expected = {"attention_backend": "auto", "query_chunk_size": 1024}
    assert exp.replay_options(setup) == expected
    report = {"replay_options": expected, "passed": True}
    assert exp.require_replay_options(report, setup) is report
    for stale in ({}, {"replay_options": exp.replay_options({})}):
        with pytest.raises(ValueError, match="fresh output directory"):
            exp.require_replay_options(stale, setup)
    for bad in (
        {"replay_attention_backend": "unknown"},
        {"replay_query_chunk_size": 0},
        {"replay_query_chunk_size": True},
    ):
        with pytest.raises(ValueError):
            exp.replay_options(bad)


@pytest.mark.parametrize("backend,chunk_size", [("torch", 64), ("auto", 1024)])
def test_prepared_replay_settings_survive_shared_loader(backend, chunk_size):
    from dataclasses import asdict

    from minisgl.training.trace import ForwardBatch, ForwardRequest, Trace
    from minisgl_examples.training.doom_health import experiment as exp
    from minisgl_examples.training.student import load_example

    trace = Trace(
        batches=[ForwardBatch(0, "prefill", (ForwardRequest((1, 2), 0, 0, ((0, 0),)),))], steps=1
    )
    data = {
        "trace": asdict(trace),
        "targets": [-100, 2],
        "setup": {"replay_attention_backend": backend, "replay_query_chunk_size": chunk_size},
    }
    prepared = exp.prepare_replay_source(data)
    restored = load_example(prepared)["trace"].to("cpu")
    assert restored.attention_backend == backend
    assert restored.query_chunk_size == chunk_size
    assert restored.num_tokens == 2
    assert asdict(restored) == data["trace"]


def test_train_and_reload_pass_the_same_configured_trace(tmp_path, monkeypatch):
    import hashlib
    from dataclasses import asdict

    import torch
    from minisgl.training.trace import ForwardBatch, ForwardRequest, Trace
    from minisgl_examples.training import student
    from minisgl_examples.training.doom_health import experiment as exp

    trace = Trace(
        batches=[ForwardBatch(0, "prefill", (ForwardRequest((1, 2), 0, 0, ((0, 0),)),))], steps=1
    )
    setup = {
        "dtype": "float16",
        "replay_attention_backend": "auto",
        "replay_query_chunk_size": 1024,
    }
    example = {
        "runtime": {"kind": "async_llm_cuda"},
        "setup": setup,
        "trace": asdict(trace),
        "targets": [-100, 2],
        "assessment": {"goal_achieved": True},
        "prompts": {},
        "provenance": {},
        "steps": [{"current_frame_hash": "same"}],
    }
    teacher, baseline = tmp_path / "teacher.json", tmp_path / "baseline.json"
    exp.save(teacher, {**example, "model": "teacher"})
    exp.save(baseline, {**example, "model": "student"})
    monkeypatch.setattr(exp, "require_free_gpu", lambda: None)
    monkeypatch.setattr(exp, "apply_memory_limit", lambda: None)
    monkeypatch.setattr(exp, "parameter_delta", lambda *args: {"max_abs_delta": 1.0})
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    def parity(source, output, **kwargs):
        exp.save(
            output,
            {
                "parity": {"passed": True},
                "replay_options": exp.replay_options(setup),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        )

    seen = []

    def capture(source, stage):
        actual = student.load_example(source)["trace"]
        seen.append((stage, actual.attention_backend, actual.query_chunk_size))

    def train(source, model, checkpoint, **kwargs):
        capture(source, "train")
        return {"checkpoint": str(checkpoint), "final": {"loss": 0.25}}

    def reload(source, checkpoint, **kwargs):
        capture(source, "reload")
        return {"loss": 0.25}

    monkeypatch.setattr(exp, "check_replay", parity)
    monkeypatch.setattr(student, "train_student", train)
    monkeypatch.setattr(student, "verify_checkpoint", reload)
    output = tmp_path / "trained"
    trained = exp.train(teacher, "student", baseline, output, steps=1)
    verified = tmp_path / "verified.json"
    exp.verify(teacher, output / "checkpoint", output / "training.json", verified)
    assert seen == [("train", "auto", 1024), ("reload", "auto", 1024)]
    assert trained["replay_options"] == exp.replay_options(setup)
    assert exp.load(verified)["replay_options"] == trained["replay_options"]
    assert exp.load(verified)["passed"]


@pytest.mark.parametrize("explicit_sharing", [False, True])
def test_notebook_requires_opt_in_for_an_occupied_gpu(monkeypatch, explicit_sharing):
    import ast
    import json
    import os
    from pathlib import Path

    from minisgl_examples.training.doom_health import experiment, gpu

    monkeypatch.delenv("DOOM_ALLOW_SHARED_GPU", raising=False)
    monkeypatch.setenv("DOOM_GPU_LIMIT_GIB", "32")
    if explicit_sharing:
        monkeypatch.setenv("DOOM_ALLOW_SHARED_GPU", "1")
    notebook = Path(experiment.__file__).with_name("doom_health_overfit.ipynb")
    cell = next(c for c in json.loads(notebook.read_text())["cells"] if c["cell_type"] == "code")
    defaults = [
        node
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "setdefault"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "DOOM_ALLOW_SHARED_GPU"
    ]
    exec(compile(ast.Module(body=defaults, type_ignores=[]), str(notebook), "exec"), {"os": os})
    monkeypatch.setattr(
        gpu.subprocess,
        "check_output",
        lambda args, **kwargs: (
            "GPU-test, 123456789\n"
            if "--query-compute-apps=gpu_uuid,pid" in args
            else "1, GPU-test, 13843, 67237, 81559, 50\n"
        ),
    )
    assert gpu.gpu_state()[0]["available"] is explicit_sharing
