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
