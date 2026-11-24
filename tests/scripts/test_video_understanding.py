"""Guard against accepting answer-only or prematurely timed video reproductions."""

from copy import deepcopy

import pytest
from minisgl_examples.training.video_understanding import experiment as exp


def result():
    return {
        "task": exp.task(),
        "control": "normal",
        "reply_time": 12.0,
        "answer_text": "Smoke is billowing from the building.",
        "hit_eos": True,
        "gates": [{"reply": False}, {"reply": False}, {"reply": True}],
    }


def test_correct_event_and_timing():
    score = exp.assess(result())
    assert score["behavior_reproduced"]
    assert not score["facts"]["explosion_or_fire"]  # Do not invent a stronger semantic match.


@pytest.mark.parametrize(
    "field,value",
    [
        ("reply_time", 5.0),
        ("reply_time", 20.0),
        ("answer_text", "Vehicles are parked in the square."),
        ("hit_eos", False),
    ],
)
def test_answer_or_timing_alone_is_insufficient(field, value):
    run = result()
    run[field] = value
    assert not exp.assess(run)["behavior_reproduced"]


def test_early_gate_is_not_a_success_even_with_a_correct_later_answer():
    run = result()
    run["gates"] = [{"reply": True}]
    assert not exp.assess(run)["behavior_reproduced"]


def test_frozen_stream_must_finish_waiting():
    run = result()
    run.update(control="freeze", answer_text="", reply_time=None, hit_eos=False)
    run["gates"] = [{"reply": False} for _ in run["task"]["frames"]]
    assert exp.assess(run)["correct_wait"]
    incomplete = deepcopy(run)
    incomplete["gates"].pop()
    assert not exp.assess(incomplete)["correct_wait"]
    run["gates"][-1]["reply"] = True
    assert not exp.assess(run)["correct_wait"]


def test_frame_control_preserves_initial_history_and_removes_future_frames():
    assert exp.frame_indices(0, "normal") == (0, 0)
    assert exp.frame_indices(2, "normal") == (1, 2)
    assert exp.frame_indices(2, "freeze") == (1, 1)
    assert exp.frame_indices(3, "freeze") == (1, 1)
    with pytest.raises(ValueError):
        exp.frame_indices(2, "other")


def test_manifest_is_pinned_and_reply_sample_in_reference_window():
    data = exp.task()
    t = data["frames"][data["expected_first_reply_frame"]]["time"]
    a, b = data["annotation"]["answer"][0]["reply_timespan"]
    assert a <= t <= b
    assert data["revision"] == "7f59bce2a6cdf40b8a895767aac7a52a92136c5f"


def test_explicit_replay_settings_and_cached_report_guard():
    from minisgl_examples.training.video_understanding import experiment as exp

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
    from minisgl_examples.training.student import load_example
    from minisgl_examples.training.video_understanding import experiment as exp

    trace = Trace(
        batches=[ForwardBatch(0, "prefill", (ForwardRequest((1, 2), 0, 0, ((0, 0),)),))], steps=1
    )
    data = {
        "trace": asdict(trace),
        "targets": [-100, 2],
        "setup": {"replay_attention_backend": backend, "replay_query_chunk_size": chunk_size},
    }
    prepared = {**data, "trace": exp.prepare_trace(data, "cpu")}
    restored = load_example(prepared)["trace"].to("cpu")
    assert restored.attention_backend == backend
    assert restored.query_chunk_size == chunk_size
    assert restored.num_tokens == 2
    assert asdict(restored) == data["trace"]
