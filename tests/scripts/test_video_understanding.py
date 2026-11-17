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
