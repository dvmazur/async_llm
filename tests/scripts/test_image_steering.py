"""Acceptance checks must distinguish an answer from a reproduced trajectory."""

import pytest
from minisgl_examples.training.image_steering import experiment as exp


def rollout(*, old=True, new=True, revision=True, eos=True, changed=True, answer="-3"):
    before = "September is 20, August is 17." if old else "Let me read the table."
    after = ("The corrected picture " if revision else "") + ("shows 14. " if new else "")
    after += "\\boxed{" + answer + "}"
    return {
        "before_update_text": before,
        "after_update_text": after,
        "text": before + after,
        "image_replaced": changed,
        "hit_eos": eos,
        "task": {"answer_before": "3", "answer_after": "-3"},
    }


@pytest.mark.parametrize("missing", ["old", "new", "eos"])
def test_correct_answer_alone_does_not_pass(missing):
    result = exp.stages(rollout(**{missing: False}))
    assert result["correct_answer"]
    assert not result["behavior_reproduced"]


def test_complete_reaction_and_last_box():
    result = rollout()
    result["text"] = "\\boxed{3}" + result["text"]
    assert exp.stages(result)["behavior_reproduced"]
    assert exp.stages(result)["final_answer"] == "-3"


def test_no_update_has_old_answer_and_is_not_a_revision():
    result = exp.stages(rollout(changed=False, answer="3"))
    assert result["correct_answer"]
    assert not result["behavior_reproduced"]


def test_missing_box_does_not_pass():
    result = rollout()
    result["text"] = "The answer is -3."
    assert not exp.stages(result)["correct_answer"]


def test_outputs_cannot_silently_overwrite(tmp_path):
    path = tmp_path / "rollout.json"
    exp.save(path, {"version": 1})
    with pytest.raises(FileExistsError):
        exp.save(path, {"version": 2})
    assert exp.load(path) == {"version": 1}


def test_visual_revision_need_not_use_a_specific_acknowledgement_phrase():
    result = exp.stages(rollout(revision=False))
    assert result["behavior_reproduced"]
    assert not result["revision_language_after_event"]


def test_explicit_replay_settings_and_cached_report_guard():
    from minisgl_examples.training.image_steering import experiment as exp

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
    from minisgl_examples.training.image_steering import experiment as exp
    from minisgl_examples.training.student import load_example

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
