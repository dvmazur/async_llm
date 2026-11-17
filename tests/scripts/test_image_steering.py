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
