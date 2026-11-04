"""Saved and freshly collected inputs share the same student-training contract."""

import copy
import json
from dataclasses import asdict

import pytest
from minisgl.training import Trace
from minisgl_examples.training import student
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tests.scripts.training_helpers import source as source


def test_trace_roundtrip_preserves_targets_and_rollout_metadata(source, tmp_path):
    model_path, trace_path = source
    payload = json.loads(trace_path.read_text())
    payload["examples"][0].update(
        prompt="test", prompt_ids=[1, 2], logprobs=[None] * 6, targets=[-100, 2, 3, 4, 5, 6]
    )
    original = copy.deepcopy(payload)
    example = student.load_example(payload)
    assert isinstance(example["trace"], Trace)
    assert example["model"] == str(model_path)
    assert payload == original
    destination = student.save_example(example, tmp_path / "saved.json")
    saved = json.loads(destination.read_text())
    assert saved["model"] == str(model_path)
    assert len(saved["examples"]) == 1
    assert saved["examples"][0]["trace"] == json.loads(json.dumps(asdict(example["trace"])))
    reloaded = student.load_example(destination)
    assert reloaded == example
    assert reloaded["targets"] == [-100, 2, 3, 4, 5, 6]
    assert reloaded["prompt_ids"] == [1, 2] and reloaded["logprobs"] == [None] * 6
    with pytest.raises(FileExistsError):
        student.save_example(example, destination)


@pytest.mark.parametrize("count", [0, 2])
def test_single_trace_demos_reject_wrong_example_count(source, count):
    _, trace_path = source
    payload = json.loads(trace_path.read_text())
    payload["examples"] *= count
    with pytest.raises(ValueError):
        student.load_example(payload)


@pytest.mark.parametrize("targets", [[1], [-100] * 6, [32] * 6, [True] * 6])
@pytest.mark.parametrize("from_disk", [False, True], ids=["memory", "disk"])
def test_invalid_targets_rejected_before_model_allocation(source, monkeypatch, targets, from_disk):
    model_path, trace_path = source
    payload = json.loads(trace_path.read_text())
    payload["examples"][0]["targets"] = targets
    trace_path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        student.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("Allocated model before validating supervision"),
    )
    with pytest.raises(ValueError, match="Invalid or misaligned"):
        student.load_student(trace_path if from_disk else payload, str(model_path), device="cpu")


def test_tokenizer_mismatch_rejected_before_model_allocation(source, tmp_path, monkeypatch):
    _, trace_path = source
    other = tmp_path / "different-tokenizer"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"other": 0, "1": 1}, unk_token="other"))
    )
    tokenizer.save_pretrained(other)
    monkeypatch.setattr(
        student.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("Allocated model before validating tokenizer"),
    )
    with pytest.raises(ValueError, match="tokenizers differ"):
        student.load_student(trace_path, str(other), device="cpu")
