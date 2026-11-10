"""Standalone MATH-500 CLI and helpers, independent of the removed delayed-input eval."""
import json
import sys

import pytest

from eval import math500
from eval.grading import boxed, grade_one


def test_model_only_cli_uses_recorded_async_demo(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(math500, "run", lambda args: seen.append(
        math500.read_config(args.config, model_path=args.model, backend=args.backend)))
    monkeypatch.setattr(sys, "argv", ["math500", "run", "--model", "local-gemma",
                                     "--output", str(tmp_path / "results"), "--limit", "1"])
    math500.main()
    config, = seen
    assert config["model_path"] == "local-gemma"
    assert config["mode"] == "async_thoughts" and config["record_training_trace"]
    assert config["max_think_tokens"] == config["max_answer_tokens"] == 8192
    assert config["sampling"] == dict(temperature=1.0, top_p=0.95, top_k=64)


def test_optional_config_and_model_override(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(dict(model_path="old", max_think_tokens=128,
                                   max_answer_tokens=64, sampling=dict(temperature=0.7))))
    config = math500.read_config(path, model_path="new")
    assert config["model_path"] == "new" and config["max_new_tokens"] == 192
    assert config["sampling"] == dict(temperature=0.7, top_p=0.95, top_k=64)
    assert math500.read_config(path)["model_path"] == "old"


@pytest.mark.parametrize("metadata,expected", [
    ({"dtype": "bfloat16"}, None),
    ({"quantization_config": None}, None),
    ({"quantization_config": {"quant_method": "compressed-tensors"}}, "fp8"),
    ({"quantization_config": {"quant_method": "fp8"}}, "fp8"),
])
def test_local_checkpoint_selects_quantization(tmp_path, metadata, expected):
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    config = math500.read_config(model_path=str(tmp_path))
    assert config["engine"]["quantization"] == expected
    # The external baseline leaves checkpoint detection to SGLang itself.
    baseline = math500.read_config(model_path=str(tmp_path), backend="sglang")
    assert "quantization" not in baseline["engine"]


def test_explicit_quantization_override_is_preserved(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"dtype": "bfloat16"}))
    override = tmp_path / "run.json"
    override.write_text(json.dumps({"engine": {"quantization": "fp8"}}))
    config = math500.read_config(override, model_path=str(tmp_path))
    assert config["engine"]["quantization"] == "fp8"  # Loader will reject incompatible weights.


def test_missing_model_rejected_before_loading(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["math500", "run", "--output", "unused"])
    with pytest.raises(SystemExit) as exc:
        math500.main()
    assert exc.value.code == 2
    with pytest.raises(ValueError, match="--model"):
        math500.read_config()


@pytest.mark.parametrize("overrides", [
    None, [], {"do_sample": True}, {"temperature": -1}, {"temperature": float("nan")},
    {"temperature": float("inf")}, {"temperature": True}, {"top_p": 0}, {"top_p": 1.1},
    {"top_p": "0.95"}, {"top_k": 0}, {"top_k": -2}, {"top_k": 64.5},
    {"top_k": True}, {"temperature": 0},
])
def test_invalid_sampling(overrides):
    with pytest.raises(ValueError, match="sampling|greedy"):
        math500.sampling_config(overrides)


def test_sampling_overrides():
    assert math500.sampling_config(dict(temperature=0, top_p=1, top_k=1)) == dict(
        temperature=0, top_p=1, top_k=1)


def test_question_partitions_cover_range_without_overlap():
    cases = list(range(21))
    parts = [math500.select_cases(cases, start=3, limit=11, shard_index=i, num_shards=4)
             for i in range(4)]
    assert sorted(x for part in parts for x in part) == list(range(3, 14))


@pytest.mark.parametrize("options", [dict(start=-1), dict(limit=0), dict(start=100),
    dict(shard_index=0), dict(num_shards=2), dict(shard_index=2, num_shards=2),
    dict(shard_index=0, num_shards=0)])
def test_invalid_selection(options):
    with pytest.raises(ValueError):
        math500.select_cases(list(range(10)), **options)


def test_boxed_nested_and_incomplete():
    assert boxed(r"earlier \boxed{3}; final \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert boxed(r"\boxed{3}; incomplete \boxed{") == "3"
    assert boxed("no final answer") is None
    assert grade_one(dict(writer_text="none", answer="2"))["correct"] is False


def test_math_verify_grading():
    pytest.importorskip("math_verify")
    assert grade_one(dict(writer_text=r"\boxed{\frac{1}{2}}", answer="0.5"))["correct"]
    assert not grade_one(dict(writer_text=r"\boxed{3}", answer="2"))["correct"]
