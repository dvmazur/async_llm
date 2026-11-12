"""Parallel-thinking scheduling, trace supervision, notebooks, and process launching."""

import ast
import asyncio
import hashlib
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import pytest
import torch
from minisgl.training import Trace
from minisgl_examples.training.parallel_thinking import demo
from minisgl_examples.training.parallel_thinking.rollout import MathFormatting
from minisgl_examples.training.student import load_example, save_example

from tests.scripts.training_helpers import collect_example


@pytest.fixture(autouse=True)
def cpu_profiling(monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())


@pytest.mark.parametrize("synchronized", [True, False])
def test_recorded_schedules_preserve_samples_and_publications(synchronized):
    result, example = collect_example(synchronized)
    trace, targets = example["trace"], example["targets"]
    assert result["generated_tokens"] == {"Alice": 4, "Bob": 4}
    assert result["completed_steps"] == 4
    if synchronized:
        assert result["max_token_gap"] == 1
    else:
        assert result["max_token_gap"] > 1
    assert len(targets) == trace.num_tokens == len(example["logprobs"])
    assert sum(target != -100 for target in targets) == 10
    assert any(update.refs for update in trace.updates)
    assert any(not update.refs for update in trace.updates)
    assert any(
        len(request.cache_view) == 6 for batch in trace.batches for request in batch.requests
    )
    assert Trace.from_dict(asdict(trace)) == trace
    assert all(
        target == (token.token_id + 2) % 32
        for token, target in zip(trace.tokens, targets)
        if target != -100
    )
    if synchronized:
        assert any(len(batch.requests) == 2 for batch in trace.batches if batch.kind == "decode")


def test_collection_retains_metadata_through_disk_roundtrip(tmp_path, monkeypatch):
    result, example = collect_example(True)

    async def solve(*args, **kwargs):
        return {
            **result,
            "example": {
                **example,
                "model": args[0],
                "prompt_ids": result["prompt_ids"],
                "rollout": result,
            },
        }

    monkeypatch.setattr(demo, "solve", solve)
    source = asyncio.run(demo.collect_trace("teacher"))
    loaded = load_example(save_example(source, tmp_path / "trace.json"))
    assert loaded["trace"] == source["trace"]
    assert loaded["targets"] == source["targets"]
    assert loaded["model"] == "teacher"
    assert loaded["prompt_ids"] == result["prompt_ids"]
    assert loaded["rollout"]["answer_text"] == result["answer_text"]


@pytest.mark.parametrize("from_disk", [False, True])
def test_training_accepts_collected_or_saved_trace(tmp_path, monkeypatch, from_disk):
    _, example = collect_example(True)
    source = save_example(example, tmp_path / "trace.json") if from_disk else example
    calls = []

    def train(data, model, output, **options):
        calls.append((data, model, output, options))
        return {"checkpoint": str(output)}

    monkeypatch.setattr(demo.student, "train_student", train)
    result = demo.train_student("student", source, tmp_path / "checkpoint", steps=3, chunk_size=4)
    data, model, output, options = calls[0]
    assert data["trace"] == example["trace"] and data["targets"] == example["targets"]
    assert data["trace"].query_chunk_size == 4 and data["trace"].attention_checkpointing
    assert model == "student" and result["checkpoint"] == str(output)
    assert options["optimizer_device"] == "cpu"
    assert options["dtype"] == options["master_dtype"] == torch.bfloat16
    assert options["steps"] == 3


def test_saved_trace_workflow_skips_collection_and_reuses_prompt(tmp_path, monkeypatch):
    result, example = collect_example(False)
    source = {**example, "model": "teacher", "rollout": result, "prompt_ids": result["prompt_ids"]}
    path = save_example(source, tmp_path / "source.json")
    monkeypatch.setattr(demo, "collect_trace", lambda *a, **k: pytest.fail("recollected teacher"))
    calls = []

    def isolated_step(function, *args, **kwargs):
        calls.append((function, args, kwargs))
        if function is demo.train_student:
            assert args[1]["trace"] == example["trace"]
            return {"checkpoint": str(args[2])}
        assert function is demo.solve
        assert kwargs["prompt_ids"] == result["prompt_ids"]
        return {**result, "example": example}

    monkeypatch.setattr(demo, "isolated", isolated_step)
    report = demo.run_workflow(
        "unused", "student", tmp_path / "run", trace=path, synchronized=False
    )
    assert len(calls) == 2 and report["teacher"]["answer_text"] == result["answer_text"]
    assert load_example(tmp_path / "run" / "trace.json")["trace"] == example["trace"]
    assert [call[2]["log_path"].name for call in calls] == ["training.log", "student.log"]


@pytest.mark.parametrize("command", ["record", "train", "verify", "run"])
def test_cli_routes_each_stage(tmp_path, monkeypatch, command):
    _, example = collect_example(True)
    trace = save_example(example, tmp_path / "source.json")
    calls = []

    def invoke(*args, **kwargs):
        calls.append((args, kwargs))
        return example if command == "record" else {"loss": 0.1}

    function = {
        "record": "collect_trace",
        "train": "train_student",
        "verify": "verify_checkpoint",
        "run": "run_workflow",
    }[command]
    monkeypatch.setattr(demo, function, invoke)
    monkeypatch.setattr(
        demo, "isolated", lambda function, *args, **kwargs: function(*args, **kwargs)
    )
    if command == "record":
        arguments = [command, "teacher", str(tmp_path / "recorded"), "--no-cuda-graphs"]
    elif command == "run":
        arguments = [command, "--trace", str(trace), "--output", str(tmp_path / "run")]
    else:
        arguments = [command, str(trace), "student", str(tmp_path / "output"), "--device", "cpu"]
    demo.main(arguments)
    assert len(calls) == 1
    if command == "run":
        assert calls[0][1]["trace"] == trace
    elif command == "record":
        assert not calls[0][1]["cuda_graphs"]
    else:
        assert calls[0][1]["device"] == "cpu"


@pytest.mark.parametrize("path", sorted(Path(demo.__file__).parent.glob("*.ipynb")))
def test_notebook_runs_collection_training_and_generation(path, tmp_path, monkeypatch):
    from minisgl_examples.training import process

    calls = []

    def isolated_step(function, *args, **kwargs):
        calls.append(function)
        if function in (demo.solve, demo.collect_trace):
            result, example = collect_example(kwargs.get("synchronized", True))
            if function is demo.collect_trace:
                return {**example, "model": args[0], "rollout": result}
            return {**result, "example": example}
        assert function is demo.train_student
        return function(*args, **kwargs)

    def train(source, model, output, **options):
        assert isinstance(source["trace"], Trace) and source["targets"]
        return {"initial": {"loss": 1.0}, "final": {"loss": 0.5}, "reload_exact": True}

    display = ModuleType("IPython.display")
    display.Markdown, display.display = str, lambda *args: None
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.setattr(process, "isolated", isolated_step)
    monkeypatch.setattr(demo.student, "train_student", train)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            code = compile(
                "".join(cell["source"]), str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )
            exec(code, namespace)
    assert calls.count(demo.train_student) == 1
    assert calls.count(demo.collect_trace) == 1
    assert calls.count(demo.solve) == (4 if path.stem == "no_barrier" else 1)


def test_refactored_prompt_and_step_boundaries_are_preserved():
    class Tokenizer:
        vocab = {"SEP": 1, "xSEP": 2, "text": 3}

        def encode(self, text, **kwargs):
            return [1]

        def decode(self, tokens):
            return {1: "done.\n\n", 2: "``` incomplete.\n\n", 3: "waiting"}[tokens[-1]]

        def apply_chat_template(self, *args, **kwargs):
            return json.dumps([args, kwargs], ensure_ascii=False)

    formatting = MathFormatting(Tokenizer(), enable_thinking=False)
    prompt = formatting.apply_chat_template("test")
    assert (
        hashlib.sha256(prompt.encode()).hexdigest()
        == "861bf8883d092d6fe3f2ec8cb4f096140835e5e4f5e27518d7c1793ebe70e8c0"
    )
    assert formatting.is_end_of_step([1])
    assert not formatting.is_end_of_step([2]) and not formatting.is_end_of_step([3])
    assert formatting.get_final_answer(r"\boxed{12} then \boxed{bad}") == 12
