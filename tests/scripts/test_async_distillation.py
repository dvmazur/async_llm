"""Distillation precision, trace checks, and portable process launching."""

import ast
import asyncio
import copy
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from minisgl.training import RecordedAsyncLLM, Trace
from minisgl.training.trace import ForwardBatch, ForwardRequest
from minisgl_examples.demos.async_thoughts.demo import Prompting
from minisgl_examples.training import process as processes
from minisgl_examples.training.distillation import demo, rollout

from tests.core.test_async_llm import _make_llm
from tests.training.workloads import mutation_trace


@pytest.fixture(autouse=True)
def cpu_profiling(monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())


class AnswerTokenizer:
    def encode(self, text, **kwargs):
        prompting = Prompting("test")
        return {
            prompting.thinker_output_prefix: [4, 5],
            prompting.writer_output_prefix: [6, 7],
            prompting.mode_switching_question: [8, 9],
        }[text]

    def decode(self, tokens, **kwargs):
        return {1: r"\boxed{391}", 2: r"\boxed{390}", 3: r"\boxed{391} \boxed{391}"}[tokens[0]]


def async_example():
    rows = [
        ("prefill", (1, 2, 3), 10, 0, ((10, 0),)),
        ("prefill", (4, 5), 20, 0, ((10, 3), (20, 0))),
        ("prefill", (6, 7), 30, 0, ((10, 3), (20, 2), (30, 0))),
        ("decode", (8,), 20, 2, ((10, 3), (20, 2))),
        ("prefill", (8, 9), 40, 0, ((10, 3), (20, 3), (30, 2), (40, 0))),
        ("decode", (8,), 30, 2, ((10, 3), (20, 3), (30, 2))),
        ("decode", (9,), 20, 3, ((10, 3), (20, 3))),
        ("decode", (9,), 30, 3, ((10, 3), (20, 4), (30, 3))),
        ("decode", (10,), 20, 4, ((10, 3), (20, 4))),
    ]
    trace = Trace(
        batches=[
            ForwardBatch(i, kind, (ForwardRequest(ids, block, start, view),))
            for i, (kind, ids, block, start, view) in enumerate(rows)
        ],
        steps=len(rows),
    )
    return {
        "prompt": "test",
        "thinker_tokens": [8, 9, 10],
        "writer_tokens": [1],
        "trace": asdict(trace),
        "targets": [
            token.token_id + 1 if token.kind == "decode" else -100 for token in trace.tokens
        ],
    }


@pytest.mark.parametrize("tokens, status", [([1], "PASS"), ([2], "FAIL"), ([3], "FAIL")])
def test_answer_requires_exactly_one_correct_box(tokens, status):
    assert (
        demo.check_answer({"writer_tokens": tokens}, AnswerTokenizer(), [391])["status"] == status
    )


def test_exact_rollout_divergence_is_separate_from_answer_and_async_correctness(tmp_path):
    teacher, student = async_example(), async_example()
    student["thinker_tokens"][-1] += 1
    paths = [tmp_path / name for name in ("teacher.json", "student.json")]
    for path, example in zip(paths, (teacher, student)):
        path.write_text(json.dumps({"examples": [example]}))
    report = demo.compare_rollouts(*paths, AnswerTokenizer(), [391], tmp_path / "checks.json")
    assert all(
        report[key]["status"] == "PASS"
        for key in ("teacher_answer", "student_answer", "teacher_async", "student_async")
    )
    assert report["exact_recorded_rollout"]["status"] == "FAIL"
    invalid = copy.deepcopy(teacher)
    invalid["trace"]["batches"][3]["requests"][0]["write_start"] = 99
    assert rollout.check_async_behavior(invalid, AnswerTokenizer())["status"] == "FAIL"


def test_trace_canonicalization_preserves_cache_mutations():
    original = asdict(mutation_trace())
    renamed = copy.deepcopy(original)
    for batch in renamed["batches"]:
        for request in batch["requests"]:
            request["write_to"] += 1000
            request["cache_view"] = [(block + 1000, size) for block, size in request["cache_view"]]
    for update in renamed["updates"]:
        update["block"] += 1000
        if update["refs"] is not None:
            update["refs"] = [(block + 1000, size) for block, size in update["refs"]]
    assert rollout.normalize_trace(original) == rollout.normalize_trace(renamed)
    renamed["updates"][0]["refs"] = tuple(reversed(renamed["updates"][0]["refs"]))
    assert rollout.normalize_trace(original) != rollout.normalize_trace(renamed)


def test_launcher_inherits_environment_and_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-parent")
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    log = tmp_path / "worker.log"
    demo.run_process(
        tmp_path,
        [
            "-c",
            "import json, os, sys; print(json.dumps([sys.executable, os.environ['CUDA_VISIBLE_DEVICES'], os.environ['OMP_NUM_THREADS'], os.environ['HF_HOME']]))",
        ],
        log,
    )
    assert json.loads(log.read_text()) == [
        sys.executable,
        "GPU-parent",
        "16",
        str(tmp_path / "cache"),
    ]


def test_cli_runs_workflow_from_callers_directory(tmp_path, monkeypatch):
    run = MagicMock(return_value={})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(demo, "run_workflow", run)
    demo.main(["run", "--output", "experiment"])
    assert run.call_args.args == (tmp_path.resolve(),)
    assert run.call_args.kwargs["output"] == Path("experiment")


def test_launcher_interrupt_terminates_and_reaps_its_child(tmp_path, monkeypatch):
    process = MagicMock()
    process.stdout.__iter__.side_effect = KeyboardInterrupt
    process.poll.return_value = None
    monkeypatch.setattr(processes.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(KeyboardInterrupt):
        demo.run_process(tmp_path, ["-c", "pass"], tmp_path / "interrupted.log")
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=10)
    process.kill.assert_not_called()
    process.stdout.close.assert_called_once_with()


def test_full_workflow_stops_before_training_when_teacher_is_wrong(tmp_path, monkeypatch):
    trace = tmp_path / "teacher.json"
    example = async_example()
    example["writer_tokens"] = [2]
    trace.write_text(json.dumps({"examples": [example]}))
    monkeypatch.setattr(demo, "record_rollout", lambda *args, **kwargs: trace)
    monkeypatch.setattr(demo.AutoTokenizer, "from_pretrained", lambda model: AnswerTokenizer())
    monkeypatch.setattr(
        demo, "fit_student", lambda *args, **kwargs: pytest.fail("trained with wrong teacher")
    )
    with pytest.raises(RuntimeError, match="Teacher answer"):
        demo.run_workflow(tmp_path, "teacher", "student", tmp_path / "run")


@pytest.mark.parametrize("overfit,reload_loss", [(False, 0.005), (True, 0.0051)])
def test_full_workflow_preserves_overfit_and_reload_requirements(
    tmp_path, monkeypatch, overfit, reload_loss
):
    trace = tmp_path / "teacher.json"
    trace.write_text(json.dumps({"examples": [async_example()]}))
    calls = []

    def record(*args, **kwargs):
        calls.append(args)
        assert len(calls) == 1, "Independent generation started before checkpoint acceptance"
        return trace

    monkeypatch.setattr(demo, "record_rollout", record)
    monkeypatch.setattr(demo.AutoTokenizer, "from_pretrained", lambda model: AnswerTokenizer())
    monkeypatch.setattr(
        demo,
        "fit_student",
        lambda *args, **kwargs: {
            "overfit_achieved": overfit,
            "checkpoint": str(tmp_path / "checkpoint"),
            "final": {"accuracy": 1.0, "loss": 0.005},
        },
    )
    monkeypatch.setattr(
        demo,
        "verify_checkpoint",
        lambda *args, **kwargs: {
            "accuracy": 1.0,
            "loss": reload_loss,
        },
    )
    with pytest.raises(RuntimeError, match="Overfit not reached|Saved checkpoint"):
        demo.run_workflow(tmp_path, "teacher", "student", tmp_path / "run")


def test_notebook_has_executable_cells_without_stale_outputs():
    path = Path(demo.__file__).with_name("simple_distillation.ipynb")
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None
            compile(
                "".join(cell["source"]), str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )


@pytest.mark.parametrize("fail", [False, True])
def test_collection_returns_trace_without_file_and_closes_engine(monkeypatch, fail):
    llm, _ = _make_llm()
    llm.tokenizer = AnswerTokenizer()
    monkeypatch.setattr(rollout, "AsyncLLM", lambda *args, **kwargs: llm)
    monkeypatch.setattr(rollout.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        rollout.AutoConfig, "from_pretrained", lambda path: SimpleNamespace(model_type="qwen3")
    )

    async def run_loop(config, recorded, tokenizer, prompting, samples):
        assert isinstance(recorded, RecordedAsyncLLM)
        block = await recorded.create_block()
        await recorded.forward([1, 2], write_to=block)
        if fail:
            raise RuntimeError("collection failed")
        samples[1] = (3, -0.25)
        await recorded.free_block(block)
        return [3], [1]

    monkeypatch.setattr(rollout.async_thoughts, "_run_loop", run_loop)
    if fail:
        with pytest.raises(RuntimeError, match="collection failed"):
            asyncio.run(demo.collect_trace("teacher"))
    else:
        example = asyncio.run(demo.collect_trace("teacher"))
        assert isinstance(example["trace"], Trace)
        assert example["model"] == "teacher"
        assert example["targets"] == [-100, 3]
        assert example["logprobs"] == [None, -0.25]
        assert example["thinker_tokens"] == [3] and example["writer_tokens"] == [1]
    assert llm._closed and llm._loop_task is None


@pytest.mark.parametrize("on_disk", [False, True])
def test_workflow_accepts_saved_or_in_memory_teacher_without_mining(tmp_path, monkeypatch, on_disk):
    example = {**async_example(), "model": "recorded-teacher"}
    source = demo.save_example(example, tmp_path / "source.json") if on_disk else example
    calls = []

    def record(root, model, prompt, directory, **kwargs):
        calls.append((model, prompt))
        return demo.save_example(example, directory / "trace.json")

    monkeypatch.setattr(demo, "record_rollout", record)
    monkeypatch.setattr(demo.AutoTokenizer, "from_pretrained", lambda model: AnswerTokenizer())
    monkeypatch.setattr(
        demo,
        "fit_student",
        lambda *args, **kwargs: {
            "overfit_achieved": True,
            "checkpoint": "trained-student",
            "final": {"accuracy": 1.0, "loss": 0.005},
        },
    )
    monkeypatch.setattr(demo, "verify_checkpoint", lambda *args: {"accuracy": 1.0, "loss": 0.005})
    result = demo.run_workflow(
        tmp_path, "unused-teacher", "student", tmp_path / "run", trace=source
    )
    assert calls == [(Path("trained-student"), example["prompt"])]
    assert result["teacher_answer"]["status"] == result["student_answer"]["status"] == "PASS"
    assert (tmp_path / "run" / "checks.json").exists()


def test_cli_passes_saved_trace_to_workflow(tmp_path, monkeypatch):
    run = MagicMock(return_value={})
    monkeypatch.setattr(demo, "run_workflow", run)
    demo.main(["run", "--trace", str(tmp_path / "trace.json"), "--output", "experiment"])
    assert run.call_args.kwargs["trace"] == tmp_path / "trace.json"


def test_notebook_runs_unchanged_experiment_through_python_stages(tmp_path, monkeypatch):
    calls = []
    example = {**async_example(), "model": "teacher"}

    class NotebookTokenizer(AnswerTokenizer):
        def decode(self, tokens, **kwargs):
            return super().decode(tokens, **kwargs) if tokens[0] in {1, 2, 3} else "thoughts"

    def record(root, model, prompt, directory, **options):
        calls.append(("record", model, options))
        return demo.save_example(example, directory / "trace.json")

    final = {"accuracy": 1.0, "loss": 0.005}

    def fit(root, trace, model, directory, **options):
        calls.append(("train", model, options))
        return {
            "initial": {"accuracy": 0.0, "loss": 2.0},
            "final": final,
            "completed_steps": 100,
            "overfit_achieved": True,
            "checkpoint": str(directory / "checkpoint"),
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(demo.AutoTokenizer, "from_pretrained", lambda model: NotebookTokenizer())
    monkeypatch.setattr(demo, "record_rollout", record)
    monkeypatch.setattr(demo, "fit_student", fit)
    monkeypatch.setattr(demo, "verify_checkpoint", lambda *args: final)
    notebook = Path(demo.__file__).with_name("simple_distillation.ipynb")
    namespace = {}
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(notebook), "exec"), namespace)
    assert [call[0] for call in calls] == ["record", "train", "record"]
    assert calls[1][2] == {"steps": 100, "lr": 1e-6}
    assert calls[0][2] == calls[2][2] == namespace["LIMITS"]
    assert namespace["report"]["exact_recorded_rollout"]["status"] == "PASS"
    assert (namespace["RUN"] / "checks.json").exists()
