"""Distillation precision, trace checks, and portable process launching."""

import ast
import copy
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from minisgl.training import Trace
from minisgl.training.trace import ForwardBatch, ForwardRequest
from minisgl_examples.demos.async_thoughts.demo import Prompting
from minisgl_examples.training.distillation import sanity
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tests.scripts.training_helpers import source as source
from tests.training.workloads import mutation_trace


@pytest.fixture(autouse=True)
def cpu_profiling(monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())


@pytest.mark.parametrize("targets", [[1], [-100] * 6, [32] * 6, [True] * 6])
def test_invalid_targets_rejected_before_training(source, targets):
    model_path, trace_path = source
    data = json.loads(trace_path.read_text())
    data["examples"][0]["targets"] = targets
    trace_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid or misaligned"):
        sanity.load_student(trace_path, str(model_path), device="cpu")


def test_student_tokenizer_mismatch_rejected(source, tmp_path):
    model_path, trace_path = source
    student = tmp_path / "student"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"other": 0, "1": 1}, unk_token="other"))
    )
    tokenizer.save_pretrained(student)
    with pytest.raises(ValueError, match="tokenizers differ"):
        sanity.load_student(trace_path, str(student), device="cpu")


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
        sanity.check_answer({"writer_tokens": tokens}, AnswerTokenizer(), [391])["status"] == status
    )


def test_exact_rollout_divergence_is_separate_from_answer_and_async_correctness(tmp_path):
    teacher, student = async_example(), async_example()
    student["thinker_tokens"][-1] += 1
    paths = [tmp_path / name for name in ("teacher.json", "student.json")]
    for path, example in zip(paths, (teacher, student)):
        path.write_text(json.dumps({"examples": [example]}))
    report = sanity.compare_rollouts(*paths, AnswerTokenizer(), [391], tmp_path / "checks.json")
    assert all(
        report[key]["status"] == "PASS"
        for key in ("teacher_answer", "student_answer", "teacher_async", "student_async")
    )
    assert report["exact_recorded_rollout"]["status"] == "FAIL"
    invalid = copy.deepcopy(teacher)
    invalid["trace"]["batches"][3]["requests"][0]["write_start"] = 99
    assert sanity.check_async_behavior(invalid, AnswerTokenizer())["status"] == "FAIL"


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
    assert sanity.normalize_trace(original) == sanity.normalize_trace(renamed)
    renamed["updates"][0]["refs"] = tuple(reversed(renamed["updates"][0]["refs"]))
    assert sanity.normalize_trace(original) != sanity.normalize_trace(renamed)


def test_launcher_inherits_environment_and_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-parent")
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    log = tmp_path / "worker.log"
    sanity.run_process(
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
    monkeypatch.setattr(sanity, "run_workflow", run)
    sanity.main(["run", "--output", "experiment"])
    assert run.call_args.args == (tmp_path.resolve(),)
    assert run.call_args.kwargs["output"] == Path("experiment")


def test_launcher_interrupt_terminates_and_reaps_its_child(tmp_path, monkeypatch):
    process = MagicMock()
    process.stdout.__iter__.side_effect = KeyboardInterrupt
    process.poll.return_value = None
    monkeypatch.setattr(sanity.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(KeyboardInterrupt):
        sanity.run_process(tmp_path, ["-c", "pass"], tmp_path / "interrupted.log")
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=10)
    process.kill.assert_not_called()
    process.stdout.close.assert_called_once_with()


def test_full_workflow_stops_before_training_when_teacher_is_wrong(tmp_path, monkeypatch):
    trace = tmp_path / "teacher.json"
    example = async_example()
    example["writer_tokens"] = [2]
    trace.write_text(json.dumps({"examples": [example]}))
    monkeypatch.setattr(sanity, "record_rollout", lambda *args, **kwargs: trace)
    monkeypatch.setattr(sanity.AutoTokenizer, "from_pretrained", lambda model: AnswerTokenizer())
    monkeypatch.setattr(
        sanity, "fit_student", lambda *args, **kwargs: pytest.fail("trained with wrong teacher")
    )
    with pytest.raises(RuntimeError, match="Teacher answer"):
        sanity.run_workflow(tmp_path, "teacher", "student", tmp_path / "run")


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

    monkeypatch.setattr(sanity, "record_rollout", record)
    monkeypatch.setattr(sanity.AutoTokenizer, "from_pretrained", lambda model: AnswerTokenizer())
    monkeypatch.setattr(
        sanity,
        "fit_student",
        lambda *args, **kwargs: {
            "overfit_achieved": overfit,
            "checkpoint": str(tmp_path / "checkpoint"),
            "final": {"accuracy": 1.0, "loss": 0.005},
        },
    )
    monkeypatch.setattr(
        sanity,
        "verify_checkpoint",
        lambda *args, **kwargs: {
            "accuracy": 1.0,
            "loss": reload_loss,
        },
    )
    with pytest.raises(RuntimeError, match="Overfit not reached|Saved checkpoint"):
        sanity.run_workflow(tmp_path, "teacher", "student", tmp_path / "run")


def test_notebook_has_executable_cells_without_stale_outputs():
    path = Path(sanity.__file__).with_name("simple_distillation.ipynb")
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None
            compile(
                "".join(cell["source"]), str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )
