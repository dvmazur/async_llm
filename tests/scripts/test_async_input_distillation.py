"""Delayed-correction recording and current trace-training API integration."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from minisgl.training import Trace
from minisgl_examples.training.loss import compute_supervised_loss, evaluate

from scripts.async_input_distillation_sanity_check import async_input_rollout as rollout
from scripts.async_input_distillation_sanity_check import sanity
from tests.core.test_async_llm import _make_llm
from tests.scripts.training_helpers import source as source
from tests.training.fixtures import cpu_profiling as cpu_profiling
from tests.training.models import tiny_model
from tests.training.workloads import causal_trace


class Tokenizer:
    eos_token_id = 31

    def apply_chat_template(self, *args, **kwargs):
        return "initial prompt"

    def encode(self, text, **kwargs):
        return {
            "initial prompt": [1, 2, 3],
            "\n": [4],
            rollout.INJECTION_PREFIX + "second shard\n\n": [10, 11, 12],
        }[text]

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


@pytest.fixture
def recorded_example(tmp_path, monkeypatch):
    llm, session = _make_llm()
    monkeypatch.setattr(rollout, "AsyncLLM", lambda *args, **kwargs: llm)
    monkeypatch.setattr(rollout.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(
        rollout,
        "load_row",
        lambda *args: {"problem_shards": ["first shard", "second shard"], "answer": "7"},
    )
    path = tmp_path / "trace.json"
    args = SimpleNamespace(
        model="stub",
        precision="float16",
        dataset="stub",
        dataset_revision="pinned",
        index=0,
        split="train",
        k_steps=2,
        max_tokens=5,
        memory_ratio=0.7,
        page_size=1,
        distributed_port=0,
        output=path,
    )
    asyncio.run(rollout.generate(args))
    assert llm._closed and llm._loop_task is None
    assert llm.async_engine.session is session
    return sanity.load_example(path)


def test_recording_aligns_targets_across_pending_token_and_both_correction_routes(recorded_example):
    example = recorded_example
    trace = Trace.from_dict(example["trace"]).to("cpu")
    assert example["output_tokens"] == [6, 8, 14, 16, 18]
    assert trace.num_tokens == 14
    assert [(i, t) for i, t in enumerate(example["targets"]) if t != -100] == [
        (3, 6),
        (4, 8),
        (11, 14),
        (12, 16),
        (13, 18),
    ]
    # Flush input 8, prompt correction, and all but the last output correction row
    # have no generated continuation target.
    assert [token.token_id for token in trace.tokens[5:12]] == [8, 10, 11, 12, 10, 11, 12]
    assert example["targets"][5:11] == [-100] * 6
    assert [i for i, p in enumerate(example["logprobs"]) if p is not None] == [3, 4, 11, 12, 13]
    requests = [b.requests[0] for b in trace.batches if b.requests[0].input_ids == (10, 11, 12)]
    assert len(requests) == 2 and requests[0].write_to != requests[1].write_to
    assert requests[0].cache_view == ((requests[0].write_to, 3),)
    assert requests[1].cache_view == ((requests[0].write_to, 6), (requests[1].write_to, 3))
    assert sanity.check_async_behavior(example, Tokenizer())["status"] == "PASS"


def test_async_check_rejects_wrong_injection_timing(recorded_example):
    wrong = copy.deepcopy(recorded_example)
    wrong["k_steps"] += 1
    result = sanity.check_async_behavior(wrong, Tokenizer())
    assert result["status"] == "FAIL"
    assert "injected_after_requested_decode_count" in result["details"]["failed"]


@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
def test_low_precision_loss_backpropagates_to_fp32_masters_with_checkpointing(precision):
    model = tiny_model().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = copy.deepcopy(model)
    # Match parameter-only substitution: rotary-frequency buffers remain FP32.
    for parameter in reference.parameters():
        parameter.data = parameter.data.to(getattr(torch, precision))
    trace = causal_trace(6).to("cpu", attention_backend="torch", query_chunk_size=2)
    targets = torch.tensor([-100, 2, 3, 4, 5, 6])
    expected = compute_supervised_loss(reference, trace, targets) / targets.ne(-100).sum()
    expected.backward()
    actual = sanity.run_low_precision(sanity.TrainingStep(model), trace, targets, precision)
    assert actual == pytest.approx(expected.item(), abs=1e-6)
    for (name, parameter), (_, other) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert parameter.dtype == torch.float32, name
        if other.grad is not None:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, other.grad.float(), rtol=1e-3, atol=1e-4)
    expected_metrics = evaluate(reference.eval(), [(trace, targets)])
    actual_metrics = sanity.evaluate_student(model, trace, targets, precision)
    assert actual_metrics == {key: expected_metrics[key] for key in ("loss", "accuracy")}


@pytest.mark.parametrize("backend,chunk", [("torch", 128), ("auto", 3)])
def test_load_student_prepares_requested_trace_options(source, backend, chunk):
    model_path, trace_path = source
    model, trace, targets, _ = sanity.load_student(
        trace_path,
        str(model_path),
        device="cpu",
        attention_backend=backend,
        query_chunk_size=chunk,
    )
    assert trace.attention_backend == backend and trace.query_chunk_size == chunk
    assert targets.device.type == "cpu"
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_train_and_checkpoint_reload_keep_precision_optimizer_and_replay_settings(
    source, tmp_path, monkeypatch
):
    model_path, trace_path = source
    load = sanity.load_student
    requested = []

    def load_cpu(*args, **kwargs):
        requested.append(kwargs.copy())
        return load(*args, device="cpu", **kwargs)

    monkeypatch.setattr(sanity, "load_student", load_cpu)
    metrics = sanity.train_from_trace(
        trace_path,
        str(model_path),
        tmp_path / "trained",
        1,
        1e-6,
        precision="float16",
        optimizer_foreach=False,
        attention_backend="torch",
        query_chunk_size=3,
    )
    destination = tmp_path / "reloaded.json"
    sanity.main(
        [
            "verify",
            str(trace_path),
            metrics["checkpoint"],
            str(destination),
            "--precision",
            "float16",
            "--attention-backend",
            "torch",
            "--query-chunk-size",
            "3",
        ]
    )
    reloaded = json.loads(destination.read_text())
    options = {"attention_backend": "torch", "query_chunk_size": 3}
    assert requested == [options, options]
    assert metrics["completed_steps"] == 1 and metrics["optimizer_foreach"] is False
    assert metrics["precision"] == "float16 forward / FP32 master weights and AdamW"
    assert metrics["replay_options"] == reloaded["replay_options"] == options
    assert {key: reloaded[key] for key in ("loss", "accuracy")} == metrics["final"]


@pytest.mark.parametrize("mode", ["train", "verify"])
def test_worker_cli_preserves_replay_options(mode, tmp_path, monkeypatch):
    calls = []
    output = tmp_path / "report.json"
    if mode == "train":
        monkeypatch.setattr(sanity, "train_from_trace", lambda *a, **k: calls.append((a, k)))
    else:

        def load(*args, **kwargs):
            calls.append((args, kwargs))
            return None, None, None, None

        monkeypatch.setattr(sanity, "load_student", load)
        monkeypatch.setattr(sanity, "evaluate_student", lambda *a: {"loss": 0.005, "accuracy": 1.0})
    sanity.main(
        [
            mode,
            "source.json",
            "student",
            str(output),
            "--precision",
            "float16",
            "--attention-backend",
            "torch_optimized",
            "--query-chunk-size",
            "17",
            "--optimizer-foreach",
            "false",
        ]
    )
    assert calls[0][1] == {"attention_backend": "torch_optimized", "query_chunk_size": 17}
    if mode == "train":
        assert calls[0][0][-2:] == ("float16", False)
    else:
        assert json.loads(output.read_text())["replay_options"] == calls[0][1]


@pytest.mark.parametrize("mode", ["train", "verify"])
def test_notebook_adapters_forward_same_precision_and_replay_settings(mode, tmp_path, monkeypatch):
    calls = []
    report = tmp_path / "trained" / "metrics.json" if mode == "train" else tmp_path / "reload.json"

    def run(root, gpu, arguments, log):
        calls.append(arguments)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}")

    monkeypatch.setattr(sanity, "run_process", run)
    options = {"precision": "float16", "attention_backend": "torch", "query_chunk_size": 23}
    if mode == "train":
        sanity.fit_student(
            tmp_path,
            "0",
            Path("source.json"),
            "student",
            report.parent,
            optimizer_foreach=False,
            **options,
        )
    else:
        sanity.verify_checkpoint(
            tmp_path, "0", Path("source.json"), Path("checkpoint"), report, **options
        )
    arguments = calls[0]
    for option, value in (
        ("--precision", "float16"),
        ("--attention-backend", "torch"),
        ("--query-chunk-size", "23"),
    ):
        assert arguments[arguments.index(option) + 1] == value
    if mode == "train":
        assert arguments[arguments.index("--optimizer-foreach") + 1] == "false"
