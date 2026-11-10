"""Hugging Face Trainer integration for trace-supervised fine-tuning."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from minisgl.training import Trace
from minisgl.training.trace import ForwardBatch, ForwardRequest, TensorData
from minisgl_examples.training import sft as original_sft
from minisgl_examples.training import sft_hf
from minisgl_examples.training.loss import compute_supervised_loss
from minisgl_examples.training.trainer_hf import (
    TraceDataCollator,
    TraceDataset,
    TraceSFTTrainer,
)
from transformers import TrainingArguments

from tests.training.models import tiny_model
from tests.training.workloads import causal_trace


def examples():
    return [
        {"trace": causal_trace(4), "targets": [-100, 3, 4, 5]},
        {"trace": causal_trace(6), "targets": [-100, -100, 6, 7, 8, 9]},
    ]


def arguments(tmp_path, **overrides):
    values = {
        "output_dir": str(tmp_path / "trainer"),
        "use_cpu": True,
        "max_steps": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "eval_strategy": "no",
        "save_strategy": "no",
        "report_to": [],
        "disable_tqdm": True,
    }
    values.update(overrides)
    return TrainingArguments(**values)


def train_arguments(tmp_path, **overrides):
    values = {
        "data": tmp_path / "traces.json",
        "output": tmp_path / "checkpoint",
        "model": None,
        "steps": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": None,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 7,
        "gradient_checkpointing": False,
        "eval_steps": 1,
        "save_steps": 0,
        "verify_reload": False,
        "attention_checkpointing": Trace.attention_checkpointing,
        "loss_checkpointing": True,
        "cache_replay_plans": Trace.cache_plans,
        "query_chunk_size": Trace.query_chunk_size,
        "loss_chunk_size": 128,
        "gdn_segment_size": Trace.gdn_segment_size,
        "gdn_read_mode": Trace.gdn_read_mode,
        "gdn_cache_reused": Trace.gdn_cache_reused,
        "gdn_read_backend": Trace.gdn_read_backend,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_trace_dataset_and_collator_preserve_alignment():
    dataset = TraceDataset(examples())
    batch = TraceDataCollator()([dataset[0], dataset[1]])
    assert len(dataset) == 2 and dataset.supervised_tokens == 7
    assert [trace.num_tokens for trace in batch["traces"]] == [4, 6]
    assert batch["labels"].tolist() == [
        [-100, 3, 4, 5, -100, -100],
        [-100, -100, 6, 7, 8, 9],
    ]
    assert batch["shift_labels"] is batch["labels"]


def test_trainer_loss_matches_direct_replay(tmp_path):
    dataset = TraceDataset(examples()[:1])
    model = tiny_model().train()
    trainer = TraceSFTTrainer(
        model=model,
        args=arguments(tmp_path, gradient_accumulation_steps=1),
        train_dataset=dataset,
        data_collator=TraceDataCollator(),
        loss_checkpointing=False,
    )
    batch = TraceDataCollator()([dataset[0]])
    count = batch["labels"].ne(-100).sum()
    expected = (
        compute_supervised_loss(
            model, dataset[0]["trace"], dataset[0]["labels"], checkpointing=False
        )
        / count
    )
    actual = trainer.compute_loss(model, batch, num_items_in_batch=count)
    torch.testing.assert_close(actual, expected)


def test_trainer_runs_accumulated_optimizer_step(tmp_path):
    dataset = TraceDataset(examples())
    model = tiny_model().train()
    before = model.lm_head.weight.detach().clone()
    trainer = TraceSFTTrainer(
        model=model,
        args=arguments(tmp_path),
        train_dataset=dataset,
        data_collator=TraceDataCollator(),
        loss_checkpointing=False,
    )
    result = trainer.train()
    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert not torch.equal(model.lm_head.weight, before)


@pytest.mark.parametrize(
    "option,value,message",
    [
        ("query_chunk_size", 0, "query_chunk_size must be positive"),
        ("gdn_segment_size", 0, "gdn_segment_size must be positive"),
        ("gdn_read_mode", "invalid", "gdn_read_mode must be expanded or factorized"),
        ("gdn_read_backend", "invalid", "gdn_read_backend must be torch or triton"),
    ],
)
def test_hf_sft_rejects_invalid_replay_options_before_loading_data(
    tmp_path, option, value, message
):
    data = Mock()
    args = train_arguments(tmp_path, data=data, **{option: value})
    with pytest.raises(ValueError, match=message):
        sft_hf.train(args)
    data.read_text.assert_not_called()


@pytest.mark.parametrize("kind", ["text", "pixels", "mixed_pixels"])
def test_hf_sft_loads_and_saves_image_model_and_processor(monkeypatch, tmp_path, kind):
    request = ForwardRequest((1, 2, 3), 0, 0, ((0, 0),))
    text = Trace(batches=[ForwardBatch(0, "prefill", (request,))], steps=1)
    traces = [text]
    if kind != "text":
        request = replace(
            request,
            input_ids=(30, 29, 31),
            mm_token_type_ids=TensorData.capture(torch.tensor([0, 1, 0])),
            mrope_positions=TensorData.capture(torch.arange(3).expand(3, -1)),
            pixel_values=TensorData.capture(torch.zeros(4, 12)),
            image_grid_thw=TensorData.capture(torch.tensor([[1, 2, 2]])),
        )
        image = Trace(batches=[ForwardBatch(0, "prefill", (request,))], steps=1)
        traces = [text, image] if kind.startswith("mixed_") else [image]
    data = tmp_path / "traces.json"
    data.write_text(
        json.dumps(
            {
                "model": "tiny-image-model",
                "examples": [
                    {
                        "trace": asdict(trace),
                        "targets": [-100, -100, 2],
                        "logprobs": [None, None, -1.0],
                    }
                    for trace in traces
                ],
            }
        )
    )
    model = torch.nn.Linear(1, 1)
    config = Mock()
    loaders = {
        name: Mock()
        for name in (
            "AutoConfig",
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoTokenizer",
            "AutoProcessor",
        )
    }
    for name, loader in loaders.items():
        monkeypatch.setattr(sft_hf, name, loader)
    loaders["AutoConfig"].from_pretrained.return_value = config
    loaders["AutoModelForCausalLM"].from_pretrained.return_value = model
    loaders["AutoModelForImageTextToText"].from_pretrained.return_value = model
    evaluation = {
        "loss": 1.0,
        "accuracy": 0.0,
        "examples": [{"nll": [1.0], "loss": 1.0, "accuracy": 0.0}] * len(traces),
    }
    monkeypatch.setattr(sft_hf, "evaluate", Mock(return_value=evaluation))
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def train(self):
            return SimpleNamespace(global_step=1, training_loss=1.0, metrics={})

        def save_model(self, output):
            output.mkdir(parents=True)

        def save_state(self):
            pass

    monkeypatch.setattr(sft_hf, "TraceSFTTrainer", FakeTrainer)
    monkeypatch.setattr(sft_hf, "TrainingArguments", lambda **kwargs: SimpleNamespace(**kwargs))
    sft_hf.train(train_arguments(tmp_path, data=data))
    multimodal = kind != "text"
    selected = "AutoModelForImageTextToText" if multimodal else "AutoModelForCausalLM"
    unused = "AutoModelForCausalLM" if multimodal else "AutoModelForImageTextToText"
    loaders[selected].from_pretrained.assert_called_once_with(
        "tiny-image-model",
        config=config if multimodal else config.get_text_config.return_value,
        dtype=torch.float32,
    )
    loaders[unused].from_pretrained.assert_not_called()
    processor = "AutoProcessor" if multimodal else "AutoTokenizer"
    loaders[processor].from_pretrained.return_value.save_pretrained.assert_called_once_with(
        tmp_path / "checkpoint"
    )
    assert captured["args"].gradient_accumulation_steps == len(traces)
    optimizer = captured["optimizers"][0]
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert all(
        actual is expected
        for actual, expected in zip(
            optimizer.param_groups[0]["params"], model.parameters(), strict=True
        )
    )
    metrics = json.loads((tmp_path / "metrics_hf.json").read_text())
    assert metrics["supervised_tokens"] == len(traces)
    assert metrics["gradient_accumulation_steps"] == len(traces)


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], (True, True)),
        (["--no-attention-checkpointing"], (False, True)),
        (["--no-loss-checkpointing"], (True, False)),
        (["--no-attention-checkpointing", "--no-loss-checkpointing"], (False, False)),
        (["--attention-checkpointing", "--loss-checkpointing"], (True, True)),
    ],
)
def test_hf_sft_cli_checkpointing_flags(monkeypatch, tmp_path, flags, expected):
    received = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(sft_hf, "train", received.append)
    sft_hf.main(["train", *flags])
    args = received.pop()
    assert (args.attention_checkpointing, args.loss_checkpointing) == expected


@pytest.mark.parametrize(
    "flag,expected",
    [(None, True), ("--cache-replay-plans", True), ("--no-cache-replay-plans", False)],
)
def test_hf_sft_cli_plan_cache_flag(monkeypatch, tmp_path, flag, expected):
    received = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(sft_hf, "train", received.append)
    sft_hf.main(["train", *([flag] if flag else [])])
    assert received.pop().cache_replay_plans is expected


def _original_sft_step(model, dataset, learning_rate, started):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    count = dataset.supervised_tokens
    started.wait(timeout=30)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = torch.zeros(())
    for trace, targets in dataset.replay_examples:
        part = (
            original_sft.compute_supervised_loss(model, trace, targets, checkpointing=False) / count
        )
        part.backward()
        loss = loss + part.detach()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss


def _hf_sft_step(model, dataset, learning_rate, tmp_path, started):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = TraceSFTTrainer(
        model=model,
        args=arguments(
            tmp_path,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=len(dataset),
            learning_rate=learning_rate,
            lr_scheduler_type="constant",
            max_grad_norm=1.0,
        ),
        train_dataset=dataset,
        data_collator=TraceDataCollator(),
        loss_checkpointing=False,
        optimizers=(optimizer, None),
    )
    started.wait(timeout=30)
    return torch.tensor(trainer.train().training_loss)


def test_hf_and_original_sft_run_in_parallel_and_match(tmp_path):
    learning_rate = 1e-3
    original_model = tiny_model().train()
    hf_model = deepcopy(original_model)
    original_dataset = TraceDataset(examples())
    hf_dataset = TraceDataset(examples())
    started = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        old = executor.submit(
            _original_sft_step, original_model, original_dataset, learning_rate, started
        )
        new = executor.submit(_hf_sft_step, hf_model, hf_dataset, learning_rate, tmp_path, started)
        original_loss, hf_loss = old.result(), new.result()
    torch.testing.assert_close(hf_loss, original_loss, rtol=1e-6, atol=1e-7)
    for original, hf in zip(original_model.parameters(), hf_model.parameters(), strict=True):
        torch.testing.assert_close(hf, original, rtol=1e-6, atol=1e-7)
