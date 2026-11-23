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

from tests.training.models import tiny_hybrid, tiny_model
from tests.training.workloads import causal_trace, mutation_trace


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
        "model_revision": None,
        "steps": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": None,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 7,
        "gradient_checkpointing": False,
        "fast": False,
        "lora": False,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "eval_steps": 1,
        "eval_size": None,
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
    "capability,bf16_supported,expected",
    [
        ((7, 5), False, ({"fp16": True, "bf16": False}, "fp16")),
        ((8, 0), True, ({"fp16": False, "bf16": True}, "bf16")),
        ((9, 0), True, ({"fp16": False, "bf16": True}, "bf16")),
    ],
)
def test_fast_precision_follows_gpu_architecture(monkeypatch, capability, bf16_supported, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_supported)
    assert sft_hf._precision(True) == expected


def test_precision_defaults_to_fp32_without_querying_cuda(monkeypatch):
    capability = Mock(side_effect=AssertionError("CUDA should not be queried"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert sft_hf._precision(False) == ({"fp16": False, "bf16": False}, "fp32")
    capability.assert_not_called()


def test_eval_subset_is_seeded_and_keeps_source_order():
    values = [{"index": index} for index in range(20)]
    first = sft_hf._select_eval_examples(values, 5, seed=7)
    second = sft_hf._select_eval_examples(values, 5, seed=7)
    assert first == second
    assert [value["index"] for value in first] == sorted(value["index"] for value in first)
    assert len(first) == 5
    assert sft_hf._select_eval_examples(values, None, seed=7) is values
    assert sft_hf._select_eval_examples(values, 20, seed=7) is values


def test_model_factory_pins_huggingface_revision(monkeypatch):
    config = Mock()
    auto_config = Mock()
    monkeypatch.setattr(sft_hf, "AutoConfig", auto_config)
    auto_config.from_pretrained.return_value = config
    factory, selected = sft_hf._model_factory("student", False, "abc123")
    auto_config.from_pretrained.assert_called_once_with("student", revision="abc123")
    assert factory is sft_hf.AutoModelForCausalLM
    assert selected is config.get_text_config.return_value


@pytest.mark.parametrize("capability", [(6, 1), (7, 0)])
def test_fast_precision_rejects_unsupported_architectures(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    with pytest.raises(RuntimeError, match="Turing.*Ampere"):
        sft_hf._precision(True)


def test_lora_replay_only_trains_adapters():
    model = sft_hf._apply_lora(
        tiny_model(),
        SimpleNamespace(lora_rank=4, lora_alpha=8, lora_dropout=0.0),
    ).train()
    dataset = TraceDataset(examples()[:1])
    loss = compute_supervised_loss(
        model,
        dataset[0]["trace"],
        dataset[0]["labels"],
        checkpointing=False,
    )
    loss.backward()
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    assert trainable and all("lora_" in name for name, _ in trainable)
    assert any(parameter.grad is not None for _, parameter in trainable)
    assert frozen and all(parameter.grad is None for parameter in frozen)


def test_lora_replay_trains_qwen35_linear_attention_adapters():
    model = sft_hf._apply_lora(
        tiny_hybrid(),
        SimpleNamespace(lora_rank=4, lora_alpha=8, lora_dropout=0.0),
    ).train()
    trace = mutation_trace()
    targets = torch.arange(trace.num_tokens) % model.config.vocab_size
    targets[::3] = -100
    compute_supervised_loss(model, trace, targets, checkpointing=False).backward()
    linear_attention = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "linear_attn" in name
    ]
    assert linear_attention
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in linear_attention
    )


def test_lora_trainer_saves_reloadable_adapter(tmp_path):
    from peft import PeftModel

    model = sft_hf._apply_lora(
        tiny_model(),
        SimpleNamespace(lora_rank=4, lora_alpha=8, lora_dropout=0.0),
    ).train()
    dataset = TraceDataset(examples())
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainer = TraceSFTTrainer(
        model=model,
        args=arguments(tmp_path),
        train_dataset=dataset,
        data_collator=TraceDataCollator(),
        loss_checkpointing=False,
        optimizers=(torch.optim.AdamW(trainable, lr=1e-3), None),
    )
    assert trainer.train().global_step == 1
    expected = sft_hf.evaluate(model.eval(), dataset.replay_examples, 128)
    checkpoint = tmp_path / "adapter"
    trainer.save_model(checkpoint)

    reloaded = PeftModel.from_pretrained(tiny_model(), checkpoint).eval()
    actual = sft_hf.evaluate(reloaded, dataset.replay_examples, 128)
    assert (checkpoint / "adapter_config.json").is_file()
    assert (checkpoint / "adapter_model.safetensors").is_file()
    for left, right in zip(expected["examples"], actual["examples"], strict=True):
        torch.testing.assert_close(
            torch.tensor(left["nll"]), torch.tensor(right["nll"]), rtol=0, atol=0
        )


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


def test_hf_sft_cli_fast_lora_flags(monkeypatch, tmp_path):
    received = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(sft_hf, "train", received.append)
    sft_hf.main(
        [
            "train",
            "--fast",
            "--lora",
            "--lora-rank",
            "8",
            "--lora-alpha",
            "16",
            "--lora-dropout",
            "0.1",
            "--model-revision",
            "abc123",
        ]
    )
    args = received.pop()
    assert args.fast is True
    assert args.model_revision == "abc123"
    assert (args.lora, args.lora_rank, args.lora_alpha, args.lora_dropout) == (
        True,
        8,
        16,
        0.1,
    )


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
