import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from minisgl.training import Trace
from minisgl.training.trace import ForwardBatch, ForwardRequest, TensorData
from minisgl_examples.training import sft


@pytest.mark.parametrize(
    "options,message",
    [
        ({"query_chunk_size": 0}, "query_chunk_size must be positive"),
        ({"gdn_segment_size": 0}, "gdn_segment_size must be positive"),
        ({"gdn_read_mode": "invalid"}, "gdn_read_mode must be expanded or factorized"),
        ({"gdn_read_backend": "invalid"}, "gdn_read_backend must be torch or triton"),
    ],
)
def test_sft_rejects_invalid_replay_options_before_loading_data(options, message):
    data = Mock()
    with pytest.raises(ValueError, match=message):
        sft.train(SimpleNamespace(data=data, **options))
    data.read_text.assert_not_called()


@pytest.mark.parametrize("kind", ["text", "pixels", "mixed_pixels"])
def test_sft_loads_and_saves_image_model_and_processor(monkeypatch, tmp_path, kind):
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
    monkeypatch.setattr(model, "cuda", lambda: model)
    model.save_pretrained = Mock()
    config = Mock()
    loaders = {}
    for name in (
        "AutoConfig",
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoTokenizer",
        "AutoProcessor",
    ):
        loaders[name] = Mock()
        monkeypatch.setattr(sft, name, loaders[name])
    loaders["AutoConfig"].from_pretrained.return_value = config
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        loaders[name].from_pretrained.return_value = model
    evaluation = {
        "loss": 1.0,
        "accuracy": 0.0,
        "examples": [{"nll": [1.0], "loss": 1.0, "accuracy": 0.0}] * len(traces),
    }
    monkeypatch.setattr(sft, "evaluate", Mock(return_value=evaluation))
    monkeypatch.setattr(
        sft, "compute_supervised_loss", lambda *args, **kwargs: model.weight.square().sum()
    )
    output = tmp_path / "checkpoint"
    sft.train(
        SimpleNamespace(
            data=data, output=output, gradient_checkpointing=False, lr=1e-3, steps=1, seed=7
        )
    )
    multimodal = kind != "text"
    selected = "AutoModelForImageTextToText" if multimodal else "AutoModelForCausalLM"
    unused = "AutoModelForCausalLM" if multimodal else "AutoModelForImageTextToText"
    loaders[selected].from_pretrained.assert_called_once_with(
        "tiny-image-model",
        config=config if multimodal else config.get_text_config.return_value,
        dtype=torch.float32,
    )
    loaders[unused].from_pretrained.assert_not_called()
    assert config.get_text_config.call_count == int(not multimodal)
    processor, unused = (
        ("AutoProcessor", "AutoTokenizer") if multimodal else ("AutoTokenizer", "AutoProcessor")
    )
    loaders[processor].from_pretrained.assert_called_once_with("tiny-image-model")
    loaders[processor].from_pretrained.return_value.save_pretrained.assert_called_once_with(output)
    loaders[unused].from_pretrained.assert_not_called()
    model.save_pretrained.assert_called_once_with(output)
    assert json.loads((tmp_path / "metrics.json").read_text())["supervised_tokens"] == len(traces)


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
def test_sft_cli_checkpointing_flags(monkeypatch, tmp_path, flags, expected):
    received = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["sft", "train", *flags])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(sft, "train", received.append)
    sft.main()
    args = received.pop()
    assert (args.attention_checkpointing, args.loss_checkpointing) == expected


@pytest.mark.parametrize(
    "flag,expected",
    [(None, True), ("--cache-replay-plans", True), ("--no-cache-replay-plans", False)],
)
def test_sft_cli_plan_cache_flag(monkeypatch, tmp_path, flag, expected):
    received = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["sft", "train", *([flag] if flag else [])])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(sft, "train", received.append)
    sft.main()
    assert received.pop().cache_replay_plans is expected
