import json
import math
from copy import deepcopy
from dataclasses import asdict, replace
from math import ceil
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from minisgl.training import replay as replay_module
from minisgl.training import sft
from minisgl.training.gdn import GDNReplay
from minisgl.training.replay import Replay
from minisgl.training.trace import ForwardBatch, ForwardRequest, TensorData, Trace
from test_training import mutation_trace, tiny_model
from test_training_gdn import tiny_hybrid
from transformers import Qwen3_5ForCausalLM


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
        "min_sequence_probability": math.exp(-1.0),
        "examples": [
            {
                "nll": [1.0],
                "loss": 1.0,
                "accuracy": 0.0,
                "sequence_probability": math.exp(-1.0),
            }
        ]
        * len(traces),
    }
    monkeypatch.setattr(sft, "evaluate", Mock(return_value=evaluation))
    monkeypatch.setattr(sft, "supervised_loss", lambda *args, **kwargs: model.weight.square().sum())
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


def test_evaluate_reports_full_sequence_probability():
    model = tiny_model().eval()
    replay = Replay.from_trace(mutation_trace())
    targets = torch.arange(len(replay.input_ids)) % 32
    targets[::3] = -100

    result = sft.evaluate(model, [(replay, targets)], chunk_size=4)
    example = result["examples"][0]

    assert example["sequence_logprob"] == pytest.approx(-sum(example["nll"]))
    assert example["sequence_probability"] == pytest.approx(math.exp(-sum(example["nll"])))
    assert result["min_sequence_probability"] == example["sequence_probability"]


def test_float16_adam_epsilon_does_not_underflow():
    assert sft._adam_epsilon("float16") == 1e-6
    assert sft._adam_epsilon("float32") == 1e-8
    assert sft._adam_epsilon("float16", 1e-5) == 1e-5


@pytest.mark.parametrize("probability,passed", [(0.995, True), (0.98, False)])
def test_verify_reloads_checkpoint_and_enforces_joint_probability(
    monkeypatch, tmp_path, probability, passed
):
    data = tmp_path / "traces.json"
    data.write_text('{"model": "unused", "examples": []}')
    output = tmp_path / "verification.json"
    model = Mock()
    model.eval.return_value = model
    monkeypatch.setattr(sft, "_examples", Mock(return_value=[("replay", "targets")]))
    loader = Mock(return_value=(model, False))
    monkeypatch.setattr(sft, "_model", loader)
    monkeypatch.setattr(
        sft,
        "evaluate",
        Mock(
            return_value={
                "loss": 0.0,
                "accuracy": 1.0,
                "min_sequence_probability": probability,
                "examples": [],
            }
        ),
    )
    args = SimpleNamespace(
        data=data,
        checkpoint=tmp_path / "checkpoint",
        output=output,
        dtype="float16",
        target_sequence_probability=0.99,
        loss_chunk_size=8,
    )

    if passed:
        sft.verify(args)
    else:
        with pytest.raises(RuntimeError, match="required probability"):
            sft.verify(args)

    assert json.loads(output.read_text())["passed"] is passed
    loader.assert_called_once_with(args.checkpoint, [("replay", "targets")], torch.float16)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("cache_plans", [False, True])
def test_sft_reuses_compiled_plans_from_cpu_replay(device, cache_plans):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = tiny_hybrid().config
    config.num_hidden_layers = 3
    config.layer_types = ["linear_attention", "full_attention", "linear_attention"]
    model = Qwen3_5ForCausalLM(config).to(device).train()
    replay = Replay.from_trace(mutation_trace(), query_chunk_size=3, cache_plans=cache_plans)
    targets = torch.arange(len(replay.input_ids)) % 32
    with patch.object(GDNReplay, "from_trace", wraps=GDNReplay.from_trace) as compile:
        for i in range(2):
            model.zero_grad(set_to_none=True)
            sft.supervised_loss(model, replay, targets, chunk_size=4).backward()
            assert compile.call_count == (1 if cache_plans else i + 1)
            assert bool(replay._gdn) == bool(replay._attention) == cache_plans
        attention = next(iter(replay._attention.values())) if cache_plans else None
        sft.evaluate(model.eval(), [(replay, targets)], chunk_size=4)
        assert compile.call_count == (1 if cache_plans else 3)
    assert len(replay._gdn) == len(replay._attention) == int(cache_plans)
    if cache_plans:
        assert next(iter(replay._attention.values())) is attention
    assert replay.input_ids.device.type == "cpu"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("use_reentrant", [None, False, True])
def test_transient_plans_preserve_logits_and_checkpointed_gradients(device, use_reentrant):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    reference = tiny_hybrid().to(device).train()
    model = deepcopy(reference)
    if use_reentrant is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
    trace = mutation_trace()
    persistent = Replay.from_trace(trace, device=device, query_chunk_size=3)
    transient = Replay.from_trace(trace, query_chunk_size=3, cache_plans=False)
    targets = torch.arange(len(trace.tokens)) % 32
    targets[::3] = -100
    logits = replay_module.forward(reference, persistent)
    expected = F.cross_entropy(logits, targets.to(device), reduction="sum")
    actual = sum(sft.supervised_loss(model, transient, targets, chunk_size=4) for _ in range(2)) / 2
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for parameter, original in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, original.grad, atol=2e-6, rtol=2e-5)
    with torch.no_grad():
        actual_logits = replay_module.forward(model.eval(), transient.to(device))
    torch.testing.assert_close(actual_logits, logits, atol=2e-6, rtol=2e-5)
    assert not transient._gdn and not transient._attention


def test_switching_off_plan_cache_releases_previous_plans():
    model = tiny_hybrid().eval()
    replay = Replay.from_trace(mutation_trace())
    targets = torch.arange(len(replay.input_ids)) % 32
    expected = sft.evaluate(model, [(replay, targets)])
    caches = replay._gdn, replay._attention
    assert all(caches)
    replay.cache_plans = False
    actual = sft.evaluate(model, [(replay, targets)])
    assert actual == expected
    assert not any(caches)
    assert not replay._gdn and not replay._attention


@pytest.mark.parametrize("attention_checkpointing", [False, True])
@pytest.mark.parametrize("loss_checkpointing", [False, True])
def test_sft_checkpointing_switches_are_independent(attention_checkpointing, loss_checkpointing):
    reference, model = tiny_model().train(), tiny_model().train()
    trace = mutation_trace()
    targets = torch.arange(len(trace.tokens)) % 32
    targets[::3] = -100
    native = Replay.from_trace(trace, query_chunk_size=3, attention_checkpointing=False)
    expected = sft.supervised_loss(reference, native, targets, chunk_size=4, checkpointing=False)
    expected.backward()
    replay = Replay.from_trace(
        trace, query_chunk_size=3, attention_checkpointing=attention_checkpointing
    )
    with (
        patch.object(replay_module, "checkpoint", wraps=replay_module.checkpoint) as attention,
        patch.object(sft, "checkpoint", wraps=sft.checkpoint) as loss,
    ):
        actual = sft.supervised_loss(
            model, replay, targets, chunk_size=4, checkpointing=loss_checkpointing
        )
        assert not model.is_gradient_checkpointing
        assert attention.call_count == (
            len(model.model.layers) * ceil(len(trace.tokens) / 3) if attention_checkpointing else 0
        )
        assert loss.call_count == (
            ceil(int((targets != -100).sum()) / 4) if loss_checkpointing else 0
        )
        actual.backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for parameter, original in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, original.grad, atol=2e-6, rtol=2e-5)


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
