from math import ceil
from unittest.mock import patch

import pytest
import torch
from minisgl.training import replay as replay_module
from minisgl.training import sft
from minisgl.training.gdn import GDNReplay
from minisgl.training.replay import Replay
from test_training import mutation_trace, tiny_model
from test_training_gdn import tiny_hybrid


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sft_reuses_compiled_plans_from_cpu_replay(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = tiny_hybrid().to(device).train()
    replay = Replay.from_trace(mutation_trace(), query_chunk_size=3)
    targets = torch.arange(len(replay.input_ids)) % 32
    with patch.object(GDNReplay, "from_trace", wraps=GDNReplay.from_trace) as compile:
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            sft.supervised_loss(model, replay, targets, chunk_size=4).backward()
            assert compile.call_count == 1
        attention = next(iter(replay._attention.values()))
        sft.evaluate(model.eval(), [(replay, targets)], chunk_size=4)
        assert compile.call_count == 1
    assert len(replay._gdn) == len(replay._attention) == 1
    assert next(iter(replay._attention.values())) is attention
    assert replay.input_ids.device.type == "cpu"


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
