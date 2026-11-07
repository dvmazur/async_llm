from copy import deepcopy

import pytest
import torch
from minisgl.training import Trace, replay_trace
from minisgl.training.trace import ForwardBatch, ForwardRequest
from transformers import LlamaConfig, LlamaForCausalLM

from tests.training.models import live_llm


@pytest.mark.parametrize(
    "scaled,tied,live", [(False, False, False), (True, True, False), (True, True, True)]
)
def test_llama_trace_replay_matches_native_logits_and_gradients(scaled, tied, live):
    torch.manual_seed(42)
    rope = {"rope_type": "default", "rope_theta": 10000.0}
    if scaled:
        rope.update(
            rope_type="llama3",
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8,
        )
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_parameters=rope,
        tie_word_embeddings=tied,
        attn_implementation="eager",
    )
    native = LlamaForCausalLM(config).eval()
    # Two branches share a four-token prefix, then prefill and decode in
    # parallel. Each branch must match its own causal HF forward.
    trace = Trace(
        batches=[
            ForwardBatch(0, "prefill", (ForwardRequest((1, 2, 3, 4), 0, 0, ((0, 0),)),)),
            ForwardBatch(
                1,
                "prefill",
                (
                    ForwardRequest((5, 6), 1, 0, ((0, 4), (1, 0))),
                    ForwardRequest((7, 8), 2, 0, ((0, 4), (2, 0))),
                ),
            ),
            ForwardBatch(
                2,
                "decode",
                (
                    ForwardRequest((9,), 1, 2, ((0, 4), (1, 2))),
                    ForwardRequest((10,), 2, 2, ((0, 4), (2, 2))),
                ),
            ),
        ],
        steps=3,
    )
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 9], [1, 2, 3, 4, 7, 8, 10]])
    left, right = native(ids, use_cache=False).logits.unbind()
    expected = torch.cat([left[:6], right[4:6], left[-1:], right[-1:]])

    if live:
        model, weights, groups = live_llm(native)
    else:
        model = deepcopy(native)
        weights = dict(model.named_parameters())
        groups = {name: [name] for name in weights}
    actual = replay_trace(model, trace, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    targets = torch.tensor([2, 3, 4, 5, 6, 9, 8, 10, -100, -100])
    for logits in (expected, actual):
        torch.nn.functional.cross_entropy(logits, targets).backward()
    reference = dict(native.named_parameters())
    for name, sources in groups.items():
        grads = [reference[source].grad for source in sources]
        torch.testing.assert_close(weights[name].grad, torch.cat(grads), atol=2e-5, rtol=3e-4)
