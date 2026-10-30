import inspect
import os
from itertools import groupby

import pytest
import torch
from minisgl.training.replay import Replay, forward
from test_training import (
    example_trace,
    mutation_trace,
    repeated_view_trace,
    sequential_attention,
    single_block_trace,
)
from test_training_gdn import (
    cuda_delta_math as cuda_delta_math,
    sequential_gdn,
    tiny_hybrid,
)
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3_5ForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5


@pytest.fixture(scope="module", autouse=True)
def torch_reference_kernels():
    """Keep FP32 reference arithmetic independent of optional inference kernels."""
    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "FusedRMSNormGated",
        ):
            patch.setattr(modeling_qwen3_5, name, None)
        for name in (
            "torch_causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        ):
            patch.setattr(modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name)))
        patch.setattr(torch.backends.cudnn, "allow_tf32", False)
        yield


def sequential_hybrid_logits(model, trace):
    """Evaluate each layer's recorded steps using raw KV and direct GDN recurrence."""
    backbone = model.model
    device = backbone.embed_tokens.weight.device
    ids = torch.tensor([token.token_id for token in trace.tokens], device=device)
    order = torch.arange(len(trace.tokens), device=device)
    x = backbone.embed_tokens(ids)
    for layer in backbone.layers:
        h = layer.input_layernorm(x)
        if hasattr(layer, "linear_attn"):
            output = sequential_gdn(layer.linear_attn, h, trace)
        else:
            attn = layer.self_attn
            query, gate = attn.q_proj(h).view(len(h), -1, 2 * attn.head_dim).chunk(2, -1)
            query = attn.q_norm(query)
            key = attn.k_norm(attn.k_proj(h).view(len(h), -1, attn.head_dim))
            value = attn.v_proj(h).view(len(h), -1, attn.head_dim)
            output = sequential_attention(
                query, key, value, trace, order, backbone.rotary_emb.inv_freq, attn.scaling
            )
            output = attn.o_proj(output * gate.reshape_as(output).sigmoid())
        x = x + output
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return model.lm_head(backbone.norm(x))


def four_layer_hybrid(device):
    config = tiny_hybrid().config
    config.num_hidden_layers = 4
    config.layer_types = ["linear_attention", "full_attention"] * 2
    model = Qwen3_5ForCausalLM(config).to(device).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.A_log.fill_(-1)
    return model


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_sequential_hybrid_logits_match_cached_huggingface(request, device):
    if device == "cuda":
        request.getfixturevalue("cuda_delta_math")
    model = four_layer_hybrid(device)
    trace = single_block_trace(mixed_steps=True)
    outputs, cache = [], None
    for _, tokens in groupby(trace.tokens, key=lambda token: token.step):
        ids = torch.tensor([[token.token_id for token in tokens]], device=device)
        output = model(ids, past_key_values=cache, use_cache=True)
        outputs.append(output.logits[0])
        cache = output.past_key_values
    expected = torch.cat(outputs)
    torch.testing.assert_close(
        sequential_hybrid_logits(model, trace), expected, atol=3e-6, rtol=3e-5
    )
    actual = forward(model, Replay.from_trace(trace, device=device, query_chunk_size=3))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("device,backend", [("cpu", "torch"), ("cuda", "torch"), ("cuda", "triton")])
@pytest.mark.parametrize("trace_factory", [example_trace, mutation_trace, repeated_view_trace])
@pytest.mark.parametrize(
    "read_mode,segment_size,query_chunk_size,cache_reused",
    [("expanded", 1, 1, False), ("expanded", 8, 3, True), ("factorized", 3, 128, True)],
)
def test_parallel_hybrid_logits_match_sequential(
    request, device, backend, trace_factory, read_mode, segment_size, query_chunk_size, cache_reused
):
    if device == "cuda":
        request.getfixturevalue("cuda_delta_math")
    model = four_layer_hybrid(device).train()
    trace = trace_factory()
    with torch.no_grad():
        expected = sequential_hybrid_logits(model, trace)
    replay = Replay.from_trace(
        trace,
        device=device,
        query_chunk_size=query_chunk_size,
        gdn_segment_size=segment_size,
        gdn_read_mode=read_mode,
        gdn_cache_reused=cache_reused,
        gdn_read_backend=backend,
    )
    actual = forward(model, replay)
    assert actual.requires_grad
    assert actual.shape == (len(trace.tokens), model.config.vocab_size)
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    rows = torch.tensor([len(trace.tokens) - 1, 0, len(trace.tokens) // 2, 0], device=device)
    torch.testing.assert_close(
        forward(model, replay, rows=rows), expected[rows], atol=3e-6, rtol=3e-5
    )


@pytest.fixture(scope="module")
def pretrained_hybrid(torch_reference_kernels):
    source = os.environ.get("MINISGL_TRAINING_LOGITS_MODEL")
    if not source:
        pytest.skip("Set MINISGL_TRAINING_LOGITS_MODEL to a local or already-cached Qwen3.5 model")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = AutoConfig.from_pretrained(source, local_files_only=True).get_text_config()
    assert config.model_type == "qwen3_5_text", "A dense Qwen3.5 text model is required"
    return AutoModelForCausalLM.from_pretrained(
        source,
        config=config,
        dtype=torch.float32,
        device_map="cuda",
        attn_implementation="eager",
        local_files_only=True,
    ).eval()


@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
@pytest.mark.parametrize("trace_factory", [mutation_trace, repeated_view_trace])
@pytest.mark.usefixtures("cuda_delta_math")
@torch.no_grad()
def test_pretrained_parallel_hybrid_logits(pretrained_hybrid, trace_factory, read_mode, backend):
    trace = trace_factory()
    expected = sequential_hybrid_logits(pretrained_hybrid, trace)
    replay = Replay.from_trace(
        trace, device="cuda", query_chunk_size=3, gdn_read_mode=read_mode, gdn_read_backend=backend
    )
    actual = forward(pretrained_hybrid, replay)
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
