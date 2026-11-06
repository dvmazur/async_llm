import inspect
import os
from dataclasses import replace
from itertools import groupby
from types import SimpleNamespace as NS

import pytest
import torch
from minisgl.training import replay_trace
from test_training import (
    example_trace,
    mutation_trace,
    repeated_view_trace,
    sequential_attention,
    single_block_trace,
    tiny_model,
)
from test_training_gdn import (
    cuda_delta_math as cuda_delta_math,
)
from test_training_gdn import (
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


def test_hybrid_trace_mutation_recompiles_attention_and_recurrent_plans():
    model, trace = tiny_hybrid(), mutation_trace()
    before = replay_trace(model, trace, hidden_only=False)
    recurrent = next(iter(trace._gdn.values()))
    attention = next(iter(trace._attention.values()))
    update = trace.updates[0]
    trace.updates[0] = replace(update, refs=tuple(reversed(update.refs)))
    actual = replay_trace(model, trace, hidden_only=False)
    torch.testing.assert_close(actual, sequential_hybrid_logits(model, trace), atol=3e-6, rtol=3e-5)
    assert not torch.allclose(actual, before)
    assert next(iter(trace._gdn.values())) is not recurrent
    assert next(iter(trace._attention.values())) is not attention


def live_llm(reference):
    """Represent inference's fused leaf tensors without initializing a GPU engine."""
    from minisgl.llm import AsyncLLM

    parameters = dict(reference.named_parameters())
    groups = {name: [name] for name in parameters}
    for name in parameters:
        for projections, fused in ((("q", "k", "v"), "qkv"), (("gate", "up"), "gate_up")):
            part = f".{projections[0]}_proj."
            if part in name:
                sources = [name.replace(part, f".{projection}_proj.") for projection in projections]
                for source in sources:
                    groups.pop(source)
                groups[name.replace(part, f".{fused}_proj.")] = sources
    weights = {
        name: (
            parameters[sources[0]].detach().clone()
            if len(sources) == 1
            else torch.cat([parameters[source].detach() for source in sources])
        )
        for name, sources in groups.items()
    }
    llm = object.__new__(AsyncLLM)
    llm.engine = NS(
        model=NS(state_dict=lambda: weights),
        config=NS(hf_config=reference.config, tp_info=NS(size=1), quantization=None),
    )
    llm.async_engine = NS(has_work=False)
    return llm, weights, groups


@pytest.mark.parametrize("factory", [tiny_model, tiny_hybrid], ids=["qwen3", "qwen3.5"])
def test_async_llm_replay_uses_live_fused_weights_and_gradients(factory):
    reference = factory()
    llm, weights, groups = live_llm(reference)
    addresses = {name: weight.data_ptr() for name, weight in weights.items()}
    trace = mutation_trace()
    rows = [18, 0, 7, 7, 13]
    expected = replay_trace(reference, trace, rows=rows, hidden_only=False)
    actual = replay_trace(llm, trace, rows=rows, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    expected.square().sum().backward()
    actual.square().sum().backward()
    parameters = dict(reference.named_parameters())
    for name, sources in groups.items():
        expected_grad = (
            parameters[sources[0]].grad
            if len(sources) == 1
            else torch.cat([parameters[source].grad for source in sources])
        )
        assert weights[name].is_leaf and weights[name].data_ptr() == addresses[name]
        torch.testing.assert_close(weights[name].grad, expected_grad, atol=2e-5, rtol=3e-4)
    assert all(parameter.is_meta for parameter in llm.engine._training_model.parameters())


@pytest.mark.parametrize("factory", [tiny_model, tiny_hybrid], ids=["qwen3", "qwen3.5"])
def test_async_llm_replay_observes_weight_updates_and_defaults_to_hidden(factory):
    reference = factory()
    llm, weights, groups = live_llm(reference)
    trace = single_block_trace(mixed_steps=True)
    before = replay_trace(llm, trace)
    torch.testing.assert_close(before, replay_trace(reference, trace), atol=3e-6, rtol=3e-5)
    adapter = llm.engine._training_model
    key = next(name for name in weights if ".gate_up_proj." in name)
    with torch.no_grad():
        weights[key].add_(0.1)
        parameters = dict(reference.named_parameters())
        for source in groups[key]:
            parameters[source].add_(0.1)
    after = replay_trace(llm, trace)
    torch.testing.assert_close(after, replay_trace(reference, trace), atol=3e-6, rtol=3e-5)
    assert not torch.allclose(after, before)
    assert llm.engine._training_model is adapter
    after.square().sum().backward()
    assert weights[key].grad is not None and weights[key].grad.abs().sum() > 0
    assert weights["lm_head.weight"].grad is None


@pytest.mark.parametrize("factory", [tiny_model, tiny_hybrid], ids=["qwen3", "qwen3.5"])
def test_async_llm_replay_trains_replaced_weights_after_inference_warmup(factory):
    reference = factory()
    llm, weights, groups = live_llm(reference)
    trace = mutation_trace()
    with torch.inference_mode():
        replay_trace(llm, trace)
    key = next(name for name in weights if ".qkv_proj." in name)
    original = weights[key]
    replacement = original.detach().clone().add_(0.1)
    weights[key] = replacement
    with torch.no_grad():
        parameters = dict(reference.named_parameters())
        for source in groups[key]:
            parameters[source].add_(0.1)
    expected = replay_trace(reference, trace, hidden_only=False)
    actual = replay_trace(llm, trace, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert replacement.is_leaf and replacement.requires_grad
    torch.testing.assert_close(
        replacement.grad,
        torch.cat([parameters[source].grad for source in groups[key]]),
        atol=2e-5,
        rtol=3e-4,
    )
    assert original.grad is None


@pytest.mark.parametrize("unsupported", ["tensor_parallel", "quantization", "pending_work"])
def test_async_llm_replay_rejects_unsupported_engine_state(unsupported):
    llm, _, _ = live_llm(tiny_model())
    if unsupported == "tensor_parallel":
        llm.engine.config.tp_info.size = 2
    elif unsupported == "quantization":
        llm.engine.config.quantization = "fp8"
    else:
        llm.async_engine.has_work = True
    with pytest.raises((NotImplementedError, RuntimeError), match="AsyncLLM"):
        replay_trace(llm, mutation_trace())


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
    actual = replay_trace(model, trace.to(device=device, query_chunk_size=3), hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize(
    "device,backend", [("cpu", "torch"), ("cuda", "torch"), ("cuda", "triton")]
)
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
    plan = trace.to(
        device=device,
        query_chunk_size=query_chunk_size,
        gdn_segment_size=segment_size,
        gdn_read_mode=read_mode,
        gdn_cache_reused=cache_reused,
        gdn_read_backend=backend,
    )
    actual = replay_trace(model, plan, hidden_only=False)
    assert actual.requires_grad
    assert actual.shape == (len(trace.tokens), model.config.vocab_size)
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    rows = torch.tensor([len(trace.tokens) - 1, 0, len(trace.tokens) // 2, 0], device=device)
    torch.testing.assert_close(
        replay_trace(model, plan, rows=rows, hidden_only=False),
        expected[rows],
        atol=3e-06,
        rtol=3e-05,
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
    plan = trace.to(
        device="cuda", query_chunk_size=3, gdn_read_mode=read_mode, gdn_read_backend=backend
    )
    actual = replay_trace(pretrained_hybrid, plan, hidden_only=False)
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
