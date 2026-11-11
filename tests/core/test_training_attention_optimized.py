"""Shared KV preparation preserves the FP32 Torch replay attention contract."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from minisgl.training import replay_trace
from minisgl.training.attention import attention as reference_attention
from minisgl.training.attention.torch_optimized import attention
from test_training import causal_trace, mutation_trace, repeated_view_trace
from test_training_attention_flash import interleaved_blocks_trace, tile_boundary_trace
from test_training_images import cache_image_trace
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def inputs(replay, *, device="cpu", dtype=torch.float32, seed=211):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(replay.num_tokens, heads, 16, generator=generator)
        .to(device=device, dtype=dtype)
        .requires_grad_()
        for heads in (4, 2, 2)
    )


def compare(inputs, replay):
    reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    inv_freq = 10000 ** (-torch.arange(0, 8, 2, device=inputs[0].device).float() / 8)
    section = (2, 1, 1) if replay.multimodal else None
    expected = reference_attention(
        *reference_inputs, replay.to(attention_backend="torch"), inv_freq, 0.25, section
    )
    actual = attention(*inputs, replay, inv_freq, 0.25, section)
    upstream = torch.linspace(
        -1, 1, actual.numel(), device=actual.device, dtype=actual.dtype
    ).reshape_as(actual)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, upstream)
    gradients = torch.autograd.grad(actual, inputs, upstream)
    assert actual.dtype == expected.dtype == inputs[2].dtype
    for tensor in (actual, *gradients):
        assert torch.isfinite(tensor).all()
    return actual, expected, gradients, expected_gradients


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7])
@pytest.mark.parametrize(
    "make_trace",
    [interleaved_blocks_trace, mutation_trace, repeated_view_trace, cache_image_trace],
)
def test_prepared_kv_matches_reference_outputs_and_gradients(make_trace, chunk_size, checkpointing):
    replay = make_trace().to(
        attention_backend="torch_optimized",
        query_chunk_size=chunk_size,
        attention_checkpointing=checkpointing,
    )
    actual, expected, gradients, expected_gradients = compare(inputs(replay), replay)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    for name, observed, wanted in zip(("dQ", "dK", "dV"), gradients, expected_gradients):
        torch.testing.assert_close(observed, wanted, atol=1e-5, rtol=2e-4, msg=name)


def test_mixed_trace_with_nonzero_left_reads_matches_reference():
    replay = tile_boundary_trace().to(
        attention_backend="torch_optimized", query_chunk_size=64, attention_checkpointing=True
    )
    assert any(left > 0 for reads in replay.reads for _, left, *_ in reads)
    actual, expected, gradients, expected_gradients = compare(inputs(replay), replay)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    for name, observed, wanted in zip(("dQ", "dK", "dV"), gradients, expected_gradients):
        torch.testing.assert_close(observed, wanted, atol=2e-5, rtol=2e-4, msg=name)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_inference_plan_reuse_does_not_retain_prepared_kv_or_old_graph(checkpointing):
    with torch.inference_mode():
        replay = repeated_view_trace().to(
            attention_backend="torch_optimized",
            query_chunk_size=3,
            attention_checkpointing=checkpointing,
        )
        warmup = tuple(value.detach() for value in inputs(replay))
        attention(*warmup, replay, torch.zeros(4), 0.25)
    plans = tuple(replay._attention.values())
    assert plans
    # Both forward values and a new gradient graph must follow the new Q/K/V.
    for seed in (31, 67):
        actual, expected, gradients, expected_gradients = compare(inputs(replay, seed=seed), replay)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        for observed, wanted in zip(gradients, expected_gradients):
            torch.testing.assert_close(observed, wanted, atol=1e-5, rtol=2e-4)
        assert all(
            current is cached
            for current, cached in zip(replay._attention.values(), plans, strict=True)
        )


@pytest.mark.parametrize("cache_plans", [False, True])
def test_dense_qwen35_adapter_uses_optimized_backend_with_fresh_gradients(cache_plans):
    torch.manual_seed(37)
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
    )
    config._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(config).train()
    reference = deepcopy(model)
    trace = mutation_trace()
    original = trace.to(query_chunk_size=3)
    optimized = original.to(attention_backend="torch_optimized", cache_plans=cache_plans)
    assert original.attention_backend == "torch"
    expected = replay_trace(reference, original, hidden_only=False)
    actual = replay_trace(model, optimized, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    targets = torch.arange(trace.num_tokens) % 32
    F.cross_entropy(expected, targets).backward()
    F.cross_entropy(actual, targets).backward()
    for (name, parameter), (_, wanted) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None and wanted.grad is not None, name
        torch.testing.assert_close(parameter.grad, wanted.grad, atol=1e-6, rtol=2e-4, msg=name)
    assert bool(optimized._attention) is cache_plans


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device("cuda")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_cuda_low_precision_outputs_and_gradients(cuda_device, dtype, checkpointing):
    replay = repeated_view_trace().to(
        cuda_device,
        attention_backend="torch_optimized",
        query_chunk_size=3,
        attention_checkpointing=checkpointing,
    )
    actual, expected, gradients, expected_gradients = compare(
        inputs(replay, device=cuda_device, dtype=dtype), replay
    )
    tolerance = (
        {"atol": 1e-5, "rtol": 1e-2} if dtype == torch.bfloat16 else {"atol": 1e-5, "rtol": 2e-3}
    )
    torch.testing.assert_close(actual, expected, **tolerance)
    for name, observed, wanted in zip(("dQ", "dK", "dV"), gradients, expected_gradients):
        torch.testing.assert_close(observed, wanted, **tolerance, msg=name)


def test_cuda_bf16_inputs_preserve_distinct_fp32_scores(cuda_device):
    replay = causal_trace(2).to(
        cuda_device,
        attention_backend="torch_optimized",
        query_chunk_size=1,
        attention_checkpointing=True,
    )
    # The exact logits 256.5 and 257 both round to 256 if QK is stored in BF16.
    q = torch.tensor(
        [[[16.0, 1.0]], [[16.0, 1.0]]], device=cuda_device, dtype=torch.bfloat16
    ).requires_grad_()
    k = torch.tensor(
        [[[16.0, 0.5]], [[16.0, 1.0]]], device=cuda_device, dtype=torch.bfloat16
    ).requires_grad_()
    v = torch.tensor(
        [[[0.0, 0.0]], [[1.0, 1.0]]], device=cuda_device, dtype=torch.bfloat16
    ).requires_grad_()
    actual = attention(q, k, v, replay, torch.zeros(1, device=cuda_device), 1.0)
    probability = torch.sigmoid(torch.tensor(0.5, device=cuda_device)).to(torch.bfloat16)
    torch.testing.assert_close(actual[1], probability.expand(2), atol=0, rtol=0)
    for gradient in torch.autograd.grad(actual.sum(), (q, k, v)):
        assert torch.isfinite(gradient).all()


def test_shared_values_accumulate_chunk_gradients_before_bf16_cast():
    replay = causal_trace(3).to(
        attention_backend="torch_optimized", query_chunk_size=1, attention_checkpointing=True
    )
    q = torch.zeros((3, 1, 2), dtype=torch.bfloat16).requires_grad_()
    k = torch.zeros_like(q).requires_grad_()
    v = torch.ones_like(q).requires_grad_()
    actual = attention(q, k, v, replay, torch.zeros(1), 1.0)
    upstream = torch.zeros_like(actual)
    upstream[:, 0] = torch.tensor([1024, 2, -3072], dtype=torch.bfloat16)
    gradients = torch.autograd.grad(actual, (q, k, v), upstream)
    # Source 0 receives 1024 + 2 / 2 - 3072 / 3 = 1 across three chunks.
    assert gradients[2][0, 0, 0].item() == 1.0
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_cuda_fp16_scores_do_not_overflow_before_softmax(cuda_device):
    replay = causal_trace(3).to(
        cuda_device,
        attention_backend="torch_optimized",
        query_chunk_size=2,
        attention_checkpointing=True,
    )
    q = torch.full((3, 2, 16), 512.0, device=cuda_device, dtype=torch.float16).requires_grad_()
    k = torch.full((3, 1, 16), 512.0, device=cuda_device, dtype=torch.float16).requires_grad_()
    v = torch.arange(48, device=cuda_device, dtype=torch.float16).reshape(3, 1, 16) / 16
    v.requires_grad_()
    actual = attention(q, k, v, replay, torch.zeros(4, device=cuda_device), 0.25)
    expected = torch.stack(
        [v[:end].float().mean(0).expand(2, -1).flatten() for end in (1, 2, 3)]
    ).to(torch.float16)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)
    for gradient in torch.autograd.grad(actual.sum(), (q, k, v)):
        assert torch.isfinite(gradient).all()
