"""Precision contracts for the dense reference, independent of CUDA kernels."""

from contextlib import nullcontext

import pytest
import torch
from minisgl.training.attention import attention
from minisgl.training.attention.torch import _attention_impl
from test_training import Float64AttentionMath, causal_trace, mutation_trace, repeated_view_trace
from test_training import fp64_attention as fp64_attention
from torch.overrides import TorchFunctionMode


def _rotate(query, positions, inv_freq):
    """Text RoPE directly in the oracle's working dtype, including partial RoPE."""
    positions = torch.as_tensor(positions, dtype=query.dtype, device=query.device)
    angles = positions[:, None] * inv_freq.to(dtype=query.dtype)[None]
    cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
    width = 2 * inv_freq.numel()
    real, imaginary = query[..., :width].chunk(2, dim=-1)
    return torch.cat(
        (
            real * cosine - imaginary * sine,
            imaginary * cosine + real * sine,
            query[..., width:],
        ),
        dim=-1,
    )


def _dense_rows(q, k, v, replay, inv_freq, scaling):
    """Expand each historical view directly; no query chunks or version grouping."""
    groups = q.shape[1] // k.shape[1]
    outputs = []
    for row, reads in enumerate(replay.reads):
        logits, values = [], []
        for version, left, right, position in reads:
            sources = [
                source
                for start, end, _ in replay.compute_source_spans(version, left, right)
                for source in range(start, end)
            ]
            query = _rotate(q[row : row + 1], [position], inv_freq)[0]
            keys = _rotate(k[sources], range(left, right), inv_freq)
            keys = keys.repeat_interleave(groups, dim=1).transpose(0, 1)
            logits.append((query[:, None] * keys).sum(-1) * scaling)
            values.append(v[sources].repeat_interleave(groups, dim=1).transpose(0, 1))
        weights = torch.cat(logits, dim=-1).softmax(-1)
        outputs.append((weights[..., None] * torch.cat(values, dim=1)).sum(1).flatten())
    return torch.stack(outputs)


class _GemmDtypes(TorchFunctionMode):
    """Include checkpoint recomputation in the reference math precision check."""

    def __init__(self):
        super().__init__()
        self.observed = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func in (torch.Tensor.__matmul__, torch.Tensor.matmul, torch.matmul):
            self.observed.append((args[0].dtype, args[1].dtype, result.dtype))
        return result


def _inputs(replay, dtype, width=16):
    generator = torch.Generator().manual_seed(197)
    return tuple(
        torch.randn(replay.num_tokens, heads, width, generator=generator).to(dtype).requires_grad_()
        for heads in (4, 2, 2)
    )


def _run_against_oracle(inputs, replay, inv_freq, *, autocast=False):
    math_dtype = torch.float64 if inputs[0].dtype == torch.float64 else torch.float32
    leaves = tuple(value.detach().to(math_dtype).requires_grad_() for value in inputs)
    scaling = inputs[0].shape[-1] ** -0.5
    expected = _dense_rows(*leaves, replay, inv_freq, scaling).to(inputs[2].dtype)
    upstream = torch.linspace(-1, 1, expected.numel(), dtype=expected.dtype).reshape_as(expected)
    expected_gradients = torch.autograd.grad(expected, leaves, upstream)
    audit = _GemmDtypes()
    context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
    with context:
        with audit:
            actual = attention(*inputs, replay, inv_freq, scaling)
        if autocast:
            assert torch.is_autocast_enabled("cpu")
    with audit:
        actual_gradients = torch.autograd.grad(actual, inputs, upstream)
    assert actual.dtype == inputs[2].dtype
    assert audit.observed, "The reference must exercise QK and PV matrix products"
    assert all(item == (math_dtype, math_dtype, math_dtype) for item in audit.observed)
    return actual, expected, actual_gradients, expected_gradients


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("make_trace", [mutation_trace, repeated_view_trace])
def test_low_precision_inputs_use_fp32_attention_and_source_sums(dtype, checkpointing, make_trace):
    replay = make_trace().to(query_chunk_size=3, attention_checkpointing=checkpointing)
    inputs = _inputs(replay, dtype)
    # Only the first half of each head has rotary coordinates.
    inv_freq = 10000 ** (-torch.arange(0, 8, 2).float() / 8)
    actual, expected, gradients, expected_gradients = _run_against_oracle(inputs, replay, inv_freq)
    tolerance = {"atol": 1e-5, "rtol": 1e-2 if dtype == torch.bfloat16 else 2e-3}
    torch.testing.assert_close(actual, expected, **tolerance)
    for name, observed, wanted in zip(("q", "k", "v"), gradients, expected_gradients):
        assert observed.dtype == dtype
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, wanted.to(dtype), **tolerance, msg=name)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_cpu_autocast_does_not_downcast_reference_gemms(checkpointing):
    replay = repeated_view_trace().to(query_chunk_size=4, attention_checkpointing=checkpointing)
    inputs = _inputs(replay, torch.float32)
    inv_freq = 10000 ** (-torch.arange(0, 16, 2).float() / 16)
    actual, expected, gradients, expected_gradients = _run_against_oracle(
        inputs, replay, inv_freq, autocast=True
    )
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    for name, observed, wanted in zip(("q", "k", "v"), gradients, expected_gradients):
        torch.testing.assert_close(observed, wanted, atol=5e-6, rtol=1e-4, msg=name)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_bf16_precision_modes_preserve_or_round_distinct_logits(checkpointing):
    replay = causal_trace(2).to(query_chunk_size=1, attention_checkpointing=checkpointing)
    # The second query has exact scores 256.5 and 257. BF16 storage rounds both
    # to 256, incorrectly making attention uniform before softmax even starts.
    q = torch.tensor([[[16.0, 1.0]], [[16.0, 1.0]]], dtype=torch.bfloat16).requires_grad_()
    k = torch.tensor([[[16.0, 0.5]], [[16.0, 1.0]]], dtype=torch.bfloat16).requires_grad_()
    v = torch.tensor([[[0.0, 0.0]], [[1.0, 1.0]]], dtype=torch.bfloat16).requires_grad_()
    inv_freq = torch.zeros(1)
    leaves = tuple(value.detach().float().requires_grad_() for value in (q, k, v))
    expected = _dense_rows(*leaves, replay, inv_freq, 1.0).to(torch.bfloat16)
    actual = attention(q, k, v, replay, inv_freq, 1.0)
    with torch.no_grad():
        low_precision = _attention_impl(q, k, v, replay, inv_freq, 1.0, upcast=False)
    assert low_precision.dtype == actual.dtype == torch.bfloat16
    torch.testing.assert_close(low_precision[1], torch.full((2,), 0.5).bfloat16(), atol=0, rtol=0)
    probability = torch.sigmoid(torch.tensor(0.5)).to(torch.bfloat16)
    torch.testing.assert_close(actual[1], probability.expand(2), atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.ones_like(actual)
    gradients = torch.autograd.grad(actual, (q, k, v), upstream)
    expected_gradients = torch.autograd.grad(expected, leaves, upstream)
    for observed, wanted in zip(gradients, expected_gradients):
        torch.testing.assert_close(observed, wanted.to(torch.bfloat16), atol=0, rtol=0)


def test_fp16_qk_does_not_overflow_before_fp32_softmax():
    replay = causal_trace(3).to(query_chunk_size=2, attention_checkpointing=True)
    q = torch.full((3, 2, 16), 512.0, dtype=torch.float16).requires_grad_()
    k = torch.full((3, 1, 16), 512.0, dtype=torch.float16).requires_grad_()
    v = (torch.arange(48, dtype=torch.float16).reshape(3, 1, 16) / 16).requires_grad_()
    actual = attention(q, k, v, replay, torch.zeros(4), 0.25)
    # Equal finite logits give the mean of the visible prefix. Materializing
    # the large QK scores in FP16 instead produces inf and then NaN softmax.
    expected = torch.stack([v[:end].float().mean(0).expand(2, -1).flatten() for end in (1, 2, 3)])
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected.to(torch.float16), atol=0, rtol=0)
    for gradient in torch.autograd.grad(actual.sum(), (q, k, v)):
        assert torch.isfinite(gradient).all()


def test_source_gradients_accumulate_before_casting_to_bf16():
    replay = causal_trace(3).to(query_chunk_size=1, attention_checkpointing=True)
    q = torch.zeros((3, 1, 2), dtype=torch.bfloat16).requires_grad_()
    k = torch.zeros_like(q).requires_grad_()
    v = torch.ones_like(q).requires_grad_()
    inv_freq = torch.zeros(1)
    leaves = tuple(value.detach().float().requires_grad_() for value in (q, k, v))
    actual = attention(q, k, v, replay, inv_freq, 1.0)
    expected = _dense_rows(*leaves, replay, inv_freq, 1.0).to(torch.bfloat16)
    upstream = torch.zeros_like(actual)
    upstream[:, 0] = torch.tensor([1024, 2, -3072], dtype=torch.bfloat16)
    gradients = torch.autograd.grad(actual, (q, k, v), upstream)
    expected_gradients = torch.autograd.grad(expected, leaves, upstream)
    # Source 0 receives 1024 + 2 / 2 - 3072 / 3 = 1 across three chunks.
    # Accumulating those contributions in BF16 can discard the middle 1.
    assert gradients[2][0, 0, 0].item() == 1.0
    for observed, wanted in zip(gradients, expected_gradients):
        torch.testing.assert_close(observed, wanted.to(torch.bfloat16), atol=0, rtol=0)


def test_double_reference_preserves_double_math(fp64_attention):
    replay = mutation_trace().to(query_chunk_size=3, attention_checkpointing=True)
    inputs = _inputs(replay, torch.float64, width=8)
    inv_freq = (10000 ** (-torch.arange(0, 8, 2).double() / 8)).double()
    with Float64AttentionMath():
        actual, expected, gradients, expected_gradients = _run_against_oracle(
            inputs, replay, inv_freq
        )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for name, observed, wanted in zip(("q", "k", "v"), gradients, expected_gradients):
        assert observed.dtype == torch.float64
        torch.testing.assert_close(observed, wanted, atol=2e-12, rtol=2e-12, msg=name)
