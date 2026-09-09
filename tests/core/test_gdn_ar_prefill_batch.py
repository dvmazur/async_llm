"""Correctness tests for batched async-reasoning GDN prefills.

The reference is the former production behavior: execute
``_forward_ar_prefill_one`` independently for every request and concatenate the
rows.  These tests deliberately force the no-FLA implementation as well, since
the optimization must only change batching and must not make FLA mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import minisgl.models.qwen3_5_delta as delta_module
from minisgl.models.config import ModelConfig, RotaryConfig
from minisgl.models.qwen3_5_delta import Qwen3_5GatedDeltaNet


HIDDEN = 12
K_HEADS = 1
V_HEADS = 2
HEAD_DIM = 4
CONV_KERNEL = 4


def _config() -> ModelConfig:
    return ModelConfig(
        num_layers=1,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN,
        vocab_size=32,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(
            head_dim=HEAD_DIM,
            rotary_dim=HEAD_DIM,
            max_position=128,
            base=1e4,
            scaling=None,
        ),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        norm_topk_prob=False,
        model_type="qwen3_5",
        architectures=["Qwen3_5ForCausalLM"],
        layer_types=("linear_attention",),
        linear_num_key_heads=K_HEADS,
        linear_num_value_heads=V_HEADS,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=CONV_KERNEL,
    )


def _layer(seed: int = 71) -> Qwen3_5GatedDeltaNet:
    generator = torch.Generator().manual_seed(seed)
    layer = Qwen3_5GatedDeltaNet(_config(), linear_idx=0)
    for name, tensor in layer.state_dict().items():
        value = torch.randn(tensor.shape, generator=generator, dtype=torch.float32) * 0.05
        if name == "A_log":
            value.zero_()
        _assign(layer, name, value)
    return layer


def _assign(root, dotted: str, value: torch.Tensor) -> None:
    obj = root
    *path, leaf = dotted.split(".")
    for part in path:
        obj = getattr(obj, part)
    setattr(obj, leaf, value)


@dataclass
class _AR:
    prefill_segments: list[int]
    prior: torch.Tensor | None
    initial: torch.Tensor | None

    def __post_init__(self) -> None:
        self.captured: dict[int, tuple[torch.Tensor, ...]] = {}
        self.conv: dict[int, torch.Tensor] = {}
        self.capture_calls: list[tuple[int, ...]] = []
        offsets = [0]
        for length in self.prefill_segments:
            offsets.append(offsets[-1] + length)
        self.prefill_cu_seqlens_cpu = torch.tensor(offsets, dtype=torch.long)
        self.prefill_cu_seqlens = self.prefill_cu_seqlens_cpu

    def prior_conv_states(self, _lin: int) -> torch.Tensor | None:
        return self.prior

    def compose_initial_recurrent_state(
        self, _lin: int, dtype: torch.dtype, *, state_v_first: bool
    ) -> torch.Tensor | None:
        assert state_v_first
        return None if self.initial is None else self.initial.to(dtype)

    def capture_token_affines(
        self,
        _lin: int,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        workers=None,
    ) -> None:
        selected = list(range(len(self.prefill_segments))) if workers is None else list(workers)
        assert len(selected) == key.shape[0]
        self.capture_calls.append(tuple(selected))
        for row, worker in enumerate(selected):
            length = self.prefill_segments[worker]
            self.captured[worker] = tuple(
                item[row : row + 1, :length].detach().clone()
                for item in (key, value, alpha, beta)
            )

    def set_conv_states(self, _lin: int, conv: torch.Tensor, workers=None) -> None:
        selected = list(range(len(self.prefill_segments))) if workers is None else list(workers)
        assert len(selected) == conv.shape[0]
        for row, worker in enumerate(selected):
            self.conv[worker] = conv[row].detach().clone()


def _state(workers: int, seed: int = 72) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    conv_dim = 2 * K_HEADS * HEAD_DIM + V_HEADS * HEAD_DIM
    prior = torch.randn(workers, conv_dim, CONV_KERNEL, generator=generator) * 0.05
    initial = torch.randn(workers, V_HEADS, HEAD_DIM, HEAD_DIM, generator=generator) * 0.05
    return prior, initial


def _sequential_reference(layer, x, ar) -> torch.Tensor:
    prior = ar.prior_conv_states(0)
    initial = ar.compose_initial_recurrent_state(0, torch.float32, state_v_first=True)
    rows = []
    offset = 0
    for worker, length in enumerate(ar.prefill_segments):
        rows.append(
            layer._forward_ar_prefill_one(
                x[offset : offset + length], ar, worker, prior, initial
            )
        )
        offset += length
    return torch.cat(rows)


def _assert_side_effects_equal(actual: _AR, expected: _AR) -> None:
    assert actual.captured.keys() == expected.captured.keys()
    assert actual.conv.keys() == expected.conv.keys()
    for worker in actual.captured:
        for got, want in zip(actual.captured[worker], expected.captured[worker]):
            torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(
            actual.conv[worker], expected.conv[worker], rtol=2e-5, atol=2e-6
        )


@pytest.mark.parametrize("has_history", [False, True])
def test_equal_length_prefill_batches_the_old_per_request_reference(
    monkeypatch, has_history: bool
):
    # This is explicitly the optional-dependency fallback configuration.
    monkeypatch.setattr(delta_module, "_fla_chunk", None)
    monkeypatch.setattr(delta_module, "_fla_recurrent", None)
    layer = _layer()
    segments = [1, 1, 1]
    prior, initial = _state(len(segments))
    if not has_history:
        prior = initial = None
    generator = torch.Generator().manual_seed(73)
    x = torch.randn(sum(segments), HIDDEN, generator=generator) * 0.1

    real_recurrent = delta_module._recurrent_delta
    batch_shapes = []

    def counted_recurrent(*args, **kwargs):
        batch_shapes.append(tuple(args[0].shape))
        return real_recurrent(*args, **kwargs)

    monkeypatch.setattr(delta_module, "_recurrent_delta", counted_recurrent)
    batched_ar = _AR(segments, prior, initial)
    actual = layer._forward_ar_prefill(x, batched_ar)
    assert batch_shapes == [(3, 1, V_HEADS, HEAD_DIM)]

    batch_shapes.clear()
    reference_ar = _AR(segments, prior, initial)
    expected = _sequential_reference(layer, x, reference_ar)
    # The old reference deliberately keeps using the chunk dispatcher.
    assert batch_shapes == []

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    _assert_side_effects_equal(batched_ar, reference_ar)


class _CountedLinear:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.full_output_size = inner.full_output_size

    def forward(self, x):
        self.calls += 1
        return self.inner.forward(x)


def test_variable_length_ragged_prefill_matches_per_request_reference(monkeypatch):
    monkeypatch.setattr(delta_module, "_fla_chunk", None)
    monkeypatch.setattr(delta_module, "_AFFINE_CAPTURE_MAX_PADDING_RATIO", 3.0)
    layer = _layer()
    projections = []
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
        counted = _CountedLinear(getattr(layer, name))
        setattr(layer, name, counted)
        projections.append(counted)

    segments = [1, 2, 4]
    prior, initial = _state(len(segments), seed=74)
    generator = torch.Generator().manual_seed(75)
    x = torch.randn(sum(segments), HIDDEN, generator=generator) * 0.1
    batched_ar = _AR(segments, prior, initial)
    actual = layer._forward_ar_prefill(x, batched_ar)
    assert [projection.calls for projection in projections] == [1, 1, 1, 1]
    assert batched_ar.capture_calls == [(0, 1, 2)]

    reference_ar = _AR(segments, prior, initial)
    expected = _sequential_reference(layer, x, reference_ar)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    _assert_side_effects_equal(batched_ar, reference_ar)


@pytest.mark.parametrize(
    ("segments", "expected_capture_calls"),
    [
        ([1, 1, 8], [(0, 1, 2)]),
        ([1, 1, 1, 1, 1, 16], [(0,), (1,), (2,), (3,), (4,), (5,)]),
    ],
)
def test_ragged_affine_capture_dispatch_matches_sequential_reference(
    monkeypatch, segments, expected_capture_calls
):
    monkeypatch.setattr(delta_module, "_fla_chunk", None)
    monkeypatch.setattr(delta_module, "_AFFINE_CAPTURE_MAX_PADDING_RATIO", 3.0)
    layer = _layer(seed=77)
    prior, initial = _state(len(segments), seed=78)
    generator = torch.Generator().manual_seed(79)
    x = torch.randn(sum(segments), HIDDEN, generator=generator) * 0.1

    batched_ar = _AR(segments, prior, initial)
    actual = layer._forward_ar_prefill(x, batched_ar)
    assert batched_ar.capture_calls == expected_capture_calls

    reference_ar = _AR(segments, prior, initial)
    expected = _sequential_reference(layer, x, reference_ar)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    _assert_side_effects_equal(batched_ar, reference_ar)


@pytest.mark.skipif(
    not torch.cuda.is_available() or delta_module._fla_chunk is None,
    reason="CUDA and flash-linear-attention are required",
)
def test_fla_varlen_scan_matches_independent_sequence_reference():
    lengths = [17, 9, 23]
    total = sum(lengths)
    heads, dim = 4, 16
    generator = torch.Generator(device="cuda").manual_seed(76)
    q = torch.randn(1, total, heads, dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = -torch.rand(1, total, heads, device="cuda", dtype=torch.float32, generator=generator)
    beta = torch.rand(1, total, heads, device="cuda", dtype=torch.bfloat16, generator=generator)
    initial = torch.randn(
        len(lengths), heads, dim, dim, device="cuda", dtype=torch.float32, generator=generator
    )
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_cpu = torch.tensor(offsets, dtype=torch.long)
    cu = cu_cpu.cuda()

    actual_out, actual_state = delta_module._chunk_delta(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial,
        state_v_first=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )
    expected_out = []
    expected_state = []
    for worker, (start, end) in enumerate(zip(offsets, offsets[1:])):
        out, state = delta_module._chunk_delta(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            initial_state=initial[worker : worker + 1],
            state_v_first=True,
        )
        expected_out.append(out)
        expected_state.append(state)

    torch.testing.assert_close(actual_out, torch.cat(expected_out, dim=1), rtol=0, atol=0)
    torch.testing.assert_close(actual_state, torch.cat(expected_state, dim=0), rtol=0, atol=0)
