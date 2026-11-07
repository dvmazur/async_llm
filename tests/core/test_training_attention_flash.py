"""FA4/CuTe replay errors against FP32 and low-precision Torch attention."""

import math
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from minisgl.training import Trace, replay_trace
from minisgl.training.attention import attention
from minisgl.training.attention.torch import _attention_impl
from minisgl.training.recording import _TraceRecorder
from minisgl.training.trace import ForwardBatch, ForwardRequest
from test_training import Session, causal_trace, mutation_trace, repeated_view_trace
from test_training_images import cache_image_trace


def historical_trace():
    session = _TraceRecorder(Session())
    a, b = session.create_block(), session.create_block()
    session.prefill_batch(
        [NS(block=a, input_ids=torch.tensor([1, 2]), context=[], pixel_values=None)]
    )
    for block, view, token in (
        (b, [a, b], 3),
        (a, [a], 4),
        (b, [a, b], 5),
        (b, [a, b], 6),
    ):
        session.decode_step([NS(output_block=block, cache_view=view)], torch.tensor([token]))
    return session.trace


def singleton_parts_trace():
    session = _TraceRecorder(Session())
    a, b = session.create_block(), session.create_block()
    session.prefill_batch([NS(block=a, input_ids=torch.tensor([1]), context=[], pixel_values=None)])
    session.decode_step([NS(output_block=b, cache_view=[a, b])], torch.tensor([2]))
    return session.trace


def interleaved_blocks_trace():
    """Record a1 a2, b1 b2, a3, b3; the first two B queries cannot see a3."""
    session = _TraceRecorder(Session())
    a, b = session.create_block(), session.create_block()
    session.prefill_batch(
        [NS(block=a, input_ids=torch.tensor([1, 2]), context=[], pixel_values=None)]
    )
    session.prefill_batch(
        [NS(block=b, input_ids=torch.tensor([3, 4]), context=[a], pixel_values=None)]
    )
    session.decode_step([NS(output_block=a, cache_view=[a])], torch.tensor([5]))
    session.decode_step([NS(output_block=b, cache_view=[a, b])], torch.tensor([6]))
    return session.trace


def tile_boundary_trace():
    """Unequal varlen sequences, repeated reads, historical prefixes and implicit self."""
    session = _TraceRecorder(Session())
    a, b, c = [session.create_block() for _ in range(3)]

    def prefill(block, count, context=()):
        session.prefill_batch(
            [
                NS(
                    block=block,
                    input_ids=torch.arange(count) % 32,
                    context=context,
                    pixel_values=None,
                )
            ]
        )

    prefill(a, 137)
    prefill(b, 11, [a, a])
    prefill(c, 3, [b])
    prefill(a, 17)
    prefill(b, 131, [a, c, a])
    prefill(c, 2, [b])
    # The implicit self read uses a nonzero lower bound inside C's packed keys.
    session.decode_step([NS(output_block=c, cache_view=[a, b])], torch.tensor([7]))
    session.decode_step([NS(output_block=b, cache_view=[b, a, b])], torch.tensor([8]))
    return session.trace


def _position(value):
    return tuple(value) if isinstance(value, list) else value


def _expected_edges(replay):
    edges = Counter()
    for query, reads in enumerate(replay.reads):
        for version, left, right, query_position in reads:
            positions = iter(range(left, right))
            spans = replay.compute_source_spans(
                version, left, right, with_rope=replay.rope_positions is not None
            )
            for span in spans:
                start, end = span[:2]
                for source in range(start, end):
                    key_position = next(positions)
                    if replay.rope_positions is not None:
                        key_position = tuple((replay.rope_positions[source] + span[3]).tolist())
                    edges[(query, source, query_position, key_position)] += 1
    return edges


def _plan_edges(plan):
    edges = Counter()
    queries, keys = plan.q_indices.tolist(), plan.k_indices.tolist()
    query_positions, key_positions = plan.q_positions.tolist(), plan.k_positions.tolist()
    starts, ends = plan.q_starts.tolist(), plan.q_ends.tolist()
    cu_q, cu_k = plan.cu_q.tolist(), plan.cu_k.tolist()
    assert plan.q_starts.dtype == plan.q_ends.dtype == torch.int32
    assert len(starts) == len(ends) == len(queries)
    assert len(cu_q) == len(cu_k)
    assert cu_q[0] == cu_k[0] == 0
    assert cu_q[-1] == len(queries) and cu_k[-1] == len(keys)
    assert plan.max_q == max(right - left for left, right in zip(cu_q, cu_q[1:]))
    assert plan.max_k == max(right - left for left, right in zip(cu_k, cu_k[1:]))
    for qleft, qright, kleft, kright in zip(cu_q, cu_q[1:], cu_k, cu_k[1:]):
        for row in range(qleft, qright):
            assert 0 <= starts[row] < ends[row] <= kright - kleft
            for column, source in enumerate(keys[kleft:kright]):
                if not starts[row] <= column < ends[row]:
                    continue
                edges[
                    (
                        queries[row],
                        source,
                        _position(query_positions[row]),
                        _position(key_positions[kleft + column]),
                    )
                ] += 1
    assert sorted(plan.merge_indices.tolist()) == list(range(len(queries)))
    pointers = plan.merge_indptr.tolist()
    assert len(pointers) == plan.num_queries + 1
    for row, (left, right) in enumerate(zip(pointers, pointers[1:])):
        assert right > left
        assert all(queries[part] == row for part in plan.merge_indices[left:right].tolist())
    return edges


@pytest.mark.parametrize(
    "make_trace",
    [
        historical_trace,
        interleaved_blocks_trace,
        mutation_trace,
        repeated_view_trace,
        cache_image_trace,
        tile_boundary_trace,
    ],
)
def test_varlen_plan_preserves_visibility_positions_and_multiplicity(make_trace):
    from minisgl.training.attention.fa4.plan import build_flash_attention_plan

    replay = make_trace().to()
    plan = build_flash_attention_plan(replay, torch.device("cpu"))
    assert plan.num_queries == replay.num_tokens
    versions = {version for reads in replay.reads for version, *_ in reads}
    assert len(plan.cu_k) == len(versions) + 1
    assert _plan_edges(plan) == _expected_edges(replay)


def test_historical_prefix_is_not_replaced_by_final_block_length():
    from minisgl.training.attention.fa4.plan import build_flash_attention_plan

    replay = historical_trace().to()
    edges = _plan_edges(build_flash_attention_plan(replay, torch.device("cpu")))
    visible = torch.zeros(6, 6, dtype=torch.bool)
    for query, source, _, _ in edges:
        visible[query, source] = True
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(visible, expected)


def test_interleaved_blocks_share_keys_without_exposing_future_tokens():
    from minisgl.training.attention.fa4.plan import build_flash_attention_plan

    replay = interleaved_blocks_trace().to()
    plan = build_flash_attention_plan(replay, torch.device("cpu"))
    # Packed order is a1 a2 a3 b1 b2 b3, while recording interleaves A and B.
    assert replay.order.tolist() == [0, 1, 4, 2, 3, 5]
    assert plan.cu_k.tolist() == [0, 3, 6]
    assert plan.cu_q.tolist() == [0, 6, 9]
    assert plan.k_indices.tolist() == list(range(6))
    visible = torch.zeros(6, 6, dtype=torch.bool)
    for query, source, _, _ in _plan_edges(plan):
        visible[query, source] = True
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(visible, expected)


def test_long_decode_packs_one_key_per_token_instead_of_all_prefixes():
    from minisgl.training.attention.fa4.plan import build_flash_attention_plan

    count = 4096
    replay = Trace(
        batches=[
            ForwardBatch(
                step,
                "decode",
                (ForwardRequest((step % 32,), 0, step, ((0, step),)),),
            )
            for step in range(count)
        ]
    ).to()
    plan = build_flash_attention_plan(replay, torch.device("cpu"))
    assert plan.cu_q.tolist() == plan.cu_k.tolist() == [0, count]
    assert plan.k_indices.tolist() == plan.q_indices.tolist() == list(range(count))
    assert plan.q_starts.tolist() == [0] * count
    assert plan.q_ends.tolist() == list(range(1, count + 1))
    assert plan.merge_indices.tolist() == list(range(count))
    assert plan.merge_indptr.tolist() == list(range(count + 1))


def test_attention_backend_option_is_explicit_and_preserves_default():
    original = causal_trace(3).to()
    assert original.attention_backend == "torch"
    configured = original.to(attention_backend="fa4")
    assert configured.attention_backend == "fa4"
    assert original.attention_backend == "torch"
    assert configured.batches == original.batches
    torch.testing.assert_close(configured.input_ids, original.input_ids)
    with pytest.raises(ValueError, match="attention_backend"):
        original.to(attention_backend="unknown")


@pytest.fixture
def hopper():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA4 replay requires a Hopper GPU")
    pytest.importorskip(
        "flash_attn.cute", reason="Install the optional FA4 dependencies from the training README"
    )
    return torch.device("cuda")


def _inputs(count, heads, kv_heads, width, dtype, device):
    torch.manual_seed(37)
    return tuple(
        torch.randn(count, number, width, device=device, dtype=dtype).requires_grad_()
        for number in (heads, kv_heads, kv_heads)
    )


def _assert_flash_error(actual, reference, baseline, *, name):
    """Compare isolated operator errors using the upstream FA4 criterion."""
    assert actual.shape == reference.shape == baseline.shape, name
    assert actual.dtype == reference.dtype == baseline.dtype, name
    for label, tensor in (("FA4", actual), ("reference", reference), ("baseline", baseline)):
        assert torch.isfinite(tensor).all(), f"{name}: {label} contains nonfinite values"
    # Keep reference arithmetic in its original dtype, as in the upstream test.
    atol = 2 * (reference + 0.3 - 0.3 - reference).abs().max().item()
    error = (actual - reference).abs().max().item()
    baseline_error = (baseline - reference).abs().max().item()
    assert all(
        math.isfinite(value) for value in (atol, error, baseline_error)
    ), f"{name}: error calculation contains nonfinite values"
    limit = 2 * baseline_error + atol
    # Official FA4: max_error_FA <= 2 * max_error_low_precision_torch + roundoff.
    # For output/dQ/dK/dV, both errors use the FP32 reference with identical Q/K/V and dO.
    # https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/testing.py#L495-L503
    assert error <= limit, (
        f"{name}: FA4 max error {error} exceeds 2 * low-precision Torch error "
        f"{baseline_error} + roundoff {atol} = {limit}"
    )


def test_flash_error_criterion_accepts_boundary_and_rejects_excess():
    reference = torch.zeros(2)
    baseline = torch.tensor([0.125, 0.0])
    actual = torch.tensor([0.25, 0.0])
    _assert_flash_error(actual, reference, baseline, name="boundary")
    with pytest.raises(AssertionError, match="exceeds"):
        _assert_flash_error(actual + 0.125, reference, baseline, name="excess")


def test_flash_error_criterion_rejects_nonfinite_values_and_errors():
    for path in ("FA4", "reference", "baseline"):
        tensors = {label: torch.zeros(2) for label in ("FA4", "reference", "baseline")}
        tensors[path][0] = float("nan")
        with pytest.raises(AssertionError, match=f"{path} contains nonfinite"):
            _assert_flash_error(
                tensors["FA4"], tensors["reference"], tensors["baseline"], name="nonfinite"
            )
    # Finite operands can still overflow subtraction, making inf <= inf pass.
    reference = torch.tensor([65504.0], dtype=torch.float16)
    baseline = actual = -reference
    with pytest.raises(AssertionError, match="error calculation contains nonfinite"):
        _assert_flash_error(actual, reference, baseline, name="overflow")


def _assert_forward_backward(inputs, replay, rotary_width, section=None):
    """Compare all three operators with identical inputs and one upstream gradient."""
    from minisgl.training.attention.fa4 import flash_attention

    inv_freq = 10000 ** (
        -torch.arange(0, rotary_width, 2, device=inputs[0].device).float() / rotary_width
    )
    expected_inputs = tuple(x.detach().clone().requires_grad_() for x in inputs)
    baseline_inputs = tuple(x.detach().clone().requires_grad_() for x in inputs)
    expected = attention(*expected_inputs, replay, inv_freq, inputs[0].shape[-1] ** -0.5, section)
    baseline = _attention_impl(
        *baseline_inputs, replay, inv_freq, inputs[0].shape[-1] ** -0.5, section, upcast=False
    )
    actual = flash_attention(*inputs, replay, inv_freq, inputs[0].shape[-1] ** -0.5, section)
    _assert_flash_error(actual, expected, baseline, name="output")
    grad_output = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, expected_inputs, grad_output)
    baseline_grads = torch.autograd.grad(baseline, baseline_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, inputs, grad_output)
    for name, observed, reference, low_precision in zip(
        ("dQ", "dK", "dV"), actual_grads, expected_grads, baseline_grads
    ):
        _assert_flash_error(observed, reference, low_precision, name=name)
    return actual_grads


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_historical_visibility_forward_backward(hopper, dtype):
    replay = historical_trace().to(hopper, attention_checkpointing=False)
    _assert_forward_backward(_inputs(6, 4, 2, 64, dtype, hopper), replay, 64)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("make_trace", [interleaved_blocks_trace, tile_boundary_trace])
def test_flash_masked_trajectory_forward_backward(hopper, dtype, make_trace):
    replay = make_trace().to(hopper, attention_checkpointing=False, query_chunk_size=64)
    inputs = _inputs(replay.num_tokens, 4, 2, 64, dtype, hopper)
    _assert_forward_backward(inputs, replay, 32)


@pytest.mark.parametrize(
    "make_trace,heads,kv_heads,width,rotary_width,section",
    [
        (lambda: causal_trace(33), 4, 2, 128, 128, None),
        (mutation_trace, 8, 2, 256, 64, None),
        (lambda: causal_trace(17), 24, 4, 256, 64, None),
        (repeated_view_trace, 4, 1, 128, 32, None),
        (cache_image_trace, 4, 2, 64, 24, (4, 4, 4)),
    ],
    ids=[
        "causal-full-rope",
        "copies-clears-partial-rope",
        "gqa-six-partial-rope",
        "repeated-and-implicit-self",
        "mrope",
    ],
)
def test_flash_forward_backward(hopper, make_trace, heads, kv_heads, width, rotary_width, section):
    replay = make_trace().to(hopper, attention_checkpointing=False)
    inputs = _inputs(replay.num_tokens, heads, kv_heads, width, torch.bfloat16, hopper)
    _assert_forward_backward(inputs, replay, rotary_width, section)


def test_flash_singleton_parts_backpropagate_normalization(hopper):
    replay = singleton_parts_trace().to(hopper, attention_checkpointing=False)
    inputs = _inputs(2, 4, 2, 64, torch.bfloat16, hopper)
    grads = _assert_forward_backward(inputs, replay, 64)
    # Each local one-key output is independent of Q/K. All these derivatives
    # require the LSE path through the merge and the FA kernel's backward.
    assert grads[0][1].abs().sum() > 0
    assert grads[1].abs().sum() > 0


def test_flash_no_grad_and_attention_checkpointing(hopper):
    from minisgl.training.attention.fa4 import flash_attention

    replay = historical_trace().to(hopper, attention_checkpointing=False)
    inputs = _inputs(6, 4, 2, 64, torch.bfloat16, hopper)
    inv_freq = 10000 ** (-torch.arange(0, 64, 2, device=hopper).float() / 64)
    plain = flash_attention(*inputs, replay, inv_freq, 0.125)
    gradient = torch.randn_like(plain)
    expected_grads = torch.autograd.grad(plain, inputs, gradient)
    checkpointed = replay.to(attention_checkpointing=True)
    actual = flash_attention(*inputs, checkpointed, inv_freq, 0.125)
    actual_grads = torch.autograd.grad(actual, inputs, gradient)
    torch.testing.assert_close(actual, plain, atol=0, rtol=0)
    for observed, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(observed, expected, atol=1e-3, rtol=1e-2)
    with torch.no_grad():
        actual = flash_attention(*inputs, checkpointed, inv_freq, 0.125)
        expected = attention(*inputs, replay, inv_freq, 0.125)
        baseline = _attention_impl(*inputs, replay, inv_freq, 0.125, upcast=False)
    assert not actual.requires_grad
    _assert_flash_error(actual, expected, baseline, name="no-grad output")


@pytest.mark.parametrize("cache_plans", [False, True])
@pytest.mark.parametrize("architecture", ["qwen3", "qwen3_5"])
def test_supported_adapter_selects_flash_and_backpropagates(hopper, cache_plans, architecture):
    from transformers import (
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    torch.manual_seed(103)
    config_type, model_type = (
        (Qwen3Config, Qwen3ForCausalLM)
        if architecture == "qwen3"
        else (Qwen3_5TextConfig, Qwen3_5ForCausalLM)
    )
    extra = (
        {}
        if architecture == "qwen3"
        else {
            "layer_types": ["full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [3, 3, 2],
            },
        }
    )
    config = config_type(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        **extra,
    )
    config._attn_implementation = "eager"
    model = model_type(config).to(device=hopper, dtype=torch.bfloat16).eval()
    reference = deepcopy(model)
    replay = historical_trace().to(
        hopper,
        attention_backend="fa4",
        attention_checkpointing=False,
        cache_plans=cache_plans,
    )
    expected = replay_trace(reference, replay.to(attention_backend="torch"), hidden_only=False)
    actual = replay_trace(model, replay, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=3e-2)
    assert bool(replay._attention) is cache_plans
    targets = torch.arange(replay.num_tokens, device=hopper) % config.vocab_size
    expected_loss = F.cross_entropy(expected.float(), targets)
    actual_loss = F.cross_entropy(actual.float(), targets)
    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-3, rtol=1e-3)
    expected_grads = torch.autograd.grad(expected_loss, tuple(reference.parameters()))
    actual_grads = torch.autograd.grad(actual_loss, tuple(model.parameters()))
    for (name, _), observed, wanted in zip(model.named_parameters(), actual_grads, expected_grads):
        assert torch.isfinite(observed).all(), name
        torch.testing.assert_close(observed, wanted, atol=5e-3, rtol=5e-2, msg=name)
    if cache_plans:
        key, plan = next(iter(replay._attention.items()))
        assert key[0] == "fa4"
        with torch.no_grad():
            repeated = replay_trace(model, replay, hidden_only=False)
        assert replay._attention[key] is plan
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
