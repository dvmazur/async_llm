"""Independent recurrent GDN VJPs for recorded traces and whole-model diagnostics.

The replacement keeps the production forward, but differentiates a direct recurrent
oracle with ordinary autograd. Query chunks and recurrent checkpoint intervals bound
the oracle's live state graphs. No production GDN derivative code is used.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from itertools import groupby
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from minisgl.training import gdn
from minisgl.training.replay import Replay
from minisgl.training.sft import supervised_loss
from minisgl.training.trace import Trace
from test_training import apply_updates, example_trace, mutation_trace
from test_training_gdn import Float64DeltaMath, sequential_delta, tiny_hybrid
from test_training_logits import torch_reference_kernels as torch_reference_kernels
from torch.utils.checkpoint import checkpoint


@pytest.fixture(autouse=True)
def exact_float32_matmuls(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def history_paths(trace, order):
    """Expand original-token snapshots, independently of the affine DAG/partitions."""
    order = torch.as_tensor(order).cpu().tolist()
    tokens = trace.tokens
    if sorted(order) != list(range(len(tokens))):
        raise ValueError("order must permute every original trace row")
    inverse = {original: packed for packed, original in enumerate(order)}
    blocks, paths = {}, [None] * len(tokens)
    for step, entries in groupby(enumerate(tokens), key=lambda entry: entry[1].step):
        entries = list(entries)
        apply_updates(trace, step, blocks)
        snapshot = {block: tuple(rows) for block, rows in blocks.items()}
        writers = {token.block for _, token in entries}
        pending = []
        for row, token in entries:
            visible = []
            for block, length in token.view:
                length -= (
                    int(block in writers) if token.kind == "decode" else int(block == token.block)
                )
                source = blocks if token.kind == "prefill" and block == token.block else snapshot
                members = source.get(block, ())
                if len(members) != length:
                    raise ValueError("Reference view does not match the pre-update snapshot")
                visible.extend(members)
            paths[inverse[row]] = tuple(inverse[source] for source in (*visible, row))
            if token.kind == "decode":
                pending.append((token.block, row))
            else:
                blocks.setdefault(token.block, []).append(row)
        for block, row in pending:
            blocks.setdefault(block, []).append(row)
    return tuple(paths)


def _advance(state, key, value, alpha, beta, sources):
    for column in range(sources.shape[1]):
        source = sources[:, column]
        active = source >= 0
        source = source.clamp_min(0)
        k, v = key[source], value[source]
        decayed = state * alpha[source, :, None, None]
        error = v - (decayed * k[..., None]).sum(-2)
        updated = decayed + k[..., None] * (beta[source, :, None] * error)[..., None, :]
        state = torch.where(active[:, None, None, None], updated, state)
    return state


def _recurrent_chunk(inputs, paths, rows, checkpoint_steps, math_dtype):
    query, key, value, alpha, beta = (x.to(math_dtype) for x in inputs)
    query, key = (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (query, key))
    selected = [paths[row] for row in rows]
    width = max(map(len, selected))
    sources = torch.tensor(
        [(*path, *([-1] * (width - len(path)))) for path in selected],
        dtype=torch.long,
        device=key.device,
    )
    state = value.new_zeros(len(rows), key.shape[1], key.shape[2], value.shape[2])
    stride = checkpoint_steps or width
    for start in range(0, width, stride):
        section = sources[:, start : start + stride]
        if checkpoint_steps:
            state = checkpoint(
                _advance,
                state,
                key,
                value,
                alpha,
                beta,
                section,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            state = _advance(state, key, value, alpha, beta, section)
    return (state * query[rows, ..., None]).sum(-2) * key.shape[-1] ** -0.5


def recurrent_vjp(
    inputs,
    trace,
    order,
    grad_output,
    *,
    rows=None,
    query_chunk_size=8,
    checkpoint_steps=32,
    math_dtype=None,
    paths=None,
):
    """Return selected packed-row outputs and gradients of all five raw inputs.

    ``grad_output`` has the full [N,H,V] shape even when ``rows`` is a subset.
    A subset computes its contribution to every input gradient; omitted output
    rows must have zero adjoints when using this as a complete backward.
    """
    if query_chunk_size < 1 or checkpoint_steps < 0:
        raise ValueError("Positive query chunk size and nonnegative checkpoint interval required")
    if paths is None:
        paths = history_paths(trace, order)
    if rows is None:
        rows = list(range(len(paths)))
    elif isinstance(rows, torch.Tensor):
        rows = rows.detach().cpu().tolist()
    else:
        rows = list(rows)
    math_dtype = math_dtype or inputs[0].dtype
    leaves = tuple(x.detach().requires_grad_() for x in inputs)
    gradients = [torch.zeros_like(x) for x in leaves]
    outputs = []
    for start in range(0, len(rows), query_chunk_size):
        selected = rows[start : start + query_chunk_size]
        with torch.enable_grad():
            output = _recurrent_chunk(leaves, paths, selected, checkpoint_steps, math_dtype)
            contributions = torch.autograd.grad(
                output,
                leaves,
                grad_outputs=grad_output[selected].to(output.dtype),
            )
        outputs.append(output.detach())
        for total, contribution in zip(gradients, contributions, strict=True):
            total.add_(contribution)
    result = (
        torch.cat(outputs)
        if outputs
        else inputs[2].new_empty((0, *inputs[2].shape[1:]), dtype=math_dtype)
    )
    return result, tuple(gradients)


def _metrics(actual, expected):
    difference = (actual.detach() - expected.detach()).double()
    reference = expected.detach().double()
    return {
        "max_abs": difference.abs().max().item() if difference.numel() else 0.0,
        "norm_relative": (
            difference.norm() / reference.norm().clamp_min(torch.finfo(torch.float64).tiny)
        ).item(),
    }


def assert_parameter_gradients_close(model, expected, *, atol=2e-4, rtol=2e-4):
    """Check current model gradients against detached named reference gradients."""
    parameters = dict(model.named_parameters())
    assert expected, "The reference must include at least one parameter gradient"
    assert expected.keys() <= parameters.keys(), "Unknown reference parameter names"
    missing = sorted(
        name
        for name, parameter in parameters.items()
        if (parameter.grad is not None) != (name in expected)
    )
    assert not missing, missing
    total_error = total_reference = total_actual = dot = 0.0
    maximum = reference_maximum = 0.0
    worst = []
    for name, before in expected.items():
        actual = parameters[name].grad
        assert actual.shape == before.shape, name
        assert torch.isfinite(actual).all() and torch.isfinite(before).all(), name
        error = actual - before
        error_sq = error.double().square().sum().item()
        reference_sq = before.double().square().sum().item()
        total_error += error_sq
        total_reference += reference_sq
        total_actual += actual.double().square().sum().item()
        dot += (actual.double() * before.double()).sum().item()
        absolute = error.abs().max().item() if error.numel() else 0.0
        maximum = max(maximum, absolute)
        reference_maximum = max(reference_maximum, before.abs().max().item())
        worst.append((absolute, name, (error_sq / max(reference_sq, 1e-30)) ** 0.5))
    report = {
        "parameters_with_gradients": len(expected),
        "relative_l2": (total_error / max(total_reference, 1e-30)) ** 0.5,
        "cosine": (
            dot / (total_actual * total_reference) ** 0.5
            if total_actual and total_reference
            else float(total_actual == total_reference)
        ),
        "max_abs": maximum,
        "reference_max_abs": reference_maximum,
        "worst_parameters": sorted(worst, reverse=True)[:10],
    }
    assert report["relative_l2"] < rtol, report
    assert maximum < atol + rtol * reference_maximum, report
    return report


def reference_delta(trace, order, query_chunk_size=8, checkpoint_steps=32, *, atol=None, rtol=None):
    """Make a delta-compatible callable with ordinary-autograd recurrent backward.

    Use ``patch.object(gdn, "delta", replacement)`` around forward AND backward.
    ``replacement.metrics`` receives one output-parity report per backward call.
    Zero-adjoint output rows are skipped; their histories are still available as
    gradient sources. Production forward is invoked under no_grad, including its
    single Q/K normalization; the recurrent VJP differentiates raw inputs directly.
    """
    production_delta = gdn.delta
    paths = history_paths(trace, order)
    reports = []

    class ReferenceDelta(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, value, alpha, beta, replay):
            inputs = (query, key, value, alpha, beta)
            with torch.no_grad():
                output = production_delta(*inputs, replay)
            ctx.save_for_backward(*inputs, output)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            if torch.is_grad_enabled():
                raise RuntimeError("Reference delta supports first-order gradients only")
            *inputs, output = ctx.saved_tensors
            rows = grad_output.ne(0).flatten(1).any(1).nonzero().flatten()
            with torch.autocast(output.device.type, enabled=False):
                expected, gradients = recurrent_vjp(
                    inputs,
                    trace,
                    order,
                    grad_output,
                    rows=rows,
                    query_chunk_size=query_chunk_size,
                    checkpoint_steps=checkpoint_steps,
                    math_dtype=output.dtype,
                    paths=paths,
                )
            report = {
                "queries": len(output),
                "nonzero_adjoint_queries": len(rows),
                "dtype": str(output.dtype),
                **_metrics(output[rows], expected),
            }
            reports.append(report)
            print(f"Independent GDN backward {len(reports)}: {json.dumps(report)}", flush=True)
            double = output.dtype == torch.float64
            torch.testing.assert_close(
                output[rows],
                expected,
                atol=atol if atol is not None else (2e-11 if double else 2e-5),
                rtol=rtol if rtol is not None else (2e-11 if double else 2e-4),
            )
            return (*gradients, None)

    def replacement(query, key, value, alpha, beta, replay):
        return ReferenceDelta.apply(query, key, value, alpha, beta, replay)

    replacement.metrics = reports
    return replacement


@pytest.mark.parametrize("checkpoint_steps", [0, 3])
def test_chunked_recurrence_vjp_matches_existing_sequential_reference(checkpoint_steps):
    torch.manual_seed(17)
    trace = mutation_trace()
    n = trace.num_tokens
    order = torch.randperm(n)
    inputs = (
        torch.randn(n, 2, 3, dtype=torch.float64),
        torch.randn(n, 2, 3, dtype=torch.float64),
        torch.randn(n, 2, 4, dtype=torch.float64),
        torch.rand(n, 2, dtype=torch.float64),
        torch.rand(n, 2, dtype=torch.float64),
    )
    inputs[3][0] = 0
    inputs[4][1] = 1
    inputs = tuple(x.requires_grad_() for x in inputs)
    expected = sequential_delta(*inputs, trace, order)
    weights = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, inputs, weights)
    actual, actual_grads = recurrent_vjp(
        inputs, trace, order, weights, query_chunk_size=3, checkpoint_steps=checkpoint_steps
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("checkpoint_layers", [False, True])
def test_reference_backward_matches_full_hybrid_parameter_gradients(checkpoint_layers):
    trace = example_trace()
    replay = Replay.from_trace(trace, gdn_segment_size=2, gdn_read_mode="factorized")
    model = tiny_hybrid().train()
    if checkpoint_layers:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = torch.full((trace.num_tokens,), -100)
    targets[-4:] = torch.tensor([7, 9, 11, 13])
    parameters = tuple(model.parameters())
    baseline = supervised_loss(model, replay, targets, chunk_size=2)
    baseline_grads = torch.autograd.grad(baseline, parameters)
    replacement = reference_delta(trace, replay.order, query_chunk_size=3, checkpoint_steps=4)
    with (
        patch.object(gdn, "delta", replacement),
        patch.object(
            gdn._Delta, "backward", side_effect=AssertionError("Production backward was invoked")
        ),
    ):
        actual = supervised_loss(model, replay, targets, chunk_size=2)
        actual_grads = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    assert replacement.metrics and all(report["max_abs"] < 2e-6 for report in replacement.metrics)
    for actual_grad, baseline_grad in zip(actual_grads, baseline_grads, strict=True):
        torch.testing.assert_close(actual_grad, baseline_grad, atol=3e-6, rtol=3e-4)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_recorded_demo_core_gradients_against_recurrence(dtype, record_property):
    """Opt-in: real trace topology, and real model head dimensions in the FP32 case."""
    path = os.environ.get("MINISGL_DEMO_BACKWARD_TRACE")
    if not path:
        pytest.skip("Set MINISGL_DEMO_BACKWARD_TRACE to a saved demo training JSON")
    data = json.loads(Path(path).read_text())
    trace = Trace.from_dict(data["examples"][0]["trace"])
    device = os.environ.get("MINISGL_DEMO_BACKWARD_DEVICE", "cpu")
    heads, dk, dv = 2, 7, 5
    if dtype == torch.float32:
        if os.environ.get("MINISGL_DEMO_BACKWARD_FULL_DIMS") != "1":
            pytest.skip("Set MINISGL_DEMO_BACKWARD_FULL_DIMS=1 for the synthetic full-width case")
        if not device.startswith("cuda"):
            pytest.skip(
                "Full model dimensions require explicit MINISGL_DEMO_BACKWARD_DEVICE=cuda:0"
            )
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            os.environ.get("MINISGL_DEMO_BACKWARD_MODEL", data["model"]), local_files_only=True
        ).get_text_config()
        heads = config.linear_num_value_heads
        dk, dv = config.linear_key_head_dim, config.linear_value_head_dim
    torch.manual_seed(19)
    n = trace.num_tokens
    order = Replay.from_trace(trace).order.to(device)
    replay = gdn.GDNReplay.from_trace(
        trace, order, device=device, segment_size=8, read_mode="expanded"
    )
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(n, heads, dk, dtype=dtype, device=device),
            torch.randn(n, heads, dk, dtype=dtype, device=device),
            torch.randn(n, heads, dv, dtype=dtype, device=device),
            0.9 + 0.1 * torch.rand(n, heads, dtype=dtype, device=device),
            0.1 + 0.4 * torch.rand(n, heads, dtype=dtype, device=device),
        )
    )
    weights = torch.randn_like(inputs[2])
    with Float64DeltaMath() if dtype == torch.float64 else nullcontext():
        actual = gdn.delta(*inputs, replay)
        actual_grads = torch.autograd.grad(actual, inputs, weights)
    expected, expected_grads = recurrent_vjp(
        inputs,
        trace,
        order,
        weights,
        query_chunk_size=int(os.environ.get("MINISGL_DEMO_BACKWARD_QUERY_CHUNK", "8")),
        checkpoint_steps=32,
    )
    report = {
        "tokens": n,
        "heads": heads,
        "dk": dk,
        "dv": dv,
        "output": _metrics(actual, expected),
        "gradients": {
            name: _metrics(a, b)
            for name, a, b in zip(
                ("query", "key", "value", "alpha", "beta"),
                actual_grads,
                expected_grads,
                strict=True,
            )
        },
    }
    record_property("recurrent_comparison", json.dumps(report))
    tolerance = 2e-11 if dtype == torch.float64 else 3e-4
    absolute = 2e-11 if dtype == torch.float64 else 2e-5
    torch.testing.assert_close(actual, expected, atol=absolute, rtol=tolerance)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=absolute, rtol=tolerance)


def test_pretrained_demo_parameter_gradients_against_recurrence(record_property):
    """Opt-in full-model parity on pretrained activations and the recorded SFT loss."""
    if os.environ.get("MINISGL_DEMO_BACKWARD_PRETRAINED") != "1":
        pytest.skip("Set MINISGL_DEMO_BACKWARD_PRETRAINED=1 for the pretrained model comparison")
    path = os.environ.get("MINISGL_DEMO_BACKWARD_TRACE")
    if not path:
        pytest.fail("MINISGL_DEMO_BACKWARD_PRETRAINED requires MINISGL_DEMO_BACKWARD_TRACE")
    device = os.environ.get("MINISGL_DEMO_BACKWARD_DEVICE", "cpu")
    if not device.startswith("cuda"):
        pytest.fail("The pretrained comparison requires MINISGL_DEMO_BACKWARD_DEVICE=cuda:0")
    from transformers import AutoConfig, AutoModelForCausalLM

    data = json.loads(Path(path).read_text())
    example = data["examples"][0]
    trace = Trace.from_dict(example["trace"])
    targets = torch.tensor(example["targets"])
    count = int((targets != -100).sum())
    assert count > 0
    model_path = os.environ.get("MINISGL_DEMO_BACKWARD_MODEL", data["model"])
    config = AutoConfig.from_pretrained(model_path, local_files_only=True).get_text_config()
    expected_layers = config.layer_types.count("linear_attention")
    assert expected_layers > 0, "The pretrained model must contain GDN layers"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, dtype=torch.float32, local_files_only=True
    ).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    replay = Replay.from_trace(trace)
    baseline_loss = supervised_loss(model, replay, targets) / count
    baseline_loss.backward()
    expected = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    replacement = reference_delta(
        trace,
        replay.order,
        query_chunk_size=int(os.environ.get("MINISGL_DEMO_BACKWARD_QUERY_CHUNK", "8")),
        checkpoint_steps=32,
    )
    with (
        patch.object(gdn, "delta", replacement),
        patch.object(
            gdn._Delta, "backward", side_effect=AssertionError("Production backward was invoked")
        ),
    ):
        actual_loss = supervised_loss(model, replay, targets) / count
        actual_loss.backward()
    torch.testing.assert_close(actual_loss, baseline_loss, atol=0, rtol=0)
    assert len(replacement.metrics) == expected_layers
    report = assert_parameter_gradients_close(model, expected)
    record_property(
        "pretrained_recurrent_comparison",
        json.dumps(
            {
                "model": model_path,
                "tokens": trace.num_tokens,
                "targets": count,
                "loss": actual_loss.item(),
                "parameter_gradients": report,
                "layers": replacement.metrics,
            }
        ),
    )
