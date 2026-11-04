from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from minisgl.training import Trace, gdn
from minisgl_examples.training.loss import compute_supervised_loss

from tests.training.fixtures import torch_reference_kernels as torch_reference_kernels
from tests.training.models import tiny_hybrid
from tests.training.references import Float64DeltaMath
from tests.training.test_demo_training_backward import (
    _metrics,
    assert_parameter_gradients_close,
    recurrent_vjp,
    reference_delta,
)
from tests.training.workloads import example_trace


@pytest.mark.parametrize("checkpoint_layers", [False, True])
def test_reference_backward_matches_full_hybrid_parameter_gradients(checkpoint_layers):
    trace = example_trace()
    replay = trace.to(gdn_segment_size=2, gdn_read_mode="factorized")
    model = tiny_hybrid().cuda().train()
    if checkpoint_layers:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = torch.full((trace.num_tokens,), -100, device="cuda")
    targets[-4:] = torch.tensor([7, 9, 11, 13], device="cuda")
    parameters = tuple(model.parameters())
    baseline = compute_supervised_loss(model, replay, targets, chunk_size=2)
    baseline_grads = torch.autograd.grad(baseline, parameters)
    replacement = reference_delta(trace, replay.order, query_chunk_size=3, checkpoint_steps=4)
    with (
        patch.object(gdn, "delta", replacement),
        patch.object(
            gdn._Delta, "backward", side_effect=AssertionError("Production backward was invoked")
        ),
    ):
        actual = compute_supervised_loss(model, replay, targets, chunk_size=2)
        actual_grads = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    assert replacement.metrics and all(
        (report["max_abs"] < 2e-06 for report in replacement.metrics)
    )
    for actual_grad, baseline_grad in zip(actual_grads, baseline_grads, strict=True):
        torch.testing.assert_close(actual_grad, baseline_grad, atol=3e-06, rtol=0.0003)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_recorded_demo_core_gradients_against_recurrence(dtype, record_property):
    """Opt-in: real trace topology, and real model head dimensions in the FP32 case."""
    path = os.environ.get("MINISGL_DEMO_BACKWARD_TRACE")
    if not path:
        pytest.skip("Set MINISGL_DEMO_BACKWARD_TRACE to a saved demo training JSON")
    data = json.loads(Path(path).read_text())
    trace = Trace.from_dict(data["examples"][0]["trace"])
    device = os.environ.get("MINISGL_DEMO_BACKWARD_DEVICE", "cuda")
    if not device.startswith("cuda"):
        pytest.fail("This GPU comparison requires a CUDA device")
    heads, dk, dv = (2, 7, 5)
    if dtype == torch.float32:
        if os.environ.get("MINISGL_DEMO_BACKWARD_FULL_DIMS") != "1":
            pytest.skip("Set MINISGL_DEMO_BACKWARD_FULL_DIMS=1 for the synthetic full-width case")
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            os.environ.get("MINISGL_DEMO_BACKWARD_MODEL", data["model"]), local_files_only=True
        ).get_text_config()
        heads = config.linear_num_value_heads
        dk, dv = (config.linear_key_head_dim, config.linear_value_head_dim)
    torch.manual_seed(19)
    n = trace.num_tokens
    order = trace.to().order.to(device)
    replay = gdn.GDNReplay.from_trace(
        trace, order, device=device, segment_size=8, read_mode="expanded"
    )
    inputs = tuple(
        (
            x.requires_grad_()
            for x in (
                torch.randn(n, heads, dk, dtype=dtype, device=device),
                torch.randn(n, heads, dk, dtype=dtype, device=device),
                torch.randn(n, heads, dv, dtype=dtype, device=device),
                0.9 + 0.1 * torch.rand(n, heads, dtype=dtype, device=device),
                0.1 + 0.4 * torch.rand(n, heads, dtype=dtype, device=device),
            )
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
    tolerance = 2e-11 if dtype == torch.float64 else 0.0003
    absolute = 2e-11 if dtype == torch.float64 else 2e-05
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
    device = os.environ.get("MINISGL_DEMO_BACKWARD_DEVICE", "cuda")
    if not device.startswith("cuda"):
        pytest.fail("The pretrained comparison requires a CUDA device")
    from transformers import AutoConfig, AutoModelForCausalLM

    data = json.loads(Path(path).read_text())
    example = data["examples"][0]
    trace = Trace.from_dict(example["trace"])
    targets = torch.tensor(example["targets"], device="cuda")
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
    replay = trace.to()
    baseline_loss = compute_supervised_loss(model, replay, targets) / count
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
        actual_loss = compute_supervised_loss(model, replay, targets) / count
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
