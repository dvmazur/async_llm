"""Replay backend selection preserves reference and explicit-device semantics."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from minisgl.training import Trace
from minisgl.training.attention import adapter
from minisgl.training.attention.fa4 import backend as fa4_backend
from minisgl.training.trace import ForwardBatch, ForwardRequest


@pytest.fixture(autouse=True)
def clear_optional_backend_probe(monkeypatch):
    monkeypatch.delenv("FLASH_ATTENTION_HD256_GENERIC_BWD", raising=False)
    monkeypatch.setattr(fa4_backend, "_flash_device_unsupported_reason", lambda *args: None)
    adapter._load_optional_flash_attention.cache_clear()
    yield
    adapter._load_optional_flash_attention.cache_clear()


@pytest.mark.parametrize(
    "device,capability,dtype,head_size,expected",
    [
        ("cpu", None, torch.float32, 64, "reference"),
        ("cpu", None, torch.bfloat16, 256, "reference"),
        ("cuda:2", (7, 5), torch.float16, 64, "optimized"),
        ("cuda:2", (8, 0), torch.bfloat16, 256, "optimized"),
        ("cuda:2", (8, 9), torch.float16, 128, "optimized"),
        ("cuda:2", (9, 0), torch.bfloat16, 256, "fa4"),
        ("cuda:2", (9, 0), torch.float16, 8, "fa4"),
        ("cuda:2", (9, 0), torch.float16, 192, "fa4"),
        ("cuda:2", (9, 0), torch.float32, 256, "optimized"),
        ("cuda:2", (9, 0), torch.float64, 256, "optimized"),
        ("cuda:2", (9, 0), torch.float16, 7, "optimized"),
        ("cuda:2", (9, 0), torch.float16, 10, "optimized"),
        ("cuda:2", (9, 0), torch.float16, 264, "optimized"),
        ("cuda:2", (10, 0), torch.bfloat16, 256, "optimized"),
        ("cuda:2", (11, 0), torch.bfloat16, 64, "optimized"),
        ("cuda:2", (12, 0), torch.bfloat16, 256, "optimized"),
    ],
)
def test_auto_backend_respects_hardware_dtype_and_head_dimension(
    monkeypatch, device, capability, dtype, head_size, expected
):
    query = SimpleNamespace(device=torch.device(device), dtype=dtype, shape=(1, 8, 3, head_size))
    hardware = Mock(return_value=capability)
    loader = Mock(return_value=object())
    monkeypatch.setattr(torch.cuda, "get_device_capability", hardware)
    monkeypatch.setattr(fa4_backend, "_flash_attention_function", loader)
    implementation = adapter._select_attention_backend("auto", query, query, query)
    assert (
        implementation
        is {
            "reference": adapter.attention,
            "optimized": adapter.optimized_attention,
            "fa4": fa4_backend.flash_attention,
        }[expected]
    )
    if hardware.called:
        hardware.assert_called_once_with(query.device)
    if device == "cpu":
        hardware.assert_not_called()
    assert loader.call_count == int(expected == "fa4")


@pytest.mark.parametrize(
    "head_size,generic_backward,expected",
    [
        (8, None, "fa4"),
        (128, None, "fa4"),
        (136, None, "optimized"),
        (248, "1", "optimized"),
        (256, None, "optimized"),
        (256, "0", "optimized"),
        (256, "true", "optimized"),
        (256, "1", "fa4"),
        (264, "1", "optimized"),
    ],
)
def test_b300_auto_respects_head_dimensions_and_lse_backward(
    monkeypatch, head_size, generic_backward, expected
):
    if generic_backward is not None:
        monkeypatch.setenv("FLASH_ATTENTION_HD256_GENERIC_BWD", generic_backward)
    query = SimpleNamespace(
        device=torch.device("cuda:2"), dtype=torch.bfloat16, shape=(1, 8, 3, head_size)
    )
    hardware = Mock(return_value=(10, 3))
    loader = Mock(return_value=object())
    monkeypatch.setattr(torch.cuda, "get_device_capability", hardware)
    monkeypatch.setattr(fa4_backend, "_flash_attention_function", loader)
    implementation = adapter._select_attention_backend("auto", query, query, query)
    assert (
        implementation
        is {
            "fa4": fa4_backend.flash_attention,
            "optimized": adapter.optimized_attention,
        }[expected]
    )
    hardware.assert_called_once_with(query.device)
    assert loader.call_count == int(expected == "fa4")
    assert os.environ.get("FLASH_ATTENTION_HD256_GENERIC_BWD") == generic_backward


@pytest.mark.parametrize(
    "head_size,reason",
    [
        (136, "head dimension between 8 and 128, or 256"),
        (256, "FLASH_ATTENTION_HD256_GENERIC_BWD=1.*LSE gradients"),
    ],
)
def test_explicit_b300_rejects_unsupported_configurations_before_execution(
    monkeypatch, head_size, reason
):
    query = SimpleNamespace(
        ndim=3, device=torch.device("cuda:2"), dtype=torch.bfloat16, shape=(3, 8, head_size)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    with pytest.raises(ValueError, match=reason):
        fa4_backend.flash_attention(query, query, query, None, None, 1.0)
    assert "FLASH_ATTENTION_HD256_GENERIC_BWD" not in os.environ


def test_b300_backward_requirement_is_not_cached(monkeypatch):
    query = SimpleNamespace(
        device=torch.device("cuda:2"), dtype=torch.bfloat16, shape=(1, 8, 3, 256)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    monkeypatch.setattr(fa4_backend, "_flash_attention_function", Mock(return_value=object()))
    assert (
        adapter._select_attention_backend("auto", query, query, query)
        is adapter.optimized_attention
    )
    monkeypatch.setenv("FLASH_ATTENTION_HD256_GENERIC_BWD", "1")
    assert (
        adapter._select_attention_backend("auto", query, query, query)
        is fa4_backend.flash_attention
    )
    monkeypatch.delenv("FLASH_ATTENTION_HD256_GENERIC_BWD")
    assert (
        adapter._select_attention_backend("auto", query, query, query)
        is adapter.optimized_attention
    )


@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.bfloat16, torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.float16, torch.bfloat16, torch.float16),
    ],
)
def test_auto_uses_torch_for_mixed_input_dtypes(monkeypatch, dtypes):
    inputs = [
        SimpleNamespace(device=torch.device("cuda"), dtype=dtype, shape=(1, 8, 3, 64))
        for dtype in dtypes
    ]
    loader = Mock(side_effect=AssertionError("FA4 requires matching Q/K/V dtypes"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(fa4_backend, "_flash_attention_function", loader)
    assert adapter._select_attention_backend("auto", *inputs) is adapter.optimized_attention
    loader.assert_not_called()


@pytest.mark.parametrize("name", ["torch", "torch_optimized", "fa4"])
def test_explicit_backend_does_not_probe_or_change_selection(monkeypatch, name):
    forbidden = Mock(side_effect=AssertionError("Explicit selection must not probe availability"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    monkeypatch.setattr(fa4_backend, "_flash_attention_function", forbidden)
    assert (
        adapter._select_attention_backend(name, None, None, None)
        is {
            "torch": adapter.attention,
            "torch_optimized": adapter.optimized_attention,
            "fa4": fa4_backend.flash_attention,
        }[name]
    )
    forbidden.assert_not_called()


def test_auto_caches_missing_optional_dependencies_without_hiding_explicit_errors(monkeypatch):
    query = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16, shape=(1, 8, 3, 256))
    loader = Mock(side_effect=ImportError("FA4 dependencies unavailable"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(fa4_backend, "_flash_attention_function", loader)
    for _ in range(2):
        assert (
            adapter._select_attention_backend("auto", query, query, query)
            is adapter.optimized_attention
        )
    loader.assert_called_once()
    implementation = adapter._select_attention_backend("fa4", query, query, query)
    assert implementation is fa4_backend.flash_attention
    with pytest.raises(ImportError, match="FA4 dependencies unavailable"):
        fa4_backend._forward(None, None, None, None, None, 1.0, None)


def test_auto_does_not_swallow_optional_backend_runtime_errors(monkeypatch):
    query = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16, shape=(1, 8, 3, 256))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(
        fa4_backend,
        "_flash_attention_function",
        Mock(side_effect=RuntimeError("broken kernel setup")),
    )
    with pytest.raises(RuntimeError, match="broken kernel setup"):
        adapter._select_attention_backend("auto", query, query, query)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown replay attention backend"):
        adapter._select_attention_backend("unknown", None, None, None)


def test_cpu_auto_preserves_reference_output_and_gradients(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", Mock(side_effect=AssertionError("CPU must stay eager"))
    )
    trace = Trace(
        batches=[ForwardBatch(0, "prefill", (ForwardRequest((1, 2, 3), 0, 0, ((0, 0),)),))],
        steps=1,
    ).to(query_chunk_size=2, attention_checkpointing=False)
    assert trace.attention_backend == "auto"
    torch.manual_seed(7)
    inputs = [torch.randn(1, heads, 3, 8) for heads in (4, 1, 1)]
    inv_freq = torch.tensor([1.0, 0.01])
    results = []
    for backend in ("torch", "auto"):
        q, k, v = (x.clone().requires_grad_() for x in inputs)
        output, weights = adapter.replay_attention(
            None,
            q,
            k,
            v,
            None,
            8**-0.5,
            replay=trace.to(attention_backend=backend),
            inv_freq=inv_freq,
        )
        assert weights is None
        output.square().sum().backward()
        results.append((output.detach(), q.grad, k.grad, v.grad))
    for actual, reference in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
