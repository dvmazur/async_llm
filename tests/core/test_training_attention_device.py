"""FA4's process-wide architecture cache must not override the input device."""

import sys
from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from minisgl.training.attention import adapter
from minisgl.training.attention.fa4 import backend


class TensorSpec(NS):
    def __len__(self):
        return self.shape[0]


def query(device):
    return TensorSpec(ndim=3, shape=(2, 4, 64), dtype=torch.bfloat16, device=torch.device(device))


@pytest.fixture
def cuda_architectures(monkeypatch):
    state = NS(current=0, capabilities=((9, 0), (10, 3)), probes=[])

    @contextmanager
    def device_context(device):
        previous = state.current
        state.current = torch.device(device).index
        try:
            yield
        finally:
            state.current = previous

    def capability(device=None):
        index = state.current if device is None else torch.device(device).index
        return state.capabilities[index]

    @lru_cache(maxsize=None)
    def get_device_arch():
        state.probes.append(state.current)
        major, minor = capability()
        return major * 10 + minor

    state.get_device_arch = get_device_arch
    state.interface = NS(_get_device_arch=get_device_arch)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", state.interface)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(backend, "_flash_attention_function", Mock(return_value=Mock()))
    adapter._load_optional_flash_attention.cache_clear()
    yield state
    adapter._load_optional_flash_attention.cache_clear()


def test_auto_initializes_fa4_on_the_input_device(cuda_architectures):
    q = query("cuda:1")
    assert adapter._select_attention_backend("auto", q, q, q) is backend.flash_attention
    assert cuda_architectures.get_device_arch() == 103
    assert cuda_architectures.probes == [1]
    assert cuda_architectures.current == 0


@pytest.mark.parametrize("first", [0, 1])
def test_other_architecture_falls_back_without_invalidating_live_graphs(cuda_architectures, first):
    original, other = query(f"cuda:{first}"), query(f"cuda:{1 - first}")
    assert (
        adapter._select_attention_backend("auto", original, original, original)
        is backend.flash_attention
    )
    selected_arch = cuda_architectures.get_device_arch()
    cuda_architectures.current = 1 - first
    assert (
        adapter._select_attention_backend("auto", other, other, other)
        is adapter.optimized_attention
    )
    with pytest.raises(ValueError, match="initialized for SM.*requires SM"):
        backend.flash_attention(other, other, other, None, None, 1.0)
    # A pending backward for the original architecture must retain its dispatch.
    assert cuda_architectures.get_device_arch() == selected_arch
    assert cuda_architectures.probes == [first]
    assert (
        adapter._select_attention_backend("auto", original, original, original)
        is backend.flash_attention
    )
    assert cuda_architectures.current == 1 - first


def test_auto_respects_architecture_selected_by_another_fa4_user(cuda_architectures):
    assert cuda_architectures.get_device_arch() == 90
    q = query("cuda:1")
    assert adapter._select_attention_backend("auto", q, q, q) is adapter.optimized_attention
    assert cuda_architectures.probes == [0]
    assert cuda_architectures.current == 0


def test_architecture_probe_errors_propagate_and_restore_device(cuda_architectures, monkeypatch):
    monkeypatch.setattr(
        cuda_architectures.interface, "_get_device_arch", Mock(side_effect=RuntimeError("probe"))
    )
    q = query("cuda:1")
    with pytest.raises(RuntimeError, match="probe"):
        adapter._select_attention_backend("auto", q, q, q)
    assert cuda_architectures.current == 0


@pytest.mark.parametrize("fail", [False, True])
def test_fa4_forward_uses_input_device_and_restores_caller(cuda_architectures, monkeypatch, fail):
    q = query("cuda:1")
    plan = NS(
        mask_kind="full",
        q_indices=None,
        q_positions=None,
        k_indices=None,
        k_positions=None,
        cu_q=None,
        cu_k=None,
        max_q=2,
        max_k=2,
        merge_indptr=None,
        merge_indices=None,
    )

    def flash_varlen(*args, **kwargs):
        assert cuda_architectures.current == 1
        assert cuda_architectures.get_device_arch() == 103
        if fail:
            raise RuntimeError("kernel failure")
        return object(), object()

    monkeypatch.setattr(backend, "_flash_attention_function", lambda: flash_varlen)
    monkeypatch.setattr(backend, "_pack_and_rotate_inputs", Mock(return_value=(q, q, q)))
    monkeypatch.setattr(
        backend, "merge_attention_parts", Mock(return_value=torch.empty((2, 4, 64)))
    )
    if fail:
        with pytest.raises(RuntimeError, match="kernel failure"):
            backend._forward(q, q, q, None, plan, 1.0, None)
    else:
        assert backend._forward(q, q, q, None, plan, 1.0, None).shape == (2, 256)
    assert cuda_architectures.current == 0
