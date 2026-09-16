"""Automatic selection must respect the architectures supported by pinned kernels."""
from types import SimpleNamespace

import pytest
import torch

from minisgl.engine.engine import _adjust_config
from minisgl.utils.arch import _get_torch_cuda_version


@pytest.fixture(autouse=True)
def primary_rank(monkeypatch):
    from minisgl.distributed import info
    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(0, 1))


@pytest.mark.parametrize("arch,expected", [
    ((8, 0), "fi"), ((9, 0), "fa,fi"),
    ((10, 0), "trtllm"), ((10, 3), "trtllm"), ((10, 7), "trtllm"),
    ((12, 0), "fi"), ((12, 1), "fi"),
])
def test_auto_attention_matches_supported_architectures(monkeypatch, arch, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: arch)
    _get_torch_cuda_version.cache_clear()
    config = SimpleNamespace(attention_backend="auto", tp_info=SimpleNamespace(rank=0),
                             model_config=SimpleNamespace(is_hybrid=False, is_moe=False),
                             page_size=1)
    try:
        _adjust_config(config)
        assert config.attention_backend == expected
        assert config.page_size == (64 if expected == "trtllm" else 1)
    finally:
        _get_torch_cuda_version.cache_clear()


def test_explicit_attention_selection_does_not_probe_hardware(monkeypatch):
    def forbidden(*args):
        raise AssertionError("explicit backend must not probe/select another backend")
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    config = SimpleNamespace(attention_backend="fi", model_config=SimpleNamespace(
        is_hybrid=False, is_moe=False), page_size=1)
    _adjust_config(config)
    assert config.attention_backend == "fi" and config.page_size == 1
