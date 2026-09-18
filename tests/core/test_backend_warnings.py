"""Fallbacks remain functional, but their reason must be visible without CUDA."""
import builtins
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import warnings

import pytest
import torch

from minisgl.moe.fused import _use_torch_moe_fallback


@pytest.fixture(autouse=True)
def clear_backend_caches():
    _use_torch_moe_fallback.cache_clear()
    yield
    _use_torch_moe_fallback.cache_clear()


@pytest.mark.parametrize('error', [ImportError('libnvrtc.so.12 missing'), OSError('ABI symbol missing')])
def test_moe_import_failure_warns_once_and_keeps_fallback(monkeypatch, error):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (12, 0))
    original = builtins.__import__
    def broken(name, *args, **kwargs):
        if name == 'sgl_kernel':
            raise error
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', broken)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        for _ in range(3):
            assert _use_torch_moe_fallback(torch.device('cuda:0'))
    assert len(records) == 1
    message = str(records[0].message)
    assert str(error) in message and type(error).__name__ in message
    assert 'routing/alignment' in message and 'expert GEMMs are unchanged' in message


def test_healthy_native_backend_has_no_warning(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (12, 0))
    monkeypatch.setitem(sys.modules, 'sgl_kernel', SimpleNamespace())
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        assert not _use_torch_moe_fallback(torch.device('cuda:0'))
    assert not records


def test_gb10_guard_remains_explicit_and_does_not_import_extension(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (12, 1))
    original = builtins.__import__
    def forbidden(name, *args, **kwargs):
        assert name != 'sgl_kernel', 'GB10 compatibility guard must run before extension import'
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', forbidden)
    with pytest.warns(RuntimeWarning, match='SM121.*intentional'):
        assert _use_torch_moe_fallback(torch.device('cuda:0'))


def test_cpu_moe_fallback_is_visible():
    with pytest.warns(RuntimeWarning, match='Torch fallback on cpu'):
        assert _use_torch_moe_fallback(torch.device('cpu'))


@pytest.mark.parametrize('error', [ImportError('fla missing'), RuntimeError('Triton ABI failure')])
def test_fla_import_failure_preserves_torch_path_and_reason(monkeypatch, error):
    import minisgl.models.qwen3_5_delta as module
    original = builtins.__import__
    def broken(name, *args, **kwargs):
        if name == 'fla.ops.gated_delta_rule':
            raise error
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', broken)
    # Separate namespace: do not corrupt the imported production module for other tests.
    with pytest.warns(RuntimeWarning, match='GDN.*Torch.*FLA') as records:
        result = runpy.run_path(str(Path(module.__file__)))
    assert str(error) in str(records[0].message)
    assert result['_fla_chunk'] is None and result['_fla_recurrent'] is None
    assert callable(result['_chunk_gated_delta_rule'])
