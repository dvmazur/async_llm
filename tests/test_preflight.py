import pytest

from environment import check as preflight


def test_missing_jit_tool_fails_before_importing_cuda(monkeypatch):
    import builtins
    import shutil
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'torch':
            pytest.fail('must reject missing ninja before importing CUDA libraries')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(RuntimeError, match='ninja is not on PATH'):
        preflight.main()
