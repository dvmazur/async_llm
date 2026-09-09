"""Allocator setup must respect user settings and never initialize CUDA."""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import minisgl.engine.memory as memory


@pytest.mark.parametrize("raw", ["", "max_split_size_mb:512", "roundup_power2_divisions:[256:1,512:2,>:4]"])
def test_append_preserves_complete_config(monkeypatch, raw):
    applied=[]
    fake=SimpleNamespace(cuda=SimpleNamespace(get_allocator_backend=lambda:'native',
        is_initialized=lambda:False,
        memory=SimpleNamespace(_snapshot=lambda:{'allocator_settings':{'PYTORCH_CUDA_ALLOC_CONF':raw}})),
        _C=SimpleNamespace(_accelerator_setAllocatorSettings=applied.append))
    monkeypatch.setattr(memory,'torch',fake)
    assert memory.prefer_expandable_segments(True)
    assert applied==[(raw+',' if raw else '')+'expandable_segments:True']


@pytest.mark.parametrize("raw", ['expandable_segments:False', 'expandable_segments:True',
                                 'max_split_size_mb:512, expandable_segments :False'])
def test_explicit_global_choice_wins(monkeypatch,raw):
    def forbidden(*args):raise AssertionError('must not override caller')
    fake=SimpleNamespace(cuda=SimpleNamespace(get_allocator_backend=lambda:'native',
        is_initialized=lambda:False,
        memory=SimpleNamespace(_snapshot=lambda:{'allocator_settings':{'PYTORCH_CUDA_ALLOC_CONF':raw}})),
        _C=SimpleNamespace(_accelerator_setAllocatorSettings=forbidden))
    monkeypatch.setattr(memory,'torch',fake)
    assert not memory.prefer_expandable_segments(True)


def test_opt_out_and_other_backend_do_not_query_snapshot(monkeypatch):
    monkeypatch.setattr(memory,'torch',SimpleNamespace(
        cuda=SimpleNamespace(get_allocator_backend=lambda:'cudaMallocAsync')))
    assert not memory.prefer_expandable_segments(False)
    assert not memory.prefer_expandable_segments(True)


def test_missing_older_torch_api_keeps_defaults(monkeypatch):
    fake=SimpleNamespace(cuda=SimpleNamespace(get_allocator_backend=lambda:'native',
        is_initialized=lambda:False,memory=SimpleNamespace(_snapshot=lambda:{})),_C=SimpleNamespace())
    monkeypatch.setattr(memory,'torch',fake)
    assert not memory.prefer_expandable_segments(True)


def test_too_late_is_an_error(monkeypatch):
    monkeypatch.setattr(memory,'torch',SimpleNamespace(cuda=SimpleNamespace(
        get_allocator_backend=lambda:'native',is_initialized=lambda:True)))
    with pytest.raises(RuntimeError,match='before CUDA'):
        memory.prefer_expandable_segments(True)


def test_actual_torch_configuration_preserved_without_cuda_init():
    # Isolated subprocess: do not alter allocator state for other GPU tests.
    env=os.environ.copy()
    env.pop('PYTORCH_ALLOC_CONF',None)
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    code='''
import torch
from minisgl.engine.memory import prefer_expandable_segments
torch._C._accelerator_setAllocatorSettings("max_split_size_mb:512")
assert not torch.cuda.is_initialized()
assert prefer_expandable_segments(True)
s=torch.cuda.memory._snapshot()["allocator_settings"]
assert s["expandable_segments"] is True
assert s["max_split_size"]==512*2**20
assert not torch.cuda.is_initialized()
assert not prefer_expandable_segments(True)
'''
    subprocess.run([sys.executable,'-c',code],env=env,check=True)


@pytest.mark.parametrize("hybrid,tp,explicit,expected", [
    (True,1,None,True), (False,1,None,False), (True,2,None,False),
    (True,1,False,False), (True,1,True,True),
])
def test_async_default_only_for_owned_tp1_hybrid(monkeypatch,hybrid,tp,explicit,expected):
    import minisgl.llm.async_llm as frontend
    from minisgl.engine.config import EngineConfig
    from minisgl.distributed import DistributedInfo
    import transformers

    monkeypatch.setattr(EngineConfig,'model_config',property(lambda self:
        SimpleNamespace(is_hybrid=hybrid,is_multimodal=False)))
    captured=[]
    def make_engine(config):
        captured.append(config)
        return SimpleNamespace(config=config)
    monkeypatch.setattr(frontend,'Engine',make_engine)
    monkeypatch.setattr(frontend,'AsyncCacheEngine',lambda engine:SimpleNamespace())
    monkeypatch.setattr(frontend,'load_tokenizer',lambda path:object())
    kwargs=dict(tp_info=DistributedInfo(rank=0,size=tp),
                generation_config=transformers.GenerationConfig())
    if explicit is not None:kwargs['prefer_expandable_segments']=explicit
    frontend.AsyncLLM('fake-local-model',**kwargs)
    assert captured[0].prefer_expandable_segments is expected
    captured.clear()
    # An externally owned engine never gets silently rebuilt/reconfigured.
    supplied=SimpleNamespace(config=object())
    result=frontend.AsyncLLM(engine=supplied)
    assert result.engine is supplied and not captured
