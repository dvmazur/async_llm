"""Catalogue-wide workspace sizing, including the 64x16 decode regression."""
from types import SimpleNamespace

import pytest
import torch

from minisgl.shared_cache.attention import SharedCacheAttention
from minisgl.shared_cache.attention_graph import (
    SharedDecodeAttentionBuffers, SharedPrefillAttentionBuffers,
)


def cpu_attention(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    attention = SharedCacheAttention.__new__(SharedCacheAttention)
    attention.device = torch.device('cpu')
    attention._workspace = torch.empty(8, dtype=torch.uint8)
    attention._workspace_prepared = False
    return attention


@pytest.mark.parametrize('reverse', [False, True])
def test_all_wrappers_share_maximum_float_but_not_int_storage(monkeypatch, reverse):
    attention = cpu_attention(monkeypatch)
    wrappers = [SimpleNamespace() for _ in range(4)]
    for wrapper in wrappers:
        def reset(floats, ints, w=wrapper):
            w.floats, w.ints = floats, ints
        wrapper.reset_workspace_buffer = reset
    decode = SimpleNamespace(attention=attention, workspace_requirements=lambda: [
        (wrappers[0], 32, 24), (wrappers[1], 16, 0)])
    prefill = SimpleNamespace(attention=attention, workspace_requirements=lambda: [
        (wrappers[2], 64, 48), (wrappers[3], 8, 16)])
    profiles = [prefill, decode] if reverse else [decode, prefill]
    attention.prepare_workspace(profiles)
    assert attention._workspace.numel() == 64
    assert all(w.floats is attention._workspace for w in wrappers)
    assert [w.ints.numel() for w in wrappers] == [24, 16, 48, 16]
    assert len({w.ints.data_ptr() for w in wrappers}) == 4
    with pytest.raises(RuntimeError, match='once before capture'):
        attention.prepare_workspace(profiles)


def test_no_rebinding_during_capture_or_after_allocation_failure(monkeypatch):
    attention = cpu_attention(monkeypatch)
    original = attention._workspace
    calls = []
    wrapper = SimpleNamespace(reset_workspace_buffer=lambda *args: calls.append(args))
    profile = SimpleNamespace(attention=attention, workspace_requirements=lambda: [
        (wrapper, 64, 24), (wrapper, 32, 48)])
    empty = torch.empty
    count = 0
    def allocate(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise torch.OutOfMemoryError('test allocation')
        return empty(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'empty', allocate)
        with pytest.raises(torch.OutOfMemoryError):
            attention.prepare_workspace([profile])
    assert not calls and attention._workspace is original
    assert not attention._workspace_prepared
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    with pytest.raises(RuntimeError, match='before capture'):
        attention.prepare_workspace([profile])
    assert not calls


def test_empty_catalogue_and_foreign_owner(monkeypatch):
    attention = cpu_attention(monkeypatch)
    attention.prepare_workspace([])
    assert not attention._workspace_prepared
    with pytest.raises(ValueError, match='belong'):
        attention.prepare_workspace([SimpleNamespace(attention=object())])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FlashInfer workspace and replay')
@pytest.mark.parametrize('heads,kv_heads,dim', [(16, 2, 256), (4, 4, 128)])
@pytest.mark.parametrize('with_prefill', [False, True])
@torch.inference_mode()
def test_large_decode_catalogue_plans_and_replays(heads, kv_heads, dim, with_prefill):
    device, dtype, workers, depth, page = torch.device('cuda'), torch.bfloat16, 64, 16, 16
    attention = SharedCacheAttention(SimpleNamespace(), torch.zeros(1, dim, device=device),
        heads, kv_heads, dim, page, dtype, device)
    slots = torch.arange(workers, device=device, dtype=torch.int32) * page
    decode = SharedDecodeAttentionBuffers(attention, workers, depth, 2048, slots)
    profiles = [decode]
    if with_prefill:
        profiles.append(SharedPrefillAttentionBuffers(attention, 4, 3, 128, 256, slots[:4]))
    requirements = [r for p in profiles for r in p.workspace_requirements()]
    if heads // kv_heads >= 4:
        assert requirements[0][1] > 128 * 1024 * 1024  # Original 128MiB fails here.
    attention.prepare_workspace(profiles)
    workspace = attention._workspace
    assert workspace.numel() >= max(r[1] for r in requirements)
    # Exercise sparse active requests but all 1024 fixed slots in the planner.
    def plan(length):
        pages = list(range((length + page - 1) // page))
        decode.plan([0, workers-1], [0, 0], [0, depth-1], [pages, pages],
                    [len(pages)]*2, [length]*2, [length-(len(pages)-1)*page]*2,
                    [], [], [])
    keys = torch.zeros(workers, page, kv_heads, dim, device=device, dtype=dtype)
    values = torch.ones_like(keys)
    query = torch.zeros(workers*depth, heads, dim, device=device, dtype=dtype)
    def run():
        return (decode.main.run(query, (keys, values)),
                decode.aux.run(query[:workers],
                               (keys.view(-1, 1, kv_heads, dim),
                                values.view(-1, 1, kv_heads, dim))))
    plan(1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    for length in (31, 1, 16):
        plan(length)
        graph.replay()
        for out in outputs:
            torch.testing.assert_close(out, torch.ones_like(out), atol=2e-3, rtol=0)
        assert attention._workspace is workspace
