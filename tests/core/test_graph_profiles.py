import pytest
from types import SimpleNamespace

import torch

from minisgl.engine.graph import GraphRunner, _determine_cuda_graph_bs


@pytest.mark.parametrize('prefill_rows', [[], [128, 512]])
def test_workspace_preparation_includes_all_profiles_before_capture(monkeypatch, prefill_rows):
    import minisgl.shared_cache.graph as module
    created, prepared = [], []
    attention = SimpleNamespace()
    def profile(kind):
        def create(owner, workers, depth, *args):
            item = SimpleNamespace(kind=kind, attention=owner, workers=workers, depth=depth)
            created.append(item)
            return item
        return create
    def reserve(profiles):
        profiles = list(profiles)
        assert profiles == created, 'reserve only after the entire catalogue exists'
        prepared.append(profiles)
    attention.prepare_workspace = reserve
    monkeypatch.setattr(module, 'SharedDecodeAttentionBuffers', profile('decode'))
    monkeypatch.setattr(module, 'SharedPrefillAttentionBuffers', profile('prefill'))
    monkeypatch.setattr(module, '_PrefillInputs', lambda *args: object())
    session = SimpleNamespace(sc_attn=attention, sc_gdn=None, page_size=16,
        engine=SimpleNamespace(dtype=torch.bfloat16), device=torch.device('cpu'),
        page_allocator=SimpleNamespace(alloc_pages=lambda n: torch.arange(n)*16),
        kv_cache=SimpleNamespace(num_layers=1, k_cache=lambda _: torch.zeros(128),
                                 v_cache=lambda _: torch.zeros(128)))
    io = module.SharedGraphIO(session, 16, prefill_rows)
    io.init_capture_graph(64, [1, 64])
    assert len(prepared) == 1
    assert [p.kind for p in prepared[0]] == ['decode', 'decode'] + ['prefill']*len(prefill_rows)
    assert [p.workers for p in prepared[0]][:2] == [1, 64]


@pytest.mark.parametrize('limit,expected', [(0, []), (1, [1]), (2, [1, 2]),
                                          (3, [1, 2]), (4, [1, 2, 4]), (9, [1, 2, 4, 8])])
def test_automatic_profiles_respect_small_limits(limit, expected):
    assert _determine_cuda_graph_bs(None, limit, 0) == expected


def test_explicit_profiles_and_global_off():
    assert _determine_cuda_graph_bs([1, 3, 6], None, 0) == [1, 3, 6]
    assert _determine_cuda_graph_bs([1, 3, 6], 0, 0) == []
    assert _determine_cuda_graph_bs([], None, 0) == []


def test_completed_refs_retire_across_inactive_decode_and_prefill_profiles():
    import weakref
    from minisgl.shared_cache.graph import SharedGraphIO
    from minisgl.shared_cache.gdn_decode import GDNDecodeBuffers
    class Retained:
        pass
    completed, inflight, prefill = Retained(), Retained(), Retained()
    weak_done, weak_live, weak_prefill = map(weakref.ref, (completed, inflight, prefill))
    ready = [False]
    decode = GDNDecodeBuffers.__new__(GDNDecodeBuffers)
    decode._pending = [(SimpleNamespace(query=lambda: True), [completed]),
                       (SimpleNamespace(query=lambda: ready[0]), [inflight])]
    pf = GDNDecodeBuffers.__new__(GDNDecodeBuffers)
    pf._pending = [(SimpleNamespace(query=lambda: True), [prefill])]
    io = SharedGraphIO(None, 1)
    io.gdn, io.prefill_gdn = {4: decode}, {128: pf}
    del completed, inflight, prefill
    io.retire_completed()
    assert weak_done() is None and weak_prefill() is None
    assert weak_live() is not None
    ready[0] = True
    io.retire_completed()
    assert weak_live() is None and not decode._pending


def test_all_profile_retirement_precedes_new_state_allocation():
    from minisgl.shared_cache.session import SharedCacheSession
    order = []
    session = SharedCacheSession.__new__(SharedCacheSession)
    session.engine = SimpleNamespace(ctx=SimpleNamespace(gdn_ar=None),
                                     config=SimpleNamespace(dtype=torch.float32))
    session.graph_io = SimpleNamespace(gdn={1: None},
                                       retire_completed=lambda: order.append('retire'))
    session.graph_runner = object()
    session._model_config = SimpleNamespace(num_linear_layers=1)
    def prepare(*a, **kw):
        assert order == ['retire']
        raise RuntimeError('stop before GPU allocation')
    session.sc_gdn = SimpleNamespace(set_context=lambda *a, **kw: None,
        prepare_decode=prepare, finish_decode=lambda *a: None, finish_prefill=lambda *a: None)
    with pytest.raises(RuntimeError, match='stop before GPU allocation'):
        session._forward(SimpleNamespace(is_decode=True, padded_size=1), [[object()]])


def test_failed_later_capture_destroys_graphs_before_buffers(monkeypatch):
    destroyed = []
    class Graph:
        def __del__(self):
            destroyed.append('graph')
    def fail(self, *args):
        self.graph_map = {1: Graph()}
        raise RuntimeError('capture failed')
    monkeypatch.setattr(GraphRunner, '_capture_graphs', fail)
    backend = SimpleNamespace(destroy_capture_graph=lambda: destroyed.append('buffers'))
    with pytest.raises(RuntimeError, match='capture failed'):
        GraphRunner(stream=None, device=torch.device('cpu'), model=None, attn_backend=backend,
            cuda_graph_bs=[1], cuda_graph_max_bs=1, free_memory=0, max_seq_len=32,
            vocab_size=8, dummy_req=None, prefill_graph_rows=[32])
    assert destroyed == ['graph', 'buffers']
