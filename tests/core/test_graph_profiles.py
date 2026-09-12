import pytest
from types import SimpleNamespace

import torch

from minisgl.engine.graph import GraphRunner, _determine_cuda_graph_bs


@pytest.mark.parametrize('limit,expected', [(0, []), (1, [1]), (2, [1, 2]),
                                          (3, [1, 2]), (4, [1, 2, 4]), (9, [1, 2, 4, 8])])
def test_automatic_profiles_respect_small_limits(limit, expected):
    assert _determine_cuda_graph_bs(None, limit, 0) == expected


def test_explicit_profiles_and_global_off():
    assert _determine_cuda_graph_bs([1, 3, 6], None, 0) == [1, 3, 6]
    assert _determine_cuda_graph_bs([1, 3, 6], 0, 0) == []
    assert _determine_cuda_graph_bs([], None, 0) == []


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
