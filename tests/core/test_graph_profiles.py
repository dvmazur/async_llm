import pytest

from minisgl.engine.graph import _determine_cuda_graph_bs


@pytest.mark.parametrize('limit,expected', [(0, []), (1, [1]), (2, [1, 2]),
                                          (3, [1, 2]), (4, [1, 2, 4]), (9, [1, 2, 4, 8])])
def test_automatic_profiles_respect_small_limits(limit, expected):
    assert _determine_cuda_graph_bs(None, limit, 0) == expected


def test_explicit_profiles_and_global_off():
    assert _determine_cuda_graph_bs([1, 3, 6], None, 0) == [1, 3, 6]
    assert _determine_cuda_graph_bs([1, 3, 6], 0, 0) == []
    assert _determine_cuda_graph_bs([], None, 0) == []
