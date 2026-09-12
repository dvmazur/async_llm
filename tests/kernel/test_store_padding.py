"""Graph padding cannot overwrite live KV rows or the surrounding storage."""
import pytest
import torch

from minisgl.kernel import store_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA KV store')
@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_negative_slots_are_masked_under_replay(index_dtype, dtype):
    # Include guard rows in the same owned allocation so a regression cannot
    # accidentally write outside this test's GPU allocation.
    backing = torch.full((12, 2, 128), 7, device='cuda', dtype=dtype)
    cache = backing[2:-1]
    qkv = torch.randn(4, 4*128, device='cuda', dtype=dtype)
    key, value = qkv[:, :128], qkv[:, 128:256]
    indices = torch.tensor([0, -1, 3, -2], device='cuda', dtype=index_dtype)

    def forward():
        store_cache(cache[:, 0], cache[:, 1], indices, key, value)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()

    for slots in ([0, -1, 3, -2], [-1, 8, -1, 2], [-1, -1, -1, -1], [1, 2, 3, 4]):
        backing.fill_(7)
        qkv.normal_()
        indices.copy_(torch.tensor(slots, device='cuda', dtype=index_dtype))
        graph.replay()
        expected = torch.full_like(backing, 7)
        for row, slot in enumerate(slots):
            if slot >= 0:
                expected[slot+2, 0] = key[row]
                expected[slot+2, 1] = value[row]
        torch.testing.assert_close(backing, expected, atol=0, rtol=0)
