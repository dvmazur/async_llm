"""CPU-built page tables consumed by real FA2 eager and CUDA Graph replay."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from minisgl.kernel import store_cache
from minisgl.shared_cache.attention import SharedCacheAttention
from minisgl.shared_cache.attention_graph import SharedDecodeAttentionBuffers
from minisgl.shared_cache.session import SharedCacheSession
from minisgl.shared_cache.shared_block import CacheBlock
from minisgl.shared_cache.worker_group import WorkerGroup


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA shared decode Attention')
@pytest.mark.parametrize('page', [1, 16])
@pytest.mark.parametrize('mrope', [False, True])
@torch.inference_mode()
def test_decode_pages_and_cross_readers_change_without_recapture(page, mrope):
    torch.manual_seed(391)
    device, dtype = torch.device('cuda'), torch.bfloat16
    workers, hq, hk, dim = 4, 8, 2, 64
    keys, values = [torch.randn(128, page, hk, dim, device=device, dtype=dtype) for _ in range(2)]
    def store(k, v, locations, layer):
        store_cache(keys.view(-1, hk*dim), values.view(-1, hk*dim), locations, k, v)
    pool = SimpleNamespace(k_cache=lambda layer: keys, v_cache=lambda layer: values, store_kv=store)
    rotary = 32 if mrope else dim
    angles = torch.arange(4096, device=device)[:, None] * (10000 ** (
        -torch.arange(0, rotary, 2, device=device).float() / rotary))[None, :]
    attention = SharedCacheAttention(pool, torch.cat([angles.cos(), angles.sin()], -1),
        hq, hk, dim, page, dtype, device, rotary_dim=rotary,
        mrope_section=(8, 4, 4) if mrope else None, rope_base=10000.)
    dummy = torch.arange(124, 128, device=device, dtype=torch.int32)*page
    keys[124:].zero_()
    values[124:].zero_()
    buffers = SharedDecodeAttentionBuffers(attention, workers, 5, 256, dummy)
    attention.prepare_workspace([buffers])
    common = CacheBlock(device, page)
    common.page_starts, common.num_tokens = [page, 2*page], page+1
    if mrope:
        common.mrope_span_override = 1
    tails = [CacheBlock(device, page) for _ in range(workers)]
    tails[0].page_starts, tails[0].num_tokens = [3*page], page
    if page > 1:
        tails[2].page_starts, tails[2].num_tokens = [4*page], page-1
    session = SharedCacheSession.__new__(SharedCacheSession)
    session.device, session.page_size = device, page
    next_page = 10
    def allocate(n):
        nonlocal next_page
        ids = torch.arange(next_page, next_page+n, device=device, dtype=torch.int32)*page
        next_page += n
        return ids
    session.page_allocator = SimpleNamespace(alloc_pages=allocate)
    q = torch.randn(workers, hq*dim, device=device, dtype=dtype)
    k, v = [torch.randn(workers, hk*dim, device=device, dtype=dtype) for _ in range(2)]
    batch = SimpleNamespace(attn_metadata=buffers.meta,
        positions=torch.zeros(workers, dtype=torch.int64, device=device),
        out_loc=torch.full((workers,), -1, dtype=torch.int32, device=device),
        mrope_positions=None)

    def prepare(n):
        group = WorkerGroup(cache_structure=[
            [common, *[b for j, b in enumerate(tails[:n]) if j != i or i % 2 == 0]]
            for i in range(n)], write_to=tails[:n])
        pending, slots, positions = session._plan_decode_writes(group)
        batch.positions.zero_()
        batch.out_loc.fill_(-1)
        batch.positions[:n] = torch.tensor(positions, device=device)
        batch.out_loc[:n] = slots
        with patch.object(CacheBlock, 'page_numbers_tensor',
                          side_effect=AssertionError('per-block device page table')):
            attention.prepare(group, pending, slots, buffers)
        return group, pending, slots

    def forward():
        return attention.forward(q, k, v, 0, batch)

    prepare(4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()

    for n in (2, 4, 1, 3, 4):
        group, pending, slots = prepare(n)
        q.normal_()
        k.normal_()
        v.normal_()
        graph.replay()
        actual = output.clone()
        torch.testing.assert_close(actual, forward(), atol=0, rtol=0)
        meta = attention.prepare(group, pending, slots)
        eager_batch = SimpleNamespace(attn_metadata=meta, positions=batch.positions[:n],
                                      out_loc=slots, mrope_positions=None)
        expected = attention.forward(q[:n], k[:n], v[:n], 0, eager_batch)
        # Padded/unpadded FA2 may choose different split-KV plans. Integer
        # metadata is tested exactly; allow the existing one-BF16-step layer gate.
        torch.testing.assert_close(actual[:n], expected, atol=2e-3, rtol=torch.finfo(dtype).eps)
        assert torch.count_nonzero(actual[n:]) == 0
        for b in group.write_to:
            b.append_token(pending[id(b)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='pinned Attention metadata DMA')
def test_bulk_page_upload_retains_staging_until_stream_completion():
    from minisgl.shared_cache.attention_metadata import upload_page_indices
    parts = [[7, 2, 11]*4096, [], [4]*1024]
    expected = torch.tensor([p for part in parts for p in part], dtype=torch.int32)
    refs = []
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        indices = upload_page_indices(parts, torch.device('cuda'), refs)
        copied = indices.clone()
        done = torch.cuda.Event()
        done.record()
    assert refs[0].is_pinned() and refs[1] is indices
    parts.clear()
    done.synchronize()
    refs.clear()
    torch.testing.assert_close(copied.cpu(), expected, atol=0, rtol=0)
