"""Host page bookkeeping vs pre-change FIFO semantics, including CUDA stalls."""
import random

import pytest
import torch

from minisgl.kvcache import PageAllocator
from minisgl.shared_cache import CacheBlock, SharedCacheSession, WorkerGroup


class LegacyPageAllocator:
    # Literal pre-change allocation/free bodies; test-only reference.
    def __init__(self, num_pages, page_size, device):
        self.num_pages = num_pages
        self.page_size = page_size
        self.device = device
        self._free = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size

    @property
    def num_free_pages(self):
        return int(self._free.numel())

    def alloc_pages(self, num_pages):
        if num_pages > self.num_free_pages:
            raise RuntimeError(
                f"PageAllocator out of pages: requested {num_pages}, "
                f"only {self.num_free_pages} free"
            )
        allocated = self._free[:num_pages].clone()
        self._free = self._free[num_pages:]
        return allocated

    def free_pages(self, page_starts):
        if page_starts.numel() == 0:
            return
        page_starts = page_starts.to(device=self._free.device, dtype=self._free.dtype)
        self._free = torch.cat([self._free, page_starts])


@pytest.mark.parametrize('page_size', [1, 4, 16])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_host_and_legacy_clients_share_one_fifo_pool(page_size, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    device = torch.device(device)
    ref = LegacyPageAllocator(64, page_size, device)
    actual = PageAllocator(64, page_size, device)
    borrowed = [(ref.alloc_pages(9), actual.alloc_pages(9))]
    actual.enable_host_metadata()  # migrate with nine pages already in use
    rng = random.Random(321)
    for step in range(150):
        if borrowed and (rng.random() < .45 or not ref.num_free_pages):
            left, right = borrowed.pop(rng.randrange(len(borrowed)))
            ref.free_pages(left)
            if step % 2:
                actual.free_pages_cpu(right.cpu())
            else:
                actual.free_pages(right.to(device))  # device-only API still valid
        else:
            n = rng.randrange(min(7, ref.num_free_pages) + 1)
            left = ref.alloc_pages(n)
            right = actual.alloc_pages_cpu(n) if step % 2 else actual.alloc_pages(n)
            assert right.device.type == ('cpu' if step % 2 else device.type)
            assert left.cpu().tolist() == right.cpu().tolist()
            borrowed.append((left, right))
        assert actual.num_free_pages == ref.num_free_pages
        assert actual.free_page_starts.cpu().tolist() == ref._free.cpu().tolist()
    for left, right in borrowed:
        ref.free_pages(left)
        actual.free_pages_cpu(right.cpu())
    assert actual.num_free_pages == 64
    assert sorted(actual.free_page_starts.cpu().tolist()) == list(range(0, 64*page_size, page_size))


def test_failed_metadata_allocation_does_not_consume_pages(monkeypatch):
    allocator = PageAllocator(8, 4, torch.device('cpu'))
    allocator.enable_host_metadata()
    def fail(*a, **kw):
        raise RuntimeError('injected host allocation failure')
    with monkeypatch.context() as m:
        m.setattr(torch, 'tensor', fail)
        with pytest.raises(RuntimeError, match='injected'):
            allocator.alloc_pages_cpu(3)
    assert allocator.num_free_pages == 8
    assert allocator.alloc_pages_cpu(3).tolist() == [0, 4, 8]
    with pytest.raises(RuntimeError, match='out of pages'):
        allocator.alloc_pages_cpu(6)
    assert allocator.num_free_pages == 5


@pytest.mark.parametrize('page_size', [1, 4, 16])
def test_partial_tail_and_rollback_keep_page_order(page_size):
    session = SharedCacheSession.__new__(SharedCacheSession)
    session.device = torch.device('cpu')
    session.sc_gdn = None
    session.page_size = page_size
    session.page_allocator = PageAllocator(64, page_size, session.device)
    session.page_allocator.enable_host_metadata()
    a, b = [CacheBlock(session.device, page_size) for _ in range(2)]
    pages, slots = session._alloc_token_storage(3, a)
    assert pages.device.type == 'cpu'
    a.grow_pages(pages, 3)
    assert slots.tolist() == [0, 1, 2]
    group = WorkerGroup(cache_structure=[[a], [b]])
    before = session.page_allocator.num_free_pages
    allocations = []
    new, slots, positions = session._plan_decode_writes(group, allocations=allocations)
    assert all(t.device.type == 'cpu' for t in allocations)
    assert positions == [3, 0]
    assert slots.tolist()[0] == 3
    assert new[id(a)] is None if page_size > 1 else new[id(a)] == 3
    assert a.num_tokens == 3 and b.num_tokens == 0  # planning is not commit
    for pages in allocations:
        session.page_allocator.free_pages_cpu(pages)
    assert session.page_allocator.num_free_pages == before
    session.free_block(a)
    session.free_block(b)
    assert session.page_allocator.num_free_pages == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_page_metadata_hot_path_has_no_d2h_or_stream_barriers():
    session = SharedCacheSession.__new__(SharedCacheSession)
    session.device = torch.device('cuda')
    session.page_size = 1
    session.page_allocator = PageAllocator(4096, 1, session.device)
    session.page_allocator.enable_host_metadata()
    blocks = [CacheBlock(session.device, 1) for _ in range(18)]
    group = WorkerGroup(cache_structure=[[b] for b in blocks])
    def one_round():
        allocations = []
        new, slots, _ = session._plan_decode_writes(group, allocations=allocations)
        for b in blocks:
            b.append_token(new[id(b)])
            b.token_slots_tensor()
            b.page_numbers_tensor()
        return slots
    for _ in range(4):
        one_round()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        slots = one_round()
        pages, prefill_slots = session._alloc_token_storage(20)
        session.page_allocator.free_pages_cpu(pages)
        torch.cuda.synchronize()
    keys = [e.key for e in prof.key_averages()]
    assert 'cudaStreamSynchronize' not in keys
    assert not any('Memcpy DtoH' in k or 'DtoH' in k for k in keys)
    assert slots.numel() == 18 and prefill_slots.numel() == 20
    for b in blocks:
        session.page_allocator.free_pages_cpu(b.clear())
    assert session.page_allocator.num_free_pages == 4096
