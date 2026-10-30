"""Page-table semantics on CPU; actual FlashInfer/replay lives in the layer tests."""
from types import SimpleNamespace

import pytest
import torch

from minisgl.shared_cache.attention import PrefillSpec, SharedCacheAttention
from minisgl.shared_cache.attention_graph import (
    SharedDecodeAttentionBuffers, SharedPrefillAttentionBuffers,
)
from minisgl.shared_cache.attention_metadata import page_numbers, upload_page_indices
from minisgl.shared_cache.session import SharedCacheSession
from minisgl.shared_cache.shared_block import CacheBlock
from minisgl.shared_cache.worker_group import WorkerGroup


class CountedPages(list):
    reads = 0

    def __iter__(self):
        self.reads += 1
        return super().__iter__()


def block(page, pages, length):
    result = CacheBlock(torch.device('cpu'), page)
    result.page_starts = CountedPages([p * page for p in pages])
    result.num_tokens = length
    return result


@pytest.mark.parametrize('page', [1, 4, 16])
def test_snapshot_is_per_prepare_and_includes_pending_page_once(page):
    b = block(page, [7, 2], 2 * page)
    snapshot = {}
    first = page_numbers(b, snapshot, 11 * page)
    assert first == [7, 2, 11]
    assert page_numbers(b, snapshot, 11 * page) is first
    assert b.page_starts.reads == 1
    assert b.page_starts == [7 * page, 2 * page]  # prepare does not publish
    b.append_token(11 * page)
    b.page_starts[0] = 13 * page  # even ordinary list mutations need no invalidation
    assert page_numbers(b, {}) == [13, 2, 11]
    assert first == [7, 2, 11]  # prior snapshot remains owned
    b.clear()
    assert page_numbers(b, {}) == []
    assert page_numbers(b, {}, 19 * page) == [19]


@pytest.mark.parametrize('parts', [[], [[], []], [[7, 2], [], [7, 2], [11]]])
def test_bulk_index_table_order_empty_and_owners(parts):
    refs = []
    indices = upload_page_indices(parts, torch.device('cpu'), refs)
    assert indices.dtype == torch.int32 and indices.ndim == 1
    assert indices.tolist() == [p for part in parts for p in part]
    assert len(refs) == 2 and refs[1] is indices
    before = indices.clone()
    for part in parts:
        part.clear()
    torch.testing.assert_close(indices, before, atol=0, rtol=0)


class Plans:
    def plan(self, **kwargs):
        self.last = kwargs


@pytest.fixture
def cpu_plans(monkeypatch):
    # Exercise the real CPU preparation without initializing CUDA/FlashInfer.
    for name in ('tensor', 'arange', 'ones'):
        original = getattr(torch, name)
        def cpu_tensor(*args, _original=original, **kwargs):
            kwargs.pop('pin_memory', None)
            assert torch.device(kwargs.get('device', 'cpu')).type == 'cpu'
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, cpu_tensor)
    monkeypatch.setattr(torch.Tensor, 'pin_memory', lambda self: self)
    def forbidden(*args):
        raise AssertionError('per-block device page tensors must not be constructed')
    monkeypatch.setattr(CacheBlock, 'page_numbers_tensor', forbidden)
    a = SharedCacheAttention.__new__(SharedCacheAttention)
    a.device, a.dtype = torch.device('cpu'), torch.bfloat16
    a.num_qo_heads, a.num_kv_heads, a.head_dim = 8, 2, 64
    a._plan_event = SimpleNamespace(synchronize=lambda: None, record=lambda: None)
    a.wrapper, a.aux_wrapper = Plans(), Plans()
    a.prefill_ctx_wrapper, a.prefill_self_wrapper = Plans(), Plans()
    return a


def buffers(a, prefill):
    """CPU storage with the same fixed shapes; GPU tests cover real constructors."""
    w, d, r = 4, 4, 64
    cls = SharedPrefillAttentionBuffers if prefill else SharedDecodeAttentionBuffers
    b = cls.__new__(cls)
    b.attention, b.workers, b.depth, b.rows = a, w, d, r
    b.main_count, b.ctx_rows, b.max_pages = w*d, r*d, 256
    b.use_3d = True
    b.dummy_parts = [[90+i] for i in range(w)]
    b.dummy_cpu = [(90+i)*a.page_size for i in range(w)]
    b.main, b.aux, b.event, b.refs = Plans(), Plans(), a._plan_event, []
    total = (r if prefill else w)*(d+1)
    b.valid = torch.zeros(total, dtype=torch.bool)
    b._all_slots = torch.arange(total)
    b.meta = SimpleNamespace(sub_worker=torch.zeros(total, dtype=torch.int64),
        sub_loc=torch.zeros((3, total) if prefill else total, dtype=torch.int64),
        pad_slot=torch.zeros(total, dtype=torch.int64), last_indices=torch.zeros(w, dtype=torch.int64))
    return b


def entries(plan):
    ids = plan.get('indices', plan.get('paged_kv_indices')).tolist()
    offsets = plan.get('indptr', plan.get('paged_kv_indptr')).tolist()
    return [ids[a:b] for a, b in zip(offsets, offsets[1:])]


@pytest.mark.parametrize('page', [1, 4, 16])
@pytest.mark.parametrize('graph', [False, True])
def test_decode_cross_readers_empty_blocks_duplicates_and_implicit_self(cpu_plans, page, graph):
    a = cpu_plans
    a.page_size = page
    common, left, right, empty = (block(page, [7, 2], page+1), block(page, [11], page),
                                  block(page, [], 0), block(page, [], 0))
    common.mrope_span_override = 1
    group = WorkerGroup(cache_structure=[[common, empty, right, left],
        [common, left, common]], write_to=[left, right])
    pending = {id(left): 13*page, id(right): 17*page}
    b = buffers(a, False) if graph else None
    meta = a.prepare(group, pending, torch.tensor([13*page, 17*page]), b)
    assert common.page_starts.reads == left.page_starts.reads == right.page_starts.reads == 1
    main, aux = (b.main.last, b.aux.last) if graph else (a.wrapper.last, a.aux_wrapper.last)
    expected = [[7, 2], [17], [11, 13], [7, 2], [11, 13], [7, 2]]
    lens = [page+1, 1, page+1, page+1, page+1, page+1]
    loc = [page+2, page+1, page, page+2, page+1, 0]
    if graph:
        valid = b.valid[:b.main_count]
        assert [v for v, active in zip(entries(main), valid) if active] == expected
        assert main['seq_lens'][valid].tolist() == lens
        assert meta.sub_loc[:b.main_count][valid].tolist() == loc
        assert entries(aux)[1] == [17*page]
        assert b.valid[b.main_count:].tolist() == [False, True, False, False]
    else:
        assert entries(main) == expected
        assert main['seq_lens'].tolist() == lens
        assert meta.sub_loc.tolist() == loc + [0]
        assert entries(aux) == [[17*page]]
    assert 'max_token_per_sequence' not in main  # decode API has no such hint
    assert left.page_starts == [11*page] and right.page_starts == []


@pytest.mark.parametrize('graph', [False, True])
@pytest.mark.parametrize('context', [False, True])
def test_prefill_public_q_max_tracks_real_rows_not_capacity(cpu_plans, graph, context):
    a = cpu_plans
    a.page_size = 4
    shared = block(4, [7, 2], 5)
    shared.mrope_span_override = 2
    b = buffers(a, True) if graph else None
    for lengths in ([3, 9], [1], [7, 13, 5]):
        shared.page_starts.reads = 0
        specs = [PrefillSpec([shared, shared] if context else [],
            torch.arange(20+i*8, 20+i*8+(2+n+3)//4)*4, n, 2, 1,
            torch.arange(n).expand(3, -1)) for i, n in enumerate(lengths)]
        a.prepare_prefill_batch(specs, b)
        assert shared.page_starts.reads == int(context)
        main, aux = (b.main.last, b.aux.last) if graph else (
            a.prefill_ctx_wrapper.last if context else None, a.prefill_self_wrapper.last)
        for plan in (main, aux):
            if plan is None:
                continue
            qlens = plan['qo_indptr'].diff().tolist()
            assert plan['max_token_per_sequence'] == max(qlens)
            assert 'max_sequence_kv' not in plan  # do not bypass FA2's required preparation
        assert aux['max_token_per_sequence'] == max(lengths)
        assert int(aux['qo_indptr'][-1]) == sum(lengths)
        assert entries(aux)[:len(specs)] == [[int(p)//4 for p in s.self_page_starts] for s in specs]
        if context:
            real = main['qo_indptr'].diff() > 0
            assert [e for e, active in zip(entries(main), real) if active] == [[7, 2]]*(2*len(specs))
        elif graph:
            assert main['max_token_per_sequence'] == 0  # all context slots inactive
        shared.page_starts = CountedPages([28, 8])  # replaced between prepares, same IDs


def test_decode_slot_allocator_uses_one_bulk_readback(cpu_plans, monkeypatch):
    page = 4
    a, b, c = block(page, [3], 4), block(page, [8], 2), block(page, [], 0)
    session = SharedCacheSession.__new__(SharedCacheSession)
    session.page_size, session.device = page, torch.device('cpu')
    calls = []
    def allocate(n):
        calls.append(n)
        return torch.tensor([40, 52])
    session.page_allocator = SimpleNamespace(alloc_pages=allocate)
    def no_scalar(*args):
        raise AssertionError('per-slot scalar readback')
    monkeypatch.setattr(torch.Tensor, 'item', no_scalar)
    pending, slots, positions = session._plan_decode_writes(
        WorkerGroup(cache_structure=[[a], [b], [c]]))
    assert calls == [2] and pending == {id(a): 40, id(b): None, id(c): 52}
    assert slots.tolist() == [40, 34, 52] and positions == [4, 2, 0]
