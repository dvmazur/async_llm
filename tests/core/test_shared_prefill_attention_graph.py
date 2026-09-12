"""Prepared prefill Attention vs the existing segmented Attention, with replay."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from minisgl.kernel import store_cache
from minisgl.shared_cache.attention import PrefillSpec, SharedCacheAttention
from minisgl.shared_cache.attention_graph import SharedPrefillAttentionBuffers
from minisgl.shared_cache.shared_block import CacheBlock


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA shared prefill Attention')
@pytest.mark.parametrize('mrope', [False, True])
@torch.inference_mode()
def test_prefill_attention_changes_contexts_rows_and_pages_without_recapture(mrope):
    torch.manual_seed(734)
    workers, rows, depth, page, hq, hkv, dim = 4, 256, 3, 16, 4, 2, 64
    device, dtype = torch.device('cuda'), torch.bfloat16
    keys, values = [torch.randn(64, page, hkv, dim, device=device, dtype=dtype) for _ in range(2)]
    def store(k, v, locations, layer):
        store_cache(keys.view(-1, hkv*dim), values.view(-1, hkv*dim), locations, k, v)
    pool = SimpleNamespace(k_cache=lambda layer: keys, v_cache=lambda layer: values, store_kv=store)
    rotary = 32 if mrope else dim
    angles = torch.arange(2048, device=device)[:, None] * (10000 ** (
        -torch.arange(0, rotary, 2, device=device).float() / rotary))[None, :]
    attention = SharedCacheAttention(pool, torch.cat([angles.cos(), angles.sin()], -1),
        hq, hkv, dim, page, dtype, device, rotary_dim=rotary,
        mrope_section=(8, 4, 4) if mrope else None, rope_base=10000.)
    dummy = torch.arange(60, 64, device=device, dtype=torch.int32) * page
    keys[60:].zero_()
    values[60:].zero_()
    buffers = SharedPrefillAttentionBuffers(attention, workers, depth, rows, 128, dummy)
    common = [CacheBlock(device, page) for _ in range(3)]
    for block, start, length in zip(common, (0, 2, 5), (17, 35, 9)):
        block.grow_pages(torch.arange(start, start+(length+page-1)//page, device=device)*page, length)
        if mrope:
            block.mrope_span_override = length//2+1
    q = torch.randn(rows, hq*dim, device=device, dtype=dtype)
    k, v = [torch.randn(rows, hkv*dim, device=device, dtype=dtype) for _ in range(2)]
    batch = SimpleNamespace(positions=torch.zeros(rows, device=device, dtype=torch.int64),
        out_loc=torch.full((rows,), -1, device=device, dtype=torch.int32),
        mrope_positions=torch.zeros(3, rows, device=device, dtype=torch.int64) if mrope else None,
        attn_metadata=buffers.meta)

    def prepare(lengths, turn, full_context=False):
        specs, offset = [], 0
        batch.out_loc.fill_(-1)
        batch.positions.zero_()
        if mrope:
            batch.mrope_positions.zero_()
        for worker, length in enumerate(lengths):
            prefix = (worker*7+turn) % 14
            pages = torch.arange(10+worker*12, 10+worker*12+(prefix+length+page-1)//page,
                                 device=device, dtype=torch.int32)*page
            rel = None
            if mrope and (worker+turn) % 2:
                positions = torch.arange(length)
                rel = torch.stack([positions//8, positions//4, positions%4])
            own_span = prefix//2 if mrope else prefix
            context = [common[(turn+j)%3] for j in range(depth if full_context else (worker+turn) % 4)]
            specs.append(PrefillSpec(context, pages, length, prefix, own_span, rel))
            batch.positions[offset:offset+length] = torch.arange(prefix, prefix+length, device=device)
            slots = (pages[:, None]+torch.arange(page, device=device)).flatten()[prefix:prefix+length]
            batch.out_loc[offset:offset+length] = slots
            if mrope:
                positions = torch.arange(length).expand(3, -1) if rel is None else rel
                batch.mrope_positions[:, offset:offset+length] = (positions+own_span).to(device)
            offset += length
        planned_rows = []
        def record_plan(original):
            def plan(*args, **kwargs):
                planned_rows.append(int(kwargs['qo_indptr'][-1]))
                return original(*args, **kwargs)
            return plan
        with patch.object(buffers.main, 'plan', record_plan(buffers.main.plan)), patch.object(
                buffers.aux, 'plan', record_plan(buffers.aux.plan)):
            attention.prepare_prefill_batch(specs, buffers)
        # Regression: fake padding queries distort FlashInfer's KV-split
        # estimate even if their final outputs are masked away.
        assert planned_rows == [sum(s.num_new*len(s.context) for s in specs), offset]
        return specs, offset

    prepare([64]*4, 0, full_context=True)
    def forward():
        return attention.forward(q, k, v, 0, batch)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    for turn, lengths in ((0, [1]), (2, [63, 65, 3]), (3, [138, 91, 1]), (4, [64]*4), (5, [9, 7])):
        specs, n = prepare(lengths, turn)
        assert buffers.used_self_rows == n
        assert buffers.used_ctx_rows == sum(s.num_new*len(s.context) for s in specs)
        q.normal_()
        k.normal_()
        v.normal_()
        before_k, before_v = keys.clone(), values.clone()
        graph.replay()
        actual = output.clone()
        # The same prepared body can also run eagerly with smaller real rows.
        torch.testing.assert_close(actual, forward(), atol=0, rtol=0)
        old_meta = attention.prepare_prefill_batch(specs)
        old_batch = SimpleNamespace(attn_metadata=old_meta, positions=batch.positions[:n],
            out_loc=batch.out_loc[:n], mrope_positions=batch.mrope_positions[:, :n] if mrope else None)
        expected = attention.forward(q[:n], k[:n], v[:n], 0, old_batch)
        new_q = attention._rope(q.reshape(rows, hq, dim)[buffers.meta.sub_worker], buffers.meta.sub_loc)
        old_q = attention._rope(q[:n].reshape(n, hq, dim)[old_meta.sub_worker], old_meta.sub_loc)
        c = old_meta.n_ctx_rows
        torch.testing.assert_close(new_q[:c], old_q[:c], atol=0, rtol=0)
        torch.testing.assert_close(new_q[buffers.ctx_rows:buffers.ctx_rows+n], old_q[c:], atol=0, rtol=0)
        # FA2's padded and unpadded plans may select different tiling. The
        # observed difference originates inside the context wrapper, not RoPE
        # or merge (one BF16 step); allow one relative BF16 epsilon here. The
        # same prepared plan's eager/replay comparison above stays bit-exact,
        # and full-model tests separately enforce the reference-relative gates.
        torch.testing.assert_close(actual[:n], expected, atol=2e-3, rtol=torch.finfo(dtype).eps)
        assert torch.count_nonzero(actual[n:]) == 0
        torch.testing.assert_close(buffers.meta.pad_slot.sort().values,
                                   torch.arange(rows*(depth+1), device=device), atol=0, rtol=0)
        changed = torch.zeros(64*page, device=device, dtype=torch.bool)
        changed[batch.out_loc[:n].long()] = True
        torch.testing.assert_close(keys.view(-1, hkv, dim)[~changed],
                                   before_k.view(-1, hkv, dim)[~changed], atol=0, rtol=0)
        torch.testing.assert_close(values.view(-1, hkv, dim)[~changed],
                                   before_v.view(-1, hkv, dim)[~changed], atol=0, rtol=0)
