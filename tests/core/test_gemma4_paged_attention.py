"""Geometry and direct GPU checks for paged Gemma, independently of logits."""
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from minisgl.attention.gemma4 import (
    Gemma4AttentionBackend, GemmaMetadata, Segment, attention_ranges,
)
from test_gemma4_text import models


def test_ranges_exactly_cover_dense_causal_window_mask():
    for qa, nq, ka, nk, causal, window in product(
            (-1, 0, 3, 7), (1, 3, 7), (-1, 0, 2, 8), (0, 1, 4, 11),
            (False, True), (None, 1, 4, 8)):
        expected = {(q, k) for q in range(qa, qa+nq) for k in range(ka, ka+nk)
                    if (not causal or k <= q) and (window is None or k > q-window)}
        actual = set()
        for a, b, c, d, use_causal, left in attention_ranges(
                qa, qa+nq, ka, ka+nk, causal=causal, window=window):
            assert left >= 0 or left == -1
            for q in range(a, b):
                aligned_q = d - (b-q)
                for k in range(c, d):
                    if (not use_causal or k <= aligned_q) and (left < 0 or k >= aligned_q-left):
                        assert (q, k) not in actual
                        actual.add((q, k))
        assert actual == expected, (qa, nq, ka, nk, causal, window)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('image', [False, True])
@pytest.mark.parametrize('layer_id', [0, 1])
@pytest.mark.parametrize('magnitude', [.1, 1.])
def test_paged_outputs_match_dense_reference_and_never_gather_kv(models, monkeypatch, image, layer_id, magnitude):
    from torch.utils._python_dispatch import TorchDispatchMode
    from transformers.models.gemma4.modeling_gemma4 import get_block_sequence_ids_for_mask
    _, model, ctx = models
    backend = ctx.attn_backend
    layer = model.model.layers[layer_id].self_attn
    # Noncontiguous physical pages, a repeated block, and queries inside an
    # image group. Keep source keys raw only in the independent reference.
    hkv, dim = ctx.kv_cache.k_cache(layer_id).shape[-2:]
    torch.manual_seed(517)
    q = torch.randn(13, 4, dim, device='cuda', dtype=torch.bfloat16) * magnitude
    raw = torch.randn(19, hkv, dim, device='cuda', dtype=torch.bfloat16) * magnitude
    val = torch.randn_like(raw)
    prefix_slots = torch.tensor([31, 4, 18, 77, 56, 91], device='cuda', dtype=torch.int32)
    write_slots = torch.tensor([103, 60, 2, 112, 113, 29, 37, 52, 53, 82, 95, 99, 101],
                               device='cuda', dtype=torch.int32)
    ctx.kv_cache.store_kv(backend._rope(raw[:6], torch.arange(6, device='cuda'), layer.layer_type).flatten(1),
                          val[:6].flatten(1), prefix_slots, layer_id)
    parts = [Segment(prefix_slots, 0), Segment(prefix_slots, 6), Segment(write_slots, 12)]
    meta = GemmaMetadata([backend._view(0, 13, parts)])
    types = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2], device='cuda') if image else None
    batch = SimpleNamespace(attn_metadata=meta, positions=torch.arange(13, device='cuda'),
                            out_loc=write_slots, mm_token_type_ids=types)
    if image:
        backend.text.use_bidirectional_attention = 'vision'
    pool_pointers = {f(layer_id).untyped_storage().data_ptr()
                     for f in (ctx.kv_cache.k_cache, ctx.kv_cache.v_cache)}
    class NoKVGather(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.index.Tensor, torch.ops.aten.index_select.default):
                assert args[0].untyped_storage().data_ptr() not in pool_pointers
            return func(*args, **(kwargs or {}))
    # Guard against quietly routing BF16 to the diagnostic implementation.
    monkeypatch.setattr(backend, '_reference', lambda *a: pytest.fail('BF16 used fallback'))
    with torch.inference_mode(), NoKVGather():
        got = backend.forward(q, raw[6:], val[6:], layer_id, batch, layer=layer)
    values = torch.cat((val[:6], val[:6], val[6:]))
    qp, kp = torch.arange(12, 25, device='cuda'), torch.arange(25, device='cuda')
    # Independent dense dot over the SAME BF16 stored keys/query frames.
    # This isolates page addressing/window/merge. Comparing a different RoPE
    # frame after BF16 rounding belongs to the unchanged HF model-level tests.
    scores = []
    repeats = q.shape[1] // hkv
    for origin, source in ((0, raw[:6]), (6, raw[:6]), (12, raw[6:])):
        query = backend._rope(q, qp-origin, layer.layer_type).transpose(0, 1).float()
        keys = backend._rope(source, torch.arange(len(source), device='cuda'), layer.layer_type)
        keys = keys.repeat_interleave(repeats, 1).transpose(0, 1).float()
        scores.append(query @ keys.transpose(1, 2))
    allowed = kp[None] <= qp[:, None]
    if layer.is_sliding:
        if image:
            groups = get_block_sequence_ids_for_mask(types[None], ctx.kv_cache.device)[0]
            allowed[:, -13:] |= (groups[:, None] == groups[None]) & (groups[:, None] >= 0)
        allowed &= kp[None] > qp[:, None] - layer.sliding_window
    scores = torch.cat(scores, -1).masked_fill(~allowed, -torch.inf)
    values = values.repeat_interleave(repeats, 1).transpose(0, 1).float()
    expected = (scores.softmax(-1) @ values).transpose(0, 1)
    torch.testing.assert_close(got.float(), expected, atol=.012, rtol=.015)
    assert meta.plans[layer.layer_type][0]


def test_rejects_non_translation_invariant_rope(models):
    _, model, _ = models
    model.config.hf_config.rope_parameters['full_attention']['rope_type'] = 'dynamic'
    with pytest.raises(ValueError, match='default/proportional'):
        Gemma4AttentionBackend(model.config)
