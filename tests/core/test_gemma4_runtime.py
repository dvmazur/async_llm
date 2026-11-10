"""Eager attention resource/dispatch contracts, with independent dense oracles."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.gemma4 import Gemma4AttentionBackend, GemmaMetadata, Segment
from minisgl.core import Batch
from minisgl.kvcache import create_kvcache_pool
from test_gemma4_shared_cache import session_for
from test_gemma4_text import models as models
from test_gemma4_vision import inputs as image_inputs
from test_gemma4_vision import models as _vision_models
from torch.utils._python_dispatch import TorchDispatchMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
vision_models = _vision_models


def paged_case(actual, ctx, layer_id, decode, image=False):
    """Three ragged workers, different window offsets, real 16-token pages."""
    ctx.page_size = 16
    ctx.kv_cache = create_kvcache_pool(actual.config, 64, 16, torch.bfloat16, torch.device("cuda"))
    ctx.attn_backend = backend = Gemma4AttentionBackend(actual.config)
    layer = actual.model.layers[layer_id].self_attn
    layer.sliding_window = 19 if layer.is_sliding else None
    backend.text.use_bidirectional_attention = "vision"
    hkv, dim = ctx.kv_cache.k_cache(layer_id).shape[-2:]
    views, sources, qs, ks, vs, positions, outloc, all_types = [], [], [], [], [], [], [], []
    start, page = 0, 0
    for wi, (n1, n2, prior, count) in enumerate(((17, 5, 3, 9), (31, 7, 5, 7), (21, 13, 2, 11))):
        count = 1 if decode else count
        q = torch.randn(count, 4, dim, device="cuda", dtype=torch.bfloat16) * 0.15
        parts, data, origin = [], [], 0
        for length in (n1, n2, prior + count):
            pages = torch.arange(page, page + (length + 15) // 16, dtype=torch.int32)
            page += pages.numel()
            part = Segment(pages, origin, page_size=16, length=length)
            raw = torch.randn(length, hkv, dim, device="cuda", dtype=torch.bfloat16) * 0.15
            value = torch.randn_like(raw)
            local = torch.arange(length, device="cuda")
            ctx.kv_cache.store_kv(
                backend._rope(raw, local, layer.layer_type).flatten(1),
                value.flatten(1),
                part.slots.cuda().int(),
                layer_id,
            )
            parts.append(part)
            data.append((raw, value))
            origin += length
        views.append(backend._view(start, count, parts))
        sources.append(data)
        qs.append(q)
        ks.append(data[-1][0][-count:])
        vs.append(data[-1][1][-count:])
        positions.append(torch.arange(prior, prior + count, device="cuda"))
        outloc.append(parts[-1].slots[-count:].cuda().int())
        types = torch.zeros(count, dtype=torch.int64)
        if image and count > 3:
            types[1:-1] = 1
        all_types.append(types)
        start += count
    batch = SimpleNamespace(
        attn_metadata=GemmaMetadata(views, decode=decode),
        positions=torch.cat(positions),
        out_loc=torch.cat(outloc),
        mm_token_type_ids=torch.cat(all_types).cuda(),
    )
    backend.prepare_inputs(batch, types_cpu=torch.cat(all_types))
    q, k, v = map(torch.cat, (qs, ks, vs))
    expected = []
    for view, query, source, types in zip(views, qs, sources, all_types):
        qp = torch.arange(view.query_start, view.query_start + view.count, device="cuda")
        scores, values = [], []
        for part, (raw, value) in zip(view.segments, source):
            rq = backend._rope(query, qp - part.origin, layer.layer_type).float().transpose(0, 1)
            rk = backend._rope(raw, torch.arange(len(raw), device="cuda"), layer.layer_type)
            rk = rk.repeat_interleave(4 // hkv, 1).float().transpose(0, 1)
            scores.append(rq @ rk.transpose(1, 2))
            values.append(value.repeat_interleave(4 // hkv, 1).float().transpose(0, 1))
        kp = torch.arange(view.num_tokens, device="cuda")
        allowed = kp[None] <= qp[:, None]
        if layer.is_sliding:
            if image:
                is_image = types.cuda() == 1
                allowed[:, -view.count :] |= is_image[:, None] & is_image[None]
            allowed &= kp[None] > qp[:, None] - layer.sliding_window
        score = torch.cat(scores, -1).masked_fill(~allowed, -torch.inf)
        expected.append((score.softmax(-1) @ torch.cat(values, 1)).transpose(0, 1))
    return backend, layer, batch, (q, k, v), torch.cat(expected)


@pytest.mark.parametrize("layer_id", [0, 1])
@pytest.mark.parametrize("decode,image", [(False, False), (False, True), (True, False)])
def test_ragged_page16_matches_dense_and_bounds_calls(models, monkeypatch, layer_id, decode, image):
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    if decode:
        init = BatchDecodeWithPagedKVCacheWrapper.__init__

        def checked(*args, **kwargs):
            assert kwargs["use_tensor_cores"] is True
            return init(*args, **kwargs)

        monkeypatch.setattr(BatchDecodeWithPagedKVCacheWrapper, "__init__", checked)
    _, actual, ctx = models
    torch.manual_seed(661)
    backend, layer, batch, inputs, expected = paged_case(actual, ctx, layer_id, decode, image)
    with torch.inference_mode():
        got = backend.forward(*inputs, layer_id, batch, layer=layer)
    torch.testing.assert_close(got.float(), expected, atol=0.012, rtol=0.015)
    plans, _ = batch.attn_metadata.plans[layer.layer_type]
    assert len(plans) <= (2 if decode or not layer.is_sliding else 3)
    assert any(p.page_size == 16 for p in plans)
    for p in plans:
        assert p.indices.dtype == torch.int32
        assert p.indices.numel() == sum(
            (a.kv_len + p.page_size - 1) // p.page_size for a in p.parts
        )
        if decode:
            assert isinstance(p.slot.wrapper, BatchDecodeWithPagedKVCacheWrapper)
            if p.page_size == 1:
                assert all(a.kv_len <= 15 for a in p.parts)
        if p.custom_mask is not None:
            assert p.custom_mask.numel() < len(p.parts) * 19 * (19 + 16)


def test_wrappers_rebind_old_batch_and_reuse_across_streams(models):
    from dataclasses import replace

    _, actual, ctx = models
    backend, layer, batch, inputs, _ = paged_case(actual, ctx, 0, False, True)

    def run(b):
        return backend.forward(*inputs, 0, b, layer=layer)

    with torch.inference_mode():
        expected = run(batch).clone()
        identities = {k: id(v.wrapper) for k, v in backend.wrappers.items()}
        workspace = backend.workspace.data_ptr()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for i in range(8):
            # Different offsets used to create one wrapper per window offset.
            views = []
            for view in batch.attn_metadata.views:
                first, middle, last = view.segments
                shift = i % 3
                middle = middle._replace(length=middle.num_tokens - shift)
                last = last._replace(origin=last.origin - shift)
                views.append(
                    view._replace(
                        query_start=view.query_start - shift, segments=[first, middle, last]
                    )
                )
            other = SimpleNamespace(**vars(batch))
            other.attn_metadata = replace(batch.attn_metadata, views=views, plans={})
            with torch.cuda.stream(stream):
                run(other)
            result = run(batch)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            assert {k: id(v.wrapper) for k, v in backend.wrappers.items()} == identities
            assert backend.workspace.data_ptr() == workspace


def test_no_unused_hf_rope_and_exact_text_prelude_equivalence(models, monkeypatch):
    _, actual, ctx = models
    ids = torch.arange(3, 12, dtype=torch.int32)
    req = SimpleNamespace(input_ids=ids, device_len=len(ids), cached_len=0, table_idx=0)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids, batch.positions = ids.cuda(), torch.arange(len(ids), device="cuda")
    batch.out_loc = ctx.page_table[0, : len(ids)]
    ctx.attn_backend.prepare_metadata(batch)
    with torch.inference_mode(), ctx.forward_batch(batch):
        # Old adapter: call the actual HF TextModel, whose layers still delegate
        # to our attention. This is not a copied Torch reference implementation.
        output = actual.model(
            input_ids=batch.input_ids[None].long(),
            position_ids=batch.positions[None],
            use_cache=False,
        )
        expected = actual._hf.lm_head(output.last_hidden_state[:, -1:])
        cap = actual.model.config.final_logit_softcapping
        if cap is not None:
            expected = (expected / cap).tanh() * cap
        monkeypatch.setattr(
            actual.model.rotary_emb,
            "forward",
            lambda *a, **k: pytest.fail("unused HF rotary prelude called"),
        )
        got = actual.forward()
    torch.testing.assert_close(got, expected[0], atol=0, rtol=0)


class NoGpuReadback(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if args and isinstance(args[0], torch.Tensor) and args[0].is_cuda:
            assert func != torch.ops.aten._local_scalar_dense.default
            if func == torch.ops.aten._to_copy.default:
                device = kwargs.get("device")
                assert device is None or torch.device(device).type != "cpu"
        return func(*args, **kwargs)


def test_public_image_prefill_prepares_cpu_metadata_without_readback(vision_models, monkeypatch):
    _, actual, ctx = vision_models
    data = image_inputs(2)
    tokens, types = data["input_ids"][0].cpu(), data["mm_token_type_ids"][0].cpu()
    embeds = actual.encode_images(data["pixel_values"], data["image_position_ids"])
    # The vision tower itself is HF code, outside the text-adapter contract.
    monkeypatch.setattr(actual, "encode_images", lambda *a, **k: embeds)
    session = session_for(actual, ctx)
    # Check the complete normal no-context path including preparation. The
    # boundary-context custom-mask planner's FI-owned readback is separate.
    with NoGpuReadback():
        result = session.prefill_block(
            session.create_block(),
            tokens,
            pixel_values=data["pixel_values"],
            image_position_ids=data["image_position_ids"],
            mm_token_type_ids=types,
        )
    assert torch.isfinite(result).all()
    assert not ctx.attn_backend._warned_gpu_inputs


def test_boundary_mask_does_not_grow_with_full_context(models):
    _, actual, ctx = models
    backend = ctx.attn_backend
    layer = actual.model.layers[0].self_attn
    layer.sliding_window = 1024
    prefix, count, page = 1_000_000, 8192, ctx.page_size
    # Metadata-only: repeated physical pages are intentional, no huge KV pool
    # or model is allocated. The boundary plan must not materialize T x T.
    parts = [
        Segment(
            torch.arange(prefix // page, dtype=torch.int32) % 64, 0, page_size=page, length=prefix
        ),
        Segment(
            torch.arange(count // page, dtype=torch.int32) % 64,
            prefix,
            page_size=page,
            length=count,
        ),
    ]
    meta = GemmaMetadata([backend._view(0, count, parts)])
    batch = SimpleNamespace(attn_metadata=meta, mm_token_type_ids=None)
    plans, _ = backend._build_plans(meta, batch, layer)
    masked = [p for p in plans if p.custom_mask is not None]
    assert len(plans) == 2 and len(masked) == 1
    assert masked[0].custom_mask.numel() < 1024 * (1024 + page)
    assert masked[0].indices.numel() <= (1024 + page - 1) // page
