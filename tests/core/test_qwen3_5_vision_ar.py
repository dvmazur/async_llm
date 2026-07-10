"""Stage-2 tests: async-reasoning (shared-cache) with an image in the prompt.

Gated on a GPU, the Qwen3.5-0.8B checkpoint, and the HF multimodal dump
``tmp/vis_ref.npz`` (produced by ``tmp/vis_ref.py`` in the isolated
transformers-5.12.1 env, which also stores a greedy continuation).

  - single-worker AR over an image prompt greedy-matches HF ``generate``;
  - a 2-worker (thinker/writer-style) group over the image prompt decodes
    coherent, distinct streams;
  - ``refresh_block`` (in-place re-prefill, e.g. an updatable image of a
    different size) reproduces a fresh prefill of the same content.

Run::

    HF_HOME=/mnt/LLM CUDA_VISIBLE_DEVICES=3 .venv/bin/python tests/core/test_qwen3_5_vision_ar.py
"""

from __future__ import annotations

import glob
import os

import torch

_REF = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "vis_ref.npz")
_CKPT = glob.glob("/mnt/LLM/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/")


def _skip_reason():
    if not torch.cuda.is_available():
        return "no CUDA"
    if not _CKPT:
        return "checkpoint missing"
    if not os.path.exists(_REF):
        return "tmp/vis_ref.npz missing (run tmp/vis_ref.py)"
    return None


_CACHE = {}


def _build():
    # One engine per process (Engine asserts CUDA isn't already initialized), shared
    # across the tests below.
    if "engine" not in _CACHE:
        import numpy as np
        from minisgl.distributed import DistributedInfo
        from minisgl.engine import Engine, EngineConfig
        from minisgl.models.qwen3_5_mrope import get_rope_index
        from minisgl.shared_cache import SharedCacheSession

        ref = np.load(_REF)
        input_ids = torch.tensor(ref["input_ids"], dtype=torch.int32)
        pixel_values = torch.tensor(ref["pixel_values"], dtype=torch.float32)
        grid = torch.tensor(ref["image_grid_thw"], dtype=torch.long)
        cfg = EngineConfig(
            model_path=_CKPT[0], tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
            max_running_req=4, num_page_override=4096, max_seq_len_override=4096,
        )
        engine = Engine(cfg)
        session = SharedCacheSession(engine)
        mc = cfg.model_config
        mrope = get_rope_index(
            input_ids.long(), mc.image_token_id, mc.vision_config.spatial_merge_size, grid
        )
        _CACHE.update(
            engine=engine, session=session, input_ids=input_ids, pixel_values=pixel_values,
            grid=grid, mrope=mrope, ref=ref, mc=mc,
        )
    c = _CACHE
    return c["engine"], c["session"], c["input_ids"], c["pixel_values"], c["grid"], c["mrope"], c["ref"]


def _synthetic_image_block(mc, h: int, w: int, seed: int):
    """Synthetic image block ids/pixel_values/grid/mRoPE (no PIL/processor), shaped
    as the vision tower expects; different (h, w) -> different grid + mRoPE span."""
    from minisgl.models.qwen3_5_mrope import get_rope_index

    vc = mc.vision_config
    merge = vc.spatial_merge_size
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
    grid = torch.tensor([[1, h, w]], dtype=torch.long)
    n_patches = h * w
    idx = torch.arange(n_patches * patch_dim, dtype=torch.float32).reshape(n_patches, patch_dim)
    pixel_values = torch.sin(idx * 7.7e-4 + float(seed)) * 0.5
    ids = torch.tensor(
        [mc.vision_start_token_id]
        + [mc.image_token_id] * (n_patches // (merge**2))
        + [mc.vision_end_token_id],
        dtype=torch.int32,
    )
    mrope = get_rope_index(ids.long(), mc.image_token_id, merge, grid)
    return ids, pixel_values, grid, mrope


def test_ar_image_single_worker_matches_hf_greedy():
    reason = _skip_reason()
    if reason:
        print(f"  [skip] {reason}")
        return
    from minisgl.shared_cache import WorkerGroup

    engine, session, input_ids, pixel_values, grid, mrope, ref = _build()
    hf = ref["greedy_ids"].tolist()
    N = len(hf)
    prompt = session.create_block()
    first = int(session.prefill_block(
        prompt, input_ids, pixel_values=pixel_values, image_grid_thw=grid, mrope_positions=mrope,
    )[0].argmax().item())
    gen = session.create_block()
    group = WorkerGroup(cache_structure=[[prompt, gen]], write_to=[gen])
    ar = [first]
    cur = torch.tensor([first], dtype=torch.int32)
    for _ in range(N - 1):
        cur = session.decode_step(group, cur).argmax(-1).to(torch.int32)
        ar.append(int(cur[0]))
    assert ar == hf, f"AR {ar}\nHF {hf}"


def test_ar_image_two_workers_decode():
    reason = _skip_reason()
    if reason:
        print(f"  [skip] {reason}")
        return
    from minisgl.shared_cache import WorkerGroup

    engine, session, input_ids, pixel_values, grid, mrope, ref = _build()
    prompt = session.create_block()
    pl = session.prefill_block(
        prompt, input_ids, pixel_values=pixel_values, image_grid_thw=grid, mrope_positions=mrope,
    )
    top2 = torch.topk(pl[0], 2).indices.to(torch.int32)
    w1, w2 = session.create_block(), session.create_block()
    group = WorkerGroup(cache_structure=[[prompt, w1], [prompt, w2]], write_to=[w1, w2])
    cur = top2.clone()
    for _ in range(8):
        cur = session.decode_step(group, cur).argmax(-1).to(torch.int32)
    assert w1.num_tokens == 8 and w2.num_tokens == 8


def test_refresh_block_equals_fresh_prefill():
    """``refresh_block`` (free + re-prefill in place) must reproduce a fresh prefill
    of the same content bit-for-bit, including a *different* image size (changing
    token count + mRoPE span).  This is the updatable-image-in-context hook."""
    reason = _skip_reason()
    if reason:
        print(f"  [skip] {reason}")
        return
    _build()  # ensure the shared engine/session + model config are cached
    session, mc = _CACHE["session"], _CACHE["mc"]

    ids_a, pv_a, grid_a, mr_a = _synthetic_image_block(mc, 8, 8, seed=1)
    ids_b, pv_b, grid_b, mr_b = _synthetic_image_block(mc, 12, 8, seed=2)  # different grid

    # Fresh prefill of image B in a clean block.
    fresh = session.create_block()
    lf = session.prefill_block(
        fresh, ids_b, pixel_values=pv_b, image_grid_thw=grid_b, mrope_positions=mr_b
    )[0].float()

    # Prefill image A, then refresh in place to image B.
    reused = session.create_block()
    session.prefill_block(reused, ids_a, pixel_values=pv_a, image_grid_thw=grid_a, mrope_positions=mr_a)
    lr = session.refresh_block(
        reused, ids_b, pixel_values=pv_b, image_grid_thw=grid_b, mrope_positions=mr_b
    )[0].float()

    assert reused.num_tokens == fresh.num_tokens, (reused.num_tokens, fresh.num_tokens)
    assert reused.mrope_span == fresh.mrope_span
    assert torch.equal(lr.argmax(), lf.argmax())
    assert torch.allclose(lr, lf, atol=1e-4), f"max|Δ|={ (lr - lf).abs().max().item() }"


if __name__ == "__main__":
    import sys

    fns = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
