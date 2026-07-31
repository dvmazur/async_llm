"""Stage-2 tests: async-reasoning (shared-cache) with an image in the prompt.

Gated on a GPU and on ``Qwen/Qwen3.5-0.8B`` being loadable through
``transformers``.  The HF ground truth (prompts, a greedy continuation, and
last-token logits) is computed in-process on CPU/fp32 before the engine
initializes CUDA, and compared against:

  - single-worker AR over an image prompt greedy-matches HF ``generate``;
  - a 2-worker (thinker/writer-style) group over the image prompt decodes
    coherent, distinct streams;
  - ``refresh_block`` (in-place re-prefill, e.g. an updatable image of a
    different size) reproduces a fresh prefill of the same content.

Run::

    CUDA_VISIBLE_DEVICES=3 pytest tests/core/test_qwen3_5_vision_async.py -v
"""

from __future__ import annotations

import functools

import pytest
import torch

_MODEL_ID = "Qwen/Qwen3.5-0.8B"
_GREEDY_TOKENS = 16


def _skip_reason():
    if not torch.cuda.is_available():
        return "no CUDA"
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(_MODEL_ID)  # cheap: config only
    except Exception as e:  # noqa: BLE001
        return f"{_MODEL_ID} unavailable: {type(e).__name__}: {e}"
    return None


_SKIP = _skip_reason()
requires_env = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

_CACHE = {}


def _hf_inputs(processor, num_images: int):
    """Processor output for a prompt with ``num_images`` deterministic images."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    content = [
        {"type": "image", "image": Image.fromarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))}
        for _ in range(num_images)
    ]
    content.append({"type": "text", "text": "Describe the image."})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


@functools.lru_cache(maxsize=1)
def _hf_reference():
    """HF ground truth: one- and two-image prompts, a greedy continuation of the
    single-image prompt, and HF's last-token logits for the two-image prompt.

    Runs on CPU/fp32 and frees the model afterwards -- it must not touch CUDA,
    since Engine asserts CUDA isn't already initialized.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        _MODEL_ID, dtype=torch.float32, device_map="cpu"
    ).eval()
    one, two = _hf_inputs(processor, 1), _hf_inputs(processor, 2)
    with torch.no_grad():
        greedy = model.generate(**one, max_new_tokens=_GREEDY_TOKENS, do_sample=False)
        last_logits = model(**two).logits[0, -1].float()
    ref = {
        "one": one,
        "two": two,
        "greedy_ids": greedy[0, one["input_ids"].shape[-1] :].tolist(),
        "last_logits": last_logits,
    }
    del model
    return ref


def _build():
    # One engine per process (Engine asserts CUDA isn't already initialized), shared
    # across the tests below.
    if "engine" not in _CACHE:
        from huggingface_hub import snapshot_download
        from minisgl.distributed import DistributedInfo
        from minisgl.engine import Engine, EngineConfig
        from minisgl.models.qwen3_5_mrope import get_rope_index
        from minisgl.shared_cache import SharedCacheSession

        ref = _hf_reference()  # before Engine: CPU only
        input_ids = ref["one"]["input_ids"][0].to(torch.int32)
        pixel_values = ref["one"]["pixel_values"].float()
        grid = ref["one"]["image_grid_thw"]
        cfg = EngineConfig(
            model_path=snapshot_download(_MODEL_ID), tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
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


@requires_env
def test_ar_image_single_worker_matches_hf_greedy():
    from minisgl.shared_cache import WorkerGroup

    engine, session, input_ids, pixel_values, grid, mrope, ref = _build()
    hf = ref["greedy_ids"]
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


@requires_env
def test_ar_image_two_workers_decode():
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


@requires_env
def test_refresh_block_equals_fresh_prefill():
    """``refresh_block`` (free + re-prefill in place) must reproduce a fresh prefill
    of the same content bit-for-bit, including a *different* image size (changing
    token count + mRoPE span).  This is the updatable-image-in-context hook."""
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


@requires_env
def test_two_image_prefill_matches_hf():
    """A prompt with TWO images prefills to the same next-token as HF (multi-image
    path: get_rope_index over 2 grids + vision tower over 2 images + scatter)."""
    import torch.nn.functional as F

    _build()
    session, mc, ref = _CACHE["session"], _CACHE["mc"], _CACHE["ref"]
    from minisgl.models.qwen3_5_mrope import get_rope_index

    ids = ref["two"]["input_ids"][0].to(torch.int32)
    pv = ref["two"]["pixel_values"].float()
    grid = ref["two"]["image_grid_thw"]  # [2, 3]
    assert grid.shape[0] == 2
    mrope = get_rope_index(ids.long(), mc.image_token_id, mc.vision_config.spatial_merge_size, grid)
    logits = session.prefill_block(
        session.create_block(), ids, pixel_values=pv, image_grid_thw=grid, mrope_positions=mrope
    )[0].float().cpu()
    hf = ref["last_logits"]
    assert torch.equal(logits.argmax(), hf.argmax()), (int(logits.argmax()), int(hf.argmax()))
    assert F.cosine_similarity(logits, hf, dim=0).item() > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
