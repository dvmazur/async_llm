"""Test Qwen3_5 async engine (shared-cache) with an image in the prompt.

Gated on a GPU and on ``Qwen/Qwen3.5-0.8B`` being loadable through
``transformers``.  The HF ground truth (prompts, a greedy continuation, and
last-token logits) is computed in-process in fp32 on CUDA if available (else
CPU) before the engine initializes its KV pool -- the reference model is then
freed -- and compared against:

  - single-worker AR over an image prompt greedy-matches HF ``generate``;
  - a 2-worker (thinker/writer-style) group over the image prompt decodes coherent, distinct streams;

Run::

    CUDA_VISIBLE_DEVICES=3 pytest tests/core/test_qwen3_5_vision_async.py -v
"""

from __future__ import annotations

import asyncio
import functools

import pytest
import torch
import torch.nn.functional as F
import transformers

from huggingface_hub import snapshot_download
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.llm import AsyncLLM
from minisgl.scheduler import AsyncCacheEngine
from minisgl.shared_cache import SharedCacheSession
from minisgl.shared_cache import WorkerGroup

from test_qwen3_5_vision_encoder import _make_hf_model, _make_hf_inputs, _DEVICE


_MODEL_ID = "Qwen/Qwen3.5-0.8B"
_GREEDY_TOKENS = 10


@functools.lru_cache(maxsize=1)
def _make_hf_reference():
    """HF ground truth: one- and two-image prompts, a greedy continuation of the
    single-image prompt, HF's last-token logits for the two-image prompt and incremental encoding.

    Returns CPU tensors; the reference model is released before we return so its
    fp32 weights are not resident when the engine sizes its KV pool.
    """
    model, processor = _make_hf_model()
    one, two = _make_hf_inputs(processor, 1), _make_hf_inputs(processor, 2)
    follow_up_1 = processor.apply_chat_template(
        [{"role": "user", "content": "How many unique colours are there on the previous image?"}],
        add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt",
    )
    follow_up_2 = _make_hf_inputs(processor, 1, text="What is the difference?", add_generation_prompt=True)
    follow_up_1.pop("attention_mask"); follow_up_2.pop("attention_mask")  # attention_mask confuses HF with cache
    with torch.no_grad():
        greedy = model.generate(**one.to(_DEVICE), max_new_tokens=_GREEDY_TOKENS, do_sample=False)
        last_logits_two_images = model(**two.to(_DEVICE)).logits[0, -1].float().cpu()

        # incremental conditional forward testing scenario: image+text -> text-only -> another image+text
        cache = transformers.DynamicCache(config=model.config)
        model(**one, use_cache=True, past_key_values=cache).logits[0, -1].float().cpu()
        last_logits_ckpt_1 = model(**follow_up_1.to(_DEVICE), use_cache=True, past_key_values=cache
                                   ).logits[0, -1].float().cpu()
        last_logits_ckpt_2 = model(**follow_up_2.to(_DEVICE), use_cache=True, past_key_values=cache
                                   ).logits[0, -1].float().cpu()
    return dict(processor=processor, one=one, two=two,
                greedy_ids=greedy[0, one["input_ids"].shape[-1] :].tolist(),
                last_logits_two_images=last_logits_two_images,
                follow_up_1=follow_up_1, follow_up_2=follow_up_2,
                last_logits_ckpt_1=last_logits_ckpt_1, last_logits_ckpt_2=last_logits_ckpt_2)


@functools.lru_cache(maxsize=1)
def _build_async_engine():
    engine_config = EngineConfig(
        model_path=snapshot_download(_MODEL_ID), tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16, max_running_req=4, num_page_override=4096, max_seq_len_override=4096,
    )

    engine = Engine(engine_config)  # Engine asserts that cuda is not initialized, so it is created first
    ref = _make_hf_reference()  # compute reference model after engine is already initialized
    session = SharedCacheSession(engine)
    config = engine_config.model_config
    async_engine = AsyncCacheEngine(engine, session=session)
    llm = AsyncLLM(_MODEL_ID, engine=engine, async_engine=async_engine, dtype=torch.bfloat16)
    return llm, engine, session, ref, config


def _make_synthetic_image_block(config, h: int, w: int, seed: int):
    """Synthetic image block ids/pixel_values/grid/mRoPE (no PIL/processor), shaped
    as the vision tower expects; different (h, w) -> different grid + mRoPE span."""
    vision_config = config.vision_config
    merge = vision_config.spatial_merge_size
    patch_dim = vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2
    grid = torch.tensor([[1, h, w]], dtype=torch.long)
    n_patches = h * w
    idx = torch.arange(n_patches * patch_dim, dtype=torch.float32).reshape(n_patches, patch_dim)
    pixel_values = torch.sin(idx * 7.7e-4 + float(seed)) * 0.5
    ids = torch.tensor(
        [config.vision_start_token_id]
        + [config.image_token_id] * (n_patches // (merge ** 2))
        + [config.vision_end_token_id],
        dtype=torch.int32,
    )
    mm_token_type_ids = torch.where(ids.long() == config.image_token_id, 1, 0)  # 0 = text, 1 = image, 2 = ...
    return ids, pixel_values, grid, mm_token_type_ids


def test_ar_image_single_worker_matches_hf_greedy():
    _, engine, session, ref, mc = _build_async_engine()
    input_ids = ref["one"]["input_ids"][0].to(torch.int32)
    mm_token_type_ids = ref["one"]["mm_token_type_ids"][0]
    pixel_values = ref["one"]["pixel_values"].float()
    grid = ref["one"]["image_grid_thw"]

    hf_greedy_ids = ref["greedy_ids"]
    N = len(hf_greedy_ids)
    prompt = session.create_block()
    first = int(session.prefill_block(
        prompt, input_ids, mm_token_type_ids=mm_token_type_ids, pixel_values=pixel_values, image_grid_thw=grid,
    )[0].argmax().item())
    gen = session.create_block()
    group = WorkerGroup(cache_structure=[[prompt, gen]], write_to=[gen])
    ar_greedy_ids = [first]
    cur = torch.tensor([first], dtype=torch.int32)
    for _ in range(N - 1):
        cur = session.decode_step(group, cur).argmax(-1).to(torch.int32)
        ar_greedy_ids.append(int(cur[0]))
    assert ar_greedy_ids == hf_greedy_ids, f"AR {ar_greedy_ids}\nHF {hf_greedy_ids}"


def test_ar_image_two_workers_decode():
    _, engine, session, ref, mc = _build_async_engine()
    input_ids = ref["one"]["input_ids"][0].to(torch.int32)
    mm_token_type_ids = ref["one"]["mm_token_type_ids"][0]
    pixel_values = ref["one"]["pixel_values"].float()
    grid = ref["one"]["image_grid_thw"]
    prompt = session.create_block()
    pl = session.prefill_block(
        prompt, input_ids, mm_token_type_ids=mm_token_type_ids, pixel_values=pixel_values, image_grid_thw=grid,
    )
    top2 = torch.topk(pl[0], 2).indices.to(torch.int32)
    w1, w2 = session.create_block(), session.create_block()
    group = WorkerGroup(cache_structure=[[prompt, w1], [prompt, w2]], write_to=[w1, w2])
    cur = top2.clone()
    for _ in range(8):
        cur = session.decode_step(group, cur).argmax(-1).to(torch.int32)
    assert w1.num_tokens == 8 and w2.num_tokens == 8


def test_repeated_block_prefill_equals_fresh_prefill():
    """Prefill after clear must reproduce a fresh prefill
    of the same content bit-for-bit, including a *different* image size (changing
    token count + mRoPE span).  This is the updatable-image-in-context hook."""
    _, engine, session, ref, mc = _build_async_engine()
    ids_a, pv_a, grid_a, mm_a = _make_synthetic_image_block(mc, 8, 8, seed=1)
    ids_b, pv_b, grid_b, mm_b = _make_synthetic_image_block(mc, 12, 8, seed=2)  # different grid

    # Fresh prefill of image B in a clean block.
    fresh = session.create_block()
    lf = session.prefill_block(
        fresh, ids_b, mm_token_type_ids=mm_b, pixel_values=pv_b, image_grid_thw=grid_b,
    )[0].float()

    # Prefill image A, then refresh (clear-prefill) in place to image B.
    reused = session.create_block()
    session.prefill_block(reused, ids_a, mm_token_type_ids=mm_a, pixel_values=pv_a, image_grid_thw=grid_a)

    session.free_block(reused)  # clear() (keeps the object) + return pages
    lr = session.prefill_block(
        reused, ids_b, mm_token_type_ids=mm_b, pixel_values=pv_b, image_grid_thw=grid_b
    )[0].float()
    assert reused.num_tokens == fresh.num_tokens, (reused.num_tokens, fresh.num_tokens)
    assert reused.mrope_span == fresh.mrope_span
    assert torch.equal(lr.argmax(), lf.argmax())
    assert torch.allclose(lr, lf, atol=1e-4), f"max|Δ|={ (lr - lf).abs().max().item() }"


def test_two_image_prefill_matches_hf():
    """A prompt with TWO images prefills to the same next-token as HF (multi-image
    path: get_rope_index over 2 grids + vision tower over 2 images + scatter)."""
    llm, engine, session, ref, mc = _build_async_engine()
    ids = ref["two"]["input_ids"][0].to(torch.int32)
    mm_token_type_ids = ref["two"]["mm_token_type_ids"][0]
    pv = ref["two"]["pixel_values"].float()
    grid = ref["two"]["image_grid_thw"]  # [2, 3]
    assert grid.shape[0] == 2
    async def _compute_logits():
        block = await llm.create_block()
        return await llm.forward(ids, [block], mm_token_type_ids=mm_token_type_ids, pixel_values=pv,
                                 image_grid_thw=grid, write_to=block)
    logits = asyncio.run(_compute_logits()).logits.float().cpu()
    hf = ref["last_logits_two_images"]
    assert torch.equal(logits.argmax(), hf.argmax()), (int(logits.argmax()), int(hf.argmax()))
    assert F.cosine_similarity(logits, hf, dim=0).item() > 0.99


def test_contextual_image_prefill_matches_hf():
    """Encode several conversation turns in consecutive blocks and via append, check against incremental HF forward"""
    llm, engine, session, ref, mc = _build_async_engine()
    async def _compute_logits():
        # stage 0: encode the common prompt; it is not tested directly, but it affects all subsequent checkpoints
        block_A, block_B, block_C, block_D = await asyncio.gather(*(llm.create_block() for _ in range(4)))
        await llm.forward(ref["one"]["input_ids"][0].to(torch.int32), mm_token_type_ids=ref["one"]["mm_token_type_ids"][0],
                          pixel_values=ref["one"]["pixel_values"].float(), image_grid_thw=ref["one"]["image_grid_thw"],
                          cache_view=[block_A], write_to=block_A, return_logits=False);

        ckpt_1 = await llm.forward(  # stage 1: non-image conditioned on image
            ref['follow_up_1']['input_ids'][0], cache_view=[block_A, block_B], write_to=block_B, return_logits=True)
        assert block_B.num_tokens == len(ref['follow_up_1']['input_ids'][0])

        ckpt_1_control = await llm.forward(  # stage 1 control group: without condition, should NOT be close enough
            ref['follow_up_1']['input_ids'][0], cache_view=[block_C], write_to=block_C, return_logits=True)
        assert block_C.num_tokens == len(ref['follow_up_1']['input_ids'][0])

        # ckpt_2_separate = await llm.forward(  # stage 2: image conditioned on stage 1, write to new block
        #     ref['follow_up_2']['input_ids'][0], mm_token_type_ids=ref["follow_up_1"]["mm_token_type_ids"][0],
        #     pixel_values=ref["follow_up_2"]["pixel_values"].float(), image_grid_thw=ref["follow_up_2"]["image_grid_thw"],
        #     cache_view=[block_A, block_B, block_D], write_to=block_D, return_logits=True)
        #
        # ckpt_2_appended = await llm.forward(  # stage 2 alternative method: append new KVs to block B
        #     ref['follow_up_2']['input_ids'][0], mm_token_type_ids=ref["follow_up_1"]["mm_token_type_ids"][0],
        #     pixel_values=ref["follow_up_2"]["pixel_values"].float(), image_grid_thw=ref["follow_up_2"]["image_grid_thw"],
        #     cache_view=[block_A, block_B], write_to=block_B, return_logits=True)
        # assert block_B.num_tokens == len(ref['follow_up_1']['input_ids'][0]) + len(ref['follow_up_2']['input_ids'][0])
        ckpt_2_separate = ckpt_2_appended = ckpt_1_control  # TODO
        return [ckpt.logits.float().cpu() for ckpt in (ckpt_1, ckpt_1_control, ckpt_2_separate, ckpt_2_appended)]

    ckpt_1_logits, ckpt_1_control_logits, ckpt_2_separate_logits, ckpt_2_appended_logits = asyncio.run(_compute_logits())
    assert F.cosine_similarity(ckpt_1_logits, ref["last_logits_ckpt_1"], dim=0).item() > 0.99
    assert F.cosine_similarity(ckpt_1_control_logits, ref["last_logits_ckpt_1"], dim=0).item() < 0.99
    # assert F.cosine_similarity(ckpt_2_separate_logits, ref["ckpt_2_last_logits"], dim=0).item() > 0.99
    # assert F.cosine_similarity(ckpt_2_appended_logits, ref["ckpt_2_last_logits"], dim=0).item() > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
