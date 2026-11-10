"""Image/window intersections: exact geometry, then small-model HF oracle.

FP32 small weights isolate masking/indexing from BF16 row-shape rounding.
The corresponding learned FP8 panel runs separately against vanilla SG and HF.
"""
import copy

import pytest
import torch

from test_gemma4_vision import models as vision_models, inputs
from test_gemma4_shared_cache import session_for
from tests.e2e.fp8.gemma_images import CASES, LAYOUTS, window_cases, validate_geometry, materialize
from tests.e2e.fp8.worker import image_inputs


def test_window_fixture_geometry_rejects_off_by_one(tmp_path):
    data = dict(mm_token_type_ids=torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]]),
                pixel_values=torch.zeros(1, 36, 12), image_position_ids=torch.zeros(1, 36, 2))
    cases = window_cases(dict(prompt_ids=[3, 4, 99, 99, 99, 99, 5, 6],
                              teacher_tokens=[7, 8, 9], cold_steps=[0, 1, 2]),
                         data, [7, 8], 16, tmp_path)
    for index in (1, 2):
        broken = copy.deepcopy(cases)
        broken[index]["prompt_ids"].append(7)
        with pytest.raises(AssertionError):
            validate_geometry(broken)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vision_models", [(16, torch.float32)], indirect=True)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("case_name", CASES)
def test_image_window_layout_and_decode_matches_hf(vision_models, tmp_path, layout, case_name):
    from minisgl.shared_cache import WorkerGroup
    reference, actual, ctx = vision_models
    data = inputs(1)
    data["pixel_values"] = data["pixel_values"].float()
    cases = window_cases(dict(prompt_ids=data["input_ids"][0].tolist(),
                              teacher_tokens=list(range(7, 15)), cold_steps=[0, 3, 7]),
                         {k: v.cpu() for k, v in data.items() if k != "input_ids"},
                         [7, 8], 16, tmp_path)
    case = next(c for c in cases if c["name"] == case_name)
    session = session_for(actual, ctx)
    logits, blocks, record = materialize(session, case, case["prompt_ids"],
                                         image_inputs(case, flatten=True), layout, chunk_rows=3)
    group = WorkerGroup(cache_structure=[blocks], write_to=[blocks[-1]])
    if layout == "image-chunked":
        assert len(record["forwards"]) > 1
        assert sum(f["rows"] for f in record["forwards"]) == len(case["prompt_ids"])
    for step, token in enumerate(case["teacher_tokens"]):
        history = case["prompt_ids"] + case["teacher_tokens"][:step]
        with torch.inference_mode():
            expected = reference(input_ids=torch.tensor([history], device="cuda"),
                                 **image_inputs(case, step, device="cuda"), use_cache=False).logits[:, -1]
        torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)
        if step + 1 < len(case["teacher_tokens"]):
            logits = session.decode_step(group, torch.tensor([token], device="cuda", dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vision_models", [(16, torch.float32)], indirect=True)
def test_image_group_larger_than_window_is_not_split(vision_models):
    """Bidirectional image attention and the sliding cutoff must both apply."""
    from minisgl.shared_cache import PrefillJob, WorkerGroup
    reference, actual, ctx = vision_models
    data = inputs(1, 12, 15)  #20soft tokens >16-token window.
    data["pixel_values"] = data["pixel_values"].float()
    session = session_for(actual, ctx, cap=3)
    block = session.create_block()
    job = PrefillJob(block, data["input_ids"][0].cpu(),
                     **{k: v for k, v in data.items() if k != "input_ids"})
    chunks = session._split_job(job, 3)
    assert [int((j.mm_token_type_ids == 1).sum()) for j in chunks
            if j.mm_token_type_ids is not None and bool((j.mm_token_type_ids == 1).any())] == [20]
    logits = session.prefill_batch([job])[0]
    group = WorkerGroup(cache_structure=[[block]])
    for step in range(8):
        with torch.inference_mode():
            expected = reference(**data, use_cache=False).logits[:, -1]
        torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)
        if step < 7:
            token = torch.tensor([[7+step]], device="cuda")
            logits = session.decode_step(group, token[0].int())
            data["input_ids"] = torch.cat((data["input_ids"], token), 1)
            data["mm_token_type_ids"] = torch.cat((data["mm_token_type_ids"], token*0), 1)


def _image_runtime(models):
    reference, actual, ctx = models
    session = session_for(actual, ctx)
    data = inputs(1)
    data["pixel_values"] = data["pixel_values"].float()
    image = session.create_block()
    session.prefill_block(image, data["input_ids"][0].cpu(),
                          **{k: v for k, v in data.items() if k != "input_ids"})
    return reference, session, data, image


def _kv_snapshot(session, block):
    slots = block.token_slots_tensor().long()
    return [cache(layer).flatten(0, 1)[slots].clone()
            for layer in range(session.kv_cache.num_layers)
            for cache in (session.kv_cache.k_cache, session.kv_cache.v_cache)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vision_models", [(16, torch.float32)], indirect=True)
def test_shared_image_readers_have_independent_windows_and_batch_order(vision_models):
    from minisgl.shared_cache import WorkerGroup
    reference, session, data, image = _image_runtime(vision_models)
    histories = [list(range(10, 14)), list(range(10, 30))]
    tokens = [7, 8]
    expected = []
    for history, token in zip(histories, tokens):
        ids = torch.tensor([image.token_ids + history + [token]], device="cuda")
        types = torch.cat((data["mm_token_type_ids"], ids.new_zeros(1, len(history)+1)), 1)
        with torch.inference_mode():
            expected.append(reference(**dict(data, input_ids=ids, mm_token_type_ids=types),
                                      use_cache=False).logits[:, -1])
    for order in ([0, 1], [1, 0]):
        tails = []
        for i in order:
            tail = session.create_block()
            session.prefill_block(tail, torch.tensor(histories[i], dtype=torch.int32), context=[image])
            tails.append(tail)
        got = session.decode_step(WorkerGroup(cache_structure=[[image, tail] for tail in tails]),
                                  torch.tensor([tokens[i] for i in order], dtype=torch.int32))
        for row, i in zip(got, order):
            torch.testing.assert_close(row[None], expected[i], atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vision_models", [(16, torch.float32)], indirect=True)
@pytest.mark.parametrize("tail_length", [3, 4, 5])
def test_repeated_image_view_merge_and_self_append_preserve_same_kv(vision_models, tail_length):
    from minisgl.shared_cache import WorkerGroup
    _, session, _, image = _image_runtime(vision_models)
    text = session.create_block()
    session.prefill_block(text, torch.arange(7, 7+tail_length, dtype=torch.int32), context=[image])
    seed = session.merge_blocks(image, text)  #11/12/13tokens straddle a4-token KV page boundary.
    snapshot = _kv_snapshot(session, seed)
    merged = session.merge_blocks(seed, seed)
    appended = session.merge_blocks(seed, session.create_block())
    session.append_block(appended, appended)
    views = [[seed, session.create_block(), seed], [merged], [appended]]
    tails = [session.create_block() for _ in views]
    logits = session.decode_step(WorkerGroup(cache_structure=[view+[tail] for view, tail in zip(views, tails)]),
                                 torch.tensor([11, 11, 11], dtype=torch.int32))
    torch.testing.assert_close(logits, logits[0:1].expand_as(logits), atol=1e-4, rtol=1e-4)
    for before, after in zip(snapshot, _kv_snapshot(session, seed)):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vision_models", [(16, torch.float32)], indirect=True)
def test_second_image_in_one_branch_does_not_mutate_sibling(vision_models):
    from minisgl.shared_cache import WorkerGroup
    _, session, _, image = _image_runtime(vision_models)
    a, b = session.create_block(), session.create_block()
    session.prefill_block(a, torch.tensor([7, 8, 9], dtype=torch.int32), context=[image])
    session.prefill_block(b, torch.tensor([10, 11, 12], dtype=torch.int32), context=[image])
    control = session.merge_blocks(b, session.create_block())
    token = torch.tensor([13], dtype=torch.int32)
    expected = session.decode_step(WorkerGroup(cache_structure=[[image, control]]), token)
    snapshot = _kv_snapshot(session, image)
    second = inputs(1, 6, 9)
    second["pixel_values"] = second["pixel_values"].float()
    session.prefill_block(a, second["input_ids"][0].cpu(), context=[image],
                          **{k: v for k, v in second.items() if k != "input_ids"})
    got = session.decode_step(WorkerGroup(cache_structure=[[image, b]]), token)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
    for before, after in zip(snapshot, _kv_snapshot(session, image)):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vision_models", [(16, torch.float32)], indirect=True)
def test_image_copy_survives_source_free_and_physical_slot_reuse(vision_models):
    from minisgl.shared_cache import WorkerGroup
    _, session, data, image = _image_runtime(vision_models)
    saved = session.merge_blocks(image, session.create_block())
    tail = session.create_block()
    token = torch.tensor([7], dtype=torch.int32)
    expected = session.decode_step(WorkerGroup(cache_structure=[[saved, tail]]), token)
    session.free_block(tail)
    pages = set(image.page_starts)
    held = session.page_allocator.alloc_pages(session.page_allocator.num_free_pages)
    try:
        session.free_block(image)
        replacement = session.create_block()
        session.prefill_block(replacement, data["input_ids"][0].cpu(),
                              **dict({k: v for k, v in data.items() if k != "input_ids"},
                                     pixel_values=-data["pixel_values"]))
        assert set(replacement.page_starts) == pages, "test must actually reuse the old physical pages"
    finally:
        session.page_allocator.free_pages(held)
    got = session.decode_step(WorkerGroup(cache_structure=[[saved, session.create_block()]]), token)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vision_models", [(16, torch.float32)], indirect=True)
def test_queued_image_prefill_precedes_reader_decode(vision_models):
    from minisgl.engine.sample import Sampler
    from minisgl.scheduler import AsyncCacheEngine
    from minisgl.shared_cache import AsyncContext, WorkerGroup
    _, session, data, control = _image_runtime(vision_models)
    runner = AsyncCacheEngine(session=session, sampler=Sampler(session.device, 512))
    image, tail = session.create_block(), session.create_block()
    # Queue decode FIRST; prefill still has priority across the two queues.
    decoded = runner.submit_decode(AsyncContext([image, tail], tail), 7, return_logits=True)
    prefetched = runner.submit_prefill(data["input_ids"][0].cpu(), write_to=image, return_logits=True,
                                      **{k: v for k, v in data.items() if k != "input_ids"})
    with pytest.raises(RuntimeError, match="queued request"):
        runner.free_block(image)
    assert runner.tick() == "prefill" and prefetched.done() and not decoded.done()
    assert runner.tick() == "decode" and decoded.done()
    expected = session.decode_step(WorkerGroup(cache_structure=[[control, session.create_block()]]),
                                    torch.tensor([7], dtype=torch.int32))
    torch.testing.assert_close(decoded.result()[1], expected[0], atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vision_models", [(16, torch.float32)], indirect=True)
def test_invalid_image_batch_resolves_futures_without_partial_cache_writes(vision_models):
    from minisgl.engine.sample import Sampler
    from minisgl.scheduler import AsyncCacheEngine
    _, session, data, image = _image_runtime(vision_models)
    runner = AsyncCacheEngine(session=session, sampler=Sampler(session.device, 512))
    snapshot = _kv_snapshot(session, image)
    available = session.page_allocator.num_free_pages
    blocks = [session.create_block(), session.create_block()]
    requests = []
    for index, block in enumerate(blocks):
        ids, types = data["input_ids"][0].cpu().clone(), data["mm_token_type_ids"][0].cpu().clone()
        if index:
            ids[2], types[2] = 7, 0  #three placeholders for four actual image features.
        requests.append(runner.submit_prefill(ids, write_to=block, cache_view=[image],
                        pixel_values=data["pixel_values"], image_position_ids=data["image_position_ids"],
                        mm_token_type_ids=types, return_logits=True))
    with pytest.raises(AssertionError, match="tower produced"):
        runner.tick()
    assert not runner.has_work and all(request.done() for request in requests)
    assert all(isinstance(request.exception(), AssertionError) for request in requests)
    assert session.page_allocator.num_free_pages == available
    assert all(block.num_tokens == 0 for block in blocks)
    for before, after in zip(snapshot, _kv_snapshot(session, image)):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
