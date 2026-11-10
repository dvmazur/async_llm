"""Run the existing cache/API scenarios on the learned Gemma model.

MINISGL_E2E_MODEL=/path/to/gemma-4-31B-it-FP8-block pytest this_file -s
One real Engine is shared by this module, with an explicit 4096-token KV pool.
State/lifetime assertions stay unchanged. Logit comparisons use the explicitly
provisional Gemma thresholds below; these are not numerical equivalence proofs.
"""
import asyncio
import os

import pytest
import torch

from tests.gemma_tolerances import GEMMA_API_LOGIT_ATOL, GEMMA_API_PROBABILITY_ATOL
from test_shared_cache import TestSharedCacheE2E as _Shared
from test_shared_cache_prefill_batch import (
    TestPrefillBatchEquivalence as _Batched,
    TestPrefillBatchRejects as _Rejects,
    TestPrefillBatchThenDecode as _Decode,
)
from test_chunked_prefill import TestChunkedEquivalence as _Chunked
from test_async_llm import (
    test_two_streams_match_lockstep_group,
    test_forward_streams_match_lockstep_group,
    test_forward_extend_matches_lockstep_decode,
    test_forward_conditional_repeated_blocks,
    test_sample_works,
)
from test_async_cache_engine import TestAsyncCacheEngineE2E as _AsyncEngine
from test_cache_block_merge import (
    test_merged_block_matches_split_view,
    test_append_block_matches_copied_merge,
)

MODEL = os.environ.get("MINISGL_E2E_MODEL", "")
pytestmark = pytest.mark.skipif(not MODEL or not torch.cuda.is_available(),
                                reason="Set MINISGL_E2E_MODEL to the Gemma checkpoint")


@pytest.fixture(scope="module")
def real_engine(tmp_path_factory):
    from minisgl.engine import Engine, EngineConfig
    from minisgl.distributed import DistributedInfo
    from tests.e2e.fp8.worker import gemma_engine_options, is_gemma
    assert is_gemma(MODEL), "This module tests Gemma, not another checkpoint"
    engine = Engine(EngineConfig(model_path=MODEL, quantization="fp8", dtype=torch.bfloat16,
                    tp_info=DistributedInfo(0, 1), max_running_req=8, max_seq_len_override=4096,
                    use_pynccl=False, **gemma_engine_options(MODEL),
                    distributed_addr=(tmp_path_factory.mktemp("gemma") / "dist").as_uri()))
    yield engine
    engine.shutdown()


@pytest.fixture
def session(real_engine):
    from minisgl.shared_cache import SharedCacheSession
    session = SharedCacheSession(real_engine)
    # Original batch tests deliberately retain their local blocks. Record only
    # ownership here so a failure cannot exhaust this module's shared KV pool.
    create, blocks = session.create_block, []
    def tracked_create():
        block = create()
        blocks.append(block)
        return block
    session.create_block = tracked_create
    yield session
    for block in blocks:
        session.free_block(block)


@pytest.fixture
def engine_and_session(real_engine, session):
    return real_engine, session


@pytest.fixture(scope="module")
def runtime(real_engine):
    from minisgl.llm import AsyncLLM
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    llm = AsyncLLM(MODEL, engine=real_engine)
    yield loop, llm
    loop.run_until_complete(llm.close())
    loop.close()
    asyncio.set_event_loop(None)


class TestGemmaSharedCache(_Shared):
    extension_logit_atol = GEMMA_API_LOGIT_ATOL


class TestGemmaAsyncEngine(_AsyncEngine):
    pass


class TestGemmaPrefillBatch(_Batched):
    logit_atol = GEMMA_API_LOGIT_ATOL


class TestGemmaRejectConflicts(_Rejects):
    pass


class TestGemmaBatchedThenDecode(_Decode):
    logit_atol = GEMMA_API_LOGIT_ATOL


class TestGemmaChunkedText(_Chunked):
    logit_atol = GEMMA_API_LOGIT_ATOL


def test_gemma_demo_coroutines(real_engine, monkeypatch):
    # Reuse the demo/role test unchanged; only its model factory must opt into
    # FP8 and the bounded eager Engine already owned by this module.
    from test_async_thoughts_demo import test_demo_smoke
    import async_thoughts.engine as demo_engine
    from minisgl.llm import AsyncLLM
    monkeypatch.setattr(demo_engine, "build_async_llm",
                        lambda model, **kwargs: AsyncLLM(model, engine=real_engine))
    test_demo_smoke()


def _image_job(engine, session, size, *, block=None, context=()):
    from PIL import Image
    from transformers import AutoProcessor
    from minisgl.shared_cache import PrefillJob
    processor = AutoProcessor.from_pretrained(engine.config.model_path)
    data = processor.apply_chat_template([dict(role="user", content=[
        dict(type="image", image=Image.new("RGB", size, color=(30, 90, 180))),
        dict(type="text", text="Describe the image.")])], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt")
    return PrefillJob(block=block if block is not None else session.create_block(),
                      input_ids=data["input_ids"][0], context=list(context),
                      pixel_values=data["pixel_values"], image_position_ids=data["image_position_ids"],
                      mm_token_type_ids=data["mm_token_type_ids"][0])


def test_learned_image_batch_and_refresh(real_engine, session):
    """Qwen image batch/refresh counterparts on the actual31B checkpoint."""
    from minisgl.shared_cache import PrefillJob
    image_a = _image_job(real_engine, session, (96, 64))
    image_b = _image_job(real_engine, session, (128, 192))
    text = PrefillJob(session.create_block(), torch.tensor([11, 12, 13], dtype=torch.int32))
    batched = session.prefill_batch([image_a, text, image_b])
    for row, job in zip(batched, [image_a, text, image_b]):
        from dataclasses import replace
        single = session.prefill_batch([replace(job, block=session.create_block())])[0]
        # Original Qwen two-image batch contract, same logits units.
        torch.testing.assert_close(row, single, atol=.3, rtol=0)
    session.free_block(image_a.block)
    refreshed = session.prefill_batch([
        _image_job(real_engine, session, (128, 192), block=image_a.block)])[0]
    fresh = session.prefill_batch([_image_job(real_engine, session, (128, 192))])[0]
    torch.testing.assert_close(refreshed, fresh, atol=1e-4, rtol=0)


def test_learned_second_image_append_matches_separate(real_engine, session):
    from test_cache_block_merge import assert_same_distribution
    a = _image_job(real_engine, session, (96, 64))
    session.prefill_batch([a])
    b = session.create_block()
    session.prefill_block(b, torch.tensor([11, 12, 13], dtype=torch.int32), context=[a.block])
    c = _image_job(real_engine, session, (128, 192), context=[a.block, b])
    separate = session.prefill_batch([c])[0]
    appended = session.prefill_batch([
        _image_job(real_engine, session, (128, 192), block=b, context=[a.block])])[0]
    assert_same_distribution(separate, appended, atol=GEMMA_API_PROBABILITY_ATOL)


def test_learned_image_two_workers(real_engine, session):
    from minisgl.shared_cache import WorkerGroup
    image = _image_job(real_engine, session, (96, 64))
    logits = session.prefill_batch([image])[0]
    left, right = session.create_block(), session.create_block()
    group = WorkerGroup(cache_structure=[[image.block, left], [image.block, right]])
    tokens = logits[0].topk(2).indices.int()
    for _ in range(8):
        tokens = session.decode_step(group, tokens).argmax(-1).int()
    assert left.num_tokens == right.num_tokens == 8
    assert left.token_ids[0] != right.token_ids[0]
