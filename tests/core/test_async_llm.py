"""
Tests for ``minisgl.llm.AsyncLLM`` (the asyncio frontend).

* **Unit tests** (always run) drive the event-loop plumbing against the stub
  session from ``test_async_cache_engine``: lazy loop start, prefill results,
  step chaining, two-stream batching (the tick-pacing contract), stream
  break/resume, seeding errors, and tick-failure recovery.

* **End-to-end test** (requires ``MINISGL_E2E_MODEL`` + CUDA) runs two
  concurrent agent streams plus a mid-run probe prefill and checks the token
  streams are identical to lock-step 2-worker ``SharedCacheSession`` stepping.

Run unit tests only::

    pytest tests/core/test_async_llm.py -v

Run everything (needs GPU + model weights; build one engine per process)::

    MINISGL_E2E_MODEL=Qwen/Qwen3-0.6B pytest tests/core/test_async_llm.py -v
"""

from __future__ import annotations

import asyncio
import os
from typing import List

import pytest
import torch
from minisgl.engine.sample import Sampler
from minisgl.llm import AsyncLLM, PrefillResult
from minisgl.scheduler import AsyncCacheEngine
from minisgl.shared_cache import AsyncContext
from test_async_cache_engine import CPU, VOCAB, StubSession

# =============================================================================
# Unit tests — no model/GPU needed
# =============================================================================


def _make_llm():
    session = StubSession()
    engine = AsyncCacheEngine(session=session, sampler=Sampler(CPU, VOCAB))
    return AsyncLLM(async_engine=engine), session


def test_public_imports():
    from minisgl.async_cache import AsyncContext as ac
    from minisgl.llm import AsyncLLM as al

    assert al is AsyncLLM
    assert ac is AsyncContext


def test_prefill_block():
    async def main():
        llm, session = _make_llm()
        res = await llm.prefill_block([1, 2, 3], return_logits=True)
        assert isinstance(res, PrefillResult)
        assert res.block.token_ids == [1, 2, 3]
        assert res.logits.shape == (VOCAB,)
        assert int(res.logits.argmax()) == 5  # stub decoy of last token 3
        assert len(session.prefill_calls) == 1
        await llm.free_block(res.block)

        res = await llm.prefill_block([1, 2, 3])  # logits are opt-in
        assert res.logits is None
        await llm.close()

    asyncio.run(main())


def test_generate_chain():
    async def main():
        llm, _ = _make_llm()
        res = await llm.prefill_block([1])
        ctx = AsyncContext(cache_view=[res.block])
        tokens = [t async for t in llm.async_generate(ctx, first_token_id=1, max_steps=4)]
        # stub decoy chain: t -> (t + 2) % VOCAB
        assert tokens == [3, 5, 7, 9]
        # KV-backed bookkeeping: prefill + the four fed inputs
        assert res.block.token_ids == [1, 1, 3, 5, 7]
        assert ctx.next_input_id == 9
        await llm.close()

    asyncio.run(main())


def test_forbid_ids_flow_through():
    async def main():
        llm, _ = _make_llm()
        res = await llm.prefill_block([1])
        ctx = AsyncContext(cache_view=[res.block])
        # masking the decoy (t + 2) falls back to the wanted token (t + 1)
        tokens = [
            t async for t in llm.async_generate(ctx, first_token_id=5, forbid_ids=[7], max_steps=1)
        ]
        assert tokens == [6]
        await llm.close()

    asyncio.run(main())


def test_generate_return_logits():
    async def main():
        llm, _ = _make_llm()
        res = await llm.prefill_block([1])
        ctx = AsyncContext(cache_view=[res.block])
        pairs = [
            p
            async for p in llm.async_generate(
                ctx, first_token_id=5, forbid_ids=[7], max_steps=2, return_logits=True
            )
        ]
        tokens = [t for t, _ in pairs]
        assert tokens == [6, 8]  # masked selection
        assert all(logits.shape == (VOCAB,) for _, logits in pairs)
        # raw (pre-mask) rows: the decoy is still the argmax of step 1
        assert int(pairs[0][1].argmax()) == 7
        assert ctx.next_input_id == 8  # chained on the token, not the pair
        await llm.close()

    asyncio.run(main())


def test_two_streams_batch_together():
    """The tick-pacing contract: two live consumers land in every batch."""

    async def main():
        llm, session = _make_llm()
        b1 = (await llm.prefill_block([1])).block
        b2 = (await llm.prefill_block([2])).block
        ctx1 = AsyncContext(cache_view=[b1, b2], output_block=b2)
        ctx2 = AsyncContext(cache_view=[b2, b1], output_block=b1)

        n_steps = 5
        out1: List[int] = []
        out2: List[int] = []

        async def consume(ctx, seed, out):
            async for tok in llm.async_generate(ctx, first_token_id=seed, max_steps=n_steps):
                out.append(tok)

        await asyncio.gather(consume(ctx1, 1, out1), consume(ctx2, 2, out2))

        assert len(session.decode_calls) == n_steps
        assert all(len(call["input_ids"]) == 2 for call in session.decode_calls)
        assert out1 == [3, 5, 7, 9, 11]
        assert out2 == [4, 6, 8, 10, 12]
        await llm.close()

    asyncio.run(main())


def test_break_and_resume():
    async def main():
        llm, _ = _make_llm()
        res = await llm.prefill_block([1])
        ctx = AsyncContext(cache_view=[res.block])

        got: List[int] = []
        async for tok in llm.async_generate(ctx, first_token_id=5, max_steps=10):
            got.append(tok)
            if len(got) == 2:
                break
        assert got == [7, 9]
        assert ctx.next_input_id == 9  # pending token survives the break

        async for tok in llm.async_generate(ctx, max_steps=1):  # no re-seed
            got.append(tok)
        assert got == [7, 9, 11]
        assert res.block.token_ids == [1, 5, 7, 9]
        await llm.close()

    asyncio.run(main())


def test_unseeded_context_raises():
    async def main():
        llm, _ = _make_llm()
        res = await llm.prefill_block([1])
        ctx = AsyncContext(cache_view=[res.block])
        gen = llm.async_generate(ctx)
        with pytest.raises(ValueError, match="no pending input token"):
            await gen.__anext__()
        await llm.close()

    asyncio.run(main())


def test_tick_failure_recovers():
    async def main():
        llm, session = _make_llm()
        res = await llm.prefill_block([1])
        ctx = AsyncContext(cache_view=[res.block])

        session.fail_next = RuntimeError("boom")
        gen = llm.async_generate(ctx, first_token_id=3, max_steps=2)
        with pytest.raises(RuntimeError, match="boom"):
            await gen.__anext__()

        # the loop task survives and the context is still usable
        tokens = [t async for t in llm.async_generate(ctx, max_steps=1)]
        assert tokens == [5]
        await llm.close()

    asyncio.run(main())


# =============================================================================
# End-to-end test — needs MINISGL_E2E_MODEL + CUDA
# =============================================================================

E2E_MODEL_PATH = os.environ.get("MINISGL_E2E_MODEL", "")

requires_e2e = pytest.mark.skipif(
    not E2E_MODEL_PATH or not torch.cuda.is_available(),
    reason="Set MINISGL_E2E_MODEL to an HF model path and ensure CUDA is available",
)


@pytest.fixture(scope="module")
def real_engine():
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig

    config = EngineConfig(
        model_path=E2E_MODEL_PATH,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=8,
        cuda_graph_bs=[2, 4],
        cuda_graph_max_bs=4,
        page_size=int(os.environ.get("MINISGL_TEST_PAGE_SIZE", "1")),
        memory_ratio=0.7,
        max_seq_len_override=2048,
    )
    return Engine(config)


def _encode(text: str) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(E2E_MODEL_PATH)
    return tok.encode(text, return_tensors="pt").view(-1).to(torch.int32)


@requires_e2e
def test_two_streams_match_lockstep_group(real_engine):
    """Two concurrent agent coroutines (+ a mid-run probe prefill) produce the
    exact token streams of lock-step 2-worker ``decode_step`` calls."""
    from minisgl.shared_cache import SharedCacheSession, WorkerGroup

    n_steps = 6
    prompt_ids = _encode("Q: What is 2 + 2?\n")
    a_ids = _encode("A:")
    b_ids = _encode("Hint:")
    probe_ids = _encode("Should we continue? yes or no:")
    seed_a, seed_b = 11, 13

    # ---- reference: lock-step 2-worker stepping ----------------------
    session = SharedCacheSession(real_engine)
    r_prompt, r_a, r_b = (session.create_block() for _ in range(3))
    session.prefill_block(r_prompt, prompt_ids)
    session.prefill_block(r_a, a_ids, context=[r_prompt])
    session.prefill_block(r_b, b_ids, context=[r_prompt, r_a])
    group = WorkerGroup(
        cache_structure=[[r_prompt, r_b, r_a], [r_prompt, r_a, r_b]],
        write_to=[r_a, r_b],
    )
    ref_a, ref_b = [seed_a], [seed_b]
    for _ in range(n_steps):
        logits = session.decode_step(group, torch.tensor([ref_a[-1], ref_b[-1]], dtype=torch.int32))
        ref_a.append(int(logits[0].argmax()))
        ref_b.append(int(logits[1].argmax()))

    # ---- async: two coroutines + a probe ------------------------------
    async def main():
        llm = AsyncLLM(engine=real_engine)
        prompt = (await llm.prefill_block(prompt_ids)).block
        a = (await llm.prefill_block(a_ids, context=[prompt])).block
        b = (await llm.prefill_block(b_ids, context=[prompt, a])).block
        ctx_a = AsyncContext(cache_view=[prompt, b, a], output_block=a)
        ctx_b = AsyncContext(cache_view=[prompt, a, b], output_block=b)

        out_a, out_b = [seed_a], [seed_b]

        async def consume(ctx, out):
            async for tok in llm.async_generate(ctx, first_token_id=out[0], max_steps=n_steps):
                out.append(tok)

        async def probe():
            res = await llm.prefill_block(probe_ids, capture_affine=False, return_logits=True)
            assert res.logits.numel() > 0
            await llm.free_block(res.block)

        await asyncio.gather(consume(ctx_a, out_a), consume(ctx_b, out_b), probe())
        for blk in (prompt, a, b):
            await llm.free_block(blk)
        await llm.close()
        return out_a, out_b

    out_a, out_b = asyncio.run(main())
    assert out_a == ref_a
    assert out_b == ref_b

    for blk in (r_prompt, r_a, r_b):
        session.free_block(blk)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
