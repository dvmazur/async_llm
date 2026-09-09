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
from minisgl.llm import AsyncLLM, CausalLMOutput
from minisgl.scheduler import AsyncCacheEngine
from minisgl.shared_cache import AsyncContext, SharedCacheSession, WorkerGroup
from test_async_cache_engine import CPU, VOCAB, StubSession

# =============================================================================
# Unit tests — no model/GPU needed
# =============================================================================


def _make_llm(*, batching_yield_rounds: int | None = None):
    session = StubSession()
    engine = AsyncCacheEngine(session=session, sampler=Sampler(CPU, VOCAB))
    kwargs = (
        {}
        if batching_yield_rounds is None
        else {"batching_yield_rounds": batching_yield_rounds}
    )
    return AsyncLLM(async_engine=engine, **kwargs), session


def test_public_imports():
    from minisgl.async_cache import AsyncContext as ac
    from minisgl.llm import AsyncLLM as al

    assert al is AsyncLLM
    assert ac is AsyncContext


def test_prefill_block():
    async def main():
        llm, session = _make_llm()
        res = await llm.prefill_block([1, 2, 3], return_logits=True)
        assert isinstance(res, CausalLMOutput)
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


def test_extra_yield_rounds_coalesce_nested_successor_stages():
    """A parent behind nested gather callbacks still joins its peer's batch."""

    async def main():
        llm, session = _make_llm(batching_yield_rounds=3)

        async def pipeline(label: int, nesting: int):
            first = await llm.create_block()
            await llm.prefill_block([label], write_to=first)

            async def descend(depth: int):
                if depth == 0:
                    successor = await llm.create_block()
                    await llm.prefill_block([label + 10], write_to=successor)
                    return
                await asyncio.gather(descend(depth - 1))

            await descend(nesting)

        # The second pipeline needs three event-loop turns to propagate its
        # completed first stage through nested gather callbacks.
        await asyncio.gather(pipeline(1, 0), pipeline(2, 3))

        assert len(session.prefill_batches) == 2
        assert [block.token_ids for block in session.prefill_batches[0]] == [[1], [2]]
        assert [block.token_ids for block in session.prefill_batches[1]] == [[11], [12]]
        await llm.close()

    asyncio.run(main())


def test_batching_yield_rounds_must_be_positive():
    session = StubSession()
    engine = AsyncCacheEngine(session=session, sampler=Sampler(CPU, VOCAB))
    with pytest.raises(ValueError, match="batching_yield_rounds"):
        AsyncLLM(async_engine=engine, batching_yield_rounds=0)


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
# Unit tests — the unified ``forward`` method
# =============================================================================


def test_forward_prefill_mode():
    async def main():
        llm, session = _make_llm()
        block = await llm.create_block()
        out = await llm.forward([1, 2, 3], write_to=block)
        assert isinstance(out, CausalLMOutput)
        assert out.logits.shape == (VOCAB,)
        assert int(out.logits.argmax()) == 5  # stub decoy of last token 3
        assert out.block is block
        assert block.token_ids == [1, 2, 3]
        assert session.prefill_calls[0]["context"] is None

        quiet = await llm.forward([4], write_to=await llm.create_block(), return_logits=False)
        assert quiet.logits is None  # logits are opt-in (default on)
        await llm.close()

    asyncio.run(main())


def test_forward_argument_validation():
    async def main():
        llm, _ = _make_llm()
        with pytest.raises(ValueError, match="input_ids and/or cache_view"):
            await llm.forward()
        with pytest.raises(ValueError, match="needs a write_to block"):
            await llm.forward([1, 2])
        with pytest.raises(ValueError, match="empty input_ids"):
            await llm.forward([], write_to=await llm.create_block())
        await llm.close()

    asyncio.run(main())


def test_forward_conditional_prefill():
    async def main():
        llm, session = _make_llm()
        prompt = await llm.create_block()
        await llm.forward([1, 2], write_to=prompt)

        # write_to outside the view: effective view is [*cache_view, write_to].
        fresh = await llm.create_block()
        out = await llm.forward([7, 8], [prompt], write_to=fresh)
        assert session.prefill_calls[-1]["context"] == [prompt]
        assert fresh.token_ids == [7, 8]
        assert int(out.logits.argmax()) == 10  # decoy of last token 8

        # A single token with a cache view uses decode.
        tail = await llm.create_block()
        out = await llm.forward([9], [prompt, fresh, tail])
        assert session.decode_calls[-1]["structure"] == [[prompt, fresh, tail]]
        assert session.decode_calls[-1]["input_ids"] == [9]
        assert out.block is tail  # defaulted write_to: last of the view
        assert tail.token_ids == [9]
        await llm.close()

    asyncio.run(main())


def test_forward_write_to_not_last_raises():
    async def main():
        llm, _ = _make_llm()
        prompt = await llm.create_block()
        await llm.forward([1], write_to=prompt)
        with pytest.raises(ValueError, match="last block of cache_view"):
            await llm.forward([2], [prompt, await llm.create_block()], write_to=prompt)
        await llm.close()

    asyncio.run(main())


def test_forward_extend_non_empty_block():
    async def main():
        llm, session = _make_llm()
        block = await llm.create_block()
        await llm.forward([1], write_to=block)

        # Extending the (non-empty) last-of-view block appends the tokens in one
        # prefill; the block itself is the write target, not part of the context.
        out = await llm.forward([5, 7], [block])
        assert block.token_ids == [1, 5, 7]
        assert not session.decode_calls
        assert [c["ids"] for c in session.prefill_calls] == [[1], [5, 7]]
        assert session.prefill_calls[-1]["context"] is None
        assert int(out.logits.argmax()) == 9  # raw decoy of the last fed token

        # A non-empty write block outside the view would not see its own tokens.
        other = await llm.create_block()
        await llm.forward([2], write_to=other)
        with pytest.raises(ValueError, match="outside cache_view"):
            await llm.forward([3], [block], write_to=other)
        await llm.close()

    asyncio.run(main())


def test_forward_extend_non_empty_block_in_context():
    async def main():
        llm, session = _make_llm()
        prompt = await llm.create_block()
        await llm.forward([1, 2], write_to=prompt)
        block = await llm.create_block()
        await llm.forward([3], [prompt, block])  # fresh block in context
        await llm.forward([4, 5], [prompt, block])  # extend it in the same context

        assert block.token_ids == [3, 4, 5]
        assert session.decode_calls[0]["input_ids"] == [3]
        last = session.prefill_calls[-1]
        assert last["ids"] == [4, 5]
        assert last["context"] == [prompt]  # the write block is the target, not context
        await llm.close()

    asyncio.run(main())


def test_forward_decode_mode():
    async def main():
        llm, session = _make_llm()
        block = await llm.create_block()
        await llm.forward([1], write_to=block)

        ctx = AsyncContext(cache_view=[block], next_input_id=5)
        out = await llm.forward(cache_view=ctx)
        assert int(out.logits.argmax()) == 7  # raw row: decoy of fed token 5
        assert block.token_ids == [1, 5]
        assert ctx.next_input_id is None  # consumed; caller re-seeds

        # Custom-generate chaining: sample client-side, re-seed, step again.
        ctx.next_input_id = int(out.logits.argmax())
        out = await llm.forward(cache_view=ctx, return_logits=True)
        assert block.token_ids == [1, 5, 7]

        # The token can also be passed directly with a plain cache view.
        out = await llm.forward([9], [block])
        assert int(out.logits.argmax()) == 11
        assert block.token_ids == [1, 5, 7, 9]
        assert len(session.decode_calls) == 3
        await llm.close()

    asyncio.run(main())


def test_forward_decode_needs_pending_token():
    async def main():
        llm, _ = _make_llm()
        block = await llm.create_block()
        await llm.forward([1], write_to=block)
        with pytest.raises(ValueError, match="pending input token"):
            await llm.forward(cache_view=AsyncContext(cache_view=[block]))
        with pytest.raises(ValueError, match="pending input token"):
            await llm.forward(cache_view=[block])  # plain view carries no token
        ctx = AsyncContext(cache_view=[block], next_input_id=5)
        with pytest.raises(ValueError, match="pending next_input_id"):
            await llm.forward([6], ctx)  # ambiguous: pending token AND input_ids
        await llm.close()

    asyncio.run(main())


def test_forward_streams_batch_together():
    """Two custom-generate loops over ``forward`` land in the same tick batch."""

    async def main():
        llm, session = _make_llm()
        b1 = await llm.create_block()
        await llm.forward([1], write_to=b1)
        b2 = await llm.create_block()
        await llm.forward([2], write_to=b2)
        ctx1 = AsyncContext(cache_view=[b1, b2], output_block=b2, next_input_id=1)
        ctx2 = AsyncContext(cache_view=[b2, b1], output_block=b1, next_input_id=2)

        n_steps = 5

        async def consume(ctx, out):
            for _ in range(n_steps):
                res = await llm.forward(cache_view=ctx)
                token = int(res.logits.argmax())  # client-side greedy pick
                out.append(token)
                ctx.next_input_id = token

        out1: List[int] = []
        out2: List[int] = []
        await asyncio.gather(consume(ctx1, out1), consume(ctx2, out2))

        assert len(session.decode_calls) == n_steps
        assert all(len(call["input_ids"]) == 2 for call in session.decode_calls)
        assert out1 == [3, 5, 7, 9, 11]  # raw decoy chain t -> (t + 2) % VOCAB
        assert out2 == [4, 6, 8, 10, 12]
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
        a = (await llm.prefill_block(a_ids, cache_view=[prompt])).block
        b = (await llm.prefill_block(b_ids, cache_view=[prompt, a])).block
        ctx_a = AsyncContext(cache_view=[prompt, b, a], output_block=a)
        ctx_b = AsyncContext(cache_view=[prompt, a, b], output_block=b)

        out_a, out_b = [seed_a], [seed_b]

        async def consume(ctx, out):
            async for tok in llm.async_generate(ctx, first_token_id=out[0], max_steps=n_steps):
                out.append(tok)

        async def probe():
            res = await llm.prefill_block(probe_ids, return_logits=True)
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


@requires_e2e
def test_forward_streams_match_lockstep_group(real_engine):
    """The demo's shape rebuilt purely on ``forward``: conditional prefills,
    two custom-generate decode loops (client-side argmax) and a mid-run probe
    prefill — token streams must equal lock-step 2-worker stepping."""

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

    # ---- async: everything through the unified forward ----------------
    async def main():
        llm = AsyncLLM(engine=real_engine)
        prompt, a, b = await asyncio.gather(*(llm.create_block() for _ in range(3)))
        await llm.forward(prompt_ids, write_to=prompt, return_logits=False)
        await llm.forward(a_ids, [prompt, a], return_logits=False)
        await llm.forward(b_ids, [prompt, a, b], return_logits=False)
        ctx_a = AsyncContext(cache_view=[prompt, b, a], output_block=a, next_input_id=seed_a)
        ctx_b = AsyncContext(cache_view=[prompt, a, b], output_block=b, next_input_id=seed_b)

        out_a, out_b = [seed_a], [seed_b]

        async def consume(ctx, out):
            for _ in range(n_steps):
                res = await llm.forward(cache_view=ctx)
                token = int(res.logits.argmax())
                out.append(token)
                ctx.next_input_id = token

        async def probe():
            blk = await llm.create_block()
            res = await llm.forward(probe_ids, write_to=blk)
            assert res.logits.numel() > 0
            await llm.free_block(blk)

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


@requires_e2e
def test_forward_extend_matches_lockstep_decode(real_engine):
    """Extending a non-empty block via ``forward(input_ids, cache_view)`` is
    the same computation as feeding those tokens through lock-step decode."""
    prompt_ids = _encode("The capital of France is")
    fed = [11, 13, 17]

    # Reference: prefill, then feed ``fed`` through decode steps.
    session = SharedCacheSession(real_engine)
    ref_block = session.create_block()
    session.prefill_block(ref_block, prompt_ids)
    ref_group = WorkerGroup(cache_structure=[[ref_block]])
    for tok in fed:
        ref_logits = session.decode_step(ref_group, torch.tensor([tok], dtype=torch.int32))

    async def main():
        llm = AsyncLLM(engine=real_engine)
        block = await llm.create_block()
        await llm.forward(prompt_ids, write_to=block, return_logits=False)
        out = await llm.forward(fed, [block])  # extend: one decode per token
        assert block.token_ids == prompt_ids.tolist() + fed
        await llm.free_block(block)
        await llm.close()
        return out.logits

    assert torch.nn.functional.cosine_similarity(asyncio.run(main()).flatten(), ref_logits.flatten(), dim=0) > 0.99
    session.free_block(ref_block)

@requires_e2e
def test_forward_conditional_repeated_blocks(real_engine):
    async def main():
        llm = AsyncLLM(engine=real_engine)
        prefix, repeater, repeater_copy, middle, suffix_ref, suffix_rep, suffix_control = await asyncio.gather(
            *(llm.create_block() for _ in range(7)))
        await llm([1, 2], cache_view=[prefix])
        await llm([3, 4], cache_view=[prefix, repeater])
        await llm([3, 4], cache_view=[prefix, repeater_copy])
        await llm([5, 6], cache_view=[prefix, repeater, middle])

        # cache view has (3, 4) twice: [1, 2, (3, 4), 5, 6, (3, 4), _]
        # reference: compute forward pass using two different identical cache blocks for (3, 4)
        out_ref = await llm([7], cache_view=[prefix, repeater, middle, repeater_copy, suffix_ref])
        out_rep = await llm([7], cache_view=[prefix, repeater, middle, repeater, suffix_rep])
        out_control = await llm([7], cache_view=[prefix, repeater, middle, suffix_control])
        assert torch.allclose(out_ref.logits, out_rep.logits, rtol=1e-2, atol=1e-2)
        assert not torch.allclose(out_ref.logits, out_control.logits, rtol=1e-2, atol=1e-2)
        await llm.close()

    asyncio.run(main())

@requires_e2e
def test_sample_works(real_engine):
    async def main():
        llm = AsyncLLM(engine=real_engine, model_path=E2E_MODEL_PATH)
        next_token_id = await llm.sample(await llm([1, 2], cache_view=[await llm.create_block()]))
        assert torch.as_tensor(next_token_id).shape == ()
        assert 0 <= next_token_id.item() < llm.tokenizer.vocab_size
    asyncio.run(main())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
