"""Independent recordings through the real async scheduler and CPU stub execution."""

import asyncio
from contextlib import aclosing, nullcontext
from dataclasses import asdict

import pytest
import torch
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM, Trace
from test_async_llm import _make_llm


@pytest.fixture(autouse=True)
def cpu_profiling(monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())


def copy_operations(session):
    def append(left, right):
        ids = list(right.token_ids)
        if ids:
            left.grow_pages(session._pages(len(ids)), len(ids))
        left.token_ids.extend(ids)
        return left

    def merge(left, right):
        block = session.create_block()
        append(block, left)
        return append(block, right)

    session.merge_blocks = merge
    session.append_block = append


def referenced_blocks(trace):
    blocks = set()
    for batch in trace.batches:
        for request in batch.requests:
            blocks.add(request.write_to)
            blocks.update(block for block, _ in request.cache_view)
    for update in trace.updates:
        blocks.add(update.block)
        blocks.update(block for block, _ in update.refs or ())
    return blocks


def pause_ticks(llm, monkeypatch):
    released = asyncio.Event()
    engine_loop = llm._engine_loop

    async def delayed_loop():
        await released.wait()
        await engine_loop()

    monkeypatch.setattr(llm, "_engine_loop", delayed_loop)
    return released


def test_concurrent_recordings_split_batches_preserve_views_and_skip_unused_blocks():
    async def run():
        llm, session = _make_llm()
        engine = llm.async_engine
        copy_operations(session)
        ready, arrivals = asyncio.Event(), 0

        async def rendezvous():
            nonlocal arrivals
            arrivals += 1
            if arrivals == 3:
                ready.set()
            await ready.wait()

        async def collect(offset):
            with RecordedAsyncLLM(llm) as recorded:
                prompt, a, b, empty, unused = [await recorded.create_block() for _ in range(5)]
                await recorded.free_block(unused)
                await recorded([offset + 1, offset + 2], write_to=prompt)
                await asyncio.gather(
                    recorded.forward([offset + 3, offset + 4], [prompt, a]),
                    recorded.forward([offset + 5, offset + 6], [prompt, b]),
                )
                await rendezvous()
                await asyncio.gather(
                    recorded.forward([offset + 7], [prompt, b, a]),
                    recorded.forward([offset + 8], [prompt, a, b]),
                )
                merged = await recorded.merge_blocks(a, b)
                copied = await recorded.merge_blocks(empty, merged)
                assert await recorded.append_block(copied, a) is copied
                await recorded.forward([offset + 9], [copied])
                await recorded.free_block(a)
                await recorded.prefill_block([offset + 10], write_to=a)
                active = asdict(recorded.get_trace())
                blocks = (prompt, a, b, empty, merged, copied)
                for block in blocks:
                    await recorded.free_block(block)
            trace = recorded.get_trace()
            assert active["batches"] == asdict(trace)["batches"]
            assert [token.token_id for token in trace.tokens] == list(
                range(offset + 1, offset + 11)
            )
            assert trace.steps == len(trace.batches)
            assert [batch.step for batch in trace.batches] == list(range(trace.steps))
            assert all(0 <= update.step <= trace.steps for update in trace.updates)
            assert referenced_blocks(trace) == {block.block_id for block in blocks}
            assert unused.block_id not in referenced_blocks(trace)
            decode = next(
                batch
                for batch in trace.batches
                if len(batch.requests) == 2 and batch.kind == "decode"
            )
            for request in decode.requests:
                assert dict(request.cache_view) == {
                    prompt.block_id: 2,
                    a.block_id: 2,
                    b.block_id: 2,
                }
            decoded = [token for token in trace.tokens if token.step == decode.step]
            assert len(decoded) == 2
            for token in decoded:
                assert dict(token.view) == {prompt.block_id: 2, a.block_id: 3, b.block_id: 3}
            restored = Trace.from_dict(asdict(trace)).to()
            assert restored.num_tokens == 10
            assert restored.input_ids.numel() == 10
            return trace, {block.block_id for block in blocks}

        async def unrecorded():
            block = (await llm.prefill_block([29, 30])).block
            await rendezvous()
            await llm.forward([31], [block])
            await llm.free_block(block)
            return block.block_id

        try:
            first, second, outside = await asyncio.gather(collect(0), collect(10), unrecorded())
            assert not first[1] & second[1]
            assert outside not in referenced_blocks(first[0]) | referenced_blocks(second[0])
            assert any(len(batch) >= 3 for batch in session.prefill_batches)
            assert any(len(call["input_ids"]) == 5 for call in session.decode_calls)
            assert llm.async_engine is engine and engine.session is session
            assert not llm._closed
            snapshot = asdict(first[0])
            await llm.prefill_block([28])
            assert asdict(first[0]) == snapshot
        finally:
            await llm.close()

    asyncio.run(asyncio.wait_for(run(), timeout=10))


def test_exception_exit_restores_engine_and_preserves_trace():
    async def run():
        llm, session = _make_llm()
        engine = llm.async_engine
        recorded = RecordedAsyncLLM(llm)
        try:
            with pytest.raises(RuntimeError, match="agent failed"):
                with recorded:
                    await recorded.prefill_block([1, 2])
                    raise RuntimeError("agent failed")
            assert llm.async_engine is engine and engine.session is session
            assert not llm._closed
            assert [token.token_id for token in recorded.get_trace().tokens] == [1, 2]
            await llm.prefill_block([3])
            assert recorded.get_trace().num_tokens == 2
            with pytest.raises(RuntimeError):
                with recorded:
                    pass
        finally:
            await llm.close()

    asyncio.run(run())


def test_one_recorder_can_exit_while_another_remains_active():
    async def run():
        llm, session = _make_llm()
        engine = llm.async_engine
        first, second = RecordedAsyncLLM(llm), RecordedAsyncLLM(llm)
        first.__enter__()
        second.__enter__()
        try:
            await first.prefill_block([1])
            first.__exit__(None, None, None)
            await second.prefill_block([2, 3])
            assert first.get_trace().num_tokens == 1
            assert second.get_trace().num_tokens == 2
        finally:
            second.__exit__(None, None, None)
            assert llm.async_engine is engine and engine.session is session
            await llm.close()

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["write", "view", "merge", "append", "free"])
def test_another_recorders_empty_block_cannot_be_borrowed(operation):
    async def run():
        llm, session = _make_llm()
        copy_operations(session)
        try:
            with RecordedAsyncLLM(llm) as first, RecordedAsyncLLM(llm) as second:
                foreign = await first.create_block()
                own = await second.create_block()
                with pytest.raises(ValueError):
                    if operation == "write":
                        await second.forward([1, 2], write_to=foreign)
                    elif operation == "view":
                        await second.forward([1, 2], [foreign, own])
                    elif operation == "merge":
                        await second.merge_blocks(own, foreign)
                    elif operation == "append":
                        await second.append_block(own, foreign)
                    else:
                        await second.free_block(foreign)
                assert first.get_trace().num_tokens == second.get_trace().num_tokens == 0
                assert foreign.num_tokens == own.num_tokens == 0
                await second.forward([3, 4], write_to=own)
                assert [token.token_id for token in second.get_trace().tokens] == [3, 4]
        finally:
            await llm.close()

    asyncio.run(run())


def test_adopt_empty_block_but_reject_unrecorded_prefilled_block():
    async def run():
        llm, _ = _make_llm()
        try:
            filled = (await llm.prefill_block([1, 2])).block
            empty = await llm.create_block()
            with RecordedAsyncLLM(llm) as recorded:
                with pytest.raises(ValueError):
                    await recorded.forward([3, 4], [filled, empty])
                assert empty.num_tokens == 0 and recorded.get_trace().num_tokens == 0
                await recorded.forward([5, 6], write_to=empty)
            assert referenced_blocks(recorded.get_trace()) == {empty.block_id}
            assert [token.token_id for token in recorded.get_trace().tokens] == [5, 6]
        finally:
            await llm.close()

    asyncio.run(run())


def test_external_write_to_owned_block_is_rejected():
    async def run():
        llm, _ = _make_llm()
        try:
            with RecordedAsyncLLM(llm) as recorded:
                block = (await recorded.prefill_block([1, 2])).block
                with pytest.raises(ValueError):
                    await llm.forward([3], [block])
                assert block.token_ids == [1, 2]
                assert recorded.get_trace().num_tokens == 2
                await recorded.forward([4], [block])
                assert [token.token_id for token in recorded.get_trace().tokens] == [1, 2, 4]
        finally:
            await llm.close()

    asyncio.run(run())


def test_modified_block_is_rejected_before_more_inference():
    async def run():
        llm, session = _make_llm()
        try:
            with RecordedAsyncLLM(llm) as recorded:
                block = (await recorded.prefill_block([1, 2])).block
                session.prefill_block(block, torch.tensor([3]))
                with pytest.raises(ValueError):
                    await recorded.forward([4], [block])
                assert recorded.get_trace().num_tokens == 2
                assert block.token_ids == [1, 2, 3]
        finally:
            await llm.close()

    asyncio.run(run())


def test_independent_recorder_rejects_already_recorded_session():
    from minisgl.training.recording import _TraceRecorder

    async def run():
        llm, session = _make_llm()
        llm.async_engine.session = _TraceRecorder(session)
        try:
            with pytest.raises(ValueError, match="already recording"):
                with RecordedAsyncLLM(llm):
                    pass
        finally:
            await llm.close()

    asyncio.run(run())


def test_async_generate_closes_delegate_and_resumes_pending_token(monkeypatch):
    async def run():
        llm, session = _make_llm()
        original, closed = llm.async_generate, []

        async def tracked(*args, **kwargs):
            async with aclosing(original(*args, **kwargs)) as stream:
                try:
                    async for result in stream:
                        yield result
                finally:
                    closed.append(True)

        monkeypatch.setattr(llm, "async_generate", tracked)
        try:
            with RecordedAsyncLLM(llm) as recorded:
                block = (await recorded.prefill_block([1])).block
                context = AsyncContext(cache_view=[block], next_input_id=5)
                async with aclosing(recorded.async_generate(context, max_steps=100)) as stream:
                    assert await anext(stream) == 7
                assert closed == [True]
                assert context.next_input_id == 7
                async with aclosing(
                    recorded.async_generate(context, max_steps=1, return_logits=True)
                ) as stream:
                    token, logits = await anext(stream)
                    assert token == 9 and int(logits.argmax()) == 9
                    with pytest.raises(StopAsyncIteration):
                        await anext(stream)
                assert closed == [True, True]
            assert context.next_input_id == 9 and block.token_ids == [1, 5, 7]
            assert [token.token_id for token in recorded.get_trace().tokens] == [1, 5, 7]
            assert len(session.decode_calls) == 2
        finally:
            await llm.close()

    asyncio.run(run())


def test_image_payloads_are_copied_into_their_own_traces(monkeypatch):
    async def run():
        llm, session = _make_llm()
        monkeypatch.setattr(
            session,
            "_mrope_rel",
            lambda ids, types, grid: torch.arange(ids.numel()).expand(3, -1),
            raising=False,
        )

        async def collect(value):
            pixels = torch.full((4, 6), value)
            grid = torch.tensor([[1, 2, 2]])
            types = torch.ones(4, dtype=torch.int64)
            with RecordedAsyncLLM(llm) as recorded:
                block = await recorded.create_block()
                await recorded.forward(
                    [1, 2, 3, 4],
                    write_to=block,
                    pixel_values=pixels,
                    image_grid_thw=grid,
                    mm_token_type_ids=types,
                )
            pixels.zero_()
            grid.zero_()
            types.zero_()
            trace = recorded.get_trace()
            assert trace.multimodal and trace.num_tokens == 4
            assert len(trace.batches) == len(trace.batches[0].requests) == 1
            request = trace.batches[0].requests[0]
            torch.testing.assert_close(request.pixel_values.to_tensor(), torch.full((4, 6), value))
            torch.testing.assert_close(
                request.image_grid_thw.to_tensor(), torch.tensor([[1, 2, 2]])
            )
            torch.testing.assert_close(
                request.mm_token_type_ids.to_tensor(), torch.ones(4, dtype=torch.int64)
            )
            torch.testing.assert_close(
                request.mrope_positions.to_tensor(), torch.arange(4).expand(3, -1)
            )
            assert referenced_blocks(trace) == {block.block_id}
            return trace

        try:
            first, second = await asyncio.gather(collect(1.0), collect(2.0))
            assert not referenced_blocks(first) & referenced_blocks(second)
            assert len(session.prefill_batches) == 1
        finally:
            await llm.close()

    asyncio.run(run())


def test_shared_batch_failure_does_not_commit_partial_traces():
    async def run():
        llm, session = _make_llm()
        try:
            with RecordedAsyncLLM(llm) as first, RecordedAsyncLLM(llm) as second:
                before = [asdict(recorder.get_trace()) for recorder in (first, second)]
                session.fail_next = RuntimeError("inference failed")
                results = await asyncio.gather(
                    first.prefill_block([1, 2]),
                    second.prefill_block([3, 4]),
                    return_exceptions=True,
                )
                assert all(isinstance(result, RuntimeError) for result in results)
                for recorder, original in zip((first, second), before):
                    assert asdict(recorder.get_trace()) == original
                    with pytest.raises(RuntimeError, match="Discard"):
                        await recorder.prefill_block([5])
            assert (await llm.prefill_block([6])).block.token_ids == [6]
        finally:
            await llm.close()

    asyncio.run(run())


def test_cancelled_scope_discards_only_its_requests_and_preserves_cancellation(monkeypatch):
    async def run():
        llm, session = _make_llm()
        engine = llm.async_engine
        released = pause_ticks(llm, monkeypatch)
        first, second = RecordedAsyncLLM(llm), RecordedAsyncLLM(llm)

        async def cancelled_agent():
            with first:
                await first.prefill_block([1, 2])

        try:
            with second:
                cancelled = asyncio.create_task(cancelled_agent())
                await asyncio.sleep(0)
                surviving = asyncio.create_task(second.prefill_block([3, 4]))
                await asyncio.sleep(0)
                assert len(engine._prefill_queue) == 2
                cancelled.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await cancelled
                assert [request.input_ids.tolist() for request in engine._prefill_queue] == [[3, 4]]
                assert first.get_trace().num_tokens == 0
                assert not surviving.done() and not session.prefill_calls
                released.set()
                assert (await surviving).block.token_ids == [3, 4]
                assert second.get_trace().num_tokens == 2
            assert llm.async_engine is engine and engine.session is session
            assert not engine.has_work and not llm._closed
        finally:
            released.set()
            await llm.close()

    asyncio.run(asyncio.wait_for(run(), timeout=10))


def test_exit_with_pending_calls_detaches_scope_and_cancels_queued_work(monkeypatch):
    async def run():
        llm, session = _make_llm()
        engine = llm.async_engine
        released = pause_ticks(llm, monkeypatch)
        recorded = RecordedAsyncLLM(llm)
        try:
            with pytest.raises(RuntimeError, match="Await all"):
                with recorded:
                    pending = asyncio.create_task(recorded.prefill_block([1, 2]))
                    await asyncio.sleep(0)
                    assert engine.has_work
            assert llm.async_engine is engine and engine.session is session
            assert not engine.has_work and not session.prefill_calls
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert recorded.get_trace().num_tokens == 0
            released.set()
            assert (await llm.prefill_block([3, 4])).block.token_ids == [3, 4]
        finally:
            released.set()
            await llm.close()

    asyncio.run(asyncio.wait_for(run(), timeout=10))
