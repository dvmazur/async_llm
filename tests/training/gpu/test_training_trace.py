import asyncio
import json
from dataclasses import asdict, replace

import torch
from minisgl.training import RecordedAsyncLLM, Trace, replay_trace
from minisgl.training.trace import BlockUpdate, ForwardBatch, ForwardRequest

from tests.training.helpers import Session, make_llm
from tests.training.models import tiny_model
from tests.training.references import sequential
from tests.training.workloads import mutation_trace


def test_preparation_keeps_recording_serializable_and_reuses_unchanged_cache():
    trace, model = (mutation_trace(), tiny_model().cuda())
    recorded = asdict(trace)
    expected = replay_trace(model, trace)
    chunks = next(iter(trace._attention.values()))
    actual = replay_trace(model, trace)
    assert next(iter(trace._attention.values())) is chunks
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert asdict(trace) == recorded
    assert Trace.from_dict(json.loads(json.dumps(asdict(trace)))) == trace


def test_recording_append_recompiles_and_configured_trace_keeps_snapshot():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block = await session.create_block()
                await session.prefill_block([1, 2], write_to=block)
                trace, model = (session.get_trace(), tiny_model().cuda())
                before = replay_trace(model, trace, hidden_only=False)
                chunks = next(iter(trace._attention.values()))
                configured = trace.to(query_chunk_size=1, attention_checkpointing=False)
                assert configured is not trace
                assert configured.query_chunk_size == 1 and trace.query_chunk_size == 128
                await session.prefill_block([3, 4], write_to=block)
                actual = replay_trace(model, trace, hidden_only=False)
                torch.testing.assert_close(actual, sequential(model, trace), atol=2e-06, rtol=2e-05)
                assert actual.shape[0] == trace.num_tokens == 4
                assert next(iter(trace._attention.values())) is not chunks
                assert configured.num_tokens == 2 and len(configured.batches) == 1
                torch.testing.assert_close(
                    replay_trace(model, configured, hidden_only=False), before
                )
        finally:
            await llm.close()

    return asyncio.run(run())


def test_recording_can_continue_between_replay_forward_and_backward():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block = await session.create_block()
                await session.prefill_block([1, 2], write_to=block)
                reference, model = (tiny_model().cuda(), tiny_model().cuda())
                trace = session.get_trace()
                trace.query_chunk_size = 1
                expected = replay_trace(
                    reference, Trace.from_dict(asdict(trace)), hidden_only=False
                )
                actual = replay_trace(model, trace, hidden_only=False)
                await session.prefill_block([3, 4], write_to=block)
                replay_trace(model, trace, hidden_only=False)
                expected.square().sum().backward()
                actual.square().sum().backward()
                for expected_parameter, actual_parameter in zip(
                    reference.parameters(), model.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        actual_parameter.grad, expected_parameter.grad, atol=1e-05, rtol=0.0002
                    )
        finally:
            await llm.close()

    return asyncio.run(run())


def test_replacing_recorded_batch_invalidates_cached_inputs_and_visibility():
    trace, model = (mutation_trace(), tiny_model().cuda())
    before = replay_trace(model, trace, hidden_only=False)
    chunks = next(iter(trace._attention.values()))
    batch = trace.batches[0]
    first, *rest = batch.requests
    trace.batches[0] = replace(batch, requests=(replace(first, input_ids=(21, 22)), *rest))
    actual = replay_trace(model, trace, hidden_only=False)
    torch.testing.assert_close(actual, sequential(model, trace), atol=2e-06, rtol=2e-05)
    assert not torch.allclose(actual, before)
    assert next(iter(trace._attention.values())) is not chunks


def test_row_selection_uses_token_count_after_recorded_batch_changes():
    trace = Trace(batches=[ForwardBatch(0, "prefill", (ForwardRequest((1, 2), 0, 0, ((0, 0),)),))])
    model = tiny_model().cuda()
    replay_trace(model, trace)
    batch = trace.batches[0]
    request = batch.requests[0]
    trace.batches[0] = replace(batch, requests=(replace(request, input_ids=(1, 2, 3)),))
    actual = replay_trace(model, trace, rows=[2], hidden_only=False)
    torch.testing.assert_close(actual, sequential(model, trace)[[2]], atol=2e-06, rtol=2e-05)
    assert trace.num_tokens == 3


def test_replacing_block_update_invalidates_cached_provenance():
    trace = Trace(
        batches=[
            ForwardBatch(
                0,
                "prefill",
                (ForwardRequest((1,), 0, 0, ((0, 0),)), ForwardRequest((2,), 1, 0, ((1, 0),))),
            ),
            ForwardBatch(1, "decode", (ForwardRequest((3,), 2, 1, ((2, 1),)),)),
        ],
        updates=[BlockUpdate(1, 2, refs=((0, 1),))],
    )
    model = tiny_model().cuda()
    before = replay_trace(model, trace, hidden_only=False)
    chunks = next(iter(trace._attention.values()))
    trace.updates[0] = BlockUpdate(1, 2, refs=((1, 1),))
    actual = replay_trace(model, trace, hidden_only=False)
    torch.testing.assert_close(actual, sequential(model, trace), atol=2e-06, rtol=2e-05)
    torch.testing.assert_close(actual[:2], before[:2], atol=0, rtol=0)
    assert not torch.allclose(actual[2], before[2])
    assert next(iter(trace._attention.values())) is not chunks
