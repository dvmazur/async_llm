import asyncio

import pytest
import torch
from minisgl.training import RecordedAsyncLLM, replay_trace
from transformers.models.qwen3_5 import modeling_qwen3_5

from tests.core.test_async_llm import _make_llm
from tests.training.helpers import (
    copy_operations,
    referenced_blocks,
)
from tests.training.models import live_llm, tiny_model, torch_hybrid
from tests.training.references import sequential, sequential_hybrid_logits


@pytest.mark.parametrize(
    "factory,reference_logits",
    [(tiny_model, sequential), (torch_hybrid, sequential_hybrid_logits)],
    ids=["qwen3", "qwen3.5"],
)
def test_concurrent_recording_to_live_replay_and_accumulated_backward(
    factory, reference_logits, monkeypatch
):
    """Record through the real scheduler; validate replay with independent Torch math."""
    monkeypatch.setattr(modeling_qwen3_5, "FusedRMSNormGated", None)

    async def run():
        reference = factory().cuda()
        live, weights, groups = live_llm(reference)
        llm, session = _make_llm()
        llm.engine = live.engine
        engine = llm.async_engine
        addresses = {name: weight.data_ptr() for name, weight in weights.items()}
        copy_operations(session)

        async def collect(offset):
            with RecordedAsyncLLM(llm) as recorded:
                prompt, left, right, unused = [await recorded.create_block() for _ in range(4)]
                await recorded.forward([offset + 1, offset + 2], write_to=prompt)
                await asyncio.gather(
                    recorded.forward([offset + 3, offset + 4], [prompt, left]),
                    recorded.forward([offset + 5, offset + 6], [prompt, right]),
                )
                await asyncio.gather(
                    recorded.forward([offset + 7], [prompt, right, left]),
                    recorded.forward([offset + 8], [prompt, left, right]),
                )
                merged = await recorded.merge_blocks(left, right)
                await recorded.forward([offset + 9], [prompt, merged])
                for block in (prompt, left, right, merged, unused):
                    await recorded.free_block(block)
            trace = recorded.get_trace()
            assert [token.token_id for token in trace.tokens] == list(
                range(offset + 1, offset + 10)
            )
            assert referenced_blocks(trace) == {
                block.block_id for block in (prompt, left, right, merged)
            }
            assert unused.block_id not in referenced_blocks(trace)
            return trace

        try:
            with torch.device("cpu"):
                traces = await asyncio.gather(collect(0), collect(10))
            assert not referenced_blocks(traces[0]) & referenced_blocks(traces[1])
            assert any(len(batch) == 4 for batch in session.prefill_batches)
            assert any(len(call["input_ids"]) == 4 for call in session.decode_calls)
            assert llm.async_engine is engine and engine.session is session
            assert not engine.has_work and not llm._closed

            parameters = dict(reference.named_parameters())
            for trace in traces:
                expected = reference_logits(reference, trace)
                actual = replay_trace(llm, trace, hidden_only=False)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                expected.square().sum().backward()
                actual.square().sum().backward()
                for name, sources in groups.items():
                    expected_gradient = (
                        parameters[sources[0]].grad
                        if len(sources) == 1
                        else torch.cat([parameters[source].grad for source in sources])
                    )
                    assert weights[name].is_leaf and weights[name].data_ptr() == addresses[name]
                    torch.testing.assert_close(
                        weights[name].grad, expected_gradient, atol=2e-5, rtol=3e-4
                    )
            assert weights["model.embed_tokens.weight"].grad.abs().sum() > 0
        finally:
            await llm.close()

    asyncio.run(asyncio.wait_for(run(), timeout=10))
