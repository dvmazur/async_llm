"""Demo export through the real async scheduler, with CPU stub model execution."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, nullcontext

import pytest
import torch
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM, Trace
from minisgl_examples.demos.async_thoughts import demo
from minisgl_examples.training.recording import generate_with_targets

from tests.core.test_async_llm import _make_llm
from tests.scripts.training_helpers import exported_example as exported_example


@pytest.fixture(autouse=True)
def cpu_stub_profiling(monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())


def test_demo_export_aligns_batched_samples_and_masks_nongeneration_rows(exported_example):
    state = exported_example
    data = state["data"]
    (example,) = data["examples"]
    trace = Trace.from_dict(example["trace"])
    assert data["model"] == "stub-model"
    assert example["prompt"] == "test"
    assert example["thinker_tokens"] == [6, 8] and example["writer_tokens"] == [13]
    assert trace == state["recorder"].get_trace()
    assert example["trace"]["format_version"] == 2 and "tokens" not in example["trace"]
    assert len(example["targets"]) == len(example["logprobs"]) == trace.num_tokens == 13
    assert [len(batch.requests) for batch in trace.batches if batch.kind == "decode"] == [2, 2, 1]
    assert state["decode_calls"] == 3

    a, b = state["writers"]
    # These are positions within each writer, independent of flattened trace order.
    wanted = {(a, 2): 6, (a, 3): 8, (b, 2): 13, (b, 3): 15}
    for row, token in enumerate(trace.tokens):
        target = wanted.get((token.block, token.position), -100)
        assert example["targets"][row] == target
        if target == -100:
            assert example["logprobs"][row] is None
        else:
            # Stub logits are raw: forbid 7 changes selection, not this likelihood.
            logits = torch.zeros(32)
            logits[(token.token_id + 1) % 32] = 10
            logits[(token.token_id + 2) % 32] = 20
            assert example["logprobs"][row] == pytest.approx(
                logits.log_softmax(-1)[target].item(), abs=1e-6
            )
    first_a = next(
        i for i, token in enumerate(trace.tokens) if (token.block, token.position) == (a, 2)
    )
    assert example["logprobs"][first_a] < -9  # A masked-softmax likelihood is near zero.
    assert 15 not in [token.token_id for token in trace.tokens]
    assert example["targets"].count(15) == 1
    assert sum(target != -100 for target in example["targets"]) == 4


def test_recorder_closes_underlying_generator_and_resumes_pending_token(monkeypatch):
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
            with RecordedAsyncLLM(llm) as recorder:
                samples = {}
                block = await recorder.create_block()
                await recorder.forward([1], write_to=block, return_logits=False)
                ctx = AsyncContext(cache_view=[block], next_input_id=5)
                async with aclosing(
                    generate_with_targets(recorder, ctx, samples, max_steps=100)
                ) as stream:
                    assert await anext(stream) == 7
                assert closed == [True]
                assert ctx.next_input_id == 7 and block.token_ids == [1, 5]
                async with aclosing(
                    generate_with_targets(recorder, ctx, samples, max_steps=1)
                ) as stream:
                    assert await anext(stream) == 9
                    with pytest.raises(StopAsyncIteration):
                        await anext(stream)
                assert closed == [True, True]
                assert ctx.next_input_id == 9 and block.token_ids == [1, 5, 7]
                assert len(session.decode_calls) == 2 and recorder.get_trace().num_tokens == 3
        finally:
            await llm.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["prefill", "decode", "consumer", "cancel"])
def test_failed_demo_does_not_export_an_artifact(monkeypatch, tmp_path, failure):
    path = tmp_path / "failed.json"
    state = {}

    async def run():
        llm, session = _make_llm()

        async def run_loop(config, recorder, tokenizer, prompting, samples):
            state["recorder"] = recorder
            block = await recorder.create_block()
            if failure == "prefill":
                session.fail_next = RuntimeError("forward failed")
            await recorder.forward([1, 2], write_to=block, return_logits=False)
            if failure == "decode":
                session.fail_next = RuntimeError("forward failed")
            ctx = AsyncContext(cache_view=[block], next_input_id=5)
            async with aclosing(
                generate_with_targets(recorder, ctx, samples, max_steps=2)
            ) as stream:
                await anext(stream)
                if failure == "cancel":
                    raise asyncio.CancelledError
                raise RuntimeError("consumer failed")

        monkeypatch.setattr(demo, "_run_loop", run_loop)
        config = demo.DemoConfig(model="stub-model", trace_output=path)
        error = asyncio.CancelledError if failure == "cancel" else RuntimeError
        with pytest.raises(error):
            await demo._run_demo(config, llm, demo.Prompting(config.problem))
        assert llm._closed and llm._loop_task is None
        assert not path.exists()
        expected_rows = 0 if failure == "prefill" else 2 if failure == "decode" else 3
        assert state["recorder"].get_trace().num_tokens == expected_rows

    asyncio.run(run())


def test_demo_refuses_existing_artifact_before_recording(monkeypatch, tmp_path):
    path = tmp_path / "existing.json"
    original = b'{"existing": true}\n'
    path.write_bytes(original)

    async def unexpected(*args):
        pytest.fail("An existing output must be rejected before inference")

    monkeypatch.setattr(demo, "_run_loop", unexpected)

    async def run():
        llm, session = _make_llm()
        config = demo.DemoConfig(trace_output=path)
        with pytest.raises(FileExistsError):
            await demo._run_demo(config, llm, demo.Prompting(config.problem))
        assert llm.async_engine.session is session
        assert llm._closed and llm._loop_task is None

    asyncio.run(run())
    assert path.read_bytes() == original
