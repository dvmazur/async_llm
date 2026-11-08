import asyncio
from contextlib import aclosing, nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM
from minisgl_examples.demos.async_thoughts.short_retry import (
    DraftStopObserver,
    generate_native_answer,
    retry_allowance,
    retry_short_answer,
)
from minisgl_examples.training.recording import build_training_example

from tests.core.test_async_llm import _make_llm


@pytest.fixture(autouse=True)
def cpu_profiling(monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())


@pytest.mark.parametrize(
    "length,stop,budget,expected",
    [(0, 10, 8, 7), (16, 10, 20, 3), (17, 10, 20, 0), (7, 10, 8, 0), (5, None, 8, 0)],
)
def test_retry_only_on_genuine_short_eos_with_available_samples(length, stop, budget, expected):
    assert retry_allowance(length, stop, budget) == expected


@pytest.mark.parametrize(
    "prompt,budget,expected,stop",
    [([2], 3, [4, 6, 8], None), ([2, 4], 3, [6, 8], 10), ([8], 2, [], 10)],
)
def test_native_reuses_causal_trace_and_observes_eos(prompt, budget, expected, stop):
    async def run():
        llm, session = _make_llm()
        llm.tokenizer = SimpleNamespace(eos_token_id=10)
        try:
            with RecordedAsyncLLM(llm) as recorder:
                samples = {}
                result = await generate_native_answer(recorder, prompt, budget, samples=samples)
                assert result == (expected, stop)
                sample_ids = [tok for tok, _ in samples.values()]
                assert sample_ids == expected + ([] if stop is None else [stop])
                trace = build_training_example(recorder, samples)
                inputs = [
                    int(tok)
                    for batch in trace["trace"].batches
                    for req in batch.requests
                    for tok in req.input_ids
                ]
                assert inputs[: len(prompt)] == prompt
                assert all(
                    len(req.cache_view) == 1 for b in trace["trace"].batches for req in b.requests
                )
                assert len(session.decode_calls) == len(sample_ids)
        finally:
            await llm.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "draft,stop,budget,chosen,expected,retry_stop,total",
    [
        ([2], 10, 5, "native_retry", [6, 8], 10, 5),
        ([2], 10, 3, "native_retry", [6], None, 3),
        ([2], 10, 2, "async_draft", [2], None, 2),
        ([2], None, 2, "async_draft", [2], None, 1),
    ],
)
def test_final_output_and_both_phase_sample_counts(
    draft, stop, budget, chosen, expected, retry_stop, total
):
    async def run():
        llm, _ = _make_llm()
        llm.tokenizer = SimpleNamespace(eos_token_id=10)
        try:
            outcome = await retry_short_answer(
                llm, prompt_ids=[4], draft_ids=draft, draft_stop_id=stop, max_answer_tokens=budget
            )
            assert (
                outcome.source,
                outcome.final_ids,
                outcome.retry_stop_id,
                outcome.sampled_tokens,
            ) == (chosen, expected, retry_stop, total)
        finally:
            await llm.close()

    asyncio.run(run())


def test_observer_handles_recorded_and_raw_writers_without_confusing_thinker_eos():
    async def run():
        for record in (False, True):
            llm, _ = _make_llm()
            llm.tokenizer = SimpleNamespace(eos_token_id=10)
            original = llm.async_engine
            observer = DraftStopObserver(llm)
            try:
                with RecordedAsyncLLM(observer) if record else nullcontext(observer) as target:
                    prompt, thought, writer = [await target.create_block() for _ in range(3)]
                    await target.forward([1], write_to=prompt, return_logits=False)
                    for view in ([prompt, thought], [prompt, thought, writer]):
                        context = AsyncContext(cache_view=view, next_input_id=8)
                        async with aclosing(
                            target.async_generate(context, max_steps=1, return_logits=record)
                        ) as stream:
                            item = await anext(stream)
                            assert (item[0] if record else item) == 10
                        assert observer.writer_stop_id == (10 if len(view) == 3 else None)
                    for b in (writer, thought, prompt):
                        await target.free_block(b)
                assert llm.async_engine is original
            finally:
                await llm.close()

    asyncio.run(run())


def test_native_cancellation_drains_inflight_decode():
    class Delayed:
        eos_token_ids = {2}

        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = False
            self.freed = False

        async def create_block(self):
            return SimpleNamespace()

        async def free_block(self, block):
            assert self.closed
            self.freed = True

        async def async_generate(self, ctx, **kwargs):
            self.started.set()
            try:
                await self.release.wait()
                yield 7
            finally:
                self.closed = True

    async def run():
        llm = Delayed()
        task = asyncio.create_task(generate_native_answer(llm, [8], 5))
        await llm.started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not llm.freed and not task.done()
        llm.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert llm.freed

    asyncio.run(run())


def test_buffered_cli_selects_once_and_preserves_discarded_phase(monkeypatch, tmp_path, capsys):
    import json

    from minisgl_examples.demos.async_thoughts import demo, short_retry
    from minisgl_examples.training.recording import generate_with_targets

    class Tokenizer:
        eos_token_id = 11

        def encode(self, text, **kwargs):
            assert text == "ordinary rendered prompt" and not kwargs["add_special_tokens"]
            return [7]

        def decode(self, ids, **kwargs):
            return "selected-" + ",".join(map(str, ids))

    def prompting(problem, tok, *, protocol):
        assert protocol == "test-protocol"
        return SimpleNamespace(input_prompt="ordinary rendered prompt")

    async def scripted(config, llm, tokenizer, prompting, samples):
        print("DISCARDED-DRAFT-MARKER")
        blocks = [await llm.create_block() for _ in range(3)]
        try:
            await llm.forward([1], write_to=blocks[0], return_logits=False)
            ctx = AsyncContext(cache_view=blocks, next_input_id=3)
            draft = []
            async with aclosing(generate_with_targets(llm, ctx, samples, max_steps=7)) as stream:
                async for token in stream:
                    if token in llm.eos_token_ids:
                        break
                    draft.append(token)
            return [], draft
        finally:
            for block in reversed(blocks):
                await llm.free_block(block)

    monkeypatch.setattr(demo, "LlamaPrompting", prompting)
    monkeypatch.setattr(demo, "_run_loop", scripted)
    llm, _ = _make_llm()
    llm.engine = SimpleNamespace(
        config=SimpleNamespace(
            model_config=SimpleNamespace(model_type="llama"), get_eos_token_ids=lambda tok: {11}
        )
    )
    llm.tokenizer = Tokenizer()
    trace = tmp_path / "phases" / "result.json"
    config = SimpleNamespace(
        model="tiny",
        problem="example",
        llama_protocol="test-protocol",
        trace_output=trace,
        max_answer_tokens=9,
    )
    outcome = asyncio.run(short_retry._run(config, llm))
    assert (
        outcome.source,
        outcome.draft_ids,
        outcome.draft_stop_id,
        outcome.retry_ids,
        outcome.retry_stop_id,
    ) == ("native_retry", [5, 7, 9], 11, [9], 11)
    stdout = capsys.readouterr().out
    assert "DISCARDED-DRAFT-MARKER" not in stdout and stdout.count("selected-9") == 1
    data = json.loads(trace.read_bytes())
    (selected,) = data["examples"]
    phases = data["short_answer_retry"]
    assert phases["sampled_tokens"] == 6 and phases["source"] == "native_retry"
    assert selected["writer_tokens"] == [9] and phases["discarded_draft"]["writer_tokens"] == [
        5,
        7,
        9,
    ]
    assert [t for t in selected["targets"] if t != -100] == [9, 11]
    assert [t for t in phases["discarded_draft"]["targets"] if t != -100] == [5, 7, 9, 11]
    assert llm._closed


def test_rejects_false_eos_and_overdrawn_budget():
    class LLM:
        eos_token_ids = {2}

    for draft, stop, budget, message in [
        ([3], 1, 6, "sampled checkpoint"),
        ([2], None, 4, "contain EOS"),
        ([3, 4], 2, 2, "fit"),
    ]:
        with pytest.raises(ValueError, match=message):
            asyncio.run(
                retry_short_answer(
                    LLM(),
                    prompt_ids=[1],
                    draft_ids=draft,
                    draft_stop_id=stop,
                    max_answer_tokens=budget,
                )
            )
