"""Demo export through the real async scheduler, with CPU stub model execution."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import aclosing
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from minisgl.shared_cache import AsyncContext
from minisgl.training.replay import Replay, forward
from minisgl.training.sft import restore_trace, supervised_loss
from minisgl.training.trace import RecordingSession
from test_async_llm import _make_llm
from test_training_gdn import tiny_hybrid
from test_training_logits import torch_reference_kernels as torch_reference_kernels

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts" / "async_thoughts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from async_thoughts import demo
from async_thoughts.training import DemoRecorder


@pytest.fixture
def exported_example(monkeypatch, tmp_path):
    path = tmp_path / "new" / "nested" / "demo.json"
    state = {}

    async def run_loop(config, llm, tokenizer, prompting):
        assert isinstance(llm, DemoRecorder)
        state["recorder"] = llm
        prompt, a, b, probe = [await llm.create_block() for _ in range(4)]
        state["writers"] = (a.block_id, b.block_id)
        await llm.forward([1, 2], write_to=prompt, return_logits=False)
        await asyncio.gather(
            llm.forward([3, 4], [prompt, a], return_logits=False),
            llm.forward([9, 10], [prompt, b], return_logits=False),
        )
        ca = AsyncContext(cache_view=[prompt, b, a])
        cb = AsyncContext(cache_view=[prompt, a, b], next_input_id=11)
        async with (
            aclosing(
                llm.async_generate(
                    ca, first_token_id=5, forbid_ids=[7], max_steps=2, return_logits=True
                )
            ) as ga,
            aclosing(llm.async_generate(cb, max_steps=2)) as gb,
        ):
            # Prefills run before decodes: both generators' row cursors must skip
            # this probe's input rows as well as distinguish their batched peer.
            first_a, first_b, _ = await asyncio.gather(
                anext(ga),
                anext(gb),
                llm.forward([20, 21], [prompt, a, b, probe], return_logits=True),
            )
            second_a, second_b = await asyncio.gather(anext(ga), anext(gb))
            assert first_a[0] == 6 and int(first_a[1].argmax()) == 7
            assert (second_a[0], first_b, second_b) == (8, 13, 15)
        # A budget flush consumes 8 but does not select another supervised target.
        await llm.forward(cache_view=ca, return_logits=False)
        assert ca.next_input_id is None and cb.next_input_id == 15
        assert a.token_ids == [3, 4, 5, 6, 8]
        assert b.token_ids == [9, 10, 11, 13]
        state["decode_calls"] = len(llm.async_engine.session.session.decode_calls)
        for block in (probe, a, b, prompt):
            await llm.free_block(block)
        # Treat 15 as a terminal: excluded from displayed output, still a target.
        return [6, 8], [13]

    monkeypatch.setattr(demo, "_run_loop", run_loop)

    async def run():
        llm, _ = _make_llm()
        config = demo.DemoConfig(model="stub-model", problem="test", trace_output=path)
        result = await demo._run_demo(config, llm, demo.Prompting(config.problem))
        assert result == ([6, 8], [13])
        assert llm._closed and llm._loop_task is None

    asyncio.run(run())
    state["path"] = path
    state["data"] = json.loads(path.read_text())
    return state


def test_demo_export_aligns_batched_samples_and_masks_nongeneration_rows(exported_example):
    state = exported_example
    data = state["data"]
    (example,) = data["examples"]
    trace = restore_trace(example["trace"])
    assert data["model"] == "stub-model"
    assert example["prompt"] == "test"
    assert example["thinker_tokens"] == [6, 8] and example["writer_tokens"] == [13]
    assert trace == state["recorder"].trace
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


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
def test_export_reloads_for_hybrid_forward_and_backward(exported_example, read_mode):
    (example,) = exported_example["data"]["examples"]
    trace = restore_trace(example["trace"])
    replay = Replay.from_trace(trace, gdn_segment_size=2, gdn_read_mode=read_mode)
    targets = torch.tensor(example["targets"])
    rows = (targets != -100).nonzero().flatten()
    assert not torch.equal(replay.order, torch.arange(trace.num_tokens))
    model = tiny_hybrid().train()
    with torch.no_grad():
        logits = forward(model, replay)
        expected = F.cross_entropy(logits[rows], targets[rows], reduction="sum")
    loss = supervised_loss(model, replay, targets, chunk_size=2)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for parameter in (
        model.model.embed_tokens.weight,
        model.model.layers[0].linear_attn.in_proj_qkv.weight,
        model.lm_head.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


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
        recorder = DemoRecorder(llm)
        try:
            block = await recorder.create_block()
            await recorder.forward([1], write_to=block, return_logits=False)
            ctx = AsyncContext(cache_view=[block], next_input_id=5)
            async with aclosing(recorder.async_generate(ctx, max_steps=100)) as stream:
                assert await anext(stream) == 7
            assert closed == [True]
            assert ctx.next_input_id == 7 and block.token_ids == [1, 5]
            async with aclosing(recorder.async_generate(ctx, max_steps=1)) as stream:
                assert await anext(stream) == 9
                with pytest.raises(StopAsyncIteration):
                    await anext(stream)
            assert closed == [True, True]
            assert ctx.next_input_id == 9 and block.token_ids == [1, 5, 7]
            assert len(session.decode_calls) == 2 and recorder.trace.num_tokens == 3
        finally:
            await llm.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["prefill", "decode", "consumer", "cancel"])
def test_failed_demo_does_not_export_an_artifact(monkeypatch, tmp_path, failure):
    path = tmp_path / "failed.json"
    state = {}

    async def run():
        llm, session = _make_llm()

        async def run_loop(config, recorder, tokenizer, prompting):
            state["recorder"] = recorder
            block = await recorder.create_block()
            if failure == "prefill":
                session.fail_next = RuntimeError("forward failed")
            await recorder.forward([1, 2], write_to=block, return_logits=False)
            if failure == "decode":
                session.fail_next = RuntimeError("forward failed")
            ctx = AsyncContext(cache_view=[block], next_input_id=5)
            async with aclosing(recorder.async_generate(ctx, max_steps=2)) as stream:
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
        assert state["recorder"].trace.num_tokens == expected_rows

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
        assert not isinstance(session, RecordingSession)
        assert llm._closed and llm._loop_task is None

    asyncio.run(run())
    assert path.read_bytes() == original
