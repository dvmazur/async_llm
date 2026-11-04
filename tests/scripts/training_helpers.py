"""Shared trace collection and checkpoint fixtures for demo workflow tests."""

import asyncio
import json
from contextlib import aclosing

import pytest
import torch
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM
from minisgl_examples.demos.async_thoughts import demo
from minisgl_examples.training.recording import generate_with_targets

from tests.core.test_async_llm import _make_llm


@pytest.fixture
def exported_example(monkeypatch, tmp_path):
    path = tmp_path / "new" / "nested" / "demo.json"
    state = {}

    async def run_loop(config, llm, tokenizer, prompting, samples):
        assert isinstance(llm, RecordedAsyncLLM)
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
                generate_with_targets(
                    llm,
                    ca,
                    samples,
                    first_token_id=5,
                    forbid_ids=[7],
                    max_steps=2,
                    return_logits=True,
                )
            ) as ga,
            aclosing(generate_with_targets(llm, cb, samples, max_steps=2)) as gb,
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
        state["decode_calls"] = len(state["session"].decode_calls)
        for block in (probe, a, b, prompt):
            await llm.free_block(block)
        # Treat 15 as a terminal: excluded from displayed output, still a target.
        return [6, 8], [13]

    monkeypatch.setattr(demo, "_run_loop", run_loop)

    async def run():
        llm, state["session"] = _make_llm()
        config = demo.DemoConfig(model="stub-model", problem="test", trace_output=path)
        result = await demo._run_demo(config, llm, demo.Prompting(config.problem))
        assert result == ([6, 8], [13])
        assert llm._closed and llm._loop_task is None

    with torch.device("cpu"):
        asyncio.run(run())
    state["path"] = path
    state["data"] = json.loads(path.read_text())
    return state
