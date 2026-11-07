"""Exercise writer permission with the real CPU scheduler and scripted model logits."""

import asyncio
from contextlib import nullcontext

import pytest
import torch
from minisgl_examples.demos.async_thoughts import demo

from tests.core.test_async_llm import _make_llm


@pytest.mark.parametrize("paragraph", [False, True])
def test_writer_never_advances_between_two_no_votes(monkeypatch, paragraph):
    # Prefill requests (including the probe) take precedence over ready decodes.
    # A writer already queued under a previous YES must finish before the probe
    # observes its state, rather than writing another token after the NO.
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *a, **k: nullcontext())

    class Tokenizer:
        eos_token_id = 31
        vocab = {"yes": 9, "no": 10}

        def encode(self, text, return_tensors=None, **kwargs):
            ids = {"\n": [4], "\n\n": [7], "yes": [9], "no": [10]}.get(text, [5, 6])
            return torch.tensor([ids]) if return_tensors else ids

        def decode(self, ids, **kwargs):
            return "".join("\n\n" if paragraph and token == 13 else "x" for token in ids)

    async def reproduce():
        llm, session = _make_llm()
        llm.tokenizer = Tokenizer()
        events, roles = [], {}

        def state_change(message):
            if message == "writer end-of-step -> thinker leads":
                events.append("paragraph")

        monkeypatch.setattr(demo, "_print_state_change", state_change)
        votes = iter([True, False, False, True])
        original_prefill, original_decode = session.prefill_block, session.decode_step

        def prefill(block, input_ids, context=None):
            logits = original_prefill(block, input_ids, context)
            if len(roles) < 3:
                roles[block.block_id] = ["prompt", "thinker", "writer"][len(roles)]
            elif len(context or []) == 3:
                yes = next(votes, True)
                events.append("yes" if yes else "no")
                logits.zero_()
                logits[0, 9 if yes else 10] = 100
            return logits

        writer_inputs = []

        def decode(group, ids):
            logits = original_decode(group, ids)
            logits.zero_()
            batch = []
            for i, (block, token) in enumerate(zip(group.write_to, ids.tolist())):
                role = roles[block.block_id]
                batch.append(role)
                if role == "writer":
                    writer_inputs.append(token)
                    logits[i, 11 if token == 7 else token + 1] = 100
                else:
                    logits[i, 11] = 100
            events.append(tuple(batch))
            return logits

        session.prefill_block, session.decode_step = prefill, decode
        config = demo.DemoConfig(max_steps=14, max_answer_tokens=9, probe_period=3)
        try:
            thoughts, answer = await demo._run_loop(
                config, llm, llm.tokenizer, demo.Prompting("test")
            )
        finally:
            await llm.close()
        no1 = events.index("no")
        no2 = events.index("no", no1 + 1)
        if paragraph:
            assert events.index("paragraph") < no1, events
        paused = events[no1 + 1 : no2]
        assert paused and all(batch == ("thinker",) for batch in paused), events
        assert any(isinstance(batch, tuple) and len(batch) == 2 for batch in events)
        assert answer == list(range(11, 20))
        assert writer_inputs == [7, *answer[:-1]]
        assert len(thoughts) == config.max_steps

    asyncio.run(asyncio.wait_for(reproduce(), timeout=5))
