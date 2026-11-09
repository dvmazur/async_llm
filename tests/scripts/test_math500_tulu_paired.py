import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl_examples.demos.async_thoughts.math500_tulu_paired import (
    Prompt,
    Sampler,
    run_serial,
    validate_sample_log,
    visible_at_sample_budget,
)

from tests.core.test_async_llm import _make_llm


@pytest.mark.parametrize(
    "prompt,cap,budget,expected_draft,expected_answer,expected_input,expected_stops",
    [
        # EOS follows a short private draft. It is charged and appears only once in the bridge.
        ([2, 4], 4, 6, [6, 8], [6, 8], [2, 4, 6, 8, 10, 13, 4, 6, 8], (10, 10)),
        # The private draft hit its cap. The last content token is pending and must precede the EOS delimiter.
        ([1, 2], 2, 5, [4, 6], [6, 8], [1, 2, 4, 6, 10, 13, 4, 6, 8], (None, 10)),
        # The first sample is EOS. The entire remaining sample allowance is available to the writer.
        ([1, 8], 2, 4, [], [6, 8], [1, 8, 10, 13, 4, 6, 8], (10, 10)),
    ],
)
def test_serial_reuses_cache_and_charges_real_samples(
    monkeypatch,
    prompt,
    cap,
    budget,
    expected_draft,
    expected_answer,
    expected_input,
    expected_stops,
):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())

    async def run():
        llm, _ = _make_llm()
        llm.tokenizer = SimpleNamespace(eos_token_id=10)
        try:
            with Sampler(llm) as rec:
                t, w = await run_serial(
                    rec, Prompt("question", "private", prompt, [10, 13, 4], "finish"), budget, cap
                )
                assert (t, w) == (expected_draft, expected_answer)
                counts, stops = validate_sample_log(
                    rec.events, t, w, {10}, "serial-thinking", budget, cap
                )
                assert (stops["thinker"], stops["writer"]) == expected_stops
                batches = rec.get_trace().batches
                assert [
                    x for b in batches for q in b.requests for x in q.input_ids
                ] == expected_input
                assert sum(counts.values()) == sum(b.kind == "decode" for b in batches)
                assert len({event["batch_step"] for event in rec.events}) == sum(counts.values())
                assert visible_at_sample_budget(rec.events, budget, {10}) == w
        finally:
            await llm.close()

    asyncio.run(run())


def test_pooled_prefix_preserves_whole_simultaneous_decode():
    events = [
        {"role": "thinker", "batch_step": 2, "token": 12},
        {"role": "writer", "batch_step": 4, "token": 19},
        {"role": "thinker", "batch_step": 4, "token": 32},
        {"role": "writer", "batch_step": 5, "token": 10},
    ]
    assert visible_at_sample_budget(events, 2, {10}) == []
    assert visible_at_sample_budget(events, 3, {10}) == [19]
    assert visible_at_sample_budget(events, 4, {10}) == [19]
