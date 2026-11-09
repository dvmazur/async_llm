"""Paired scheduler witnesses for native evidence turns and private pending-token input."""

import asyncio
import json
from contextlib import nullcontext

import pytest
import torch
from minisgl.training import RecordedAsyncLLM
from minisgl_examples.demos.async_thoughts import demo
from minisgl_examples.demos.async_thoughts.__main__ import parse_args

from tests.core.test_async_llm import _make_llm
from tests.scripts.test_native_llama_chat import native_tokenizer
from tests.training.helpers import copy_operations


def drive(tmp_path, monkeypatch, gate, pending=False, *, max_think=42, vote=True):
    tok = native_tokenizer()
    prompt = demo.LlamaPrompting("task", tok, protocol="native-chat")
    output = tmp_path / f"{gate}-{pending}-{max_think}.json"
    config = demo.DemoConfig(
        max_steps=max_think,
        max_answer_tokens=17,
        probe_period=4,
        llama_gate=gate,
        probe_include_pending=pending,
        probe_log=output,
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        demo.ModeSwitchProbe, "_vote", lambda self, logits: (vote, float(vote), float(not vote))
    )

    async def run():
        llm, session = _make_llm()
        copy_operations(session)
        llm.tokenizer = tok
        llm.eos_token_ids = frozenset()
        try:
            with RecordedAsyncLLM(llm) as rec:
                rec.eos_token_ids = frozenset()
                result = await demo._run_loop(config, rec, tok, prompt)
            return result, rec.get_trace()
        finally:
            await llm.close()

    (thoughts, writer), trace = asyncio.run(run())
    votes = json.loads(output.read_text())
    return thoughts, writer, trace, votes, prompt, tok


def test_each_evidence_turn_requires_a_parked_writer_and_new_notes(tmp_path, monkeypatch):
    _, writer, trace, votes, prompt, tok = drive(tmp_path, monkeypatch, "evidence-turns")
    events = votes["events"]
    assert votes["llama_gate"] == "evidence-turns"
    assert len(events) >= 4 and all(event["vote_write"] for event in events)
    assert events[0]["source"] == "standalone"
    assert all(event["source"] == "live_cache" for event in events[1:])
    assert all(not event["writer_allowed_before"] for event in events)
    for earlier, later in zip(events, events[1:]):
        assert later["committed_thought_tokens"] > earlier["committed_thought_tokens"]
        assert 0 < later["sampled_answer_tokens"] - earlier["sampled_answer_tokens"] <= 4
    assert len(writer) == 17
    live = [r for b in trace.batches for r in b.requests if len(r.cache_view) == 4]
    assert all(
        list(r.input_ids) == tok.encode(prompt.next_step_question(), add_special_tokens=False)
        for r in live
    )
    assert any(b.kind == "decode" and len(b.requests) == 2 for b in trace.batches)
    _, _, _, control, _, _ = drive(tmp_path, monkeypatch, "periodic")
    assert any(e["writer_allowed_before"] for e in control["events"] if e["source"] == "live_cache")


@pytest.mark.parametrize("gate", ["periodic", "evidence-turns"])
def test_only_private_question_receives_pending_sample(tmp_path, monkeypatch, gate):
    _, writer, trace, votes, prompt, tok = drive(tmp_path, monkeypatch, gate, pending=True)
    live = [r for b in trace.batches for r in b.requests if len(r.cache_view) == 4]
    events = [e for e in votes["events"] if e["source"] == "live_cache"]
    assert votes["probe_include_pending"] and len(events) == len(live) > 0
    question = (
        prompt.next_step_question() if gate == "evidence-turns" else prompt.mode_switching_question
    )
    suffix = tok.encode(question, add_special_tokens=False)
    for q, event in zip(live, events, strict=True):
        n = event["committed_answer_tokens"]
        assert event["query_pending_answer_token_id"] == writer[n]
        assert list(q.input_ids) == [writer[n], *suffix]
        assert event["scored_input_tokens"] == len(suffix) + 1
    writer_steps = [
        r
        for b in trace.batches
        for r in b.requests
        if b.kind == "decode" and len(r.cache_view) == 3
    ]
    assert [r.input_ids[0] for r in writer_steps] == [
        prompt.native_turns.writer_ids[-1],
        *writer[:-1],
    ]


def test_thinker_completion_releases_writer_after_wait(tmp_path, monkeypatch):
    thoughts, writer, _, votes, _, _ = drive(
        tmp_path, monkeypatch, "evidence-turns", max_think=9, vote=False
    )
    assert len(thoughts) == 9 and len(writer) == 17
    assert votes["events"] and all(not e["vote_write"] for e in votes["events"])


def test_gate_cli_and_incompatible_protocol():
    assert parse_args([]).llama_gate == "periodic"
    assert not parse_args([]).probe_include_pending
    opts = parse_args(
        [
            "--llama-protocol",
            "native-chat",
            "--llama-gate",
            "evidence-turns",
            "--probe-include-pending",
        ]
    )
    assert opts.llama_gate == "evidence-turns" and opts.probe_include_pending
    with pytest.raises(SystemExit):
        parse_args(["--llama-gate", "evidence-turns"])
