"""Probe observations use the original scored forwards and preserve gate behavior."""

import asyncio
import json

import pytest
from minisgl_examples.demos.async_thoughts import demo
from minisgl_examples.demos.async_thoughts.__main__ import parse_args

from tests.scripts.test_async_thoughts_demo import FakeLLM


def test_probe_log_matches_actual_admission_continuation_and_pending_answer(tmp_path):
    path = tmp_path / "nested" / "votes.json"
    config = parse_args(
        [
            "--probe-log",
            str(path),
            "--probe-period",
            "2",
            "--max-think-tokens",
            "40",
            "--max-answer-tokens",
            "4",
        ]
    )
    assert config.probe_log == path
    assert parse_args([]).probe_log is None

    def run(log):
        fake = FakeLLM([False, True, False] + [True] * 50)
        fake.use_llama("question")
        config.probe_log = log
        result = asyncio.run(demo._run_loop(config, fake, fake.tokenizer, fake.prompting))
        return fake, result

    plain, plain_result = run(None)
    logged, result = run(path)
    assert plain_result == result
    assert plain.probe_decisions == logged.probe_decisions
    assert plain.writer_feeds == logged.writer_feeds
    data = json.loads(path.read_text())
    assert data["llama_protocol"] == "working-notes"
    events = data["events"]
    assert [event["vote_write"] for event in events] == logged.probe_decisions
    assert events[0]["phase"] == events[1]["phase"] == "admission"
    assert events[0]["source"] == events[1]["source"] == "standalone"
    assert events[2]["source"] == "live_cache" and events[2]["phase"] == "continuation"
    assert [(e["yes_logit"], e["no_logit"]) for e in events[:3]] == [(0, 1), (1, 0), (0, 1)]
    assert [(e["writer_allowed_before"], e["writer_allowed_after"]) for e in events[:3]] == [
        (False, False),
        (False, True),
        (True, False),
    ]
    for event, observation in zip(events[:2], logged.start_observations, strict=True):
        assert event["thinker_block_tokens"] == len(observation.committed_thinker)
        assert event["committed_thought_tokens"] == len(observation.committed_thinker) - 2
    paused = [e for e in events if e["paused_at_writer_paragraph_before"]]
    assert paused
    for e in paused:
        assert e["writer_allowed_before"] is False
        assert e["sampled_answer_tokens"] == e["committed_answer_tokens"] + 1
        assert e["pending_answer_token_id"] == logged.tokenizer.vocab["\n\n"]
    assert all(e["discarded_reason"] is None for e in events)
    with pytest.raises(FileExistsError):
        run(path)


def test_late_no_is_logged_without_revoking_completion_release(tmp_path):
    path = tmp_path / "late.json"
    llm = FakeLLM([False], hold_probe=True, hold_writer=True)
    config = demo.DemoConfig(max_steps=6, max_answer_tokens=3, probe_period=1, probe_log=path)

    async def run():
        task = asyncio.create_task(demo._run_loop(config, llm, llm.tokenizer, llm.prompting))
        try:
            await asyncio.wait_for(llm.probe_entered.wait(), timeout=1)
            await asyncio.wait_for(llm.thinker_flushed.wait(), timeout=1)
            llm.probe_release.set()
            await asyncio.wait_for(llm.probe_finished.wait(), timeout=1)
        finally:
            llm.writer_release.set()
            llm.probe_release.set()
            await asyncio.wait_for(task, timeout=2)

    asyncio.run(run())
    (event,) = json.loads(path.read_text())["events"]
    assert event["source"] == "live_cache" and event["phase"] == "admission"
    assert event["vote_write"] is False
    assert event["discarded_reason"] == "thinker_done"
    assert event["writer_allowed_before"] is False and event["writer_allowed_after"] is True


def test_live_counts_match_forward_request_snapshots(monkeypatch, tmp_path):
    from contextlib import nullcontext

    import torch
    from minisgl.training import RecordedAsyncLLM

    from tests.core.test_async_llm import _make_llm
    from tests.scripts.test_native_llama_chat import native_tokenizer
    from tests.training.helpers import copy_operations

    output = tmp_path / "native.json"
    tok = native_tokenizer()
    llm, session = _make_llm()
    llm.tokenizer = tok
    copy_operations(session)
    llm.eos_token_ids = frozenset()
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())
    config = demo.DemoConfig(max_steps=28, max_answer_tokens=8, probe_period=2, probe_log=output)
    prompt = demo.LlamaPrompting("task", tok, protocol="native-chat")

    async def run():
        try:
            with RecordedAsyncLLM(llm) as rec:
                rec.eos_token_ids = frozenset()
                await demo._run_loop(config, rec, tok, prompt)
            return rec.get_trace()
        finally:
            await llm.close()

    trace = asyncio.run(run())
    data = json.loads(output.read_text())
    assert data["llama_protocol"] == "native-chat"
    live = [e for e in data["events"] if e["source"] == "live_cache"]
    requests = [r for b in trace.batches for r in b.requests if len(r.cache_view) == 4]
    assert live and len(live) == len(requests)
    for e, req in zip(live, requests):
        assert e["scored_input_tokens"] == len(req.input_ids)
        assert (e["thinker_block_tokens"], e["writer_block_tokens"]) == (
            req.cache_view[1][1],
            req.cache_view[2][1],
        )
        assert e["committed_answer_tokens"] == req.cache_view[2][1] - len(
            prompt.native_turns.writer_ids
        )
        assert e["committed_thought_tokens"] == req.cache_view[1][1] - 1
