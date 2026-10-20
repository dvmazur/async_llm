"""CPU policy checks plus an opt-in GPU smoke test for the async-thoughts demo.

The fake LLM checks real cache views, pending tokens, and coroutine lifetimes;
only model execution is replaced. GPU execution requires MINISGL_E2E_MODEL.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

_SCRIPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "async_thoughts")
)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from async_thoughts import demo


class Tokenizer:
    def __init__(self):
        self.vocab, self.text = {}, {}
        for text in ("yes", "no", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            self.encode(text)
        self.eos_token_id = self.vocab["<|im_end|>"]

    def encode(self, text, return_tensors=None, **kwargs):
        if text not in self.vocab:
            token = len(self.vocab)
            self.vocab[text], self.text[token] = token, text
        ids = [self.vocab[text]]
        return torch.tensor([ids]) if return_tensors == "pt" else ids

    def decode(self, tokens, **kwargs):
        return "".join(self.text[int(token)] for token in tokens)


class FakeLLM:
    def __init__(
        self,
        decisions=(),
        writer=("A", "\n\n", "B", "<|im_end|>"),
        thinker=None,
        fail=None,
        hold_probe=False,
        hold_writer=False,
    ):
        self.tokenizer = Tokenizer()
        self.prompting = demo.Prompting("test")
        self.question_ids = self.tokenizer.encode(self.prompting.mode_switching_question)
        self.decisions = list(decisions)
        self.writer = [self.tokenizer.encode(text)[0] for text in writer]
        self.thinker = (
            [self.tokenizer.encode(text)[0] for text in thinker] if thinker is not None else None
        )
        self.fail = fail
        self.blocks, self.freed, self.active = [], [], Counter()
        self.probe_views, self.probe_decisions, self.writer_feeds = [], [], []
        self.thinker_tokens, self.first_writer_thoughts = [], None
        self.probe_entered, self.probe_finished = asyncio.Event(), asyncio.Event()
        self.probe_release, self.writer_release = asyncio.Event(), asyncio.Event()
        self.thinker_flushed = asyncio.Event()
        self.probe_cancelled = False
        if not hold_probe:
            self.probe_release.set()
        if not hold_writer:
            self.writer_release.set()

    async def create_block(self):
        block = SimpleNamespace(token_ids=[], index=len(self.blocks))
        self.blocks.append(block)
        return block

    async def free_block(self, block):
        assert not self.active[block.index], "freed an in-flight cache view"
        assert block.index not in self.freed, "freed a block twice"
        self.freed.append(block.index)

    async def forward(self, input_ids=None, cache_view=None, *, write_to=None, return_logits=True):
        if input_ids is None:
            ctx = cache_view
            self.active.update(block.index for block in ctx.cache_view)
            try:
                await asyncio.sleep(0)
                assert not set(self.freed).intersection(block.index for block in ctx.cache_view)
                assert ctx.next_input_id is not None
                ctx.output_block.token_ids.append(ctx.next_input_id)
                ctx.next_input_id = None
                if ctx.output_block is self.blocks[1]:
                    self.thinker_flushed.set()
                return SimpleNamespace(logits=None)
            finally:
                self.active.subtract(block.index for block in ctx.cache_view)

        ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
        view = list(cache_view or [])
        if write_to is not None:
            assert write_to not in view
            view.append(write_to)
        self.active.update(block.index for block in view)
        is_probe = ids == self.question_ids
        try:
            await asyncio.sleep(0)
            assert not set(self.freed).intersection(block.index for block in view)
            view[-1].token_ids.extend(ids)
            if not is_probe:
                return SimpleNamespace(logits=None)
            assert len(view) == 4
            assert view[1:3] == self.blocks[1:3], "probe must reuse live stream KV"
            assert view[0] not in self.blocks[:3]
            assert view[-1] not in view[:-1]
            self.probe_views.append(tuple(block.index for block in view))
            self.probe_entered.set()
            try:
                await self.probe_release.wait()
            except asyncio.CancelledError:
                self.probe_cancelled = True
                raise
            if self.fail == "probe":
                raise RuntimeError("probe failed")
            allowed = self.decisions.pop(0) if self.decisions else False
            self.probe_decisions.append(allowed)
            logits = torch.zeros(len(self.tokenizer.vocab))
            logits[self.tokenizer.vocab["yes"]] = float(allowed)
            logits[self.tokenizer.vocab["no"]] = float(not allowed)
            return SimpleNamespace(logits=logits)
        finally:
            self.active.subtract(block.index for block in view)
            if is_probe:
                self.probe_finished.set()

    async def async_generate(self, ctx, *, forbid_ids, max_steps):
        writer = ctx.output_block is self.blocks[2]
        role = "writer" if writer else "thinker"
        assert ctx.cache_view == (self.blocks[:3] if writer else self.blocks[:2])
        for step in range(max_steps):
            self.active.update(block.index for block in ctx.cache_view)
            try:
                await asyncio.sleep(0)
                if writer:
                    await self.writer_release.wait()
                assert not set(self.freed).intersection(block.index for block in ctx.cache_view)
                if self.fail == role and (writer or step == 1):
                    raise RuntimeError(f"{role} failed")
            finally:
                self.active.subtract(block.index for block in ctx.cache_view)
            pending = ctx.next_input_id
            assert pending is not None
            ctx.output_block.token_ids.append(pending)
            if writer:
                if self.first_writer_thoughts is None:
                    self.first_writer_thoughts = list(self.blocks[1].token_ids)
                self.writer_feeds.append((pending, len(self.thinker_tokens), len(self.probe_views)))
                token = self.writer[step]
            else:
                token = (
                    self.thinker[step]
                    if self.thinker is not None
                    else self.tokenizer.encode(f"thought{step}")[0]
                )
                self.thinker_tokens.append(token)
            assert token not in forbid_ids
            ctx.next_input_id = token
            yield token


class TestAsyncThoughtsPolicy(unittest.IsolatedAsyncioTestCase):
    async def run_demo(self, llm, **kwargs):
        config = demo.DemoConfig(problem="test", **kwargs)
        return await asyncio.wait_for(
            demo._run_loop(config, llm, llm.tokenizer, llm.prompting), timeout=2
        )

    def assert_released(self, llm):
        self.assertEqual(sorted(llm.freed), list(range(len(llm.blocks))))
        self.assertFalse(any(llm.active.values()))
        for block in llm.blocks[:3]:
            self.assertFalse(set(llm.question_ids).intersection(block.token_ids))
        if llm.probe_views:
            self.assertEqual(len({view[0] for view in llm.probe_views}), 1)
            question_blocks = [view[-1] for view in llm.probe_views]
            self.assertEqual(len(set(question_blocks)), len(question_blocks))
            for index in question_blocks:
                self.assertEqual(llm.blocks[index].token_ids, llm.question_ids)

    async def test_live_views_and_writer_resume_after_paragraph_pause(self):
        llm = FakeLLM([True, False, False, True] + [True] * 100)
        thoughts, answer = await self.run_demo(
            llm, max_steps=100, max_answer_tokens=4, probe_period=1
        )
        self.assertEqual(answer, llm.writer[:-1])
        self.assertLess(len(thoughts), 100)
        self.assertEqual(llm.probe_decisions[:4], [True, False, False, True])
        seed = llm.tokenizer.encode("\n\n")[0]
        self.assertEqual([feed[0] for feed in llm.writer_feeds], [seed, *llm.writer[:-1]])
        self.assertGreater(llm.writer_feeds[2][2], llm.writer_feeds[1][2])
        self.assert_released(llm)

    async def test_thinker_budget_commits_last_token_and_releases_writer(self):
        llm = FakeLLM(writer=("answer", "<|im_end|>"))
        thoughts, answer = await self.run_demo(
            llm, max_steps=3, max_answer_tokens=2, probe_period=1
        )
        self.assertEqual(thoughts, llm.thinker_tokens)
        self.assertEqual(len(thoughts), 3)
        self.assertFalse(any(llm.probe_decisions))
        self.assertEqual(answer, llm.writer[:-1])
        self.assertEqual(llm.first_writer_thoughts[-3:], thoughts)
        self.assertEqual(llm.blocks[1].token_ids.count(thoughts[-1]), 1)
        self.assert_released(llm)

    async def test_natural_thinker_end_releases_writer_without_committing_terminal(self):
        for terminal in ("</think>", "<|im_end|>"):
            with self.subTest(terminal=terminal):
                llm = FakeLLM(
                    thinker=("thought0", "thought1", terminal), writer=("answer", "<|im_end|>")
                )
                thoughts, answer = await self.run_demo(
                    llm, max_steps=10, max_answer_tokens=2, probe_period=1
                )
                self.assertEqual(thoughts, llm.thinker[:-1])
                self.assertEqual(answer, llm.writer[:-1])
                self.assertEqual(llm.first_writer_thoughts[-2:], thoughts)
                self.assertNotIn(llm.thinker[-1], llm.blocks[1].token_ids)
                self.assert_released(llm)

    async def test_writer_eos_stops_background_thinking(self):
        llm = FakeLLM([True] * 100, writer=("<|im_end|>",))
        thoughts, answer = await self.run_demo(
            llm, max_steps=100, max_answer_tokens=1, probe_period=1
        )
        self.assertEqual(answer, [])
        self.assertLess(len(thoughts), 100)
        self.assert_released(llm)

    async def test_answer_budget_at_paragraph_stops_without_waiting_for_probe(self):
        llm = FakeLLM([True], writer=("\n\n",))
        thoughts, answer = await self.run_demo(
            llm, max_steps=100, max_answer_tokens=1, probe_period=1
        )
        self.assertEqual(answer, llm.writer)
        self.assertLess(len(thoughts), 100)
        self.assert_released(llm)

    async def test_probe_is_independent_and_late_no_cannot_park_finished_thinker(self):
        llm = FakeLLM([False], hold_probe=True, hold_writer=True)
        task = asyncio.create_task(
            self.run_demo(llm, max_steps=6, max_answer_tokens=4, probe_period=1)
        )
        try:
            await asyncio.wait_for(llm.probe_entered.wait(), timeout=1)
            started_at = len(llm.thinker_tokens)
            await asyncio.wait_for(llm.thinker_flushed.wait(), timeout=1)
            self.assertGreater(len(llm.thinker_tokens), started_at)
            self.assertEqual(len(llm.thinker_tokens), 6)
            self.assertFalse(task.done())
            llm.probe_release.set()
            await asyncio.wait_for(llm.probe_finished.wait(), timeout=1)
        finally:
            llm.probe_release.set()
            llm.writer_release.set()
        thoughts, answer = await task
        self.assertEqual(thoughts, llm.thinker_tokens)
        self.assertEqual(answer, llm.writer[:-1])
        self.assertEqual(llm.probe_decisions, [False])
        self.assert_released(llm)

    async def test_outer_cancellation_drains_initial_prefill(self):
        entered, release = asyncio.Event(), asyncio.Event()

        class PausedPrefill(FakeLLM):
            cancelled = False

            async def forward(self, input_ids=None, cache_view=None, **kwargs):
                if input_ids is not None and cache_view == self.blocks[:1]:
                    self.active.update(block.index for block in cache_view)
                    entered.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        self.cancelled = True
                        raise
                    finally:
                        self.active.subtract(block.index for block in cache_view)
                return await super().forward(input_ids, cache_view, **kwargs)

        llm = PausedPrefill()
        config = demo.DemoConfig(problem="test")
        task = asyncio.create_task(demo._run_loop(config, llm, llm.tokenizer, llm.prompting))
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        try:
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertFalse(llm.cancelled)
            self.assertEqual(llm.freed, [])
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        self.assertEqual(llm.thinker_tokens, [])
        self.assert_released(llm)

    async def test_outer_cancellation_drains_in_flight_probe(self):
        llm = FakeLLM([True], hold_probe=True)
        config = demo.DemoConfig(problem="test", max_steps=100, probe_period=1)
        task = asyncio.create_task(demo._run_loop(config, llm, llm.tokenizer, llm.prompting))
        await asyncio.wait_for(llm.probe_entered.wait(), timeout=1)
        task.cancel()
        try:
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertFalse(llm.probe_cancelled)
        finally:
            llm.probe_release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        self.assert_released(llm)

    async def test_failures_drain_siblings_and_release_all_blocks(self):
        for role in ("thinker", "writer", "probe"):
            with self.subTest(role=role):
                llm = FakeLLM([True] * 100, fail=role)
                with self.assertRaisesRegex(RuntimeError, f"{role} failed"):
                    await self.run_demo(llm, max_steps=100, max_answer_tokens=4, probe_period=1)
                self.assert_released(llm)


E2E_MODEL_PATH = os.environ.get("MINISGL_E2E_MODEL", "")


@pytest.mark.skipif(
    not E2E_MODEL_PATH or not torch.cuda.is_available(),
    reason="Set MINISGL_E2E_MODEL to an HF model path and ensure CUDA is available",
)
def test_demo_smoke():
    from async_thoughts.engine import build_async_llm, vocab_id_or_none
    from transformers import AutoTokenizer

    config = demo.DemoConfig(
        model=E2E_MODEL_PATH,
        max_steps=25,
        max_answer_tokens=64,
        probe_period=8,
        memory_ratio=0.7,
    )
    llm = build_async_llm(config.model, memory_ratio=config.memory_ratio)
    thinker_tokens, writer_tokens = asyncio.run(
        demo._run_demo(config, llm, demo.Prompting(config.problem))
    )

    tokenizer = AutoTokenizer.from_pretrained(E2E_MODEL_PATH, trust_remote_code=True)
    assert 0 <= len(thinker_tokens) <= config.max_steps
    assert 1 <= len(writer_tokens) <= config.max_answer_tokens
    # Return values contain generated content only; terminal markers end a stream.
    forbidden = {
        i
        for i in (
            vocab_id_or_none(tokenizer, n)
            for n in ("</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")
        )
        if i is not None
    }
    if tokenizer.eos_token_id is not None:
        forbidden.add(tokenizer.eos_token_id)
    assert not forbidden & set(thinker_tokens)
    assert not forbidden & set(writer_tokens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
