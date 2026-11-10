"""Exercise the actual demo coroutines with CPU fake engine/tokenizer only."""
import asyncio
from types import SimpleNamespace

import pytest

from eval.async_thoughts import GemmaPrompting, live_token_bound, solve
from eval.math500 import INSTRUCTION, prompt_ids, read_config
from minisgl_examples.demos.async_thoughts import demo, engine


class Tokenizer:
    def __init__(self):
        names = ["<eos>", "</think>", "<|im_start|>", "<|endoftext|>", "<turn|>",
                 "<|turn>", "<|channel>", "<channel|>", "<|tool_call>", "\n", "\n\n", "yes", "no"]
        self.vocab = {name: i for i, name in enumerate(names)}
        self.eos_token_id = self.vocab["<eos>"]
        self.templates = []

    def get_vocab(self):
        return self.vocab

    def encode(self, text, **kwargs):
        if text in self.vocab:
            return [self.vocab[text]]
        return [self.vocab.setdefault(ch, len(self.vocab)) for ch in text]

    def decode(self, ids, **kwargs):
        reverse = {value: key for key, value in self.vocab.items()}
        return "".join(reverse[i] for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        system = "<|turn>system\n<|think|><turn|>\n" if kwargs["enable_thinking"] else ""
        return "<bos>" + system + "<|turn>user\n" + messages[0]["content"].strip() + "<turn|>\n"


class Logits:
    def __init__(self, yes):
        self.yes = yes
    def __getitem__(self, indices):
        return self
    def float(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return [2 if self.yes else -2, 0]


class FakeLLM:
    def __init__(self, *, thought=None, answer=None, yes=True, probe_delay=0, fail=False):
        self.tokenizer = Tokenizer()
        self.blocks, self.calls, self.params, self.probe_views = [], [], [], []
        self.queues = {"thinker": thought or ["a"] * 100 + ["<channel|>"],
                       "writer": answer or ["b", "\n\n", "c", "<eos>"]}
        self.yes, self.probe_delay, self.fail = yes, probe_delay, fail
        self.active_decode = self.peak_decode = 0

    async def create_block(self):
        block = SimpleNamespace(index=len(self.blocks), tokens=[], active=0, freed=False)
        self.blocks.append(block)
        return block

    async def free_block(self, block):
        assert block.active == 0 and not block.freed
        block.freed = True

    async def forward(self, ids=None, cache_view=None, *, write_to=None, return_logits=True):
        context = cache_view if hasattr(cache_view, "cache_view") else None
        view = context.cache_view if context else cache_view or [write_to]
        ids = [context.next_input_id] if context else list(ids)
        for block in view:
            assert not block.freed
            block.active += 1
        try:
            for _ in range(1 + (self.probe_delay if len(view) == 4 else 0)):
                await asyncio.sleep(0)
            view[-1].tokens.extend(ids)
            self.calls.append((view[-1].index, ids))
            if context:
                context.next_input_id = None
            if len(view) == 4:
                self.probe_views.append([b.index for b in view])
            return SimpleNamespace(logits=Logits(self.yes))
        finally:
            for block in view:
                block.active -= 1

    async def async_generate(self, context, *, sampling_params, max_steps, forbid_ids,
                             first_token_id=None, return_logits=False):
        assert first_token_id is None and not return_logits
        role = "thinker" if context.cache_view[-1].index == 1 else "writer"
        self.params.append((role, sampling_params, forbid_ids))
        for text in self.queues[role][:max_steps]:
            self.active_decode += 1
            self.peak_decode = max(self.peak_decode, self.active_decode)
            try:
                await self.forward(cache_view=context, return_logits=False)
                if self.fail:
                    raise RuntimeError("fake GPU failure")
                token = self.tokenizer.encode(text)[0]
                context.next_input_id = token
            finally:
                self.active_decode -= 1
            yield token


@pytest.fixture(autouse=True)
def cpu_encode(monkeypatch):
    encode = lambda text, tok: tok.encode(text, add_special_tokens=False)
    monkeypatch.setattr(engine, "encode", encode)
    monkeypatch.setattr(demo, "encode", encode)


def config(think=128, answer=16):
    return dict(max_think_tokens=think, max_answer_tokens=answer, probe_period=2,
                max_new_tokens=think+answer, engine=dict(page_size=16))


def execute(llm, settings=None):
    params = SimpleNamespace(temperature=1.0, top_p=.95, top_k=64)
    row = asyncio.run(asyncio.wait_for(solve(llm, "FULL PROBLEM", settings or config(), params,
        {llm.tokenizer.eos_token_id}, context_factory=SimpleNamespace), timeout=2))
    assert all(p is params for _, p, _ in llm.params)
    assert all(b.freed and b.active == 0 for b in llm.blocks)
    return row


def test_real_demo_coroutines_overlap_and_probe_reuses_live_blocks():
    llm = FakeLLM()
    row = execute(llm)
    assert llm.peak_decode == 2
    assert row["stop_reason"] == "eos" and row["answer_text"] == "b\n\nc"
    assert row["async_thoughts"]["probe_calls"] > 0
    assert all(view[1:3] == [1, 2] for view in llm.probe_views)
    assert len({view[0] for view in llm.probe_views}) == 1  # One cached static probe prompt.
    assert len({view[-1] for view in llm.probe_views}) == len(llm.probe_views)
    assert any(e["event"] == "state" and "end-of-step" in e["text"] for e in row["async_thoughts"]["events"])


def test_role_budget_and_final_thinker_flush_release_writer():
    llm = FakeLLM(yes=False, thought=["a"]*20, answer=["b"]*20)
    row = execute(llm, config(think=5, answer=4))
    assert row["generated_tokens"] == 9
    assert row["async_thoughts"]["sampled"] == dict(thinker=5, writer=4)
    assert row["async_thoughts"]["flush_tokens"] == 1
    assert row["stop_reason"] == "length"
    thinker = llm.tokenizer.decode(llm.blocks[1].tokens)
    assert thinker.endswith("aaaaa")


def test_stale_negative_probe_cannot_park_writer_after_thinking_finishes():
    llm = FakeLLM(yes=False, thought=["a"]*3 + ["<channel|>"], probe_delay=30)
    row = execute(llm)
    assert row["stop_reason"] == "eos"
    assert any(e.get("ignored") for e in row["async_thoughts"]["events"] if e["event"] == "probe")


def test_error_drains_and_frees_all_blocks():
    llm = FakeLLM(fail=True)
    with pytest.raises(RuntimeError, match="fake GPU failure"):
        execute(llm)
    assert all(b.freed and b.active == 0 for b in llm.blocks)


def test_cancel_drains_pending_prefill():
    async def scenario():
        llm = FakeLLM()
        task = asyncio.create_task(solve(llm, "problem", config(), None, {0}, context_factory=SimpleNamespace))
        while not any(b.active for b in llm.blocks):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(b.freed and b.active == 0 for b in llm.blocks)
    asyncio.run(scenario())


def test_gemma_changes_framing_not_probe_prose_or_pending_headers():
    tok = Tokenizer()
    original = demo.Prompting("problem")
    gemma = GemmaPrompting("problem", tok, {0, 4})
    assert gemma.writer_output_prefix == original.writer_output_prefix.replace("</think>", "<channel|>")
    assert gemma.mode_switching_question == original.mode_switching_question
    assert tok.templates[1][0][0]["content"] == original.mode_switching_prompt.removeprefix("<|im_start|>user\n")
    assert gemma.thinker_output_prefix.endswith("thought")  # The demo feeds newline separately.
    assert gemma.input_prompt.endswith("problem\n")
    assert "<|im_start|>" not in gemma.mode_switching_prompt


def test_default_budget_and_memory_account_for_both_roles_and_probe():
    cfg = read_config(model_path="unused")
    assert cfg["max_new_tokens"] == 16384
    assert cfg["max_think_tokens"] == cfg["max_answer_tokens"] == 8192
    assert cfg["engine"]["max_running_req"] == 4
    assert live_token_bound(Tokenizer(), "problem", cfg) > cfg["max_new_tokens"]


def test_sglang_prefix_matches_demo_thinker_token_boundaries():
    tokenizer = Tokenizer()
    framing = GemmaPrompting(INSTRUCTION + "\n\nproblem", tokenizer, {0})
    expected = (tokenizer.encode(framing.input_prompt) +
                tokenizer.encode(framing.thinker_output_prefix) + tokenizer.encode("\n"))
    assert prompt_ids(tokenizer, "problem", True, demo_prefix=True) == expected
