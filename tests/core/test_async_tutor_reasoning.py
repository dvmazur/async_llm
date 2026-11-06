"""Reasoning stays private and has a separate, bounded decoding budget."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from scripts.async_tutor.async_tutor.demo import IMAGE_CAPTION, AsyncLLMBackend, Prompts, Snapshot
from scripts.async_tutor.async_tutor.trace import Trace


class SpecialTokenizer:
    pieces = {0: "<eos>", 1: "</think>", 2: "<think>", 3: "private", 4: "Final", 5: ".", 6: "\n\n"}

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join(
            self.pieces[token] for token in ids if not (skip_special_tokens and token in (0, 1, 2))
        )


class ScriptedLLM:
    def __init__(self, tokenizer, tokens, eos=0):
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=eos))
        self.tokens = iter(tokens)
        self.blocks, self.freed, self.calls = [], [], []
        self.sample_count = 0

    async def create_block(self):
        block = SimpleNamespace(num_tokens=0)
        self.blocks.append(block)
        return block

    async def free_block(self, block):
        self.freed.append(block)

    async def sample(self, output, *, temperature):
        self.sample_count += 1
        return torch.tensor(next(self.tokens))

    async def forward(self, input_ids, *, cache_view=None, write_to=None, **kwargs):
        ids = torch.as_tensor(input_ids).flatten().tolist()
        self.calls.append(ids)
        block = write_to if write_to is not None else cache_view[-1]
        block.num_tokens += len(ids)
        return SimpleNamespace(logits=torch.zeros(3))


def decode_script(tokens, *, reasoning_tokens=4, limit=2, stop_after=None):
    async def run():
        llm = ScriptedLLM(SpecialTokenizer(), tokens)
        queue = asyncio.Queue()
        backend = SimpleNamespace(
            llm=llm,
            eos={0},
            trace=Trace(queue),
            think_start_ids=[2],
            think_end_ids=[1],
            think_close_ids=[1, 6],
        )
        block = await llm.create_block()
        stopped = lambda: stop_after is not None and llm.sample_count >= stop_after
        answers = [
            text
            async for text in AsyncLLMBackend.generate(
                backend, [block], None, limit, stopped, reasoning_tokens=reasoning_tokens
            )
        ]
        records = [queue.get_nowait() for _ in range(queue.qsize())]
        return answers, records, llm

    return asyncio.run(run())


def test_special_think_delimiter_is_consumed_before_detokenizing_the_answer():
    answers, records, llm = decode_script([3, 1, 4, 5])
    assert answers == ["Final", "Final."]
    assert llm.calls == [[3], [1], [4], [5]]
    assert (
        next(record for record in records if record["event"] == "reasoning_done")["forced"] is False
    )


def test_reasoning_budget_forces_delimiter_and_preserves_the_full_answer_budget():
    answers, records, llm = decode_script([3, 3, 4, 5], reasoning_tokens=2)
    assert answers == ["Final", "Final."]
    assert llm.calls == [[3], [3], [1, 6], [4], [5]]
    done = next(record for record in records if record["event"] == "reasoning_done")
    assert done["forced"] is True
    assert done["reasoning_tokens"] == 2
    assert records[-1]["generated_tokens"] == 4


@pytest.mark.parametrize(
    "tokens, stop_after, reason", [([3, 0], None, "eos"), ([3, 3], 1, "stopped")]
)
def test_interrupted_reasoning_never_becomes_visible_text(tokens, stop_after, reason):
    answers, records, _ = decode_script(tokens, stop_after=stop_after)
    assert answers == []
    assert records[-1]["finish_reason"] == reason
    assert next(record for record in records if record["event"] == "reasoning_done")["incomplete"]


@pytest.mark.parametrize("tokens, reasoning_tokens", [([3, 1, 4, 2, 3], 4), ([3, 4, 2, 3], 1)])
def test_reopened_reasoning_stops_before_any_private_text_is_published(tokens, reasoning_tokens):
    answers, records, llm = decode_script(tokens, reasoning_tokens=reasoning_tokens, limit=8)
    assert answers == ["Final"]
    assert records[-1]["finish_reason"] == "reasoning_reopened"
    assert llm.calls[-1] == [2]


@pytest.fixture(scope="module")
def cached_qwen_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.8 tokenizer is not cached; no network or weights required")


@pytest.fixture(scope="module")
def cached_qwen_processor():
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained("Qwen/Qwen3.8-27B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.8 processor is not cached; no network or weights required")


def test_real_qwen_reply_thinks_while_probes_keep_the_same_system_prefix(cached_qwen_tokenizer):
    tokenizer = cached_qwen_tokenizer
    llm = ScriptedLLM(tokenizer, [], eos=tokenizer.eos_token_id)
    backend = AsyncLLMBackend(
        llm, prompts=Prompts(system="Tutor.", observe="Inspect.", speak="Decide.")
    )
    messages = [{"role": "user", "content": "Check 2 + 2."}]
    thinking = backend.encode_chat(messages, enable_thinking=True)
    direct = backend.encode_chat(messages)
    assert tokenizer.decode(thinking).endswith("<think>\n")
    assert tokenizer.decode(direct).endswith("<think>\n\n</think>\n\n")
    assert "Reasoning effort is set to low." in tokenizer.decode(backend.system_ids)
    expected = tokenizer.apply_chat_template(
        [{"role": "system", "content": backend.prompts.system}, *messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="low",
    )
    assert backend.system_ids + thinking == tokenizer.encode(expected, add_special_tokens=False)
    assert backend.system_ids + direct == tokenizer.encode(
        expected + "\n</think>\n\n", add_special_tokens=False
    )
    assert backend.reasoning_tokens > 0
    assert backend.reply_budget > backend.reply_tokens + backend.reasoning_tokens


@pytest.mark.parametrize("reasoning_tokens", [0, 2048])
def test_real_qwen_image_prefill_uses_the_same_system_prefix(
    cached_qwen_processor, reasoning_tokens
):
    async def run():
        processor = cached_qwen_processor
        llm = ScriptedLLM(processor.tokenizer, [], eos=processor.tokenizer.eos_token_id)
        llm.processor = processor
        backend = AsyncLLMBackend(
            llm,
            prompts=Prompts(system="Tutor.", observe="Inspect.", speak="Decide."),
            reasoning_tokens=reasoning_tokens,
        )
        image = Image.new("RGB", (96, 96), "white")
        expected = (
            processor.apply_chat_template(
                [
                    {"role": "system", "content": backend.prompts.system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": IMAGE_CAPTION},
                        ],
                    },
                ],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=bool(reasoning_tokens),
                reasoning_effort="low",
            )["input_ids"]
            .flatten()
            .tolist()
        )
        frame = None
        try:
            frame = await backend.prepare_frame(image, "None.", [])
            assert llm.calls[0] == backend.system_ids
            assert llm.calls[0] + llm.calls[1] == expected
            assert frame.num_tokens == len(expected) - len(backend.system_ids)
            system = processor.tokenizer.decode(backend.system_ids)
            assert ("Reasoning effort is set to low." in system) == bool(reasoning_tokens)
        finally:
            if frame is not None:
                await backend.free([frame])
            await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


def test_real_qwen_reply_hides_reasoning_and_frees_its_cache(cached_qwen_tokenizer):
    async def run():
        tokenizer = cached_qwen_tokenizer
        tokens = tokenizer.encode("Private calculation.</think>Correct.", add_special_tokens=False)
        llm = ScriptedLLM(tokenizer, [*tokens, tokenizer.eos_token_id], eos=tokenizer.eos_token_id)
        backend = AsyncLLMBackend(
            llm, prompts=Prompts(system="Tutor.", observe="Inspect.", speak="Decide.")
        )
        frame = await llm.create_block()
        snapshot = Snapshot(1, [frame], "TASK: 2 + 2\nWORK: 4")
        replies = [text async for text in backend.reply(snapshot, [], "Check this.", lambda: False)]
        assert replies[-1] == "Correct."
        assert all("Private" not in text and "think" not in text for text in replies)
        assert len(llm.freed) == 1
        await backend.free(snapshot.blocks)
        await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


def test_zero_reasoning_budget_closes_reply_prefix_and_negative_budget_is_rejected(
    cached_qwen_tokenizer,
):
    llm = ScriptedLLM(cached_qwen_tokenizer, [], eos=cached_qwen_tokenizer.eos_token_id)
    backend = AsyncLLMBackend(llm, reasoning_tokens=0)
    assert backend.reasoning_tokens == 0
    assert backend.reply_budget == backend.reply_tokens
    assert "Reasoning effort" not in cached_qwen_tokenizer.decode(backend.system_ids)
    text = cached_qwen_tokenizer.decode(backend.encode_chat([{"role": "user", "content": "Hi"}]))
    assert text.endswith("<think>\n\n</think>\n\n")
    with pytest.raises(ValueError, match="cannot be negative"):
        AsyncLLMBackend(llm, reasoning_tokens=-1)
