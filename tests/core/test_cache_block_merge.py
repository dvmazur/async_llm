"""End-to-end cache-block merge test on the full Qwen3.5-9B model."""

from __future__ import annotations

import asyncio

import pytest
import torch
from PIL import Image

from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext


MODEL_PATH = "Qwen/Qwen3.5-9B"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cache-block merge test requires CUDA",
)


@pytest.fixture
def runtime(tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    llm = AsyncLLM(
        MODEL_PATH,
        dtype=torch.bfloat16,
        max_running_req=2,
        memory_ratio=0.65,
        attention_backend="fi",
        distributed_addr=(tmp_path / "distributed_init").as_uri(),
    )
    yield loop, llm
    loop.run_until_complete(llm.close())
    loop.close()
    asyncio.set_event_loop(None)


def assert_same_distribution(left: torch.Tensor, right: torch.Tensor) -> None:
    left = left.float().softmax(-1)
    right = right.float().softmax(-1)
    torch.testing.assert_close(left, right, rtol=0, atol=2e-3)
    assert left.argmax().item() == right.argmax().item()


def test_merged_block_matches_split_view(runtime) -> None:
    loop, llm = runtime

    async def run() -> None:
        left, right, split_tail, merged_tail = await asyncio.gather(
            *(llm.create_block() for _ in range(4))
        )

        image_inputs = llm.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.new("RGB", (256, 192))},
                        {"type": "text", "text": "Analyze this image carefully."},
                    ],
                }
            ],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        await llm(**image_inputs, cache_view=[left], write_to=left, return_logits=False)

        right_ids = llm.tokenizer(
            "<|im_start|>assistant\n<think>\nThe image shows a structured visual pattern.",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        await llm(right_ids, cache_view=[left, right], write_to=right, return_logits=False)

        left_tokens = left.token_ids.copy()
        right_tokens = right.token_ids.copy()
        merged = await llm.merge_blocks(left, right)

        assert left.token_ids == left_tokens
        assert right.token_ids == right_tokens
        assert merged.token_ids == left_tokens + right_tokens
        assert merged.mrope_span == left.mrope_span + right.mrope_span

        probe_ids = llm.tokenizer(
            " Continue the analysis using the visual evidence:",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        split, compact = await asyncio.gather(
            llm(probe_ids, cache_view=[left, right, split_tail], write_to=split_tail),
            llm(probe_ids, cache_view=[merged, merged_tail], write_to=merged_tail),
        )
        assert_same_distribution(split.logits, compact.logits)

        next_token = int(split.logits.argmax())
        split_ctx = AsyncContext([left, right, split_tail], split_tail, next_token)
        merged_ctx = AsyncContext([merged, merged_tail], merged_tail, next_token)
        split, compact = await asyncio.gather(
            llm(cache_view=split_ctx),
            llm(cache_view=merged_ctx),
        )
        assert_same_distribution(split.logits, compact.logits)

    loop.run_until_complete(run())
