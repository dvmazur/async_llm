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


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    old_reduced_precision = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    llm = AsyncLLM(
        MODEL_PATH,
        dtype=torch.bfloat16,
        max_running_req=2,
        memory_ratio=0.65,
        attention_backend="fi",
        distributed_addr=(
            tmp_path_factory.mktemp("cache_merge") / "distributed_init"
        ).as_uri(),
    )
    yield loop, llm
    loop.run_until_complete(llm.close())
    loop.close()
    asyncio.set_event_loop(None)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced_precision


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

        for block in (left, right, split_tail, merged, merged_tail):
            await llm.free_block(block)

    loop.run_until_complete(run())


def test_merge_consume_modes(runtime) -> None:
    loop, llm = runtime

    async def run() -> None:
        allocator = llm.async_engine.session.page_allocator

        for consume_left, consume_right in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            free_pages_before = allocator.num_free_pages
            left, right, split_tail, merged_tail = await asyncio.gather(
                *(llm.create_block() for _ in range(4))
            )

            left_ids = llm.tokenizer(
                "<|im_start|>user\nExplain why cache composition should preserve logits.",
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]
            await llm(left_ids, cache_view=[left], write_to=left, return_logits=False)

            right_ids = llm.tokenizer(
                "<|im_end|>\n<|im_start|>assistant\n<think>First inspect both cache segments.",
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]
            await llm(
                right_ids,
                cache_view=[left, right],
                write_to=right,
                return_logits=False,
            )

            left_tokens = left.token_ids.copy()
            right_tokens = right.token_ids.copy()
            left_pages = left.page_starts.copy()
            right_pages = right.page_starts.copy()
            left_conv_ptrs = {
                layer_idx: state.data_ptr()
                for layer_idx, state in left.linear_conv_state.items()
            }
            right_conv_ptrs = {
                layer_idx: state.data_ptr()
                for layer_idx, state in right.linear_conv_state.items()
            }

            probe_ids = llm.tokenizer(
                " Therefore the next conclusion is",
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]
            split = await llm(
                probe_ids,
                cache_view=[left, right, split_tail],
                write_to=split_tail,
            )
            next_token = int(split.logits.argmax())
            split_ctx = AsyncContext([left, right, split_tail], split_tail, next_token)
            split_decode = await llm(cache_view=split_ctx)

            merged = await llm.merge_blocks(
                left,
                right,
                consume_left=consume_left,
                consume_right=consume_right,
            )

            assert merged.token_ids == left_tokens + right_tokens
            if consume_left:
                assert merged.page_starts[: len(left_pages)] == left_pages
                assert left.is_consumed
                with pytest.raises(RuntimeError, match="consumed"):
                    await llm.free_block(left)
            else:
                assert not left.is_consumed
                assert left.token_ids == left_tokens
                assert left.page_starts == left_pages
                assert {
                    layer_idx: state.data_ptr()
                    for layer_idx, state in left.linear_conv_state.items()
                } == left_conv_ptrs

            if consume_right:
                adopted_right_pages = merged.page_starts[len(left_pages) :]
                assert adopted_right_pages == right_pages[: len(adopted_right_pages)]
                assert right.is_consumed
                with pytest.raises(RuntimeError, match="consumed"):
                    await llm.free_block(right)
                for layer_idx, pointer in right_conv_ptrs.items():
                    assert merged.linear_conv_state[layer_idx].data_ptr() == pointer
            else:
                assert not right.is_consumed
                assert right.token_ids == right_tokens
                assert right.page_starts == right_pages
                assert {
                    layer_idx: state.data_ptr()
                    for layer_idx, state in right.linear_conv_state.items()
                } == right_conv_ptrs
                for layer_idx, pointer in right_conv_ptrs.items():
                    assert merged.linear_conv_state[layer_idx].data_ptr() != pointer

            compact = await llm(
                probe_ids,
                cache_view=[merged, merged_tail],
                write_to=merged_tail,
            )
            assert_same_distribution(split.logits, compact.logits)

            merged_ctx = AsyncContext([merged, merged_tail], merged_tail, next_token)
            compact_decode = await llm(cache_view=merged_ctx)
            assert_same_distribution(split_decode.logits, compact_decode.logits)

            for block in (split_tail, merged, merged_tail):
                await llm.free_block(block)
            if not consume_left:
                await llm.free_block(left)
            if not consume_right:
                await llm.free_block(right)
            assert allocator.num_free_pages == free_pages_before

    loop.run_until_complete(run())
