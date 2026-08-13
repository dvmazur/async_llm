"""End-to-end cache-block merge test on the full Qwen3.5-9B model."""

from __future__ import annotations

import asyncio

import pytest
import torch
from PIL import Image

from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext, CacheBlock


MODEL_PATH = "Qwen/Qwen3.5-9B"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cache-block merge test requires CUDA",
)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            torch.backends.cuda.matmul,
            "allow_bf16_reduced_precision_reduction",
            False,
        )
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        llm = AsyncLLM(
            MODEL_PATH,
            dtype=torch.bfloat16,
            max_running_req=2,
            num_page_override=8192,
            attention_backend="fi",
            distributed_addr=(
                tmp_path_factory.mktemp("cache_merge") / "distributed_init"
            ).as_uri(),
        )
        yield loop, llm
        loop.run_until_complete(llm.close())
        loop.close()
        asyncio.set_event_loop(None)


def assert_same_distribution(left: torch.Tensor, right: torch.Tensor) -> None:
    left = left.float().softmax(-1)
    right = right.float().softmax(-1)
    torch.testing.assert_close(left, right, rtol=0, atol=2.5e-3)
    assert left.argmax().item() == right.argmax().item()


async def create_blocks(llm: AsyncLLM, count: int) -> tuple[CacheBlock, ...]:
    return tuple(await asyncio.gather(*(llm.create_block() for _ in range(count))))


async def fill_image_conversation(
    llm: AsyncLLM,
    left: CacheBlock,
    right: CacheBlock,
) -> None:
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


async def assert_matching_continuation(
    llm: AsyncLLM,
    left_view: list[CacheBlock],
    right_view: list[CacheBlock],
    left_tail: CacheBlock,
    right_tail: CacheBlock,
) -> None:
    probe_ids = llm.tokenizer(
        " Continue the analysis using the visual evidence:",
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    left_output, right_output = await asyncio.gather(
        llm(probe_ids, cache_view=[*left_view, left_tail], write_to=left_tail),
        llm(probe_ids, cache_view=[*right_view, right_tail], write_to=right_tail),
    )
    assert_same_distribution(left_output.logits, right_output.logits)

    next_token = int(left_output.logits.argmax())
    left_context = AsyncContext([*left_view, left_tail], left_tail, next_token)
    right_context = AsyncContext([*right_view, right_tail], right_tail, next_token)
    left_output, right_output = await asyncio.gather(
        llm(cache_view=left_context),
        llm(cache_view=right_context),
    )
    assert_same_distribution(left_output.logits, right_output.logits)


def assert_same_kv(llm: AsyncLLM, left: CacheBlock, right: CacheBlock) -> None:
    session = llm.async_engine.session
    left_slots = left.token_slots_tensor().to(torch.int64)
    right_slots = right.token_slots_tensor().to(torch.int64)
    for layer_idx in range(session.kv_cache.num_layers):
        for cache in (
            session.kv_cache.k_cache(layer_idx),
            session.kv_cache.v_cache(layer_idx),
        ):
            flat = cache.reshape(-1, *cache.shape[2:])
            torch.testing.assert_close(
                flat.index_select(0, left_slots),
                flat.index_select(0, right_slots),
                rtol=0,
                atol=0,
            )


async def free_blocks(llm: AsyncLLM, *blocks: CacheBlock) -> None:
    await asyncio.gather(*(llm.free_block(block) for block in blocks))


def test_merged_block_matches_split_view(runtime) -> None:
    loop, llm = runtime

    async def run() -> None:
        left, right, split_tail, merged_tail = await create_blocks(llm, 4)
        await fill_image_conversation(llm, left, right)

        left_tokens = left.token_ids.copy()
        right_tokens = right.token_ids.copy()
        merged = await llm.merge_blocks(left, right)

        assert left.token_ids == left_tokens
        assert right.token_ids == right_tokens
        assert merged.token_ids == left_tokens + right_tokens
        assert merged.mrope_span == left.mrope_span + right.mrope_span

        await assert_matching_continuation(
            llm,
            [left, right],
            [merged],
            split_tail,
            merged_tail,
        )
        await free_blocks(llm, left, right, split_tail, merged, merged_tail)

    loop.run_until_complete(run())


def test_append_block_matches_copied_merge(runtime) -> None:
    loop, llm = runtime

    async def run() -> None:
        allocator = llm.async_engine.session.page_allocator
        free_pages_before = allocator.num_free_pages
        left, right, merged_tail, appended_tail = await create_blocks(llm, 4)
        await fill_image_conversation(llm, left, right)

        left_tokens = left.token_ids.copy()
        right_tokens = right.token_ids.copy()
        left_pages = left.page_starts.copy()
        right_pages = right.page_starts.copy()
        right_span = right.mrope_span

        merged = await llm.merge_blocks(left, right)
        appended = await llm.append_block(left, right)
        assert appended is left
        assert left.token_ids == left_tokens + right_tokens
        assert left.page_starts[: len(left_pages)] == left_pages
        assert right.token_ids == right_tokens
        assert right.page_starts == right_pages

        await assert_matching_continuation(
            llm,
            [merged],
            [left],
            merged_tail,
            appended_tail,
        )

        # Aliasing has the natural copy meaning: block := old_block + old_block.
        self_copied = await llm.merge_blocks(right, right)
        self_appended = await llm.append_block(right, right)
        assert self_appended is right
        assert right.token_ids == right_tokens + right_tokens
        assert right.page_starts[: len(right_pages)] == right_pages
        assert right.mrope_span == right_span * 2

        assert_same_kv(llm, self_copied, right)

        await free_blocks(
            llm, left, right, merged, self_copied, merged_tail, appended_tail
        )
        assert allocator.num_free_pages == free_pages_before

    loop.run_until_complete(run())
