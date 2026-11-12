import asyncio
import random
from types import SimpleNamespace as NS

import torch
from minisgl.models.qwen3_5_mrope import get_rope_index
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM, Trace
from minisgl.training.trace import ForwardBatch, ForwardRequest

from tests.training.helpers import Session, make_llm


def example_trace():

    async def run():
        llm, _ = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                prompt, a, b = [await session.create_block() for _ in range(3)]

                async def prefill(block, ids, context):
                    await session.prefill_block(ids, write_to=block, cache_view=context)

                await prefill(prompt, [1, 2, 3], [])
                await prefill(a, [4, 5], [prompt])
                await prefill(b, [6, 7], [prompt])
                ca = NS(cache_view=[prompt, b, a], output_block=a)
                cb = NS(cache_view=[prompt, a, b], output_block=b)
                await asyncio.gather(
                    session.forward(cache_view=AsyncContext(ca.cache_view, ca.output_block, 8)),
                    session.forward(cache_view=AsyncContext(cb.cache_view, cb.output_block, 9)),
                )
                await prefill(a, [10, 11], [prompt, b])
                ca.cache_view = [b, prompt, a]
                await asyncio.gather(
                    session.forward(cache_view=AsyncContext(ca.cache_view, ca.output_block, 12)),
                    session.forward(cache_view=AsyncContext(cb.cache_view, cb.output_block, 13)),
                )
                ca.cache_view = [prompt, b]
                await session.forward(cache_view=AsyncContext(ca.cache_view, ca.output_block, 14))
                return session.get_trace()
        finally:
            await llm.close()

    return asyncio.run(run())


def single_block_trace(mixed_steps):

    async def run():
        llm, _ = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block = await session.create_block()
                groups = (
                    [[1, 2, 3], [4], [5, 6], [7], [8, 9]] if mixed_steps else [list(range(1, 10))]
                )
                for ids in groups:
                    if len(ids) == 1:
                        await asyncio.gather(
                            *(
                                session.forward(
                                    cache_view=AsyncContext(
                                        ctx.cache_view, ctx.output_block, int(token)
                                    )
                                )
                                for ctx, token in zip(
                                    [NS(cache_view=[block], output_block=block)],
                                    torch.tensor(ids),
                                    strict=True,
                                )
                            )
                        )
                    else:
                        await session.prefill_block(ids, write_to=block, cache_view=[])
                return session.get_trace()
        finally:
            await llm.close()

    return asyncio.run(run())


def causal_trace(length):
    return Trace(
        batches=[
            ForwardBatch(
                0,
                "prefill",
                (ForwardRequest(tuple(i % 32 for i in range(length)), 0, 0, ((0, 0),)),),
            )
        ]
    )


def mutation_trace():

    async def run():
        llm, _ = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                prompt, a, b, empty = [await session.create_block() for _ in range(4)]

                async def prefill(block, ids, context):
                    await session.prefill_block(ids, write_to=block, cache_view=context)

                await prefill(prompt, [1, 2], [])
                await prefill(a, [3, 4], [prompt])
                await prefill(b, [5, 6], [prompt, a])
                merged = await session.merge_blocks(a, b)
                copied = await session.merge_blocks(empty, merged)
                await prefill(a, [7], [prompt])
                await session.free_block(b)
                await prefill(b, [8, 9], [])
                await prefill(merged, [10, 11], [prompt])
                ca = NS(cache_view=[copied, b, merged], output_block=merged)
                cb = NS(cache_view=[prompt, merged, b], output_block=b)
                await asyncio.gather(
                    session.forward(cache_view=AsyncContext(ca.cache_view, ca.output_block, 12)),
                    session.forward(cache_view=AsyncContext(cb.cache_view, cb.output_block, 13)),
                )
                await session.append_block(merged, merged)
                await session.append_block(copied, a)
                await session.free_block(a)
                await prefill(a, [14], [])
                await prefill(merged, [15], [copied])
                ca.cache_view = [copied]
                await session.forward(cache_view=AsyncContext(ca.cache_view, ca.output_block, 16))
                await session.free_block(copied)
                await prefill(copied, [17], [b, merged])
                right_copy = await session.merge_blocks(merged, empty)
                empty_copy = await session.merge_blocks(empty, empty)
                await prefill(empty_copy, [18], [right_copy])
                await session.free_block(merged)
                await prefill(right_copy, [19], [prompt])
                return session.get_trace()
        finally:
            await llm.close()

    return asyncio.run(run())


def repeated_view_trace():

    async def run():
        llm, _ = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                a, b, c, empty = [await session.create_block() for _ in range(4)]

                async def prefill(block, ids, context):
                    await session.prefill_block(ids, write_to=block, cache_view=context)

                await prefill(a, [1, 2, 3, 4, 5], [])
                await prefill(b, [6, 7], [a, a])
                await prefill(c, [8], [a, b, a])
                await asyncio.gather(
                    session.forward(cache_view=AsyncContext([a, b, a, empty, b], a, 9)),
                    session.forward(cache_view=AsyncContext([b, a, b, a], b, 10)),
                    session.forward(cache_view=AsyncContext([a, a], c, 11)),
                )
                copied = await session.merge_blocks(a, b)
                await session.append_block(copied, copied)
                await session.free_block(a)
                await prefill(a, [12, 13], [copied, copied, b, copied])
                await asyncio.gather(
                    session.forward(cache_view=AsyncContext([a, copied, a, copied], a, 14)),
                    session.forward(cache_view=AsyncContext([copied, a, b, copied, b], b, 15)),
                )
                await session.free_block(copied)
                await prefill(copied, [16, 17], [a, b, a])
                return session.get_trace()
        finally:
            await llm.close()

    return asyncio.run(run())


def randomized_trace(seed):

    async def run():
        llm = None
        session = None
        try:
            rng = random.Random(seed)
            llm, _ = make_llm(Session())
            session = RecordedAsyncLLM(llm)
            session.__enter__()
            blocks = [await session.create_block() for _ in range(5)]

            async def prefill(block, count, context=()):
                await session.prefill_block(
                    [rng.randrange(1, 32) for _ in range(count)], write_to=block, cache_view=context
                )

            for block in blocks[:3]:
                await prefill(block, 2)
            for round_id in range(12):
                left = rng.choice(blocks)
                right = left if round_id % 3 == 0 else rng.choice(blocks)
                if round_id % 4 == 0:
                    await session.free_block(left)
                elif round_id % 4 == 1:
                    await session.append_block(left, right)
                else:
                    copied = await session.merge_blocks(left, right)
                    if round_id % 4 == 2:
                        await session.free_block(left)
                    blocks.append(copied)
                if round_id % 3 == 0:
                    output = rng.choice(blocks)
                    contexts = [block for block in blocks if block is not output]
                    await prefill(output, 2, rng.sample(contexts, min(3, len(contexts))))
                else:
                    outputs = rng.sample(blocks, 3)
                    contexts = [
                        NS(cache_view=rng.sample(blocks, rng.randrange(5)), output_block=block)
                        for block in outputs
                    ]
                    await asyncio.gather(
                        *(
                            session.forward(
                                cache_view=AsyncContext(
                                    ctx.cache_view, ctx.output_block, int(token)
                                )
                            )
                            for ctx, token in zip(
                                contexts,
                                torch.tensor([rng.randrange(1, 32) for _ in outputs]),
                                strict=True,
                            )
                        )
                    )
            return session.get_trace()
        finally:
            if session is not None:
                session.__exit__(None, None, None)
            if llm is not None:
                await llm.close()

    return asyncio.run(run())


def uneven_history_trace():

    async def run():
        llm, _ = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                tip, large, small = [await session.create_block() for _ in range(3)]

                async def prefill(block, ids, context=()):
                    await session.prefill_block(ids, write_to=block, cache_view=context)

                await prefill(tip, [1, 2, 3])
                await prefill(large, [4, 5, 6, 7, 8])
                await prefill(small, [9])
                for i in range(8):
                    right = large if i % 2 == 0 else small
                    copied = await session.merge_blocks(tip, right)
                    await session.append_block(copied, right)
                    await prefill(tip, [10 + i], [right])
                    await session.free_block(tip)
                    await prefill(tip, [20 + i])
                    await asyncio.gather(
                        session.forward(cache_view=AsyncContext([tip, copied], copied, i + 1)),
                        session.forward(cache_view=AsyncContext([copied, tip], tip, i + 2)),
                    )
                    tip = copied
                return session.get_trace()
        finally:
            await llm.close()

    return asyncio.run(run())


class ImageSession(Session):
    def create_block(self):
        block = super().create_block()
        block.mrope_span = 0
        return block

    def _mrope_rel(self, ids, types, grid):
        return get_rope_index(ids, types, 2, grid)

    def prefill_batch(self, jobs):
        for job in jobs:
            positions = getattr(job, "mrope_rel", None)
            if positions is None:
                positions = (
                    self._mrope_rel(job.input_ids, job.mm_token_type_ids, job.image_grid_thw)
                    if job.pixel_values is not None
                    else torch.arange(job.input_ids.numel()).expand(3, -1)
                )
            job.block.mrope_span += max(0, int(positions.max()) + 1)
        return super().prefill_batch(jobs)

    def decode_step(self, group, ids):
        for ctx in group:
            ctx.output_block.mrope_span += 1
        return super().decode_step(group, ids)

    def merge_blocks(self, left, right):
        block = super().merge_blocks(left, right)
        block.mrope_span = left.mrope_span + right.mrope_span
        return block

    def append_block(self, left, right):
        span = left.mrope_span + right.mrope_span
        super().append_block(left, right)
        left.mrope_span = span
        return left

    def free_block(self, block):
        super().free_block(block)
        block.mrope_span = 0


def image_inputs(device="cpu"):
    ids = torch.tensor([30, *([29] * 6), 31, 8], device=device)
    return {
        "input_ids": ids[None],
        "pixel_values": torch.sin(torch.arange(24 * 12, device=device).float()).reshape(24, 12) / 2,
        "image_grid_thw": torch.tensor([[1, 4, 6]], device=device),
        "mm_token_type_ids": (ids == 29).long()[None],
    }


async def image_prefill(session, block, context=(), inputs=None):
    inputs = image_inputs() if inputs is None else inputs
    return await session.prefill_block(
        inputs["input_ids"][0],
        write_to=block,
        cache_view=list(context),
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        mm_token_type_ids=inputs["mm_token_type_ids"][0],
    )


def single_image_trace(inputs=None):

    async def run():
        llm, _ = make_llm(ImageSession())
        try:
            with RecordedAsyncLLM(llm) as session:
                await image_prefill(session, await session.create_block(), inputs=inputs)
                return session.get_trace()
        finally:
            await llm.close()

    return asyncio.run(run())


def cache_image_trace():

    async def run():
        llm, _ = make_llm(ImageSession())
        try:
            with RecordedAsyncLLM(llm) as session:
                image, a, b = [await session.create_block() for _ in range(3)]
                await image_prefill(session, image)
                await session.prefill_block([2, 3], write_to=a, cache_view=[image, image])
                await image_prefill(session, a, context=[image])
                await session.prefill_block([4, 5], write_to=b, cache_view=[a, image])
                await asyncio.gather(
                    session.forward(cache_view=AsyncContext([b, image, a, image], a, 6)),
                    session.forward(cache_view=AsyncContext([image, a, b], b, 7)),
                )
                merged = await session.merge_blocks(a, image)
                await session.append_block(merged, merged)
                await session.free_block(image)
                await session.prefill_block(
                    [10, 11], write_to=image, cache_view=[merged, b, merged]
                )
                await asyncio.gather(
                    session.forward(cache_view=AsyncContext([merged, image, b, merged], a, 12)),
                    session.forward(cache_view=AsyncContext([b, image, b], b, 13)),
                )
                await session.prefill_block([14, 15], write_to=merged, cache_view=[b, image])
                return session.get_trace()
        finally:
            await llm.close()

    return asyncio.run(run())
