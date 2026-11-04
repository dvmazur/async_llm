"""CPU execution stubs for exercising the real AsyncLLM scheduler."""

from types import SimpleNamespace as NS

import torch
from minisgl.engine.sample import Sampler
from minisgl.llm import AsyncLLM
from minisgl.scheduler import AsyncCacheEngine


class Session:
    def __init__(self):
        self.next_block_id = 100

    def create_block(self):
        block = NS(block_id=self.next_block_id, num_tokens=0)
        self.next_block_id += 1
        return block

    def merge_blocks(self, left, right):
        block = self.create_block()
        block.num_tokens = left.num_tokens + right.num_tokens
        return block

    def append_block(self, left, right):
        left.num_tokens += right.num_tokens
        return left

    def free_block(self, block):
        block.num_tokens = 0

    def prefill_batch(self, jobs):
        for job in jobs:
            job.block.num_tokens += job.input_ids.numel()
        return [torch.zeros(1, 32, device="cpu") for _ in jobs]

    def decode_step(self, group, input_ids):
        for ctx in group:
            ctx.output_block.num_tokens += 1
        return torch.zeros(len(group), 32, device="cpu")


def make_llm(session=None):
    session = Session() if session is None else session
    engine = AsyncCacheEngine(session=session, sampler=Sampler(torch.device("cpu"), 32))
    return AsyncLLM(async_engine=engine), session


def generate_stress_example(vocab_size=32, large_size=4):
    import asyncio

    from minisgl.training import RecordedAsyncLLM
    from minisgl_examples.training.stress import generate_workload

    async def run():
        llm, _ = make_llm()
        try:
            with RecordedAsyncLLM(llm) as recorded:
                return await generate_workload(
                    recorded, vocab_size=vocab_size, large_size=large_size
                )
        finally:
            await llm.close()

    return asyncio.run(run())


def copy_operations(session):
    def append(left, right):
        ids = list(right.token_ids)
        if ids:
            left.grow_pages(session._pages(len(ids)), len(ids))
        left.token_ids.extend(ids)
        return left

    def merge(left, right):
        block = session.create_block()
        append(block, left)
        return append(block, right)

    session.merge_blocks = merge
    session.append_block = append


def referenced_blocks(trace):
    blocks = set()
    for batch in trace.batches:
        for request in batch.requests:
            blocks.add(request.write_to)
            blocks.update(block for block, _ in request.cache_view)
    for update in trace.updates:
        blocks.add(update.block)
        blocks.update(block for block, _ in update.refs or ())
    return blocks
