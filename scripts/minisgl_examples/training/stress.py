"""Mine a seeded cache-mutation workload through the AsyncLLM scheduler."""

import argparse
import asyncio
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext, CacheBlock
from minisgl.training import RecordedAsyncLLM

from minisgl_examples.training.recording import Samples, build_training_example, record_sample


async def generate_workload(
    recorded: RecordedAsyncLLM, vocab_size: int, seed: int = 7, large_size: int = 128
) -> dict[str, Any]:
    if not 1 <= large_size <= 256 or vocab_size < 2:
        raise ValueError("Require 1 <= large_size <= 256 and vocab_size >= 2")
    trace = recorded.get_trace()
    if trace.num_tokens or trace.updates:
        raise ValueError("The workload requires a fresh recording scope")
    rng = random.Random(seed)
    blocks = [await recorded.create_block() for _ in range(24)]
    empty = blocks[0]
    samples: Samples = {}
    operations = Counter()
    peak_live = 0

    def check() -> None:
        nonlocal peak_live
        peak_live = max(peak_live, sum(block.num_tokens for block in blocks))
        if peak_live > 8192:
            raise ValueError("Synthetic workload exceeds the live cache budget")

    def ids(count: int) -> torch.Tensor:
        return torch.tensor([rng.randrange(1, vocab_size) for _ in range(count)])

    async def forward(
        block: CacheBlock, input_ids: torch.Tensor, view: list[CacheBlock], *, prefill: bool = False
    ) -> None:
        batch_cursor, row_cursor = len(trace.batches), trace.num_tokens
        write_end = block.num_tokens + input_ids.numel()
        if prefill:
            result = await recorded.prefill_block(
                input_ids, cache_view=view, write_to=block, return_logits=True
            )
        else:
            context = AsyncContext(
                cache_view=view, output_block=block, next_input_id=int(input_ids.item())
            )
            result = await recorded.forward(cache_view=context)
        logits = result.logits.float().flatten()
        record_sample(
            recorded,
            samples,
            block_id=block.block_id,
            write_end=write_end,
            batch_cursor=batch_cursor,
            row_cursor=row_cursor,
            token=int(logits.argmax()),
            logits=logits,
        )

    async def prefill(jobs: list[tuple[CacheBlock, torch.Tensor, list[CacheBlock]]]) -> None:
        await asyncio.gather(
            *(forward(block, tokens, view, prefill=True) for block, tokens, view in jobs)
        )
        operations["prefill_batch"] += 1
        check()

    async def merge(left: CacheBlock, right: CacheBlock) -> CacheBlock:
        block = await recorded.merge_blocks(left, right)
        blocks.append(block)
        operations["merge"] += 1
        check()
        return block

    lengths = [
        min(n, large_size) for n in [0, 1, 2, 3, 7, 31, max(1, large_size // 2), large_size]
    ] * 3
    for start in range(0, 24, 6):
        await prefill(
            [(blocks[i], ids(lengths[i]), []) for i in range(start, start + 6) if lengths[i]]
        )
    published = []
    for turn in range(12):
        if turn % 2 == 0:
            left, right = blocks[7 + 8 * ((turn // 2) % 3)], blocks[3 + 8 * (turn % 3)]
            merged = await merge(left, right)
            copied = await (merge(empty, merged) if turn % 4 == 0 else merge(merged, empty))
            operations["empty_left_copy" if turn % 4 == 0 else "empty_right_copy"] += 1
            await recorded.append_block(copied, blocks[1 + 8 * (turn % 3)])
            operations["append"] += 1
            if turn in (0, 6):
                await recorded.append_block(copied, copied)
                operations["self_append"] += 1
            check()
            published.append(copied)
            await prefill([(left, ids(1), [empty, right])])
            operations["extend_copied_source"] += 1
            await recorded.free_block(left)
            await recorded.free_block(merged)
            operations["clear"] += 2
            await prefill([(left, ids(min(1 + turn % 7, large_size)), [copied, empty, right])])
            operations["clear_reuse"] += 1
        candidates = [*blocks[1:24], *published]
        outputs = rng.sample(candidates, 4 + turn % 5)
        contexts = []
        for worker, output in enumerate(outputs):
            others = [block for block in candidates if block is not output]
            width = 1 + (turn * 3 + worker * 5) % 12
            view = rng.sample(others, width)
            mode = (turn + worker) % 5
            if mode == 0:
                view = []
            elif mode == 2:
                view = [output]
            elif mode == 3:
                view = view[:11]
                view.insert(rng.randrange(len(view) + 1), output)
            elif mode == 4:
                view = list(reversed(outputs))
                view += rng.sample([b for b in others if b not in outputs], 12 - len(view))
            visible = sum(b.num_tokens + int(b in outputs) for b in view)
            if visible + 1 > 8192:
                raise ValueError("Synthetic workload exceeds the visible context budget")
            contexts.append(AsyncContext(cache_view=view, output_block=output))
        inputs = ids(len(outputs))
        await asyncio.gather(
            *(
                forward(context.output_block, token.view(1), list(context.cache_view))
                for context, token in zip(contexts, inputs, strict=True)
            )
        )
        operations["decode_batch"] += 1
        check()
        if turn % 3 == 1:
            outputs = rng.sample(blocks[1:24], 2)
            context = [b for b in [empty, *published[-2:]] if b not in outputs]
            await prefill([(b, ids(min(2 + turn, large_size)), context) for b in outputs])
            operations["prefill_extension"] += len(outputs)
    empty_copy = await merge(empty, empty)
    await prefill([(empty_copy, ids(2), published[-2:])])
    tokens = trace.tokens
    batches = Counter(t.step for t in tokens if t.kind == "decode")
    return {
        "workload": {"seed": seed, "large_size": large_size, "vocab_size": vocab_size},
        "trace": asdict(trace),
        **{
            key: value
            for key, value in build_training_example(recorded, samples).items()
            if key != "trace"
        },
        "coverage": {
            "blocks": len(blocks),
            "operations": dict(operations),
            "ordered_views": len({tuple(b for b, _ in t.view) for t in tokens}),
            "max_view_depth": max(len(t.view) for t in tokens),
            "max_visible_tokens": max(sum(n for _, n in t.view) for t in tokens),
            "peak_live_tokens": peak_live,
            "decode_batch_sizes": dict(sorted(Counter(batches.values()).items())),
        },
    }


async def _collect(args: argparse.Namespace) -> dict[str, Any]:
    llm = AsyncLLM(
        args.model,
        dtype=torch.float16,
        attention_backend="fi",
        cuda_graph_max_bs=0,
        num_page_override=8192,
        max_seq_len_override=8192,
        max_running_req=16,
        max_prefill_rows=64,
        distributed_addr="tcp://127.0.0.1:0",
    )
    try:
        with RecordedAsyncLLM(llm) as recorded:
            return await generate_workload(
                recorded, llm.engine.config.model_config.vocab_size, args.seed, args.large_size
            )
    finally:
        await llm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--large-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        parser.error("Select exactly one GPU with CUDA_VISIBLE_DEVICES")
    if not args.output.resolve().is_relative_to(Path.cwd()) or args.output.exists():
        parser.error("Output must be a new file inside the working directory")
    if not 1 <= args.large_size <= 256:
        parser.error("--large-size must be between 1 and 256")
    example = asyncio.run(_collect(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": args.model, "examples": [example]}))
    print(json.dumps(example["coverage"], indent=2), flush=True)
    print(f"Saved {len(example['targets'])} input rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
