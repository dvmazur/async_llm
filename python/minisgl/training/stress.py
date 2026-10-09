"""Mine a deterministic synthetic workload through the real shared-cache session."""

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from .trace import RecordingSession


def workload(session, vocab_size, seed=7, large_size=128):
    from minisgl.shared_cache import AsyncContext, PrefillJob, WorkerGroup

    if not 1 <= large_size <= 256 or vocab_size < 2:
        raise ValueError("Require 1 <= large_size <= 256 and vocab_size >= 2")
    if session.trace.tokens or session.trace.updates:
        raise ValueError("The workload requires a fresh RecordingSession")
    rng = random.Random(seed)
    blocks = [session.create_block() for _ in range(24)]
    empty = blocks[0]
    targets, logprobs, operations = {}, {}, Counter()
    peak_live = 0

    def check():
        nonlocal peak_live
        peak_live = max(peak_live, sum(block.num_tokens for block in blocks))
        if peak_live > 8192:
            raise ValueError("Synthetic workload exceeds the live cache budget")

    def ids(count):
        return torch.tensor([rng.randrange(1, vocab_size) for _ in range(count)])

    def save(rows, logits):
        if logits is None:
            return
        for row, output in zip(rows, logits, strict=True):
            output = output.float().flatten()
            target = int(output.argmax())
            targets[row] = target
            logprobs[row] = float(output.log_softmax(-1)[target])

    def prefill(jobs):
        rows, cursor = [], len(session.trace.tokens)
        for job in jobs:
            cursor += job.input_ids.numel()
            rows.append(cursor - 1)
        save(rows, session.prefill_batch(jobs))
        operations["prefill_batch"] += 1
        check()

    def merge(left, right):
        block = session.merge_blocks(left, right)
        blocks.append(block)
        operations["merge"] += 1
        check()
        return block

    lengths = [
        min(n, large_size) for n in [0, 1, 2, 3, 7, 31, max(1, large_size // 2), large_size]
    ] * 3
    for start in range(0, 24, 6):
        prefill(
            [PrefillJob(blocks[i], ids(lengths[i])) for i in range(start, start + 6) if lengths[i]]
        )
    published = []
    for turn in range(12):
        if turn % 2 == 0:
            left, right = blocks[7 + 8 * ((turn // 2) % 3)], blocks[3 + 8 * (turn % 3)]
            merged = merge(left, right)
            copied = merge(empty, merged) if turn % 4 == 0 else merge(merged, empty)
            operations["empty_left_copy" if turn % 4 == 0 else "empty_right_copy"] += 1
            session.append_block(copied, blocks[1 + 8 * (turn % 3)])
            operations["append"] += 1
            if turn in (0, 6):
                session.append_block(copied, copied)
                operations["self_append"] += 1
            check()
            published.append(copied)
            prefill([PrefillJob(left, ids(1), [empty, right])])
            operations["extend_copied_source"] += 1
            session.free_block(left)
            session.free_block(merged)
            operations["clear"] += 2
            prefill([PrefillJob(left, ids(min(1 + turn % 7, large_size)), [copied, empty, right])])
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
        cursor = len(session.trace.tokens)
        save(
            range(cursor, cursor + len(outputs)),
            session.decode_step(WorkerGroup(contexts), ids(len(outputs))),
        )
        operations["decode_batch"] += 1
        check()
        if turn % 3 == 1:
            outputs = rng.sample(blocks[1:24], 2)
            context = [b for b in [empty, *published[-2:]] if b not in outputs]
            prefill([PrefillJob(b, ids(min(2 + turn, large_size)), context) for b in outputs])
            operations["prefill_extension"] += len(outputs)
    empty_copy = merge(empty, empty)
    prefill([PrefillJob(empty_copy, ids(2), published[-2:])])
    trace = session.trace
    batches = Counter(t.step for t in trace.tokens if t.kind == "decode")
    return {
        "workload": {"seed": seed, "large_size": large_size, "vocab_size": vocab_size},
        "trace": asdict(trace),
        "targets": [targets.get(i, -100) for i in range(len(trace.tokens))],
        "logprobs": [logprobs.get(i) for i in range(len(trace.tokens))],
        "coverage": {
            "blocks": len(blocks),
            "operations": dict(operations),
            "ordered_views": len({tuple(b for b, _ in t.view) for t in trace.tokens}),
            "max_view_depth": max(len(t.view) for t in trace.tokens),
            "max_visible_tokens": max(sum(n for _, n in t.view) for t in trace.tokens),
            "peak_live_tokens": peak_live,
            "decode_batch_sizes": dict(sorted(Counter(batches.values()).items())),
        },
    }


def main():
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig
    from minisgl.shared_cache import SharedCacheSession

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
    engine = Engine(
        EngineConfig(
            model_path=args.model,
            tp_info=DistributedInfo(rank=0, size=1),
            dtype=torch.float16,
            attention_backend="fi",
            cuda_graph_max_bs=0,
            num_page_override=8192,
            max_seq_len_override=8192,
            max_running_req=16,
            max_prefill_rows=64,
        )
    )
    try:
        example = workload(
            RecordingSession(SharedCacheSession(engine)),
            engine.config.model_config.vocab_size,
            args.seed,
            args.large_size,
        )
    finally:
        engine.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": args.model, "examples": [example]}))
    print(json.dumps(example["coverage"], indent=2), flush=True)
    print(f"Saved {len(example['trace']['tokens'])} input rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
