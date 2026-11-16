"""Record one MATH-500-Sharded rollout with a mid-reasoning input injection."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import torch
from huggingface_hub import hf_hub_download
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext
from minisgl.training.trace import record
from transformers import AutoTokenizer

DATASET = "MrDarkTesla/math_500_sharded"
DATASET_REVISION = "660e78826c964737cfb75851913536b16dc2942e"
DATA_FILE = "data-00000-of-00001.arrow"
INJECTION_PREFIX = "\n\nADDITIONAL INFORMATION: "


def encode(text, tokenizer):
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False))


def load_row(dataset, revision, index):
    """Read one row from the Hub repository's Dataset.save_to_disk Arrow file."""
    path = hf_hub_download(dataset, DATA_FILE, repo_type="dataset", revision=revision)
    table = pa.ipc.open_stream(path).read_all()
    if not 0 <= index < table.num_rows:
        raise ValueError(f"Dataset index {index} is outside [0, {table.num_rows})")
    return table.slice(index, 1).to_pylist()[0]


def target_row(trace, batch_cursor, block_id, position):
    """Locate the single decode input whose logits produced a sampled token."""
    row = sum(
        len(request.input_ids)
        for batch in trace.batches[:batch_cursor]
        for request in batch.requests
    )
    matches = []
    for batch in trace.batches[batch_cursor:]:
        for request in batch.requests:
            if (
                batch.kind == "decode"
                and request.write_to == block_id
                and request.write_start == position
            ):
                matches.append(row)
            row += len(request.input_ids)
    if len(matches) != 1:
        raise RuntimeError("Could not align sampled token with one recorded decode row")
    return matches[0]


async def generate(args):
    item = load_row(args.dataset, args.dataset_revision, args.index)
    shards = list(item["problem_shards"])
    if len(shards) != 2 or not all(isinstance(value, str) and value for value in shards):
        raise ValueError("Expected exactly two nonempty problem_shards")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = AsyncLLM(
        args.model,
        dtype=torch.bfloat16,
        max_running_req=1,
        memory_ratio=args.memory_ratio,
        page_size=args.page_size,
        distributed_addr=f"tcp://127.0.0.1:{args.distributed_port}",
    )
    trace = record(llm)
    prompt = output = None
    sampled = {}
    generated = []
    injection_step = None
    try:
        prompt, output = await asyncio.gather(llm.create_block(), llm.create_block())
        initial = (
            "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
            + shards[0]
        )
        await llm.forward(encode(initial, tokenizer), write_to=prompt, return_logits=False)
        newline = tokenizer.encode("\n", add_special_tokens=False)[-1]
        context = AsyncContext([prompt, output], next_input_id=newline)

        for step in range(args.max_tokens):
            if injection_step is None and step == args.k_steps:
                injection_step = trace.steps
                correction = INJECTION_PREFIX + shards[1] + "\n\n"
                await llm.forward(
                    encode(correction, tokenizer),
                    cache_view=[prompt],
                    write_to=prompt,
                    return_logits=False,
                )

            batch_cursor = len(trace.batches)
            position = output.num_tokens
            result = await llm.forward(cache_view=context)
            token = int(result.logits.argmax().item())
            row = target_row(trace, batch_cursor, output.block_id, position)
            sampled[row] = (token, result.logits.float().log_softmax(-1)[token].item())
            generated.append(token)
            context.next_input_id = token
            if token == tokenizer.eos_token_id:
                break

        if injection_step is None:
            raise RuntimeError("Correction was not injected; require k_steps < max_tokens")
        targets, logprobs = [-100] * trace.num_tokens, [None] * trace.num_tokens
        for row, (token, logprob) in sampled.items():
            targets[row], logprobs[row] = token, logprob
        example = {
            "prompt": initial,
            "problem_shards": shards,
            "answer": str(item["answer"]),
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "dataset_split": args.split,
            "dataset_index": args.index,
            "k_steps": args.k_steps,
            "injection_trace_step": injection_step,
            "trace": asdict(trace),
            "targets": targets,
            "logprobs": logprobs,
            "output_tokens": generated,
        }
        destination = Path(args.output)
        if destination.exists():
            raise ValueError(f"Refusing to overwrite {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"model": args.model, "examples": [example]}, ensure_ascii=False) + "\n"
        )
        print(tokenizer.decode(generated, skip_special_tokens=True), flush=True)
    finally:
        if output is not None:
            await llm.free_block(output)
        if prompt is not None:
            await llm.free_block(prompt)
        await llm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--k-steps", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--memory-ratio", type=float, default=0.7)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--distributed-port", type=int, default=2361)
    args = parser.parse_args()
    if not 0 < args.k_steps < args.max_tokens:
        parser.error("Require 0 < k-steps < max-tokens")
    asyncio.run(generate(args))


if __name__ == "__main__":
    main()

