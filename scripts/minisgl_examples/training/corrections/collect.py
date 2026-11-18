"""Collect MATH-500-Sharded correction traces for later SFT replay."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import torch
from huggingface_hub import hf_hub_download
from minisgl.llm import AsyncLLM

from minisgl_examples.training.corrections.rollout import collect_correction_example

DATASET = "MrDarkTesla/math_500_sharded"
DATASET_REVISION = "660e78826c964737cfb75851913536b16dc2942e"
DATA_FILE = "data-00000-of-00001.arrow"


def cuda_graph_batch_sizes(concurrency: int) -> list[int]:
    """Return compact power-of-two graph sizes up to the request limit."""
    sizes = []
    size = 1
    while size <= concurrency:
        sizes.append(size)
        size *= 2
    if sizes[-1] != concurrency:
        sizes.append(concurrency)
    return sizes


def load_table(dataset: str, revision: str) -> pa.Table:
    """Load the pinned Dataset.save_to_disk Arrow shard from the Hub."""
    path = hf_hub_download(dataset, DATA_FILE, repo_type="dataset", revision=revision)
    return pa.ipc.open_stream(path).read_all()


async def collect_dataset_traces(args: argparse.Namespace) -> dict[str, Any]:
    """Reuse one H100 inference engine to collect a contiguous row range."""
    if torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one GPU")
    if args.start_index < 0 or args.count < 1:
        raise ValueError("Use a nonnegative start index and a positive count")
    if args.concurrency < 1:
        raise ValueError("Use a positive concurrency")
    if not args.k_steps or any(not 0 <= step < args.max_tokens for step in args.k_steps):
        raise ValueError("Require every k_step to satisfy 0 <= k_step < max_tokens")

    config = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "dataset_split": args.split,
        "start_index": args.start_index,
        "count": args.count,
        "k_steps": args.k_steps,
        "max_tokens": args.max_tokens,
        "precision": args.precision,
        "concurrency": args.concurrency,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"Existing run configuration differs: {config_path}")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    trace_dir = args.output / "traces"
    trace_dir.mkdir(exist_ok=True)

    table = load_table(args.dataset, args.dataset_revision)
    stop = args.start_index + args.count
    if stop > table.num_rows:
        raise ValueError(f"Requested rows [{args.start_index}, {stop}) from {table.num_rows} rows")

    pending = [
        (index, k_steps, trace_dir / f"{index:04d}-k{k_steps:04d}.json")
        for index in range(args.start_index, stop)
        for k_steps in args.k_steps
        if not (trace_dir / f"{index:04d}-k{k_steps:04d}.json").exists()
    ]
    if not pending:
        return {
            "output": str(args.output),
            "collected": 0,
            "skipped": args.count * len(args.k_steps),
        }

    graph_kwargs = (
        {
            "cuda_graph_bs": cuda_graph_batch_sizes(args.concurrency),
            "cuda_graph_max_bs": args.concurrency,
            "shared_cuda_graph_max_depth": 4,
            "shared_cuda_graph_prefill_rows": [32, 256],
        }
        if args.cuda_graphs
        else {"cuda_graph_bs": [], "cuda_graph_max_bs": 0, "shared_cuda_graph_prefill_rows": []}
    )
    llm = AsyncLLM(
        args.model,
        dtype=getattr(torch, args.precision),
        attention_backend="fi",
        max_running_req=args.concurrency,
        max_prefill_rows=256,
        memory_ratio=args.memory_ratio,
        max_seq_len_override=args.max_seq_len,
        page_size=args.page_size,
        distributed_addr=f"tcp://127.0.0.1:{args.distributed_port}",
        **graph_kwargs,
    )

    async def collect_one(index: int, k_steps: int, destination: Path) -> dict[str, Any]:
        item = table.slice(index, 1).to_pylist()[0]
        example = await collect_correction_example(
            llm,
            item,
            dataset=args.dataset,
            dataset_revision=args.dataset_revision,
            dataset_split=args.split,
            dataset_index=index,
            k_steps=k_steps,
            max_tokens=args.max_tokens,
            precision=args.precision,
        )
        trace = example["trace"]
        report = {
            "dataset_index": index,
            "k_steps": k_steps,
            "trace_tokens": trace.num_tokens,
            "supervised_tokens": sum(value != -100 for value in example["targets"]),
            "output_tokens": len(example["output_tokens"]),
            "label": example["answer_evaluation"]["label"],
            "answer_before_correction": example["answer_before_correction"],
            "eos_before_correction": example["eos_before_correction"],
        }
        data = {"model": args.model, "examples": [{**example, "trace": asdict(trace)}]}
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with temporary.open("x", encoding="utf-8") as output:
            output.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        print(json.dumps(report), flush=True)
        return report

    semaphore = asyncio.Semaphore(args.concurrency)

    async def limited_collect(index: int, k_steps: int, destination: Path) -> dict[str, Any]:
        async with semaphore:
            return await collect_one(index, k_steps, destination)

    try:
        reports = await asyncio.gather(
            *(
                limited_collect(index, k_steps, destination)
                for index, k_steps, destination in pending
            )
        )
    finally:
        await llm.close()

    summary = {
        "output": str(args.output),
        "collected": len(reports),
        "skipped": args.count * len(args.k_steps) - len(pending),
        "traces": reports,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", default="Qwen/Qwen3.5-4B")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--dataset", default=DATASET)
    result.add_argument("--dataset-revision", default=DATASET_REVISION)
    result.add_argument("--split", default="train")
    result.add_argument("--start-index", type=int, default=0)
    result.add_argument("--count", type=int, default=1)
    result.add_argument("--k-steps", type=int, nargs="+", default=[0, 16, 32, 64, 128, 256, 512])
    result.add_argument("--max-tokens", type=int, default=1024)
    result.add_argument("--concurrency", type=int, default=1)
    result.add_argument("--precision", choices=("bfloat16", "float16"), default="bfloat16")
    result.add_argument("--memory-ratio", type=float, default=0.7)
    result.add_argument("--max-seq-len", type=int, default=8192)
    result.add_argument("--page-size", type=int, default=1)
    result.add_argument("--distributed-port", type=int, default=2361)
    result.add_argument("--no-cuda-graphs", dest="cuda_graphs", action="store_false")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    result = asyncio.run(collect_dataset_traces(args))
    print(json.dumps(result, indent=2, allow_nan=False))
