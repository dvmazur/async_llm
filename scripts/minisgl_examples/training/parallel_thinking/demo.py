"""Collect or load an Alice/Bob trace, train an HF student, and generate freely."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from minisgl.core import SamplingParams

from minisgl_examples.training import student
from minisgl_examples.training.parallel_thinking.rollout import PROBLEM
from minisgl_examples.training.parallel_thinking.rollout import run_agents as run_agents
from minisgl_examples.training.parallel_thinking.rollout import solve as solve
from minisgl_examples.training.process import isolated
from minisgl_examples.training.student import TraceSource, load_example, save_example


async def collect_trace(model: str, **generation_options: Any) -> dict[str, Any]:
    """Return a reusable example, with its prompt and rollout metadata."""
    generation_options.setdefault(
        "sampling", SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=1)
    )
    result = await solve(model, **generation_options)
    example = result.pop("example")
    return {**example, "model": str(model), "prompt_ids": result["prompt_ids"], "rollout": result}


def train_student(
    model: str,
    example: TraceSource,
    output: str | Path,
    *,
    steps: int = 80,
    learning_rate: float = 3e-5,
    chunk_size: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Train on a collected example or saved trace, retaining the CPU optimizer."""
    example = load_example(example)
    example["trace"] = example["trace"].to(
        query_chunk_size=chunk_size, attention_checkpointing=True, cache_plans=True
    )
    return student.train_student(
        example,
        model,
        output,
        steps=steps,
        lr=learning_rate,
        device=device,
        dtype=dtype,
        master_dtype=dtype,
        optimizer_device="cpu",
        chunk_size=chunk_size,
    )


def verify_checkpoint(
    source: TraceSource,
    checkpoint: str | Path,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    chunk_size: int = 64,
) -> dict[str, Any]:
    source = load_example(source)
    source["trace"] = source["trace"].to(
        query_chunk_size=chunk_size, attention_checkpointing=True, cache_plans=True
    )
    return student.verify_checkpoint(
        source, checkpoint, device=device, dtype=dtype, chunk_size=chunk_size
    )


def run_workflow(
    teacher: str,
    student: str,
    output: str | Path,
    *,
    trace: TraceSource | None = None,
    steps: int = 80,
    learning_rate: float = 3e-5,
    **generation_options: Any,
) -> dict[str, Any]:
    """Mine a trace or reuse one, then train and assess independent generation."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if trace is None:
        example = isolated(collect_trace, teacher, **generation_options)
    else:
        example = load_example(trace)
    output.mkdir(parents=True)
    save_example(example, output / "trace.json")
    training = isolated(
        train_student,
        student,
        example,
        output / "checkpoint",
        steps=steps,
        learning_rate=learning_rate,
    )
    teacher_result = example.get("rollout", {})
    prompt_ids = example.get("prompt_ids", teacher_result.get("prompt_ids"))
    if prompt_ids is None:
        recorded = example["trace"]
        prompt_block = recorded.batches[0].requests[0].write_to
        prompt_ids = [token.token_id for token in recorded.tokens if token.block == prompt_block]
    options = dict(generation_options, prompt_ids=prompt_ids)
    generated = isolated(solve, training["checkpoint"], **options)
    generated.pop("example")
    report = {"teacher": teacher_result, "training": training, "student": generated}
    (output / "report.json").write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="Collect and save a teacher trace")
    record.add_argument("model")
    record.add_argument("output", type=Path)
    run = commands.add_parser("run", help="Collect or load, train, and generate")
    run.add_argument("--teacher", default="Qwen/Qwen3-4B")
    run.add_argument("--student", default="Qwen/Qwen3-1.7B")
    run.add_argument("--trace", type=Path, help="Reuse a saved trace instead of collecting")
    run.add_argument("--output", type=Path, required=True)
    for command in (record, run):
        command.add_argument(
            "--mode", choices=("synchronized", "no-barrier"), default="synchronized"
        )
        command.add_argument("--problem", default=PROBLEM)
        command.add_argument("--max-steps", type=int, default=384)
        command.add_argument("--answer-steps", type=int, default=64)
        command.add_argument("--bob-delay", type=float, default=0.0)
        command.add_argument("--delay-every", type=int, default=32)
        command.add_argument("--memory-ratio", type=float, default=0.65)
        command.add_argument("--max-seq-len", type=int, default=16384)
        command.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    train = commands.add_parser("train", help="Train an HF student on a saved trace")
    verify = commands.add_parser("verify", help="Evaluate an HF checkpoint on a saved trace")
    for command in (train, verify):
        command.add_argument("trace", type=Path)
        command.add_argument("model")
        command.add_argument("output", type=Path)
        command.add_argument("--device", default="cuda")
    for command in (train, run):
        command.add_argument("--steps", "--train-steps", dest="steps", type=int, default=80)
        command.add_argument(
            "--lr", "--learning-rate", dest="learning_rate", type=float, default=3e-5
        )
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0].startswith("--") and arguments[0] != "--help":
        arguments.insert(0, "run")
    args = parser.parse_args(arguments)
    if args.command == "train":
        result = train_student(
            args.model,
            args.trace,
            args.output,
            steps=args.steps,
            learning_rate=args.learning_rate,
            device=args.device,
        )
    elif args.command == "verify":
        result = verify_checkpoint(args.trace, args.model, device=args.device)
        args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    else:
        options = vars(args).copy()
        options.pop("command")
        options["synchronized"] = options.pop("mode") == "synchronized"
        if args.command == "record":
            model, output = options.pop("model"), options.pop("output")
            if output.exists():
                raise FileExistsError(output)
            example = isolated(collect_trace, model, **options)
            path = save_example(example, output / "trace.json")
            result = {"trace": str(path), "tokens": example["trace"].num_tokens}
        else:
            result = run_workflow(**options)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
