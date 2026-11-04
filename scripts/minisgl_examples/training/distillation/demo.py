"""Collect or load one async-thoughts trace, train an HF student, and verify it."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from minisgl_examples.training import student as training
from minisgl_examples.training.distillation.rollout import (
    DEFAULT_PROMPT,
    check_answer,
    collect_trace,
    compare_rollouts,
)
from minisgl_examples.training.process import isolated
from minisgl_examples.training.student import TraceSource, has_overfit, load_example, save_example


def _validate_student(source: TraceSource, model: str, device: str) -> None:
    if torch.device(device).type == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one GPU")
    if AutoConfig.from_pretrained(model).model_type != "qwen3":
        raise ValueError("This sanity check supports dense text Qwen3")
    if load_example(source)["trace"].multimodal:
        raise ValueError("This sanity check uses text only")


def train_student(
    model: str,
    source: TraceSource,
    output: str | Path,
    *,
    steps: int = 100,
    learning_rate: float = 1e-6,
    device: str = "cuda",
) -> dict[str, Any]:
    """Fit one collected or saved example using BF16 compute and FP32 AdamW."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    _validate_student(source, model, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    else:
        example = load_example(source)
        raw = json.dumps({**example, "trace": asdict(example["trace"])}, sort_keys=True).encode()
    report = training.train_student(
        source,
        model,
        output / "checkpoint",
        steps=steps,
        lr=learning_rate,
        device=device,
        dtype=torch.bfloat16,
        master_dtype=torch.float32,
        optimizer_device=device,
        stop_on_initial=True,
    )
    report.update(
        teacher_sha256=hashlib.sha256(raw).hexdigest(),
        precision="BF16 forward / FP32 master weights and AdamW",
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def verify_checkpoint(
    source: TraceSource, checkpoint: str | Path, *, device: str = "cuda"
) -> dict[str, Any]:
    _validate_student(source, str(checkpoint), device)
    torch.backends.cuda.matmul.allow_tf32 = False
    return training.verify_checkpoint(
        source, str(checkpoint), device=device, dtype=torch.bfloat16, master_dtype=torch.float32
    )


def run_workflow(
    teacher: str,
    student: str,
    output: str | Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    reference: int = 391,
    steps: int = 100,
    learning_rate: float = 1e-6,
    trace: TraceSource | None = None,
    **generation_options: Any,
) -> dict[str, Any]:
    """Require a correct teacher and replay overfit, then assess independent generation."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    example = (
        isolated(
            collect_trace,
            teacher,
            prompt=prompt,
            log_path=output / "teacher" / "generation.log",
            **generation_options,
        )
        if trace is None
        else load_example(trace)
    )
    teacher_trace = save_example(example, output / "teacher" / "trace.json")
    tokenizer = AutoTokenizer.from_pretrained(example.get("model", teacher))
    grade = check_answer(example, tokenizer, [reference])
    (output / "teacher_check.json").write_text(json.dumps(grade, indent=2))
    if grade["status"] != "PASS":
        raise RuntimeError(f"Teacher answer is incorrect or incomplete: {grade}")
    metrics = isolated(
        train_student,
        student,
        example,
        output / "trained",
        steps=steps,
        learning_rate=learning_rate,
        log_path=output / "training.log",
    )
    if not metrics["overfit_achieved"]:
        raise RuntimeError("Overfit not reached; see training.log and trained/metrics.json")
    checkpoint = metrics["checkpoint"]
    reloaded = isolated(
        verify_checkpoint,
        teacher_trace,
        checkpoint,
        log_path=output / "reloaded.log",
    )
    (output / "reloaded.json").write_text(json.dumps(reloaded, allow_nan=False))
    if not has_overfit(reloaded) or abs(reloaded["loss"] - metrics["final"]["loss"]) >= 1e-5:
        raise RuntimeError(f"Saved checkpoint does not retain the training result: {reloaded}")
    generated = isolated(
        collect_trace,
        checkpoint,
        prompt=example["prompt"],
        log_path=output / "student" / "generation.log",
        **generation_options,
    )
    student_trace = save_example(generated, output / "student" / "trace.json")
    return compare_rollouts(
        teacher_trace, student_trace, tokenizer, [reference], output / "checks.json"
    )


def main(argv: list[str] | None = None) -> None:
    """Run the complete experiment or a separate record/train/verify stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="Record a fresh inference trace")
    record.add_argument("model")
    record.add_argument("output", type=Path)
    run = commands.add_parser("run", help="Run all stages with strict acceptance checks")
    run.add_argument("--teacher", default="Qwen/Qwen3-4B")
    run.add_argument("--student", default="Qwen/Qwen3-0.6B")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--reference", type=int, default=391)
    run.add_argument("--trace", type=Path, help="Use a saved teacher trace instead of mining")
    for command in (record, run):
        command.add_argument("--prompt", default=DEFAULT_PROMPT)
        command.add_argument("--think-tokens", type=int, default=128)
        command.add_argument("--answer-tokens", type=int, default=192)
        command.add_argument("--probe-period", type=int, default=4)
        command.add_argument("--memory-ratio", type=float, default=0.7)
        command.add_argument("--max-seq-len", type=int, default=8192)
        command.add_argument("--no-cuda-graphs", dest="cuda_graphs", action="store_false")
    train = commands.add_parser("train", help="Fit a student to one recorded trace")
    verify = commands.add_parser("verify", help="Evaluate a saved checkpoint")
    for command in (train, verify):
        command.add_argument("trace", type=Path)
        command.add_argument("model")
        command.add_argument("output", type=Path)
        command.add_argument("--device", default="cuda")
    for command in (train, run):
        command.add_argument("--steps", type=int, default=100)
        command.add_argument("--lr", dest="learning_rate", type=float, default=1e-6)
    args = parser.parse_args(argv)
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
        args.output.write_text(json.dumps(result, allow_nan=False))
    else:
        options = vars(args).copy()
        options.pop("command")
        if args.command == "record":
            model, output = options.pop("model"), options.pop("output")
            if output.exists():
                raise FileExistsError(output)
            example = isolated(collect_trace, model, log_path=output / "generation.log", **options)
            path = save_example(example, output / "trace.json")
            result = {"trace": str(path), "tokens": example["trace"].num_tokens}
        else:
            result = run_workflow(**options)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
