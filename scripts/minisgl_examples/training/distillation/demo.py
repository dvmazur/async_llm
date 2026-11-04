"""Collect or load one async-thoughts trace, train an HF student, and verify it."""

import argparse
import asyncio
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
from minisgl_examples.training.process import run_process
from minisgl_examples.training.student import TraceSource, has_overfit, load_example, save_example


def record_rollout(
    root: Path,
    model: str | Path,
    prompt: str,
    directory: Path,
    *,
    think_tokens: int = 128,
    answer_tokens: int = 192,
    probe_period: int = 4,
    memory_ratio: float = 0.7,
    max_seq_len: int = 8192,
    cuda_graphs: bool = True,
) -> Path:
    """Generate one fresh rollout with the original async_thoughts demo."""
    if directory.exists():
        raise ValueError(f"Choose a new output directory: {directory}")
    trace = directory / "trace.json"
    print(f"Generating {directory.name}...", flush=True)
    run_process(
        root,
        [
            "-m",
            "minisgl_examples.training.distillation",
            "record",
            str(model),
            str(directory),
            "--prompt",
            prompt,
            "--think-tokens",
            str(think_tokens),
            "--answer-tokens",
            str(answer_tokens),
            "--probe-period",
            str(probe_period),
            "--memory-ratio",
            str(memory_ratio),
            "--max-seq-len",
            str(max_seq_len),
            *([] if cuda_graphs else ["--no-cuda-graphs"]),
        ],
        directory / "generation.log",
    )
    return trace


def generate_trace(model: str, directory: Path, **options: Any) -> Path:
    """Collect an example and save it in the common trace format."""
    destination = directory / "trace.json"
    if destination.exists():
        raise FileExistsError(destination)
    return save_example(asyncio.run(collect_trace(model, **options)), destination)


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


def evaluate_checkpoint(
    source: TraceSource, checkpoint: str | Path, *, device: str = "cuda"
) -> dict[str, Any]:
    _validate_student(source, str(checkpoint), device)
    torch.backends.cuda.matmul.allow_tf32 = False
    return training.verify_checkpoint(
        source, str(checkpoint), device=device, dtype=torch.bfloat16, master_dtype=torch.float32
    )


def fit_student(
    root: Path,
    teacher_trace: Path,
    student: str,
    directory: Path,
    *,
    steps: int = 100,
    lr: float = 1e-6,
) -> dict[str, Any]:
    run_process(
        root,
        [
            "-m",
            "minisgl_examples.training.distillation",
            "train",
            str(teacher_trace),
            student,
            str(directory),
            "--steps",
            str(steps),
            "--lr",
            str(lr),
        ],
        directory.parent / "training.log",
    )
    return dict(json.loads((directory / "metrics.json").read_text()))


def verify_checkpoint(
    root: Path, teacher_trace: Path, checkpoint: Path, destination: Path
) -> dict[str, Any]:
    run_process(
        root,
        [
            "-m",
            "minisgl_examples.training.distillation",
            "verify",
            str(teacher_trace),
            str(checkpoint),
            str(destination),
        ],
        destination.with_suffix(".log"),
    )
    return dict(json.loads(destination.read_text()))


def run_workflow(
    root: Path,
    teacher: str,
    student: str,
    output: Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    reference: int = 391,
    steps: int = 100,
    lr: float = 1e-6,
    trace: TraceSource | None = None,
    **generation_options: Any,
) -> dict[str, Any]:
    """Require a correct teacher and replay overfit, then assess independent generation."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if trace is None:
        teacher_trace = record_rollout(
            root, teacher, prompt, output / "teacher", **generation_options
        )
    else:
        teacher_trace = save_example(trace, output / "teacher" / "trace.json")
    example = load_example(teacher_trace)
    prompt = example["prompt"]
    tokenizer = AutoTokenizer.from_pretrained(example.get("model", teacher))
    grade = check_answer(example, tokenizer, [reference])
    (output / "teacher_check.json").write_text(json.dumps(grade, indent=2))
    if grade["status"] != "PASS":
        raise RuntimeError(f"Teacher answer is incorrect or incomplete: {grade}")
    metrics = fit_student(root, teacher_trace, student, output / "trained", steps=steps, lr=lr)
    if not metrics["overfit_achieved"]:
        raise RuntimeError("Overfit not reached; see training.log and trained/metrics.json")
    checkpoint = Path(metrics["checkpoint"])
    reloaded = verify_checkpoint(root, teacher_trace, checkpoint, output / "reloaded.json")
    if not has_overfit(reloaded) or abs(reloaded["loss"] - metrics["final"]["loss"]) >= 1e-5:
        raise RuntimeError(f"Saved checkpoint does not retain the training result: {reloaded}")
    student_trace = record_rollout(
        root, checkpoint, prompt, output / "student", **generation_options
    )
    return compare_rollouts(
        teacher_trace, student_trace, tokenizer, [reference], output / "checks.json"
    )


def main(argv: list[str] | None = None) -> None:
    """Run the complete experiment or a separate record/train/verify stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
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
        command.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args(argv)
    if args.mode == "train":
        train_student(
            args.model,
            args.trace,
            args.output,
            steps=args.steps,
            learning_rate=args.lr,
            device=args.device,
        )
    elif args.mode == "verify":
        result = evaluate_checkpoint(args.trace, args.model, device=args.device)
        args.output.write_text(json.dumps(result, allow_nan=False))
    else:
        options = vars(args).copy()
        options.pop("mode")
        if args.mode == "record":
            output = options.pop("output")
            generate_trace(directory=output, **options)
        else:
            result = run_workflow(Path.cwd().resolve(), **options)
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
