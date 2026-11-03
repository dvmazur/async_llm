"""Single-example Qwen3 distillation using the existing async_thoughts demo."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from minisgl.training import Trace
from torch.func import functional_call
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from minisgl_examples.training.loss import compute_supervised_loss, evaluate

DEFAULT_PROMPT = (
    r"What is 17 * 23? Give one short calculation and put only the final integer in \boxed{...}."
)


def load_example(path: Path) -> dict[str, Any]:
    """Read the single recorded rollout used in this experiment."""
    examples = json.loads(path.read_text())["examples"]
    if len(examples) != 1:
        raise ValueError("This sanity check trains on exactly one example")
    return dict(examples[0])


def run_process(root: Path, arguments: list[str], log: Path) -> None:
    """Run one model in its own process and keep the full output in a log."""
    env = os.environ.copy()
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", *arguments]
    with log.open("w") as output:
        process = subprocess.Popen(
            command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                output.write(line)
                output.flush()
                if line.startswith(("initial:", "step ")):
                    print(line.strip(), flush=True)
            output.flush()
            if process.wait():
                raise RuntimeError(f"Process failed. See {log}\n" + log.read_text()[-2000:])
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


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
            "minisgl_examples.training.distillation.sanity",
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


def generate_trace(
    model: str,
    directory: Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    think_tokens: int = 128,
    answer_tokens: int = 192,
    probe_period: int = 4,
    memory_ratio: float = 0.7,
    max_seq_len: int = 8192,
    cuda_graphs: bool = True,
) -> Path:
    """Record the original async-thoughts schedule with bounded engine resources."""
    from minisgl.llm import AsyncLLM

    from minisgl_examples.demos.async_thoughts.demo import DemoConfig, Prompting, _run_demo

    if torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one GPU")
    if AutoConfig.from_pretrained(model).model_type != "qwen3":
        raise ValueError("This sanity check supports dense text Qwen3")
    if min(think_tokens, answer_tokens, probe_period, max_seq_len) < 1:
        raise ValueError("Token budgets, probe period and sequence limit must be positive")
    if not 0 < memory_ratio < 1:
        raise ValueError("Memory ratio must be between zero and one")
    trace = directory / "trace.json"
    if trace.exists():
        raise FileExistsError(trace)
    llm = AsyncLLM(
        model,
        dtype=torch.bfloat16,
        max_running_req=4,
        cuda_graph_bs=[1, 2] if cuda_graphs else [],
        cuda_graph_max_bs=2 if cuda_graphs else 0,
        shared_cuda_graph_max_depth=4,
        shared_cuda_graph_prefill_rows=[32, 256] if cuda_graphs else [],
        max_prefill_rows=256,
        memory_ratio=memory_ratio,
        max_seq_len_override=max_seq_len,
        page_size=1,
        distributed_addr="tcp://127.0.0.1:0",
    )
    config = DemoConfig(
        model=model,
        problem=prompt,
        max_steps=think_tokens,
        max_answer_tokens=answer_tokens,
        probe_period=probe_period,
        trace_output=trace,
    )
    asyncio.run(_run_demo(config, llm, Prompting(prompt)))
    return trace


# Only these two small adapters are needed for differentiable BF16 parameter substitution.
class TrainingStep(torch.nn.Module):
    """Compute the loss and backward before BF16 parameter substitution ends."""

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, trace: Trace, targets: torch.Tensor) -> float:
        loss = compute_supervised_loss(self.model, trace, targets) / targets.ne(-100).sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        # Keep backward inside functional_call, including checkpoint recomputation.
        loss.backward()
        return float(loss.detach())


class TraceEvaluation(torch.nn.Module):
    """Return loss and accuracy for the recorded targets."""

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, trace: Trace, targets: torch.Tensor) -> dict[str, float]:
        result = evaluate(self.model, [(trace, targets)])
        return {key: float(result[key]) for key in ("loss", "accuracy")}


def run_bf16(module: torch.nn.Module, trace: Trace, targets: torch.Tensor) -> Any:
    """Use BF16 parameters temporarily while keeping the FP32 master weights."""
    weights = {name: p.to(torch.bfloat16) for name, p in module.named_parameters()}
    return functional_call(module, weights, (trace, targets), tie_weights=True)


def load_student(
    trace_path: Path, model_name: str, *, device: str = "cuda"
) -> tuple[PreTrainedModel, Trace, torch.Tensor, PreTrainedTokenizerBase]:
    if torch.device(device).type == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    data = json.loads(trace_path.read_text())
    example = load_example(trace_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoTokenizer.from_pretrained(data["model"])
    if (
        json.loads(tokenizer.backend_tokenizer.to_str())
        != json.loads(teacher.backend_tokenizer.to_str())
        or tokenizer.all_special_ids != teacher.all_special_ids
    ):
        raise ValueError("Teacher/student tokenizers differ")
    trace = Trace.from_dict(example["trace"])
    ids, valid = example["targets"], set(tokenizer.get_vocab().values())
    if (
        len(ids) != trace.num_tokens
        or not any(t != -100 for t in ids)
        or any(type(t) is not int or (t != -100 and t not in valid) for t in ids)
        or any(type(t.token_id) is not int or t.token_id not in valid for t in trace.tokens)
    ):
        raise ValueError("Invalid or misaligned token IDs/targets")
    trace = trace.to(device)
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type != "qwen3":
        raise ValueError("This sanity check supports dense text Qwen3")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        dtype=torch.float32,
    ).to(device)
    return model, trace, torch.tensor(ids, dtype=torch.long, device=device), tokenizer


def evaluate_student(
    model: PreTrainedModel, trace: Trace, targets: torch.Tensor
) -> dict[str, float]:
    with torch.no_grad():
        result: dict[str, float] = run_bf16(TraceEvaluation(model.eval()), trace, targets)
    return result


def has_overfit(result: dict[str, float]) -> bool:
    return result["accuracy"] == 1.0 and result["loss"] < 0.01


def train_from_trace(
    trace: Path, model_name: str, output: Path, steps: int, lr: float, *, device: str = "cuda"
) -> dict[str, Any]:
    """Fit the student to all supervised tokens of one recorded trace."""
    if output.exists() or steps < 1 or not 0 < lr < float("inf"):
        raise ValueError("Use a new output path and positive steps/lr")
    # One iteration trains on all supervised tokens of the same trace.
    source_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, replay, targets, tokenizer = load_student(trace, model_name, device=device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    initial = final = evaluate_student(model, replay, targets)
    history = [{"step": 0, **initial}]
    print(f"initial: {initial}", flush=True)
    step = 0
    while step < steps and not has_overfit(final):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_bf16(TrainingStep(model), replay, targets)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % 5 == 0 or step == steps:
            final = evaluate_student(model, replay, targets)
            history.append({"step": step, **final})
            print(f"step {step}: {final}", flush=True)
    checkpoint = output / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    metrics = {
        "initial": initial,
        "final": final,
        "completed_steps": step,
        "history": history,
        "overfit_achieved": has_overfit(final),
        "checkpoint": str(checkpoint.resolve()),
        "teacher_sha256": source_hash,
        "lr": lr,
        "precision": "BF16 forward / FP32 master weights and AdamW",
        "seed": 7,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


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
            "minisgl_examples.training.distillation.sanity",
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
) -> dict[str, float]:
    run_process(
        root,
        [
            "-m",
            "minisgl_examples.training.distillation.sanity",
            "verify",
            str(teacher_trace),
            str(checkpoint),
            str(destination),
        ],
        destination.with_suffix(".log"),
    )
    return dict(json.loads(destination.read_text()))


def check_result(ok: bool, reason: str, **details: Any) -> dict[str, Any]:
    return {"status": "PASS" if ok else "FAIL", "reason": reason, "details": details}


def check_answer(
    example: dict[str, Any], tokenizer: PreTrainedTokenizerBase, reference: list[int]
) -> dict[str, Any]:
    """Check the single boxed integer against an independent reference."""
    if len(reference) != 1:
        raise ValueError("This sanity check expects one final integer")
    text = tokenizer.decode(example["writer_tokens"], skip_special_tokens=True)
    boxes = re.findall(r"\\+boxed\s*\{\s*([+-]?\d+)\s*\}", text)
    actual = [int(value) for value in boxes]
    return check_result(
        actual == reference and len(re.findall(r"\\+boxed", text)) == 1,
        "Expected exactly one numeric boxed answer",
        expected=reference,
        actual=actual,
    )


def check_async_behavior(
    example: dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> dict[str, Any]:
    """Check that writer uses new thoughts while thinker is still producing tokens."""
    from minisgl_examples.demos.async_thoughts.demo import Prompting

    try:
        trace = Trace.from_dict(example["trace"])
        trace.to("cpu")
        prompting = Prompting(example["prompt"])
        thinker_prefix = tokenizer.encode(prompting.thinker_output_prefix, add_special_tokens=False)
        writer_prefix = tokenizer.encode(prompting.writer_output_prefix, add_special_tokens=False)
        question = tokenizer.encode(prompting.mode_switching_question, add_special_tokens=False)
        thinker_block = writer_block = None
        probe_steps: list[int] = []
        for batch in trace.batches:
            if batch.kind != "prefill":
                continue
            for request in batch.requests:
                if list(request.input_ids) == thinker_prefix:
                    thinker_block = request.write_to
                elif list(request.input_ids) == writer_prefix:
                    writer_block = request.write_to
                elif list(request.input_ids) == question:
                    probe_steps.append(batch.step)

        if thinker_block is None or writer_block is None:
            raise ValueError("Cannot find thinker/writer prefixes from async_thoughts")
        thinker_steps: list[int] = []
        writer_steps: list[int] = []
        visible_thoughts: list[int] = []
        for token, target in zip(trace.tokens, example["targets"], strict=True):
            if target == -100:
                continue
            if token.block == thinker_block:
                thinker_steps.append(token.step)
            elif token.block == writer_block:
                writer_steps.append(token.step)
                length = sum(n for block, n in token.view if block == thinker_block)
                visible_thoughts.append(length)

        checks = {
            "both_streams": bool(thinker_steps and writer_steps),
            "writer_before_thinker_ends": bool(
                writer_steps and thinker_steps and min(writer_steps) < max(thinker_steps)
            ),
            "probe_before_writer": bool(
                writer_steps and any(s < min(writer_steps) for s in probe_steps)
            ),
            "writer_reads_thoughts": bool(visible_thoughts)
            and min(visible_thoughts) > len(thinker_prefix),
            "writer_receives_new_thoughts": len(set(visible_thoughts)) > 1
            and visible_thoughts == sorted(visible_thoughts),
        }
        return check_result(
            all(checks.values()),
            "Observable async structure, not optimality of probe decisions",
            failed=[name for name, ok in checks.items() if not ok],
            checks=checks,
        )
    except (ValueError, KeyError, TypeError, NotImplementedError) as error:
        return check_result(False, "Invalid or unsupported trace", error=str(error))


def normalize_trace(data: dict[str, Any]) -> dict[str, Any]:
    """Rename block IDs so different allocations do not count as different behavior."""
    trace = Trace.from_dict(data)
    if trace.multimodal:
        raise ValueError("This sanity check uses text only")
    result = asdict(trace)
    block_ids: dict[int, int] = {}

    def rename(block: int) -> int:
        if block not in block_ids:
            block_ids[block] = len(block_ids)
        return block_ids[block]

    for batch in result["batches"]:
        for request in batch["requests"]:
            request["write_to"] = rename(request["write_to"])
            request["cache_view"] = [
                (rename(block), length) for block, length in request["cache_view"]
            ]
    for update in result["updates"]:
        update["block"] = rename(update["block"])
        if update["refs"] is not None:
            update["refs"] = [(rename(block), length) for block, length in update["refs"]]
    return result


def compare_rollouts(
    teacher_trace: Path,
    student_trace: Path,
    tokenizer: PreTrainedTokenizerBase,
    reference: list[int],
    destination: Path,
) -> dict[str, Any]:
    teacher, student = load_example(teacher_trace), load_example(student_trace)
    if teacher["prompt"] != student["prompt"]:
        raise ValueError("Teacher and student prompts differ")
    fields = {
        key: teacher[key] == student[key]
        for key in ("prompt", "thinker_tokens", "writer_tokens", "targets")
    }
    left, right = normalize_trace(teacher["trace"]), normalize_trace(student["trace"])
    fields["execution_trace"] = left == right
    streams = {}
    for role in ("thinker", "writer"):
        teacher_tokens = teacher[f"{role}_tokens"]
        student_tokens = student[f"{role}_tokens"]
        prefix = 0
        for teacher_token, student_token in zip(teacher_tokens, student_tokens):
            if teacher_token != student_token:
                break
            prefix += 1
        streams[role] = {
            "equal": teacher_tokens == student_tokens,
            "common_prefix_tokens": prefix,
            "teacher_length": len(teacher_tokens),
            "student_length": len(student_tokens),
        }
    report = {
        "teacher_answer": check_answer(teacher, tokenizer, reference),
        "student_answer": check_answer(student, tokenizer, reference),
        "teacher_async": check_async_behavior(teacher, tokenizer),
        "student_async": check_async_behavior(student, tokenizer),
        "exact_recorded_rollout": check_result(
            all(fields.values()),
            "Tokens, targets and cache history",
            fields=fields,
            streams=streams,
            first_trace_difference=None if left == right else "Cache execution histories differ",
        ),
    }
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


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
    **generation_options: Any,
) -> dict[str, Any]:
    """Require a correct teacher and replay overfit, then assess independent generation."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    teacher_trace = record_rollout(root, teacher, prompt, output / "teacher", **generation_options)
    tokenizer = AutoTokenizer.from_pretrained(teacher)
    grade = check_answer(load_example(teacher_trace), tokenizer, [reference])
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
        train_from_trace(
            args.trace, args.model, args.output, args.steps, args.lr, device=args.device
        )
    elif args.mode == "verify":
        model, replay, targets, _ = load_student(args.trace, args.model, device=args.device)
        result = evaluate_student(model, replay, targets)
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
