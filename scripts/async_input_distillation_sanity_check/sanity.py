"""Distill one recorded trajectory with a delayed problem correction."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from minisgl.training import Trace
from minisgl_examples.training.loss import compute_supervised_loss, evaluate
from torch.func import functional_call
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_example(path: Path) -> dict[str, Any]:
    """Read the single recorded rollout used in this experiment."""
    examples = json.loads(path.read_text())["examples"]
    if len(examples) != 1:
        raise ValueError("This sanity check trains on exactly one example")
    return dict(examples[0])


def run_process(root: Path, gpu: str, arguments: list[str], log: Path) -> None:
    """Run one model in its own process and keep the full output in a log."""
    free = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    if int(free.strip()) < 12000:
        raise RuntimeError(f"GPU {gpu}: need at least 12000 MiB free")
    venv_bin = Path(sys.executable).parent
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    path = [str(venv_bin)]
    if cuda_home.is_dir():
        path.append(str(cuda_home / "bin"))
    path.append(os.environ.get("PATH", ""))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "OMP_NUM_THREADS": "1",
        "PATH": ":".join(path),
    }
    if cuda_home.is_dir():
        env["CUDA_HOME"] = str(cuda_home)
    env.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", *arguments]
    with log.open("w") as output:
        process = subprocess.Popen(
            command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            if line.startswith(("initial:", "step ")):
                print(line.strip(), flush=True)
        output.flush()
        if process.wait():
            raise RuntimeError(f"Process failed. See {log}\n" + log.read_text()[-2000:])


def record_rollout(
    root: Path,
    gpu: str,
    model: str | Path,
    directory: Path,
    *,
    dataset: str = "MrDarkTesla/math_500_sharded",
    dataset_revision: str = "660e78826c964737cfb75851913536b16dc2942e",
    split: str = "train",
    index: int = 0,
    k_steps: int = 32,
    max_tokens: int = 512,
    precision: str = "bfloat16",
) -> Path:
    """Generate one rollout whose second problem shard arrives during decoding."""
    if directory.exists():
        raise ValueError(f"Choose a new output directory: {directory}")
    trace = directory / "trace.json"
    print(f"Generating {directory.name}...", flush=True)
    run_process(
        root,
        gpu,
        [
            str(Path(__file__).with_name("async_input_rollout.py")),
            "--model",
            str(model),
            "--dataset",
            dataset,
            "--dataset-revision",
            dataset_revision,
            "--split",
            split,
            "--index",
            str(index),
            "--k-steps",
            str(k_steps),
            "--max-tokens",
            str(max_tokens),
            "--precision",
            precision,
            "--memory-ratio",
            "0.7",
            "--output",
            str(trace),
        ],
        directory / "generation.log",
    )
    return trace


# Only these two small adapters are needed for differentiable low-precision
# parameter substitution.
class TrainingStep(torch.nn.Module):
    """Compute loss/backward before low-precision parameter substitution ends."""

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


def precision_dtype(precision: str) -> torch.dtype:
    if precision not in {"bfloat16", "float16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    return getattr(torch, precision)


def run_low_precision(
    module: torch.nn.Module,
    trace: Trace,
    targets: torch.Tensor,
    precision: str,
) -> Any:
    """Use low-precision parameters temporarily while keeping FP32 master weights."""
    dtype = precision_dtype(precision)
    weights = {name: p.to(dtype) for name, p in module.named_parameters()}
    return functional_call(module, weights, (trace, targets), tie_weights=True)


def load_student(
    trace_path: Path,
    model_name: str,
    *,
    attention_backend: str = "torch",
    query_chunk_size: int = 128,
    device: str = "cuda",
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
    trace = trace.to(device, attention_backend=attention_backend, query_chunk_size=query_chunk_size)
    config = AutoConfig.from_pretrained(model_name).get_text_config()
    if config.model_type not in {"qwen3", "qwen3_5_text"}:
        raise ValueError("This sanity check supports dense text Qwen3/Qwen3.5")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        dtype=torch.float32,
    ).to(device)
    return model, trace, torch.tensor(ids, dtype=torch.long, device=device), tokenizer


def evaluate_student(
    model: PreTrainedModel,
    replay: Trace,
    targets: torch.Tensor,
    precision: str = "bfloat16",
) -> dict[str, float]:
    with torch.no_grad():
        result: dict[str, float] = run_low_precision(
            TraceEvaluation(model.eval()), replay, targets, precision
        )
    return result


def has_overfit(result: dict[str, float]) -> bool:
    return result["accuracy"] == 1.0 and result["loss"] < 0.01


def train_from_trace(
    trace: Path,
    model_name: str,
    output: Path,
    steps: int,
    lr: float,
    precision: str = "bfloat16",
    optimizer_foreach: bool | None = None,
    *,
    attention_backend: str = "torch",
    query_chunk_size: int = 128,
) -> dict[str, Any]:
    if output.exists() or steps < 1 or not 0 < lr < float("inf"):
        raise ValueError("Use a new output path and positive steps/lr")
    """Fit the student to all supervised tokens of one recorded trace."""
    # One iteration trains on all supervised tokens of the same trace.
    source_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, replay, targets, tokenizer = load_student(
        trace, model_name, attention_backend=attention_backend, query_chunk_size=query_chunk_size
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=optimizer_foreach)
    initial = final = evaluate_student(model, replay, targets, precision)
    history = [{"step": 0, **initial}]
    print(f"initial: {initial}", flush=True)
    step = 0
    while step < steps and not has_overfit(final):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_low_precision(TrainingStep(model), replay, targets, precision)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % 5 == 0 or step == steps:
            final = evaluate_student(model, replay, targets, precision)
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
        "precision": f"{precision} forward / FP32 master weights and AdamW",
        "optimizer_foreach": optimizer_foreach,
        "replay_options": {
            "attention_backend": attention_backend,
            "query_chunk_size": query_chunk_size,
        },
        "seed": 7,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


def fit_student(
    root: Path,
    gpu: str,
    teacher_trace: Path,
    student: str,
    directory: Path,
    *,
    steps: int = 100,
    lr: float = 1e-6,
    precision: str = "bfloat16",
    optimizer_foreach: bool | None = None,
    attention_backend: str = "torch",
    query_chunk_size: int = 128,
) -> dict[str, Any]:
    foreach = "auto" if optimizer_foreach is None else str(optimizer_foreach).lower()
    run_process(
        root,
        gpu,
        [
            str(Path(__file__).resolve()),
            "train",
            str(teacher_trace),
            student,
            str(directory),
            "--steps",
            str(steps),
            "--lr",
            str(lr),
            "--precision",
            precision,
            "--optimizer-foreach",
            foreach,
            "--attention-backend",
            attention_backend,
            "--query-chunk-size",
            str(query_chunk_size),
        ],
        directory.parent / "training.log",
    )
    return dict(json.loads((directory / "metrics.json").read_text()))


def verify_checkpoint(
    root: Path,
    gpu: str,
    teacher_trace: Path,
    checkpoint: Path,
    destination: Path,
    *,
    precision: str = "bfloat16",
    attention_backend: str = "torch",
    query_chunk_size: int = 128,
) -> dict[str, Any]:
    run_process(
        root,
        gpu,
        [
            str(Path(__file__).resolve()),
            "verify",
            str(teacher_trace),
            str(checkpoint),
            str(destination),
            "--precision",
            precision,
            "--attention-backend",
            attention_backend,
            "--query-chunk-size",
            str(query_chunk_size),
        ],
        destination.with_suffix(".log"),
    )
    return dict(json.loads(destination.read_text()))


def check_result(ok: bool, reason: str, **details: Any) -> dict[str, Any]:
    return {"status": "PASS" if ok else "FAIL", "reason": reason, "details": details}


def find_last_boxed_answer(text: str) -> str | None:
    """Extract the last complete boxed expression, including nested braces."""
    prefix = r"\boxed{"
    end = len(text)
    while True:
        start = text.rfind(prefix, 0, end)
        if start < 0:
            return None
        depth = 0
        for position in range(start + len(r"\boxed"), len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    return text[start + len(prefix) : position].strip()
        end = start


def check_answer(
    example: dict[str, Any], tokenizer: PreTrainedTokenizerBase, reference: str
) -> dict[str, Any]:
    """Grade the generated MATH answer with the benchmark's symbolic verifier."""
    from math_verify import parse, verify

    text = tokenizer.decode(example["output_tokens"], skip_special_tokens=True)
    actual = find_last_boxed_answer(text)
    try:
        ok = actual is not None and bool(
            verify(parse(r"\boxed{" + reference + "}"), parse(r"\boxed{" + actual + "}"))
        )
    except Exception:
        ok = False
    return check_result(
        ok,
        "Symbolic equality of the final boxed answer",
        expected=reference,
        actual=actual,
    )


def check_async_behavior(
    example: dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> dict[str, Any]:
    """Check that the second shard was appended after k supervised decode steps."""
    try:
        trace = Trace.from_dict(example["trace"])
        trace.to("cpu")
        correction = tokenizer.encode(
            "\n\nADDITIONAL INFORMATION: " + example["problem_shards"][1] + "\n\n",
            add_special_tokens=False,
        )
        injections = []
        output_block = None
        for batch in trace.batches:
            for request in batch.requests:
                if list(request.input_ids) == correction:
                    injections.append((batch.step, request.write_to, request.write_start))
                if batch.kind == "decode" and output_block is None:
                    output_block = request.write_to
        if output_block is None:
            raise ValueError("Cannot find generated output block")
        prompt_routes = [item for item in injections if item[1] != output_block]
        stream_routes = [item for item in injections if item[1] == output_block]
        if len(prompt_routes) != 1 or len(stream_routes) != 1:
            raise ValueError("Correction must be routed once to prompt and once to live output")
        injection_step, prompt_block, prompt_start = prompt_routes[0]
        before = after = 0
        sees_extended_prompt = False
        for token, target in zip(trace.tokens, example["targets"], strict=True):
            if target == -100 or token.block != output_block:
                continue
            if token.step < injection_step:
                before += 1
            elif token.step > injection_step:
                after += 1
                sees_extended_prompt |= any(
                    block == prompt_block and length >= prompt_start + len(correction)
                    for block, length in token.view
                )
        checks = {
            "correction_routed_to_prompt_and_live_stream": len(injections) == 2,
            "injected_after_requested_decode_count": before == example["k_steps"],
            "generation_continues_after_correction": after > 0,
            "post_correction_decode_sees_extended_prompt": sees_extended_prompt,
        }
        return check_result(
            all(checks.values()),
            "Correction arrives mid-reasoning and subsequent decode sees it",
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
    reference: str,
    destination: Path,
) -> dict[str, Any]:
    teacher, student = load_example(teacher_trace), load_example(student_trace)
    if teacher["prompt"] != student["prompt"]:
        raise ValueError("Teacher and student prompts differ")
    fields = {
        key: teacher[key] == student[key]
        for key in ("prompt", "problem_shards", "answer", "k_steps", "output_tokens", "targets")
    }
    fields["precision"] = teacher.get("precision", "bfloat16") == student.get(
        "precision", "bfloat16"
    )
    left, right = normalize_trace(teacher["trace"]), normalize_trace(student["trace"])
    fields["execution_trace"] = left == right
    report = {
        "teacher_answer": check_answer(teacher, tokenizer, reference),
        "student_answer": check_answer(student, tokenizer, reference),
        "teacher_async": check_async_behavior(teacher, tokenizer),
        "student_async": check_async_behavior(student, tokenizer),
        "exact_recorded_rollout": check_result(
            all(fields.values()),
            "Tokens, targets and cache history",
            fields=fields,
            teacher_tokens=len(teacher["output_tokens"]),
            student_tokens=len(student["output_tokens"]),
            first_trace_difference=None if left == right else "Cache execution histories differ",
        ),
    }
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


def main(argv: list[str] | None = None) -> None:
    """Worker process used by the notebook for training or checkpoint verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    parser.add_argument("trace", type=Path)
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--precision", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--optimizer-foreach", choices=("auto", "true", "false"), default="auto")
    parser.add_argument(
        "--attention-backend", choices=("torch", "auto", "torch_optimized", "fa4"), default="torch"
    )
    parser.add_argument("--query-chunk-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.query_chunk_size < 1:
        parser.error("query-chunk-size must be positive")
    optimizer_foreach = {
        "auto": None,
        "true": True,
        "false": False,
    }[args.optimizer_foreach]
    if args.mode == "train":
        train_from_trace(
            args.trace,
            args.model,
            args.output,
            args.steps,
            args.lr,
            args.precision,
            optimizer_foreach,
            attention_backend=args.attention_backend,
            query_chunk_size=args.query_chunk_size,
        )
    else:
        model, replay, targets, _ = load_student(
            args.trace,
            args.model,
            attention_backend=args.attention_backend,
            query_chunk_size=args.query_chunk_size,
        )
        result = evaluate_student(model, replay, targets, args.precision)
        args.output.write_text(
            json.dumps(
                {
                    **result,
                    "replay_options": {
                        "attention_backend": args.attention_backend,
                        "query_chunk_size": args.query_chunk_size,
                    },
                },
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
