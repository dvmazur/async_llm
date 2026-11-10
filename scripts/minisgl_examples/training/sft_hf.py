"""Mine AsyncLLM traces and train them with Hugging Face Trainer."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
from minisgl.training import Trace
from minisgl.training.trace import _validate_replay_options
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    TrainingArguments,
)

from minisgl_examples.training.loss import evaluate
from minisgl_examples.training.sft import mine
from minisgl_examples.training.trainer_hf import (
    TraceDataCollator,
    TraceDataset,
    TraceSFTTrainer,
)


def _trace_options(args: argparse.Namespace) -> dict[str, Any]:
    options = {
        "query_chunk_size": args.query_chunk_size,
        "attention_checkpointing": args.attention_checkpointing,
        "cache_plans": args.cache_replay_plans,
        "gdn_segment_size": args.gdn_segment_size,
        "gdn_read_mode": args.gdn_read_mode,
        "gdn_cache_reused": args.gdn_cache_reused,
        "gdn_read_backend": args.gdn_read_backend,
    }
    _validate_replay_options(
        options["query_chunk_size"],
        options["gdn_segment_size"],
        options["gdn_read_mode"],
        options["gdn_read_backend"],
    )
    return options


def _is_multimodal(dataset: TraceDataset) -> bool:
    return any(
        request.pixel_values is not None
        for trace, _ in dataset.replay_examples
        for batch in trace.batches
        for request in batch.requests
    )


def _model_factory(model_name: str, multimodal: bool):
    config = AutoConfig.from_pretrained(model_name)
    factory = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    if not multimodal:
        config = config.get_text_config()
    return factory, config


def _parity_error(initial: dict[str, Any], raw_examples: list[dict[str, Any]]) -> float | None:
    errors = []
    for result, example in zip(initial["examples"], raw_examples, strict=True):
        logprobs = example.get("logprobs")
        if logprobs is None:
            continue
        expected = [value for value in logprobs if value is not None]
        if len(expected) != len(result["nll"]):
            raise ValueError("Recorded log probabilities do not match supervised targets")
        errors.extend(abs(nll + logprob) for nll, logprob in zip(result["nll"], expected))
    return max(errors) if errors else None


def _assert_reload_exact(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    if len(expected["examples"]) != len(actual["examples"]):
        raise AssertionError("Reloaded checkpoint changed the evaluation set")
    for left, right in zip(expected["examples"], actual["examples"], strict=True):
        torch.testing.assert_close(
            torch.tensor(left["nll"]), torch.tensor(right["nll"]), rtol=0, atol=0
        )


def train(args: argparse.Namespace) -> None:
    trace_options = _trace_options(args)
    data = json.loads(args.data.read_text())
    raw_examples = data["examples"]
    dataset = TraceDataset(raw_examples, trace_options=trace_options)
    multimodal = _is_multimodal(dataset)
    model_name = args.model or data["model"]
    factory, config = _model_factory(model_name, multimodal)
    model = factory.from_pretrained(model_name, config=config, dtype=torch.float32)
    processor_factory = AutoProcessor if multimodal else AutoTokenizer
    processor = processor_factory.from_pretrained(model_name)

    save_strategy = "no" if args.save_steps == 0 else "steps"
    accumulation_steps = args.gradient_accumulation_steps or math.ceil(
        len(dataset) / args.batch_size
    )
    training_args = TrainingArguments(
        output_dir=str(args.output),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="constant",
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
        prediction_loss_only=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy=save_strategy,
        save_steps=max(1, args.save_steps),
        logging_strategy="steps",
        logging_steps=1,
        optim="adamw_torch",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )
    # Match the original SFT loop: one AdamW group containing every parameter.
    # Trainer's default grouping excludes biases and normalization weights from decay.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = TraceSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=TraceDataCollator(),
        processing_class=processor,
        loss_chunk_size=args.loss_chunk_size,
        loss_checkpointing=args.loss_checkpointing,
        optimizers=(optimizer, None),
    )

    initial = evaluate(model.eval(), dataset.replay_examples, args.loss_chunk_size)
    max_error = None
    if model_name == data["model"]:
        max_error = _parity_error(initial, raw_examples)
        if max_error is not None and max_error > 0.1:
            raise RuntimeError(f"Inference/replay target logprob mismatch: {max_error:.6f}")
    print(
        f"initial loss={initial['loss']:.6f} accuracy={initial['accuracy']:.2%}"
        + ("" if max_error is None else f" max logprob error={max_error:.6f}"),
        flush=True,
    )

    result = trainer.train()
    final = evaluate(model.eval(), dataset.replay_examples, args.loss_chunk_size)
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    trainer.save_state()
    print(f"final loss={final['loss']:.6f} accuracy={final['accuracy']:.2%}", flush=True)

    reload_exact = False
    if args.verify_reload:
        del trainer, model
        for trace, _ in dataset.replay_examples:
            trace._attention.clear()
            trace._gdn.clear()
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = factory.from_pretrained(args.output, dtype=torch.float32).cuda().eval()
        reloaded_result = evaluate(reloaded, dataset.replay_examples, args.loss_chunk_size)
        _assert_reload_exact(final, reloaded_result)
        reload_exact = True
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()

    metrics = {
        "trainer": "huggingface",
        "model": model_name,
        "teacher_model": data["model"],
        "data": str(args.data),
        "completed_steps": result.global_step,
        "training_loss": result.training_loss,
        "supervised_tokens": dataset.supervised_tokens,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "initial": initial,
        "final": final,
        "max_logprob_error": max_error,
        "reload_exact": reload_exact,
        "log_history": result.metrics,
        "torch": torch.__version__,
    }
    (args.output.parent / "metrics_hf.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved Hugging Face Trainer checkpoint to {args.output}", flush=True)


def _add_mine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=Path("training_runs/qwen3_sft_hf/trajectories.json")
    )


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data", type=Path, default=Path("training_runs/qwen3_sft_hf/trajectories.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("training_runs/qwen3_sft_hf/checkpoint")
    )
    parser.add_argument("--model", help="Student model; defaults to the model stored in data")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Microbatches per update; defaults to one full pass over the trace dataset",
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument(
        "--save-steps", type=int, default=0, help="0 disables intermediate Trainer checkpoints"
    )
    parser.add_argument("--verify-reload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attention-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=Trace.attention_checkpointing,
    )
    parser.add_argument("--loss-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache-replay-plans", action=argparse.BooleanOptionalAction, default=Trace.cache_plans
    )
    parser.add_argument("--query-chunk-size", type=int, default=Trace.query_chunk_size)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    parser.add_argument("--gdn-segment-size", type=int, default=Trace.gdn_segment_size)
    parser.add_argument(
        "--gdn-read-mode", choices=("expanded", "factorized"), default=Trace.gdn_read_mode
    )
    parser.add_argument(
        "--gdn-cache-reused", action=argparse.BooleanOptionalAction, default=Trace.gdn_cache_reused
    )
    parser.add_argument(
        "--gdn-read-backend", choices=("torch", "triton"), default=Trace.gdn_read_backend
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    _add_mine_arguments(commands.add_parser("mine"))
    _add_train_arguments(commands.add_parser("train"))
    args = parser.parse_args(argv)
    if torch.cuda.device_count() != 1:
        parser.error("Select exactly one GPU with CUDA_VISIBLE_DEVICES")
    if not args.output.resolve().is_relative_to(Path.cwd()):
        parser.error("Outputs must be inside the working directory")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.command == "mine":
        if args.max_tokens < 1 or args.temperature <= 0:
            parser.error("--max-tokens and --temperature must be positive")
        torch.manual_seed(args.seed)
        asyncio.run(mine(args))
        return
    positive = (
        args.steps,
        args.batch_size,
        args.lr,
        args.max_grad_norm,
        args.eval_steps,
        args.query_chunk_size,
        args.loss_chunk_size,
        args.gdn_segment_size,
    )
    if (
        any(value <= 0 for value in positive)
        or (args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps <= 0)
        or args.save_steps < 0
        or args.weight_decay < 0
    ):
        parser.error(
            "Training sizes/rates must be positive; save steps and weight decay nonnegative"
        )
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
