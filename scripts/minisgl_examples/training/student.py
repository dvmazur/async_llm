"""Train an HF student on a collected example or a saved trace."""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeAlias

import torch
from minisgl.training import Trace, replay_trace
from minisgl.training.forward import REPLAY_MODEL_TYPES
from torch.func import functional_call
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from minisgl_examples.training.loss import compute_supervised_loss, evaluate

TraceSource: TypeAlias = str | Path | Mapping[str, Any]
Processor: TypeAlias = PreTrainedTokenizerBase | ProcessorMixin


def load_example(source: TraceSource) -> dict[str, Any]:
    """Accept one example directly or the on-disk ``model/examples`` envelope."""
    data = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else dict(source)
    if "examples" in data:
        if len(data["examples"]) != 1:
            raise ValueError("These demos train on exactly one example")
        example = dict(data["examples"][0])
        if data.get("model") is not None:
            example["model"] = data["model"]
    else:
        example = dict(data)
    trace = example["trace"]
    example["trace"] = trace if isinstance(trace, Trace) else Trace.from_dict(trace)
    example["targets"] = list(example["targets"])
    return example


def save_example(source: TraceSource, path: str | Path, *, model: str | None = None) -> Path:
    example = load_example(source)
    teacher = model if model is not None else example.pop("model", None)
    example.pop("model", None)
    data = {"examples": [{**example, "trace": asdict(example["trace"])}]}
    if teacher is not None:
        data["model"] = str(teacher)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def load_student(
    source: TraceSource,
    model_name: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[PreTrainedModel, Trace, torch.Tensor, Processor]:
    example = load_example(source)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if example.get("model") is not None:
        teacher = AutoTokenizer.from_pretrained(example["model"])
        if (
            json.loads(tokenizer.backend_tokenizer.to_str())
            != json.loads(teacher.backend_tokenizer.to_str())
            or tokenizer.all_special_ids != teacher.all_special_ids
        ):
            raise ValueError("Teacher/student tokenizers differ")
    trace, ids = example["trace"], example["targets"]
    valid = set(tokenizer.get_vocab().values())
    if (
        len(ids) != trace.num_tokens
        or not any(value != -100 for value in ids)
        or any(type(value) is not int or (value != -100 and value not in valid) for value in ids)
        or any(
            type(token.token_id) is not int or token.token_id not in valid for token in trace.tokens
        )
    ):
        raise ValueError("Invalid or misaligned token IDs/targets")
    trace = trace.to(device)
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type not in REPLAY_MODEL_TYPES:
        raise ValueError("Student training supports dense Llama, Qwen3 and Qwen3.5")
    multimodal = getattr(config, "vision_config", None) is not None
    factory = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    processor = AutoProcessor.from_pretrained(model_name) if multimodal else tokenizer
    model = factory.from_pretrained(model_name, config=config, dtype=dtype).to(device)
    return model, trace, torch.tensor(ids, device=device), processor


class _ReplayStep(torch.nn.Module):
    def __init__(self, model: PreTrainedModel, chunk_size: int) -> None:
        super().__init__()
        self.model, self.chunk_size = model, chunk_size

    def forward(self, trace: Trace, targets: torch.Tensor, backward: bool) -> Any:
        if not backward:
            return evaluate(self.model, [(trace, targets)], self.chunk_size)
        loss = compute_supervised_loss(self.model, trace, targets, self.chunk_size)
        loss = loss / targets.ne(-100).sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        # Checkpoint recomputation must happen before temporary parameter bindings end.
        loss.backward()
        return loss.detach().item()


def _run_step(
    model: PreTrainedModel,
    trace: Trace,
    targets: torch.Tensor,
    dtype: torch.dtype,
    chunk_size: int,
    *,
    backward: bool,
) -> Any:
    step = _ReplayStep(model, chunk_size)
    if all(parameter.dtype == dtype for parameter in model.parameters()):
        return step(trace, targets, backward)
    weights = {name: parameter.to(dtype) for name, parameter in step.named_parameters()}
    return functional_call(step, weights, (trace, targets, backward), tie_weights=True)


@torch.no_grad()
def evaluate_student(
    model: PreTrainedModel,
    trace: Trace,
    targets: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    chunk_size: int = 128,
) -> dict[str, Any]:
    return _run_step(model.eval(), trace, targets, dtype, chunk_size, backward=False)


def has_overfit(metrics: Mapping[str, Any]) -> bool:
    return metrics["accuracy"] == 1.0 and metrics["loss"] < 0.01


@torch.no_grad()
def _checkpoint_logits(
    model: PreTrainedModel, trace: Trace, rows: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    class Logits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self) -> torch.Tensor:
            return replay_trace(self.model, trace, rows=rows, hidden_only=False)

    wrapper = Logits()
    weights = {name: parameter.to(dtype) for name, parameter in wrapper.named_parameters()}
    return functional_call(wrapper, weights, (), tie_weights=True).cpu()


def train_student(
    source: TraceSource,
    model_name: str,
    checkpoint: str | Path,
    *,
    steps: int = 80,
    lr: float = 3e-5,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    master_dtype: torch.dtype | None = None,
    optimizer_device: str | None = None,
    chunk_size: int = 128,
    stop_on_initial: bool = False,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    if steps < 1 or chunk_size < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Use positive steps, learning rate and chunk size")
    torch.manual_seed(7)
    model, trace, targets, processor = load_student(
        source, model_name, device=device, dtype=master_dtype or dtype
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = dict(model.named_parameters())
    model_device = next(model.parameters()).device
    target_device = torch.device(optimizer_device) if optimizer_device is not None else model_device
    if target_device.type == model_device.type and target_device.index is None:
        target_device = model_device
    offload = target_device != model_device
    if offload:
        targets = targets.to(target_device)
    masters = (
        {
            name: torch.nn.Parameter(p.detach().to(target_device).clone())
            for name, p in parameters.items()
        }
        if offload
        else parameters
    )
    optimizer = torch.optim.AdamW(masters.values(), lr=lr, foreach=False if offload else None)
    initial = final = evaluate_student(model, trace, targets, dtype=dtype, chunk_size=chunk_size)
    print(f"initial: {initial['loss']:.8f}, accuracy={initial['accuracy']:.6f}", flush=True)
    history, losses = [{"step": 0, **initial}], []
    for step in range(1, steps + 1):
        if stop_on_initial and step == 1 and has_overfit(initial):
            break
        model.train()
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        losses.append(_run_step(model, trace, targets, dtype, chunk_size, backward=True))
        if offload:
            for name, parameter in parameters.items():
                if parameter.grad is not None:
                    masters[name].grad = parameter.grad.detach().to(target_device)
                    parameter.grad = None
        torch.nn.utils.clip_grad_norm_(masters.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if offload:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(masters[name])
        if step % 5 == 0 or step == steps:
            final = evaluate_student(model, trace, targets, dtype=dtype, chunk_size=chunk_size)
            history.append({"step": step, **final})
            print(f"step {step}: {final['loss']:.8f}, accuracy={final['accuracy']:.6f}", flush=True)
            if has_overfit(final):
                break
    rows = targets.ne(-100).nonzero().flatten()[:8]
    expected = _checkpoint_logits(model.eval(), trace, rows, dtype)
    model.cpu().save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)
    del optimizer, masters, parameters, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded, saved_trace, saved_targets, _ = load_student(
        source, str(checkpoint), device=device, dtype=master_dtype or dtype
    )
    actual = _checkpoint_logits(reloaded.eval(), saved_trace, rows, dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    del reloaded, saved_trace, saved_targets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "initial": initial,
        "final": final,
        "history": history,
        "losses": losses,
        "completed_steps": len(losses),
        "overfit_achieved": has_overfit(final),
        "checkpoint": str(checkpoint.resolve()),
        "reload_exact": True,
        "lr": lr,
        "seed": 7,
    }


def verify_checkpoint(
    source: TraceSource,
    checkpoint: str | Path,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    master_dtype: torch.dtype | None = None,
    chunk_size: int = 128,
) -> dict[str, Any]:
    model, trace, targets, _ = load_student(
        source, str(checkpoint), device=device, dtype=master_dtype or dtype
    )
    result = evaluate_student(model, trace, targets, dtype=dtype, chunk_size=chunk_size)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
