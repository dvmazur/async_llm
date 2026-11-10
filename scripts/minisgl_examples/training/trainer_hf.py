"""Hugging Face Trainer adapters for asynchronous execution traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from minisgl.training import Trace
from torch.utils.data import Dataset
from transformers import Trainer

from minisgl_examples.training.loss import compute_supervised_loss


class TraceDataset(Dataset):
    """Validated traces and row-aligned targets retained as Python objects."""

    def __init__(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        trace_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        options = dict(trace_options or {})
        for example in examples:
            value = example["trace"]
            trace = value if isinstance(value, Trace) else Trace.from_dict(value)
            targets = torch.as_tensor(example["targets"], dtype=torch.long)
            if targets.ndim != 1 or len(targets) != trace.num_tokens:
                raise ValueError("Each trace needs row-aligned one-dimensional targets")
            if not targets.ne(-100).any():
                raise ValueError("Each trace needs nonempty supervision")
            self.examples.append(
                {
                    "trace": trace.to("cpu", **options),
                    "labels": targets.clone(),
                }
            )
        if not self.examples:
            raise ValueError("Training requires at least one trace")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]

    @property
    def supervised_tokens(self) -> int:
        return sum(int(example["labels"].ne(-100).sum()) for example in self.examples)

    @property
    def replay_examples(self) -> list[tuple[Trace, torch.Tensor]]:
        return [(example["trace"], example["labels"]) for example in self.examples]


class TraceDataCollator:
    """Pad row targets while leaving variable execution graphs unstacked."""

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty trace batch")
        traces = [example["trace"] for example in examples]
        width = max(trace.num_tokens for trace in traces)
        labels = torch.full((len(examples), width), -100, dtype=torch.long)
        for row, (trace, example) in enumerate(zip(traces, examples, strict=True)):
            targets = torch.as_tensor(example["labels"], dtype=torch.long)
            if targets.ndim != 1 or len(targets) != trace.num_tokens:
                raise ValueError("Trace targets changed after dataset validation")
            labels[row, : len(targets)] = targets
        return {
            "traces": traces,
            "labels": labels,
            # Trace targets already point at the logits row that produced them.
            # Trainer must count them directly instead of applying the causal-LM shift.
            "shift_labels": labels,
        }


class TraceSFTTrainer(Trainer):
    """Trainer whose forward pass is differentiable replay of recorded traces."""

    def __init__(
        self,
        *args: Any,
        loss_chunk_size: int = 128,
        loss_checkpointing: bool = True,
        **kwargs: Any,
    ) -> None:
        if loss_chunk_size < 1:
            raise ValueError("Loss chunk size must be positive")
        super().__init__(*args, **kwargs)
        if self.args.world_size != 1:
            raise NotImplementedError("TraceSFTTrainer currently supports one process/GPU")
        self.loss_chunk_size = loss_chunk_size
        self.loss_checkpointing = loss_checkpointing
        # The loss consumes num_items_in_batch itself. This also asks Trainer to
        # count shift_labels over the complete gradient-accumulation window.
        self.model_accepts_loss_kwargs = True

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        traces = inputs["traces"]
        labels = inputs["labels"]
        if labels.ndim != 2 or len(traces) != len(labels):
            raise ValueError("Expected one padded target row per trace")
        device = next(model.parameters()).device
        loss_sum = torch.zeros((), device=device)
        for trace, targets in zip(traces, labels, strict=True):
            targets = targets[: trace.num_tokens]
            loss_sum = loss_sum + compute_supervised_loss(
                model,
                trace.to(device),
                targets,
                self.loss_chunk_size,
                checkpointing=self.loss_checkpointing,
            )
        denominator = labels.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
        denominator = torch.as_tensor(denominator, device=device)
        if denominator.item() <= 0:
            raise ValueError("Trace batch contains no supervised targets")
        loss = loss_sum / denominator
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        outputs = {"loss": loss}
        return (loss, outputs) if return_outputs else loss


__all__ = ["TraceDataCollator", "TraceDataset", "TraceSFTTrainer"]
