"""Chunked supervised loss and evaluation shared by the training demos."""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from minisgl.training import Trace, replay_trace
from torch.utils.checkpoint import checkpoint


def _supervised(
    model: torch.nn.Module, plan: Trace, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    rows = (targets != -100).nonzero().flatten()
    if len(targets) != plan.num_tokens or not len(rows):
        raise ValueError("Each trace needs aligned, nonempty supervision")
    states = replay_trace(model, plan, rows=rows)
    return states, targets[rows].to(device)


def compute_supervised_loss(
    model: torch.nn.Module,
    replay: Trace,
    targets: torch.Tensor,
    chunk_size: int = 128,
    *,
    checkpointing: bool = True,
) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("Loss chunk size must be positive")
    states, selected = _supervised(model, replay, targets)

    def chunk_loss(x, y):
        return F.cross_entropy(model.lm_head(x).float(), y, reduction="sum")

    losses = []
    for x, y in zip(states.split(chunk_size), selected.split(chunk_size), strict=True):
        losses.append(
            checkpoint(chunk_loss, x, y, use_reentrant=False, preserve_rng_state=False)
            if checkpointing and torch.is_grad_enabled()
            else chunk_loss(x, y)
        )
    return torch.stack(losses).sum()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, examples: Sequence[tuple[Trace, torch.Tensor]], chunk_size: int = 128
) -> dict[str, Any]:
    if chunk_size < 1:
        raise ValueError("Loss chunk size must be positive")
    results = []
    for plan, targets in examples:
        states, selected = _supervised(model, plan, targets)
        nll, correct = [], 0
        for x, y in zip(states.split(chunk_size), selected.split(chunk_size), strict=True):
            logits = model.lm_head(x).float()
            nll.append(F.cross_entropy(logits, y, reduction="none"))
            correct = correct + (logits.argmax(-1) == y).sum()
            del logits
        nll = torch.cat(nll)
        results.append(
            {
                "tokens": len(selected),
                "loss": nll.mean().item(),
                "accuracy": correct.item() / len(selected),
                "nll": nll.cpu().tolist(),
            }
        )
    count = sum(r["tokens"] for r in results)
    return {
        "loss": sum(r["loss"] * r["tokens"] for r in results) / count,
        "accuracy": sum(r["accuracy"] * r["tokens"] for r in results) / count,
        "examples": results,
    }
