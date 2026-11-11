"""Hugging Face registration and replay backend selection."""

from collections.abc import Sequence
from copy import copy
from typing import Any

import torch
from transformers import AttentionInterface

from minisgl.training.attention.torch import attention
from minisgl.training.attention.torch_optimized import attention as optimized_attention
from minisgl.training.trace import Trace


def replay_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    replay: Trace | None = None,
    inv_freq: torch.Tensor | None = None,
    mrope_section: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    if query.shape[0] != 1:
        raise ValueError("Replay packs all tokens into one sequence")
    implementation = attention
    if replay.attention_backend == "torch_optimized":
        implementation = optimized_attention
    elif replay.attention_backend == "fa4":
        from minisgl.training.attention.fa4 import flash_attention

        implementation = flash_attention
    output = implementation(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1),
        value[0].transpose(0, 1),
        replay,
        inv_freq,
        scaling,
        mrope_section,
    )
    return output.view(1, query.shape[2], query.shape[1], query.shape[3]), None


AttentionInterface.register("minisgl_replay", replay_attention)


def replay_self_attention(
    module: torch.nn.Module, *args: Any, replay: Trace | None = None, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if replay is None:
        return module._replay_native_forward(*args, **kwargs)
    local = copy(module)
    local.config = copy(module.config)
    local.config._attn_implementation = "minisgl_replay"
    return module._replay_native_forward.__func__(local, *args, replay=replay, **kwargs)
