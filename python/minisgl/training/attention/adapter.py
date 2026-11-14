"""Hugging Face registration and replay backend selection."""

from collections.abc import Callable, Sequence
from copy import copy
from functools import lru_cache
from typing import Any

import torch
from transformers import AttentionInterface

from minisgl.training.attention.torch import attention
from minisgl.training.attention.torch_optimized import attention as optimized_attention
from minisgl.training.trace import Trace


@lru_cache(maxsize=1)
def _load_optional_flash_attention() -> Callable[..., torch.Tensor] | None:
    """Probe optional dependencies once; execution errors must still propagate."""
    from minisgl.training.attention.fa4.backend import _flash_attention_function, flash_attention

    try:
        _flash_attention_function()
    except ImportError:
        return None
    return flash_attention


def _select_attention_backend(
    backend: str, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> Callable[..., torch.Tensor]:
    """Honor explicit selections; auto chooses from the supported CUDA paths."""
    if backend == "torch":
        return attention
    if backend == "torch_optimized":
        return optimized_attention
    if backend == "fa4":
        from minisgl.training.attention.fa4 import flash_attention

        return flash_attention
    if backend != "auto":
        raise ValueError(f"Unknown replay attention backend: {backend}")
    if query.device.type != "cuda":
        return attention
    # FA4 requires matching low-precision inputs; Torch also handles mixed dtypes.
    head_size = query.shape[-1]
    if (
        query.dtype in (torch.float16, torch.bfloat16)
        and key.dtype == query.dtype == value.dtype
        and 8 <= head_size <= 256
        and head_size % 8 == 0
        and torch.cuda.get_device_capability(query.device)[0] == 9
    ):
        flash_backend = _load_optional_flash_attention()
        if flash_backend is not None:
            return flash_backend
    return optimized_attention


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
    attention_backend = _select_attention_backend(replay.attention_backend, query, key, value)
    output = attention_backend(
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
