from __future__ import annotations

import torch
import torch.nn.functional as F

from minisgl.utils import is_turing


def _portable_gate_and_mul(
    x: torch.Tensor,
    activation,
    out: torch.Tensor | None,
) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    result = (activation(gate.float()) * up.float()).to(dtype=x.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if x.is_cuda and is_turing(x.device):
        return _portable_gate_and_mul(x, F.silu, out)
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if x.is_cuda and is_turing(x.device):
        return _portable_gate_and_mul(x, lambda value: F.gelu(value), out)

    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
