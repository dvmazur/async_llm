from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


class GemmaRMSNorm(RMSNorm):
    """Qwen3.5/Gemma norm: keep raw weights; add one inside FP32 arithmetic.

    Folding ``1 + weight`` into a BF16 checkpoint tensor loses information
    before inference. FlashInfer's Gemma kernels implement the required bias
    without that extra rounding or an additional GPU operation.
    """

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import gemma_rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = gemma_rmsnorm


class GemmaRMSNormFused(RMSNormFused):
    """Gemma normalization with the existing in-place residual contract."""

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import gemma_fused_add_rmsnorm, gemma_rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = gemma_rmsnorm
        self.fused_add_rmsnorm = gemma_fused_add_rmsnorm
