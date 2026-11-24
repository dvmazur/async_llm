from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float, *, weight_plus_one: bool = False) -> None:
        from flashinfer import gemma_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        # Same normalization operation, but Qwen's +1 must happen in FP32,
        # not be rounded into the BF16 checkpoint weight during loading.
        self.rmsnorm = gemma_rmsnorm if weight_plus_one else rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float, *, weight_plus_one: bool = False) -> None:
        from flashinfer import fused_add_rmsnorm, gemma_fused_add_rmsnorm, gemma_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = gemma_rmsnorm if weight_plus_one else rmsnorm
        self.fused_add_rmsnorm = gemma_fused_add_rmsnorm if weight_plus_one else fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
