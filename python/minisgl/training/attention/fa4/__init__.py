"""Optional Hopper FA4 backend; CUDA dependencies are loaded on use."""

from minisgl.training.attention.fa4.backend import flash_attention

__all__ = ["flash_attention"]
