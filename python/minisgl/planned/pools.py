"""Runtime-owned all-layer storage. No per-block slabs or successor pool."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PoolShape:
    layers: int
    slots: int
    heads: int
    dim: int
    conv_channels: int
    conv_window: int

    def __post_init__(self):
        if any(type(getattr(self, f)) is not int or getattr(self, f) <= 0 for f in self.__dataclass_fields__):
            raise ValueError("pool dimensions must be positive integers")


class GDNPool:
    def __init__(self, shape: PoolShape, *, device, activation_dtype=torch.bfloat16):
        if activation_dtype not in (torch.bfloat16, torch.float32, torch.float16):
            raise ValueError("unsupported conv dtype")
        self.shape = shape
        self.affine = torch.empty(shape.layers, shape.slots, 2, shape.heads, shape.dim, shape.dim,
                                  dtype=torch.float32, device=device)
        self.conv = torch.empty(shape.layers, shape.slots, shape.conv_channels, shape.conv_window,
                                dtype=activation_dtype, device=device)

    @property
    def reserved_bytes(self):
        return self.affine.numel()*self.affine.element_size() + self.conv.numel()*self.conv.element_size()

    @property
    def bytes_per_slot(self):
        return self.reserved_bytes // self.shape.slots

    def internal_views(self, layer, slot):
        """Internal borrowed views: caller must own/pin slot until last use."""
        if not 0 <= layer < self.shape.layers or not 0 <= slot < self.shape.slots:
            raise IndexError("layer or slot outside pool")
        return self.affine[layer, slot, 0], self.affine[layer, slot, 1], self.conv[layer, slot]
