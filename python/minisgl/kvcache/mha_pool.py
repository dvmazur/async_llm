from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_shapes: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        if layer_shapes is None:
            self._kv_buffer = torch.empty(
                (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
                device=device, dtype=dtype,
            )
            self._k_buffer, self._v_buffer = self._kv_buffer[0], self._kv_buffer[1]
        else:
            assert len(layer_shapes) == num_layers
            self._kv_buffer = tuple(torch.empty(
                (2, num_pages, page_size, div_even(h, tp_info.size, allow_replicate=True), d),
                device=device, dtype=dtype) for h, d in layer_shapes)
            self._k_buffer = tuple(t[0] for t in self._kv_buffer)
            self._v_buffer = tuple(t[1] for t in self._kv_buffer)
        self._num_layers = num_layers
        self._device = device
        self._dtype = dtype

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].flatten(0, 1),
            v_cache=self._v_buffer[layer_id].flatten(0, 1),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
