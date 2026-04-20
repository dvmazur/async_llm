from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from minisgl.kvcache import BaseCacheHandle


@dataclass(frozen=True)
class _NullCacheHandle(BaseCacheHandle):
    """Placeholder handle for shared-cache requests that bypass the prefix cache."""

    def get_matched_indices(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.int32)


NULL_CACHE_HANDLE = _NullCacheHandle(cached_len=0)


class SharedBlock:
    """
    A reusable block of KV cache pages that can be shared across multiple workers.

    Tracks the physical page locations in the KV pool and the RoPE positions
    that keys were stored with, so that RoPE corrections can be applied when
    the block appears at a different logical position in another worker's view.
    """

    _next_id: int = 0

    def __init__(self, device: torch.device):
        self.block_id = SharedBlock._next_id
        SharedBlock._next_id += 1
        self.device = device
        self.page_indices: List[int] = []
        self.stored_positions: List[int] = []

    @property
    def num_tokens(self) -> int:
        return len(self.page_indices)

    def grow(self, new_pages: torch.Tensor, stored_positions: torch.Tensor) -> None:
        """
        Record that new tokens were written to this block.

        Args:
            new_pages: physical page indices where the new KV states were stored.
            stored_positions: RoPE positions the keys were stored with.
        """
        assert len(new_pages) == len(stored_positions)
        self.page_indices.extend(new_pages.tolist())
        self.stored_positions.extend(stored_positions.tolist())

    def get_page_indices(self) -> torch.Tensor:
        """Return physical page indices as a device tensor."""
        return torch.tensor(self.page_indices, dtype=torch.int32, device=self.device)

    def compute_corrections(self, target_start: int) -> torch.Tensor:
        """
        Compute per-token RoPE correction offsets to place this block starting
        at ``target_start``.

        Target positions are ``[target_start, target_start + 1, ..., target_start + N - 1]``.
        ``corrections[i] = target_positions[i] - stored_positions[i]``.
        """
        n = self.num_tokens
        target = torch.arange(target_start, target_start + n, dtype=torch.int64)
        stored = torch.tensor(self.stored_positions, dtype=torch.int64)
        return target - stored

    def needs_correction(self, target_start: int) -> bool:
        """True if any token's stored RoPE position differs from its target."""
        for i, sp in enumerate(self.stored_positions):
            if target_start + i != sp:
                return True
        return False

    def clear(self) -> List[int]:
        """Reset the block and return page indices that the caller should free."""
        pages = list(self.page_indices)
        self.page_indices.clear()
        self.stored_positions.clear()
        return pages

    def __repr__(self) -> str:
        return f"SharedBlock(id={self.block_id}, tokens={self.num_tokens})"
