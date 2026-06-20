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

    Tracks the physical page locations in the KV pool.  Keys are stored at
    block-relative RoPE positions (0..len-1) and never re-rotated; the
    query-rotation decode rotates queries instead (see
    ``shared_cache.attention``).
    """

    _next_id: int = 0

    def __init__(self, device: torch.device):
        self.block_id = SharedBlock._next_id
        SharedBlock._next_id += 1
        self.device = device
        self.page_indices: List[int] = []

    @property
    def num_tokens(self) -> int:
        return len(self.page_indices)

    def grow(self, new_pages: torch.Tensor) -> None:
        """
        Record that new tokens were written to this block.

        Args:
            new_pages: physical page indices where the new KV states were stored.
        """
        self.page_indices.extend(new_pages.tolist())

    def get_page_indices(self) -> torch.Tensor:
        """Return physical page indices as a device tensor."""
        return torch.tensor(self.page_indices, dtype=torch.int32, device=self.device)

    def clear(self) -> List[int]:
        """Reset the block and return page indices that the caller should free."""
        pages = list(self.page_indices)
        self.page_indices.clear()
        return pages

    def __repr__(self) -> str:
        return f"SharedBlock(id={self.block_id}, tokens={self.num_tokens})"
