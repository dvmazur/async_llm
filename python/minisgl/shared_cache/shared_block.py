from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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
    A reusable, paged block of KV cache that can be shared across multiple
    workers.

    The block owns a list of **pages** borrowed from the engine's page
    allocator; each page holds ``page_size`` contiguous token slots.  Token
    ``i`` (block-relative, ``0 <= i < num_tokens``) lives at physical slot
    ``page_starts[i // page_size] + (i % page_size)``.  The last page may be
    partially filled.

    Keys are stored at block-relative RoPE positions (0..num_tokens-1) and
    never re-rotated; the query-rotation decode rotates queries instead (see
    ``shared_cache.attention``).
    """

    _next_id: int = 0

    def __init__(self, device: torch.device, page_size: int = 1):
        self.block_id = SharedBlock._next_id
        SharedBlock._next_id += 1
        self.device = device
        self.page_size = page_size
        # Page-start token slots (multiples of page_size), one per owned page.
        self.page_starts: List[int] = []
        self.num_tokens: int = 0

    @property
    def num_pages(self) -> int:
        return len(self.page_starts)

    @property
    def last_page_len(self) -> int:
        """Valid token count in the final page (1..page_size), 0 if empty."""
        if self.num_tokens == 0:
            return 0
        return self.num_tokens - (self.num_pages - 1) * self.page_size

    @property
    def has_capacity(self) -> bool:
        """Whether the current last page has room for one more token."""
        return self.num_tokens < self.num_pages * self.page_size

    def page_starts_tensor(self) -> torch.Tensor:
        """Page-start token slots as a device tensor ``[num_pages]``."""
        return torch.tensor(self.page_starts, dtype=torch.int32, device=self.device)

    def page_numbers_tensor(self) -> torch.Tensor:
        """Physical page numbers (= page_start // page_size) for paged kernels."""
        return self.page_starts_tensor() // self.page_size

    def token_slots_tensor(self) -> torch.Tensor:
        """Per-token physical slots ``[num_tokens]`` (flattened paged layout)."""
        if self.num_tokens == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        starts = self.page_starts_tensor()
        offsets = torch.arange(self.page_size, dtype=torch.int32, device=self.device)
        return (starts[:, None] + offsets[None, :]).flatten()[: self.num_tokens]

    def grow_pages(self, page_starts: torch.Tensor, num_new_tokens: int) -> None:
        """Record a prefill write: ``num_new_tokens`` tokens packed into the
        given freshly-allocated ``page_starts`` (``ceil(num_new_tokens/P)`` of
        them).  The block must be empty (prefill always writes a fresh block)."""
        assert self.num_tokens == 0, "grow_pages only supports prefilling a fresh block"
        self.page_starts.extend(int(p) for p in page_starts.tolist())
        self.num_tokens += num_new_tokens

    def append_token(self, new_page_start: Optional[int]) -> None:
        """Record a single decode write.  ``new_page_start`` is the page-start
        slot of a freshly-allocated page when this token starts a new page, else
        ``None`` (the token fit in the current last page)."""
        if new_page_start is not None:
            assert not self.has_capacity, "append_token got a new page but last page has room"
            self.page_starts.append(int(new_page_start))
        else:
            assert self.has_capacity, "append_token needs a new page but none was given"
        self.num_tokens += 1

    def clear(self) -> List[int]:
        """Reset the block and return the page-start slots the caller should free."""
        pages = list(self.page_starts)
        self.page_starts.clear()
        self.num_tokens = 0
        return pages

    def __repr__(self) -> str:
        return (
            f"SharedBlock(id={self.block_id}, tokens={self.num_tokens}, "
            f"pages={self.num_pages}, page_size={self.page_size})"
        )
