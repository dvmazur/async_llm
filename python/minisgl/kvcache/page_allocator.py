from __future__ import annotations

import torch


class PageAllocator:
    """Page-aligned free-list allocator over a KV-cache pool's pages.

    Hands out and reclaims *pages*.  Each page is identified by its first
    token-slot index (a multiple of ``page_size``), matching the page-aligned
    layout the scheduler's ``CacheManager`` uses (``arange(num_pages) *
    page_size``).

    This is the engine's "main page cache": the single pool every consumer
    draws from.  The scheduler's ``CacheManager`` layers the prefix cache,
    eviction, and the shared-cache borrowed-page accounting on top; this
    allocator itself only hands out and reclaims page-aligned slots.
    """

    def __init__(self, num_pages: int, page_size: int, device: torch.device) -> None:
        self.num_pages = num_pages
        self.page_size = page_size
        self.device = device
        # Page-aligned free list of page-start slots: [0, P, 2P, ...].
        self._free = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size

    @property
    def num_free_pages(self) -> int:
        return int(self._free.numel())

    @property
    def free_page_starts(self) -> torch.Tensor:
        """The current free list (page-start slots).  Read-only view for
        integrity checks; mutate via ``alloc_pages`` / ``free_pages``."""
        return self._free

    def pages_to_tokens(self, page_starts: torch.Tensor) -> torch.Tensor:
        """Expand page-start slots ``[N]`` to per-token slots ``[N * page_size]``:
        ``[s, ...] -> [s, s+1, ..., s+page_size-1, ...]``."""
        if self.page_size == 1:
            return page_starts
        offsets = torch.arange(self.page_size, device=page_starts.device, dtype=page_starts.dtype)
        return (page_starts[:, None] + offsets[None, :]).flatten()

    def alloc_pages(self, num_pages: int) -> torch.Tensor:
        """Borrow ``num_pages`` pages; returns their page-start slots ``[num_pages]``."""
        if num_pages > self.num_free_pages:
            raise RuntimeError(
                f"PageAllocator out of pages: requested {num_pages}, "
                f"only {self.num_free_pages} free"
            )
        allocated = self._free[:num_pages].clone()
        self._free = self._free[num_pages:]
        return allocated

    def free_pages(self, page_starts: torch.Tensor) -> None:
        """Return previously-borrowed pages (given by their page-start slots)."""
        if page_starts.numel() == 0:
            return
        page_starts = page_starts.to(device=self._free.device, dtype=self._free.dtype)
        self._free = torch.cat([self._free, page_starts])
