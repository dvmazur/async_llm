from __future__ import annotations

from collections import deque
from itertools import islice
from typing import Sequence

import torch


class PageAllocator:
    """Page-aligned free-list allocator over a KV-cache pool's pages.

    Hands out and reclaims *pages*.  Each page is identified by its first
    token-slot index (a multiple of ``page_size``), matching the page-aligned
    layout the scheduler's ``CacheManager`` uses (``arange(num_pages) *
    page_size``).

    This is the engine's "main page cache": the single pool that standalone
    consumers such as ``SharedCacheSession`` borrow pages from (and return them
    to) instead of fabricating their own private token pool.  Unlike
    ``CacheManager`` it carries no prefix cache / eviction -- borrowers manage
    their own blocks explicitly and hand pages back on teardown.
    """

    def __init__(self, num_pages: int, page_size: int, device: torch.device) -> None:
        self.num_pages = num_pages
        self.page_size = page_size
        self.device = device
        # Page-aligned free list of page-start slots: [0, P, 2P, ...].
        self._free = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self._host_free: deque[int] | None = None

    def enable_host_metadata(self) -> None:
        """Move free-page bookkeeping to CPU once, before shared-cache work.

        Page IDs are allocation metadata, not KV values. The ordinary scheduler
        keeps its original device free list unless a shared session opts in.
        Already borrowed pages stay borrowed; allocation/reclaim order is FIFO
        in both modes. The legacy tensor API remains usable in host mode too.
        """
        if self._host_free is None:
            self._host_free = deque(self._free.cpu().tolist())
            self._free = None

    def alloc_pages_cpu(self, num_pages: int) -> torch.Tensor:
        """Borrow pages as an owned CPU int32 tensor, with no hot-path D2H.

        CUDA sessions get immutable pinned metadata for asynchronous H2D use.
        Call enable_host_metadata during session setup, not on its first token.
        """
        self.enable_host_metadata()
        if num_pages < 0 or num_pages > self.num_free_pages:
            raise RuntimeError(
                f"PageAllocator out of pages: requested {num_pages}, "
                f"only {self.num_free_pages} free"
            )
        pages = list(islice(self._host_free, num_pages))
        result = torch.tensor(pages, dtype=torch.int32, device='cpu',
                              pin_memory=torch.device(self.device).type == 'cuda')
        # Do not consume pages if constructing/pinning the metadata failed.
        for _ in range(num_pages):
            self._host_free.popleft()
        return result

    def free_pages_cpu(self, page_starts: Sequence[int] | torch.Tensor) -> None:
        """Return CPU-known pages without constructing a device tensor."""
        self.enable_host_metadata()
        if isinstance(page_starts, torch.Tensor):
            if page_starts.device.type != 'cpu':
                raise ValueError('free_pages_cpu requires CPU page metadata')
            page_starts = page_starts.tolist()
        self._host_free.extend(int(p) for p in page_starts)

    @property
    def num_free_pages(self) -> int:
        if self._host_free is not None:
            return len(self._host_free)
        return int(self._free.numel())

    @property
    def free_page_starts(self) -> torch.Tensor:
        """The current free list (page-start slots).  Read-only view for
        integrity checks; mutate via ``alloc_pages`` / ``free_pages``."""
        if self._host_free is None:
            return self._free
        # Diagnostic snapshot only; the allocator's authoritative state is CPU.
        return torch.tensor(list(self._host_free), dtype=torch.int32, device=self.device)

    def pages_to_tokens(self, page_starts: torch.Tensor) -> torch.Tensor:
        """Expand page-start slots ``[N]`` to per-token slots ``[N * page_size]``:
        ``[s, ...] -> [s, s+1, ..., s+page_size-1, ...]``."""
        if self.page_size == 1:
            return page_starts
        offsets = torch.arange(self.page_size, device=page_starts.device, dtype=page_starts.dtype)
        return (page_starts[:, None] + offsets[None, :]).flatten()

    def alloc_pages(self, num_pages: int) -> torch.Tensor:
        """Borrow ``num_pages`` pages; returns their page-start slots ``[num_pages]``."""
        if self._host_free is not None:
            return self.alloc_pages_cpu(num_pages).to(self.device, non_blocking=True)
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
        if self._host_free is not None:
            # Compatibility for device-only clients (e.g. prefix-cache eviction).
            # SharedCacheSession uses free_pages_cpu and never takes this D2H.
            self.free_pages_cpu(page_starts.to(device='cpu', dtype=torch.int32))
            return
        page_starts = page_starts.to(device=self._free.device, dtype=self._free.dtype)
        self._free = torch.cat([self._free, page_starts])
