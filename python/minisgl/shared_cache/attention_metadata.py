"""CPU page-number snapshots and one upload per Attention index table."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .shared_block import CacheBlock


def page_numbers(
    block: CacheBlock, snapshot: dict[int, list[int]], extra_page: int | None = None,
) -> list[int]:
    """Read each distinct block once within ONE prepare, including pending writes.

    The caller owns the snapshot; it must not survive into the next prepare.
    No block mutation/invalidation contract or persistent tensor cache is added.
    """
    key = id(block)
    if key not in snapshot:
        pages = [start // block.page_size for start in block.page_starts]
        if extra_page is not None:
            pages.append(extra_page // block.page_size)
        snapshot[key] = pages
    return snapshot[key]


def upload_page_indices(parts, device: torch.device, refs: list[torch.Tensor]) -> torch.Tensor:
    """Flatten integer lists on CPU, not thousands of tiny CUDA tensors.

    Pass a device tensor to the public FlashInfer plan: its graph wrapper's
    CPU-index copy otherwise forces blocking=True even for pinned inputs.
    Keep both allocations alive through the caller's existing plan event.
    """
    host = torch.tensor([page for part in parts for page in part], dtype=torch.int32,
                        device='cpu', pin_memory=device.type == 'cuda')
    indices = host.to(device, non_blocking=True)
    refs.extend((host, indices))
    return indices
