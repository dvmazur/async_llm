from __future__ import annotations

from typing import List, Optional, Sequence

from .shared_block import SharedBlock


class WorkerGroup:
    """
    A group of concurrent workers sharing cache blocks.

    Mirrors AsyncReasoning's ``SharedCacheManager``: each worker has an ordered
    sequence of ``SharedBlock`` refs that form its KV cache.  The same
    ``SharedBlock`` can appear in multiple workers' sequences at different
    positions.

    Example::

        prompt, w1, w2 = SharedBlock(dev), SharedBlock(dev), SharedBlock(dev)
        group = WorkerGroup(
            cache_structure=[
                [prompt, w2, w1],   # worker 0 sees prompt → w2 → itself
                [prompt, w1, w2],   # worker 1 sees prompt → w1 → itself
            ],
            write_to=[w1, w2],
        )
    """

    def __init__(
        self,
        cache_structure: Sequence[Sequence[SharedBlock]],
        write_to: Optional[Sequence[SharedBlock]] = None,
    ):
        self.cache_structure: List[List[SharedBlock]] = [list(s) for s in cache_structure]
        self.write_to: List[SharedBlock] = (
            list(write_to) if write_to is not None else [seq[-1] for seq in self.cache_structure]
        )
        assert len(self.write_to) == self.num_workers

    @property
    def num_workers(self) -> int:
        return len(self.cache_structure)

    def worker_cache_length(self, worker_idx: int) -> int:
        """Total cached tokens for a given worker (sum over its blocks)."""
        return sum(b.num_tokens for b in self.cache_structure[worker_idx])

    def max_cache_length(self) -> int:
        return max(
            (self.worker_cache_length(i) for i in range(self.num_workers)),
            default=0,
        )

    def __repr__(self) -> str:
        return (
            f"WorkerGroup(workers={self.num_workers}, "
            f"blocks={set(id(b) for seq in self.cache_structure for b in seq).__len__()})"
        )
