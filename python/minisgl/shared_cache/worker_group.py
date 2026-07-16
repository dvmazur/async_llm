from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Union, overload

from .async_context import AsyncContext
from .shared_block import CacheBlock


class WorkerGroup:
    """
    A group of concurrent workers sharing cache blocks — the unit of one
    batched decode step.

    Each worker is an :class:`AsyncContext`: an ordered ``cache_view`` of
    ``CacheBlock`` refs forming its KV cache, plus the ``output_block`` its new
    token is written to.  The same ``CacheBlock`` can appear in multiple
    workers' views at different positions.

    Example::

        prompt, w1, w2 = CacheBlock(dev), CacheBlock(dev), CacheBlock(dev)
        group = WorkerGroup([
            AsyncContext(cache_view=[prompt, w2, w1]),  # writes w1, sees w2
            AsyncContext(cache_view=[prompt, w1, w2]),  # writes w2, sees w1
        ])

    The legacy keyword form ``WorkerGroup(cache_structure=..., write_to=...)``
    builds the contexts for you (``write_to`` defaults to each view's last
    block).
    """

    def __init__(
        self,
        workers: Optional[Sequence[AsyncContext]] = None,
        *,
        cache_structure: Optional[Sequence[Sequence[CacheBlock]]] = None,
        write_to: Optional[Sequence[CacheBlock]] = None,
    ):
        if workers is not None:
            assert cache_structure is None and write_to is None, (
                "pass either workers or cache_structure/write_to, not both"
            )
            self.workers: List[AsyncContext] = list(workers)
        else:
            assert cache_structure is not None, "WorkerGroup needs workers or cache_structure"
            views = [list(view) for view in cache_structure]
            outs = list(write_to) if write_to is not None else [view[-1] for view in views]
            assert len(outs) == len(views)
            self.workers = [
                AsyncContext(cache_view=view, output_block=out) for view, out in zip(views, outs)
            ]

        seen_ids = set()
        for ctx in self.workers:
            if id(ctx.output_block) in seen_ids:
                raise ValueError("WorkerGroup has two workers writing the same block in one step")
            seen_ids.add(id(ctx.output_block))

    @overload
    def __getitem__(self, key: int) -> AsyncContext: ...
    @overload
    def __getitem__(self, key: slice) -> "WorkerGroup": ...

    def __getitem__(self, key: Union[int, slice]) -> "AsyncContext | WorkerGroup":
        """Index a specific worker or slice a couple of them."""
        if isinstance(key, slice):
            return WorkerGroup(self.workers[key])
        return self.workers[key]

    def __len__(self) -> int:
        return len(self.workers)

    def __iter__(self) -> Iterator[AsyncContext]:
        return iter(self.workers)

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    @property
    def cache_structure(self) -> List[List[CacheBlock]]:
        return [list(ctx.cache_view) for ctx in self.workers]

    @property
    def write_to(self) -> List[CacheBlock]:
        return [ctx.output_block for ctx in self.workers]

    def worker_cache_length(self, worker_idx: int) -> int:
        """Total cached tokens for a given worker (sum over its blocks)."""
        return self.workers[worker_idx].num_cached_tokens

    def max_cache_length(self) -> int:
        return max((ctx.num_cached_tokens for ctx in self.workers), default=0)

    def __repr__(self) -> str:
        num_blocks = len({id(b) for ctx in self.workers for b in ctx.cache_view})
        return f"WorkerGroup(workers={self.num_workers}, blocks={num_blocks})"
