from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .shared_block import CacheBlock

# A worker's ordered view of the cache: the blocks it attends to, concatenated.
CacheView = List[CacheBlock]


@dataclass
class AsyncContext:
    """
    One agent's decoding context: the ``CacheView`` it reads and the block its
    new tokens are written to.

    ``output_block`` usually is the last block of ``cache_view`` (the agent
    appends to what it sees); a write block *outside* the view is also allowed
    and takes the attention's aux path (the worker still sees its own
    current-step token).
    """

    cache_view: CacheView = field(default_factory=list)
    output_block: CacheBlock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.output_block is None:
            assert self.cache_view, "AsyncContext needs a cache_view or an explicit output_block"
            self.output_block = self.cache_view[-1]

    @property
    def num_cached_tokens(self) -> int:
        """Total tokens this context attends to (sum over its view)."""
        return sum(b.num_tokens for b in self.cache_view)
