"""Bounded second-touch LRU for composed GDN prefix states."""

from __future__ import annotations

from collections import OrderedDict
from typing import Hashable, Optional

import torch


class GDNComposeStateCache:
    """GPU-state cache with second-touch admission and byte-bounded LRU eviction.

    The first observation enters a bounded ghost LRU.  A second observation
    admits a detached copy into the resident cache.  This filters the many
    one-shot mutable prefixes produced by async branching.
    """

    def __init__(self, max_bytes: int, ghost_multiplier: int = 4):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self.ghost_multiplier = int(ghost_multiplier)
        self._resident: OrderedDict[Hashable, torch.Tensor] = OrderedDict()
        self._ghost: OrderedDict[Hashable, None] = OrderedDict()
        self._resident_bytes = 0
        self.stats = {
            "lookups": 0,
            "hits": 0,
            "ghost_first_touches": 0,
            "admissions": 0,
            "evictions": 0,
            "skipped_write_prefixes": 0,
        }

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    @property
    def resident_entries(self) -> int:
        return len(self._resident)

    def get(self, key: Hashable) -> Optional[torch.Tensor]:
        self.stats["lookups"] += 1
        value = self._resident.get(key)
        if value is not None:
            self.stats["hits"] += 1
            self._resident.move_to_end(key)
        return value

    def _trim_ghost(self, state_bytes: int) -> None:
        resident_capacity = max(1, self.max_bytes // state_bytes)
        ghost_capacity = self.ghost_multiplier * resident_capacity
        while len(self._ghost) > ghost_capacity:
            self._ghost.popitem(last=False)

    def consider(
        self,
        key: Hashable,
        state: torch.Tensor,
        *,
        current_write_prefix: bool,
    ) -> None:
        if current_write_prefix:
            self.stats["skipped_write_prefixes"] += 1
            return
        if key in self._resident:
            self._resident.move_to_end(key)
            return

        state_bytes = state.numel() * state.element_size()
        if state_bytes > self.max_bytes:
            return
        if key not in self._ghost:
            self._ghost[key] = None
            self.stats["ghost_first_touches"] += 1
            self._trim_ghost(state_bytes)
            return

        del self._ghost[key]
        while self._resident and self._resident_bytes + state_bytes > self.max_bytes:
            _, evicted = self._resident.popitem(last=False)
            self._resident_bytes -= evicted.numel() * evicted.element_size()
            self.stats["evictions"] += 1
        cached = state.detach().clone()
        self._resident[key] = cached
        self._resident_bytes += state_bytes
        self.stats["admissions"] += 1


__all__ = ["GDNComposeStateCache"]
