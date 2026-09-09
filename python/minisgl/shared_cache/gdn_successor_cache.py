"""Bounded last-decode states, accounting for complete retained allocations.

Rows keep logical block/layer identity, never scheduler batch indices. Outputs
of one recurrent call stay packed: replacing/evicting the last live row releases
the allocation. Retaining one row still charges the entire backing storage.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import weakref

import torch


@dataclass(eq=False)
class StateBatch:
    tensor: torch.Tensor
    keys: set = field(default_factory=set)

    @property
    def nbytes(self):
        return self.tensor.untyped_storage().nbytes()


@dataclass
class StateRow:
    batch: StateBatch
    row: int
    signature: tuple
    owner: weakref.ReferenceType | None

    def tensor(self):
        return self.batch.tensor[self.row : self.row + 1]


class GDNSuccessorCache:
    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("successor cache budget must be positive")
        self.max_bytes = int(max_bytes)
        self.resident_bytes = 0
        self._entries: dict[tuple[int, int], StateRow] = {}
        self._batches: OrderedDict[int, StateBatch] = OrderedDict()
        self.stats = dict(hits=0, misses=0, invalidations=0, stores=0,
                          evicted_batches=0, skipped_oversize=0)

    @property
    def resident_entries(self):
        return len(self._entries)

    def _discard(self, key):
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        batch = entry.batch
        batch.keys.remove(key)
        if not batch.keys:
            self._batches.pop(id(batch))
            self.resident_bytes -= batch.nbytes

    def discard_block(self, block):
        for key in [key for key in self._entries if key[0] == id(block)]:
            self._discard(key)

    def get(self, block, layer: int, signature: tuple):
        key = (id(block), layer)
        entry = self._entries.get(key)
        if entry is not None and (entry.owner() is not block or entry.signature != signature):
            self._discard(key)
            self.stats["invalidations"] += 1
            entry = None
        self.stats["hits" if entry is not None else "misses"] += 1
        if entry is not None:
            self._batches.move_to_end(id(entry.batch))
        return entry

    def put(self, layer: int, state: torch.Tensor, records):
        """records: (output row, write block, verified post-capture signature)."""
        if not records:
            return
        if state.dtype != torch.float32 or state.ndim != 4:
            raise ValueError("successor states must be FP32 [W,H,Dv,Dk]")
        # The no-FLA result can be a transposed view; normalize just once here.
        batch = StateBatch(state.detach().contiguous())
        if batch.nbytes > self.max_bytes:
            self.stats["skipped_oversize"] += 1
            return
        for _, block, _ in records:
            self._discard((id(block), layer))
        while self._batches and self.resident_bytes + batch.nbytes > self.max_bytes:
            old = next(iter(self._batches.values()))
            for key in list(old.keys):
                self._discard(key)
            self.stats["evicted_batches"] += 1
        self._batches[id(batch)] = batch
        self.resident_bytes += batch.nbytes
        cache_ref = weakref.ref(self)
        for row, block, signature in records:
            key = (id(block), layer)

            def expired(ref, key=key, cache_ref=cache_ref):
                cache = cache_ref()
                if cache is not None:
                    current = cache._entries.get(key)
                    if current is not None and current.owner is ref:
                        cache._discard(key)

            self._entries[key] = StateRow(batch, row, signature, weakref.ref(block, expired))
            batch.keys.add(key)
            self.stats["stores"] += 1


def assemble_state_rows(rows: list[StateRow], *, materialize: bool = True):
    """Use known row provenance, not a GEMM storage-layout dispatch."""
    first = rows[0]
    if all(r.batch is first.batch and r.row == first.row + i for i, r in enumerate(rows)):
        return first.batch.tensor[first.row : first.row + len(rows)]
    parts = [row.tensor() for row in rows]
    return torch.cat(parts, dim=0) if materialize else parts
