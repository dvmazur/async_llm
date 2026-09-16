"""CPU address descriptions; no compose memoization or GPU state pool."""

from dataclasses import dataclass

import numpy as np
import torch


class StateCache:
    def __init__(self):
        self.prepared = None


class StateDict(dict):
    """Keep the legacy dict API, invalidating metadata before any mutation.

    The shared cache has no reference back to a block/dict, avoiding a cycle.
    In-place tensor VALUE edits keep valid addresses. Debug code changing a
    tensor's storage/layout must call block.invalidate_gdn_state() afterwards.
    """

    def __init__(self, cache, values=()):
        super().__init__(values)
        self.cache = cache

    def __setitem__(self, key, value):
        self.cache.prepared = None
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self.cache.prepared = None
        super().__delitem__(key)

    def clear(self):
        self.cache.prepared = None
        super().clear()

    def pop(self, key, *default):
        self.cache.prepared = None
        return super().pop(key, *default)

    def popitem(self):
        self.cache.prepared = None
        return super().popitem()

    def update(self, *args, **kwargs):
        # update may fail after partially consuming an iterator.
        self.cache.prepared = None
        super().update(*args, **kwargs)

    def setdefault(self, key, default=None):
        if key not in self:
            self.cache.prepared = None
        return super().setdefault(key, default)

    def __ior__(self, other):
        self.update(other)
        return self


@dataclass(frozen=True)
class StateAddresses:
    signature: tuple
    pointers: np.ndarray  # [layers, A/B/conv], int64, CPU only
    owners: tuple  # keep raw-pointer storage alive, not the mutable block
    conv_present: np.ndarray  # key presence; a debug None still shadows earlier blocks

    @classmethod
    def from_slabs(cls, signature, slabs):
        layout = np.array([(t.data_ptr(), t.stride(0)*t.element_size()) for t in slabs],
                          dtype=np.int64)
        pointers = layout[:, 0] + np.arange(signature[1])[:, None]*layout[:, 1]
        return cls(signature, pointers, tuple(slabs), np.ones(signature[1], dtype=bool))


def owns_state_dicts(block):
    return all(isinstance(d, StateDict) and d.cache is block._gdn_state_cache
               for d in (block.linear_affine, block.linear_conv_state))


def prepare_state(block, signature):
    """Inspect tensors once per revision, not once per reader/layer/forward.

    Plain externally assigned dicts remain aliased and uncached. Converted
    debug tensors are also uncached: an in-place edit of their original must
    be reflected by a fresh conversion on the next forward.
    """
    cacheable = owns_state_dicts(block)
    cached = block._gdn_state_cache.prepared if cacheable else None
    if cached is not None and cached.signature == signature:
        return cached
    device, layers, h, dk, dv, conv_shape, dtype = signature
    shapes = ((1, h, dk, dk), (1, h, dv, dk), conv_shape)
    pointers, owners = np.zeros((layers, 3), dtype=np.int64), []
    for layer in range(layers):
        pair = block.linear_affine.get(layer, (None, None))
        for kind, tensor in enumerate((*pair, block.linear_conv_state.get(layer))):
            if tensor is None:
                continue
            if tuple(tensor.shape) != shapes[kind]:
                raise ValueError(f'Unexpected GDN state shape {tensor.shape}, expected {shapes[kind]}')
            prepared = tensor.to(device=device, dtype=dtype if kind == 2 else torch.float32).contiguous()
            cacheable &= prepared is tensor
            owners.append(prepared)
            pointers[layer, kind] = prepared.data_ptr()
    present = np.array([layer in block.linear_conv_state for layer in range(layers)])
    state = StateAddresses(signature, pointers, tuple(owners), present)
    if cacheable:
        block._gdn_state_cache.prepared = state
    return state
