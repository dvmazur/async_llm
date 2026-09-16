"""Prepared decode metadata for SharedCacheGDN, not a separate executor.

Blocks still own their tensors. Only addresses/topology have stable device
storage; the existing model and affine functions do the computation.
"""

from __future__ import annotations

import numpy as np
import torch

from minisgl.kernel.gdn_io import collect_states, gather_rows, scatter_rows
from .gdn_affine import update_affine_summary

from .gdn_state import StateAddresses, owns_state_dicts, prepare_state


def prefix_links(chains, workers: int, depth: int):
    """Deduplicate complete block prefixes, including absent per-layer states."""
    if len(chains) > workers or any(len(c) > depth for c in chains):
        raise ValueError("GDN decode exceeds the prepared worker/depth capacity")
    levels = [[] for _ in range(depth)]
    parents = [[0] * workers for _ in range(depth)]
    terminals = [[-1] * workers for _ in range(depth)]
    memo = {}
    for worker, chain in enumerate(chains):
        prefix, parent = (), 0
        for level, block in enumerate(chain):
            prefix += (id(block),)
            if prefix not in memo:
                node = len(levels[level])
                memo[prefix] = node
                levels[level].append(block)
                parents[level][node] = parent
            parent = memo[prefix]
        if chain:
            terminals[len(chain) - 1][worker] = parent
    return levels, parents, terminals


class GDNDecodeBuffers:
    """One capacity's persistent metadata and per-layer reused scratch.

    ``prepare`` and ``publish`` execute on the host, never inside capture.
    ``compose``, ``conv``, ``capture`` and ``store_conv`` are GPU-only bodies.
    """

    def __init__(self, ar, layers: int, workers: int, depth: int, dtype):
        self.workers, self.depth, self.layers = workers, depth, layers
        self.dtype, self.device = dtype, ar.device
        self.h, self.dk, self.dv = ar.num_heads, ar.head_k_dim, ar.head_v_dim
        self.conv_shape = (ar.conv_dim, ar.conv_kernel)
        self._state_signature = (self.device, layers, self.h, self.dk, self.dv,
                                 self.conv_shape, dtype)
        args = dict(device=ar.device)
        self.affine_ptrs = torch.zeros(layers, depth, workers, 2, dtype=torch.int64, **args)
        self.read_ptrs = torch.zeros(layers, workers, 3, dtype=torch.int64, **args)
        self.write_ptrs = torch.zeros_like(self.read_ptrs)
        self.parents = torch.zeros(depth, workers, dtype=torch.int64, **args)
        self.terminals = torch.full((depth, workers), -1, dtype=torch.int32, **args)
        self.b = torch.empty(workers, self.h, self.dv, self.dk, dtype=torch.float32, **args)
        self.a = torch.empty(workers, self.h, self.dk, self.dk, dtype=torch.float32, **args)
        self.initial = torch.empty(workers, self.h, self.dk, self.dv, dtype=torch.float32, **args)
        self.conv_input = torch.empty(workers, *self.conv_shape, dtype=dtype, **args)
        self._pending = []
        self._current = None
        self.prepare_count = 0
        self.compose_count = 0

    def prepare(self, chains, targets):
        if self._current is not None:
            raise RuntimeError("Previous GDN decode has not been published")
        if len(chains) != len(targets) or len({id(t) for t in targets}) != len(targets):
            raise ValueError("GDN decode needs one distinct write target per worker")
        levels, parents, terminals = prefix_links(chains, self.workers, self.depth)
        # Keep tensor and pinned upload storage alive until its last raw-pointer
        # consumer completes. This does not synchronize or delay enqueueing.
        self.retire_completed()
        affine = np.zeros(self.affine_ptrs.shape, dtype=np.int64)
        reads = np.zeros(self.read_ptrs.shape, dtype=np.int64)
        writes = np.zeros(self.write_ptrs.shape, dtype=np.int64)
        refs = []
        # One all-layer allocation per worker: a frozen/stopped worker must
        # not retain the outputs of an entire previous batch through its view.
        # New outputs still preserve readers of the previous tensors.
        out_a = [torch.empty(self.layers, 1, self.h, self.dk, self.dk,
                             dtype=torch.float32, device=self.device) for _ in targets]
        out_b = [torch.empty(self.layers, 1, self.h, self.dv, self.dk,
                             dtype=torch.float32, device=self.device) for _ in targets]
        out_conv = [torch.empty(self.layers, *self.conv_shape,
                                dtype=self.dtype, device=self.device) for _ in targets]
        refs.extend((*out_a, *out_b, *out_conv))
        # These are our own contiguous all-layer allocations. Compute layer
        # addresses without temporary tensor views; publish still creates the
        # per-layer views exposed by blocks after execution.
        output_addresses = [tuple((t.data_ptr(), t.stride(0) * t.element_size())
                                  for t in outputs)
                            for outputs in zip(out_a, out_b, out_conv)]

        states = {}

        def state(block):
            if id(block) not in states:
                states[id(block)] = prepare_state(block, self._state_signature)
                # Strong owners are released ONLY after publish's completion
                # event. This protects raw-pointer reads without per-view
                # record_stream calls, including cross-stream input storage.
                refs.append(states[id(block)])
            return states[id(block)].pointers

        for level, blocks in enumerate(levels):
            for node, block in enumerate(blocks):
                affine[:, level, node] = state(block)[:, :2]
        layer_ids = np.arange(self.layers)[:, None]
        for w, target in enumerate(targets):
            reads[:, w, :2] = state(target)[:, :2]
            seen_conv = np.zeros(self.layers, dtype=bool)
            for block in reversed(chains[w]):
                conv = state(block)[:, 2]
                present = states[id(block)].conv_present
                missing = ~seen_conv & present
                reads[missing, w, 2] = conv[missing]
                seen_conv |= present
                if np.all(seen_conv):
                    break
            layout = np.asarray(output_addresses[w], dtype=np.int64)
            writes[:, w] = layout[:, 0] + layer_ids * layout[:, 1]
        for dst, src in ((self.affine_ptrs, affine), (self.read_ptrs, reads),
                         (self.write_ptrs, writes), (self.parents, parents),
                         (self.terminals, terminals)):
            host = torch.empty(dst.shape, dtype=dst.dtype, pin_memory=True)
            host.numpy()[...] = src
            refs.append(host)
            dst.copy_(host, non_blocking=True)
        self._current = (list(targets), out_a, out_b, out_conv, refs)
        self.prepare_count += 1

    def retire_completed(self):
        self._pending = [(event, refs) for event, refs in self._pending if not event.query()]

    def publish(self, success=True):
        if self._current is None:
            return
        targets, out_a, out_b, out_conv, refs = self._current
        if success:
            for layer in range(self.layers):
                for w, target in enumerate(targets):
                    target.linear_affine[layer] = (out_a[w][layer], out_b[w][layer])
                    target.linear_conv_state[layer] = out_conv[w][layer]
            for w, target in enumerate(targets):
                if owns_state_dicts(target):
                    target._gdn_state_cache.prepared = StateAddresses.from_slabs(
                        self._state_signature, (out_a[w], out_b[w], out_conv[w]))
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self._pending.append((event, refs))
        self._current = None

    def compose(self, layer):
        self.compose_count += 1  # capture counts host execution; replay counted by runner
        self.initial.zero_()
        state = None
        for level in range(self.depth):
            gather_rows(self.affine_ptrs[layer, level, :, 1], self.b)
            if level == 0:
                state = self.b.clone()
            else:
                gather_rows(self.affine_ptrs[layer, level, :, 0], self.a, self.dk)
                state = torch.matmul(state.index_select(0, self.parents[level]), self.a)
                state.add_(self.b)
            collect_states(state, self.terminals[level], self.initial)
        return self.initial

    def conv(self, layer):
        return gather_rows(self.read_ptrs[layer, :, 2], self.conv_input)

    def capture(self, layer, key, value, alpha, beta, eps):
        key_f = key.float()
        key_f = key_f * torch.rsqrt((key_f * key_f).sum(-1, keepdim=True) + eps)
        gather_rows(self.read_ptrs[layer, :, 0], self.a, self.dk)
        gather_rows(self.read_ptrs[layer, :, 1], self.b)
        a, b = update_affine_summary(A_hat=self.a, B_hat=self.b, k=key_f[:, 0],
                                     v=value[:, 0].float(), alpha=alpha[:, 0].float(),
                                     beta=beta[:, 0].float())
        scatter_rows(self.write_ptrs[layer, :, 0], a)
        scatter_rows(self.write_ptrs[layer, :, 1], b)

    def store_conv(self, layer, conv):
        scatter_rows(self.write_ptrs[layer, :, 2], conv)
