"""Prepared decode metadata for SharedCacheGDN, not a separate executor.

Blocks still own their tensors. Only addresses/topology have stable device
storage; the existing model and affine functions do the computation.
"""

from __future__ import annotations

import numpy as np
import torch

from minisgl.kernel.gdn_compose import compose_first, compose_level
from minisgl.kernel.gdn_prefill import capture_affine_scan
from minisgl.kernel.gdn_io import gather_rows, scatter_rows

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


def terminal_sinks(terminals):
    """CSR node -> terminal workers; at most W deliveries across all depths."""
    width = len(terminals[0])
    offsets, sinks = [], []
    empty = [True] * width
    for row in terminals:
        groups = [[] for _ in range(width)]
        for worker, node in enumerate(row):
            if node >= 0:
                groups[node].append(worker)
                empty[worker] = False
        prefix, flat = [0], []
        for group in groups:
            flat.extend(group)
            prefix.append(len(flat))
        offsets.append(prefix)
        sinks.append(flat + [0] * (width - len(flat)))
    return offsets, sinks, empty


class GDNDecodeBuffers:
    """One capacity's persistent metadata and per-layer reused scratch.

    ``prepare`` and ``publish`` execute on the host, never inside capture.
    ``compose``, ``conv``, ``capture`` and ``store_conv`` are GPU-only bodies.
    """

    def __init__(self, ar, layers: int, workers: int, depth: int, dtype):
        self.workers, self.depth, self.layers = workers, depth, layers
        self.dtype, self.device = dtype, ar.device
        self.state_dtype = getattr(ar, 'state_dtype', torch.float32)
        self.h, self.dk, self.dv = ar.num_heads, ar.head_k_dim, ar.head_v_dim
        self.conv_shape = (ar.conv_dim, ar.conv_kernel)
        self._state_signature = (self.device, layers, self.h, self.dk, self.dv,
                                 self.conv_shape, dtype, self.state_dtype)
        args = dict(device=ar.device)
        self.affine_ptrs = torch.zeros(layers, depth, workers, 2, dtype=torch.int64, **args)
        self.read_ptrs = torch.zeros(layers, workers, 3, dtype=torch.int64, **args)
        self.write_ptrs = torch.zeros_like(self.read_ptrs)
        self.parents = torch.zeros(depth, workers, dtype=torch.int64, **args)
        self.level_counts = torch.zeros(depth, dtype=torch.int32, **args)
        self.terminals = torch.full((depth, workers), -1, dtype=torch.int32, **args)
        self.sink_offsets = torch.zeros(depth, workers + 1, dtype=torch.int32, **args)
        self.sink_workers = torch.zeros(depth, workers, dtype=torch.int32, **args)
        self.empty_workers = torch.ones(workers, dtype=torch.bool, **args)
        self.b = torch.empty(workers, self.h, self.dv, self.dk, dtype=self.state_dtype, **args)
        # Compose ping-pongs between b and frontier. Capture writes directly
        # into block-owned outputs, without staging dense A/B scratch.
        self.frontier = torch.empty_like(self.b)
        self.initial = torch.empty(workers, self.h, self.dk, self.dv, dtype=self.state_dtype, **args)
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
        sink_offsets, sink_workers, empty = terminal_sinks(terminals)
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
                             dtype=self.state_dtype, device=self.device) for _ in targets]
        out_b = [torch.empty(self.layers, 1, self.h, self.dv, self.dk,
                             dtype=self.state_dtype, device=self.device) for _ in targets]
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
                         (self.level_counts, [len(nodes) for nodes in levels]),
                         (self.terminals, terminals), (self.sink_offsets, sink_offsets),
                         (self.sink_workers, sink_workers), (self.empty_workers, empty)):
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
        state, output = self.b, self.frontier
        for level in range(self.depth):
            if level == 0:
                compose_first(self.affine_ptrs[layer, level], self.level_counts, state,
                              self.initial, self.sink_offsets[level], self.sink_workers[level],
                              self.empty_workers)
            else:
                compose_level(self.affine_ptrs[layer, level], self.parents[level],
                              self.level_counts, level, state, output, self.initial,
                              self.sink_offsets[level], self.sink_workers[level])
                state, output = output, state
        return self.initial

    def conv(self, layer):
        return gather_rows(self.read_ptrs[layer, :, 2], self.conv_input)

    def capture(self, layer, key, value, alpha, beta, eps):
        capture_affine_scan(self.read_ptrs[layer], self.write_ptrs[layer],
                            key[:, 0], value[:, 0], alpha[:, 0], beta[:, 0], l2norm_eps=eps,
                            state_dtype=self.state_dtype)

    def store_conv(self, layer, conv):
        scatter_rows(self.write_ptrs[layer, :, 2], conv)
