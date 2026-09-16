"""Prepared ragged prefill using the same block state/addressing as decode."""

import torch

from minisgl.kernel.gdn_prefill import capture_affine_scan, chunk_gdn
from minisgl.kernel.gdn_conv import prefill_conv
from minisgl.kernel.gdn_prefill_io import pack_initial, store_affines
from .gdn_decode import GDNDecodeBuffers


class GDNPrefillBuffers(GDNDecodeBuffers):
    def __init__(self, ar, layers, workers, depth, dtype, rows):
        super().__init__(ar, layers, workers, depth, dtype)
        self.rows = rows
        chunks = (rows + 63) // 64 + workers
        args = dict(device=self.device)
        self.cu = torch.zeros(workers + 1, dtype=torch.int32, **args)
        self.chunk_offsets = torch.zeros_like(self.cu)
        self.chunk_indices = torch.zeros(chunks, 2, dtype=torch.int32, **args)
        self.row_worker = torch.zeros(rows, dtype=torch.int64, **args)
        self.row_local = torch.zeros(rows, dtype=torch.int64, **args)
        self.conv_output = torch.empty(self.conv_shape[0], rows, dtype=dtype, **args).t()
        # Joint FLA: S/A/B share token transforms, but A/B still belong to
        # individual blocks. Only these packing buffers are persistent scratch.
        width = self.dv + self.dk + self.dv
        # Empty slots are never read: both packing and FLA H are guarded.
        self.joint_initial = torch.empty(workers, self.h, self.dk, width,
                                         dtype=self.state_dtype, **args)
        self.joint_values = torch.zeros(1, rows, self.h, width, dtype=dtype, **args)
        self.prefill_count = 0

    def prepare(self, chains, targets, lengths):
        if len(lengths) != len(targets) or any(t <= 0 for t in lengths) or sum(lengths) > self.rows:
            raise ValueError('Prefill exceeds prepared request/token capacity')
        super().prepare(chains, targets)
        lengths = list(lengths) + [0] * (self.workers - len(lengths))
        cu, offsets, chunks = [0], [0], []
        owners, local = [], []
        for worker, length in enumerate(lengths):
            start = cu[-1]
            for position in range(length):
                owners.append(worker)
                local.append(position)
            count = (length + 63) // 64
            chunks.extend((worker, chunk) for chunk in range(count))
            cu.append(start + length)
            offsets.append(offsets[-1] + count)
        pad = self.rows - cu[-1]
        owners += [0] * pad
        local += [0] * pad
        for dst, values in ((self.cu, cu), (self.chunk_offsets, offsets),
                            (self.row_worker, owners), (self.row_local, local)):
            host = torch.tensor(values, dtype=dst.dtype, pin_memory=True)
            self._current[-1].append(host)
            dst.copy_(host, non_blocking=True)
        # Keep capacity/address, upload only real entries. Guards read the
        # actual count from chunk_offsets[-1] before touching this table.
        if chunks:
            host = torch.tensor(chunks, dtype=self.chunk_indices.dtype, pin_memory=True)
            self._current[-1].append(host)
            self.chunk_indices[:len(chunks)].copy_(host, non_blocking=True)

    def convolve(self, layer, qkv, weight):
        return prefill_conv(qkv, weight, self.read_ptrs[layer], self.write_ptrs[layer],
                            self.cu, self.row_worker, self.row_local, self.conv_output)

    def core_and_capture(self, layer, q, k, v, g, beta, initial, use_fla, torch_chunk):
        if not use_fla:
            out = self.core(q, k, v, g, beta, initial, False, torch_chunk)
            self.capture_prefill(layer, k, v, g.exp(), beta)
            return out
        self.prefill_count += 1
        pack_initial(self.read_ptrs[layer], initial, self.joint_initial, self.cu)
        self.joint_values[0, ..., :self.dv].copy_(v)
        self.joint_values[0, ..., self.dv + self.dk:].copy_(v)
        # One installed FLA pass, NOT a second FP32 affine pass. Its operands
        # (including those updating A/B) use activation dtype, normally BF16;
        # accumulation is FP32; final A/B use the configured state storage dtype.
        # This changes capture numerics from
        # the old FP32 token scan; external model parity is the quality gate.
        out, final = chunk_gdn(q[None], k[None], self.joint_values, g[None], beta[None],
                               self.joint_initial, self.cu, self.chunk_indices,
                               self.chunk_offsets, output_final_state=True, skip_empty_states=True)
        store_affines(self.write_ptrs[layer], final, self.dv)
        return out[0, ..., :self.dv].contiguous()

    def core(self, q, k, v, g, beta, initial, use_fla, torch_chunk):
        self.prefill_count += 1
        if use_fla:
            return chunk_gdn(q[None], k[None], v[None], g[None], beta[None],
                              initial, self.cu, self.chunk_indices, self.chunk_offsets)[0][0]
        # Optional no-FLA compatibility uses the ORIGINAL Torch chunk algorithm,
        # with explicit per-request zero padding (not the native FLA path).
        # Padding is a no-op recurrence: q/k/v/beta=0, g=0 => alpha=1.
        positions = torch.arange(self.rows, device=self.device)[None, :]
        valid = positions < (self.cu[1:] - self.cu[:-1])[:, None]
        indices = (self.cu[:-1, None] + positions).long().clamp(max=self.rows - 1)
        def pack(x, preserve_reduction_layout=False):
            mask = valid.reshape(*valid.shape, *((1,) * (x.ndim - 1)))
            result = torch.where(mask, x[indices], 0)
            # The fallback normalizes q/k in BF16 BEFORE its FP32 conversion.
            # Gathering must not change a channel-major reduction into a
            # token-major one. Repeated-head q/k are already token-major.
            if preserve_reduction_layout and x.stride(-1) != 1:
                result = result.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
            return result
        out, _ = torch_chunk(pack(q, True), pack(k, True), pack(v), pack(g), pack(beta),
                              initial_state=initial)
        result = out[self.row_worker, self.row_local]
        return result.masked_fill((torch.arange(self.rows, device=self.device) >= self.cu[-1])[:, None, None], 0)

    def capture_prefill(self, layer, k, v, alpha, beta):
        capture_affine_scan(self.read_ptrs[layer], self.write_ptrs[layer], k, v, alpha, beta, self.cu,
                            state_dtype=self.state_dtype)
