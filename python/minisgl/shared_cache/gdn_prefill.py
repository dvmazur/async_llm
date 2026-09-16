"""Prepared ragged prefill using the same block state/addressing as decode."""

import torch
import torch.nn.functional as F

from minisgl.kernel.gdn_prefill import capture_affine_scan, chunk_gdn
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
        self.conv_indices = torch.zeros(rows, self.conv_shape[1], dtype=torch.int64, **args)
        self.state_indices = torch.zeros(workers, self.conv_shape[1], dtype=torch.int64, **args)
        self.row_worker = torch.zeros(rows, dtype=torch.int64, **args)
        self.row_local = torch.zeros(rows, dtype=torch.int64, **args)
        self.prefill_count = 0

    def prepare(self, chains, targets, lengths):
        if len(lengths) != len(targets) or any(t <= 0 for t in lengths) or sum(lengths) > self.rows:
            raise ValueError('Prefill exceeds prepared request/token capacity')
        super().prepare(chains, targets)
        lengths = list(lengths) + [0] * (self.workers - len(lengths))
        cu, offsets, chunks = [0], [0], []
        conv, store, owners, local = [], [], [], []
        k = self.conv_shape[1]
        base = self.workers * k
        for worker, length in enumerate(lengths):
            start = cu[-1]
            for position in range(length):
                conv.append([base + start + t if t >= 0 else worker * k + k + t
                             for t in range(position - k + 1, position + 1)])
                owners.append(worker)
                local.append(position)
            store.append([base + start + t if t >= 0 else worker * k + k + t
                          for t in range(length - k, length)])
            count = (length + 63) // 64
            chunks.extend((worker, chunk) for chunk in range(count))
            cu.append(start + length)
            offsets.append(offsets[-1] + count)
        pad = self.rows - cu[-1]
        conv += [[0] * k] * pad
        owners += [0] * pad
        local += [0] * pad
        # All dummy chunks are outside sequence0, never duplicates of real
        # chunks. FLA masks their token accesses; h has full capacity storage.
        chunks += [(0, len(self.chunk_indices) - 1)] * (len(self.chunk_indices) - len(chunks))
        for dst, values in ((self.cu, cu), (self.chunk_offsets, offsets),
                            (self.chunk_indices, chunks), (self.conv_indices, conv),
                            (self.state_indices, store), (self.row_worker, owners), (self.row_local, local)):
            host = torch.tensor(values, dtype=dst.dtype, pin_memory=True)
            self._current[-1].append(host)
            dst.copy_(host, non_blocking=True)

    def convolve(self, layer, qkv, weight):
        prior = self.conv(layer).transpose(1, 2).reshape(-1, self.conv_shape[0])
        inputs = torch.cat([prior, qkv], dim=0)
        # Keep the original channel-major conv result. Downstream Torch key
        # normalization uses a strided reduction; changing this layout changes
        # its FP32 rounding and can amplify in later BF16 recurrent steps.
        total = torch.zeros(qkv.shape[1], qkv.shape[0], device=qkv.device, dtype=torch.float32)
        # Match the old prefill's BF16 product rounding before FP32 accumulation.
        for tap in range(self.conv_shape[1]):
            total += (inputs.index_select(0, self.conv_indices[:, tap]).t() * weight[:, 0, tap, None]).float()
        new_state = inputs[self.state_indices].transpose(1, 2)
        self.store_conv(layer, new_state)
        return F.silu(total).to(qkv.dtype).t()

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
        capture_affine_scan(self.read_ptrs[layer], self.write_ptrs[layer], k, v, alpha, beta, self.cu)
