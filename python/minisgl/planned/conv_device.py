"""Static conv binding: no per-request Python convolution or cat."""
import torch
import triton

from . import conv_kernels


class BoundConv:
    def __init__(self, conv_pool, weight, phase, rows, capacity, *, prefill, window_workspace=None):
        self.pool, self.weight, self.phase, self.rows = conv_pool, weight, phase, rows
        if conv_pool.ndim != 4 or weight.ndim != 3 or weight.shape[1] != 1:
            raise ValueError("expected [L,slots,C,K] conv pool and [C,1,K] weights")
        self.layers, self.slots, self.channels, self.window_size = conv_pool.shape
        if (not conv_pool.is_cuda or not conv_pool.is_contiguous() or not weight.is_contiguous()
                or weight.shape != (self.channels,1,self.window_size)
                or weight.device != conv_pool.device or weight.dtype != conv_pool.dtype
                or phase.device != conv_pool.device or rows.device != conv_pool.device):
            raise ValueError("incompatible conv buffers")
        self.prefill = prefill
        self.base = 0 if prefill else capacity.prefill_tokens
        self.request_base = 0 if prefill else capacity.prefill_requests
        self.row_count = capacity.prefill_tokens if prefill else capacity.decode_workers
        self.total_rows = capacity.prefill_tokens + capacity.decode_workers
        shape = (phase.width,self.channels,self.window_size)
        if window_workspace is None:
            window_workspace = torch.empty(shape,device=conv_pool.device,dtype=conv_pool.dtype)
        if (window_workspace.shape != shape or window_workspace.device != conv_pool.device
                or window_workspace.dtype != conv_pool.dtype or not window_workspace.is_contiguous()):
            raise ValueError("invalid shared conv window workspace")
        self.new_window = window_workspace

    def run(self, layer, qkv, output):
        if (qkv.shape != (self.total_rows,self.channels) or output.shape != qkv.shape
                or qkv.dtype != self.pool.dtype or output.dtype != qkv.dtype
                or qkv.device != self.pool.device or output.device != qkv.device
                or not qkv.is_contiguous() or not output.is_contiguous()
                or not 0 <= layer < self.layers):
            raise ValueError("invalid bound conv inputs")
        if self.row_count:
            conv_kernels.ragged_conv[(self.row_count,triton.cdiv(self.channels,128))](
                qkv,self.weight,self.pool,self.phase.prior_conv_slots,self.rows.active,
                self.rows.request,self.rows.local_token,self.rows.prefill_offsets,
                output,self.new_window,LAYER=layer,BASE=self.base,REQUEST_BASE=self.request_base,
                PREFILL=self.prefill,SLOTS=self.slots,C=self.channels,K=self.window_size,BLOCK_C=128)
        return self.new_window

    def publish(self, layer):
        """Baseline publication; decode will fuse this store into capture."""
        if self.phase.width:
            conv_kernels.publish_windows[(self.phase.width,triton.cdiv(self.channels*self.window_size,256))](
                self.new_window,self.pool,self.phase.write_slots,self.phase.active,LAYER=layer,
                SLOTS=self.slots,C=self.channels,K=self.window_size,BLOCK=256)
