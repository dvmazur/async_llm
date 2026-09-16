"""Prefill depthwise convolution over ragged, block-owned token histories."""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _prefill_conv(QKV, Weight, Read, Write, Cu, Owner, Local, Out,
                  T: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
                  BT: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    channel = tl.program_id(1) * BC + tl.arange(0, BC)
    # Owner/local metadata includes padding, but padding must not read row0's
    # history or publish a state. Requests never borrow another request's tail.
    owner = tl.load(Owner + row, row < T, other=0)
    time = tl.load(Local + row, row < T, other=0)
    start = tl.load(Cu + owner)
    end = tl.load(Cu + owner + 1)
    active = (row < T) & (row < end) & (row >= start)
    prior = tl.load(Read + owner * 3 + 2).to(tl.pointer_type(QKV.dtype.element_ty))
    dest = tl.load(Write + owner * 3 + 2).to(tl.pointer_type(QKV.dtype.element_ty))
    mask = active[:, None] & (channel[None, :] < C)
    acc = tl.full((BT, BC), 0, tl.float32)
    for tap in tl.static_range(K):
        pos = time - K + 1 + tap
        fresh = tl.load(QKV + (start + pos)[:, None] * C + channel[None, :],
                        mask & (pos[:, None] >= 0), other=0.)
        old = tl.load(prior[:, None] + channel[None, :] * K + (K + pos)[:, None],
                      mask & (pos[:, None] < 0) & (prior[:, None].to(tl.uint64) != 0), other=0.)
        value = tl.where(pos[:, None] >= 0, fresh, old)
        weight = tl.load(Weight + channel * K + tap, channel < C, other=0.)
        # Preserve Torch: BF16/FP16 product rounded BEFORE FP32 accumulation.
        product = (value.to(tl.float32) * weight[None, :].to(tl.float32)).to(QKV.dtype.element_ty)
        acc += product.to(tl.float32)
    result = tl.div_rn(acc, 1. + libdevice.exp(-acc))
    # Preserve the channel-major output, hence downstream reduction layout.
    tl.store(Out + channel[None, :] * T + row[:, None], tl.where(active[:, None], result, 0.),
             (row[:, None] < T) & (channel[None, :] < C))
    last = active & (row == end - 1) & (dest.to(tl.uint64) != 0)
    for tap in tl.static_range(K):
        pos = end - start - K + tap
        fresh = tl.load(QKV + (start + pos)[:, None] * C + channel[None, :],
                        last[:, None] & (channel[None, :] < C) & (pos[:, None] >= 0), other=0.)
        old = tl.load(prior[:, None] + channel[None, :] * K + (K + pos)[:, None],
                      last[:, None] & (channel[None, :] < C) & (pos[:, None] < 0)
                      & (prior[:, None].to(tl.uint64) != 0), other=0.)
        value = tl.where(pos[:, None] >= 0, fresh, old)
        tl.store(dest[:, None] + channel[None, :] * K + tap, value,
                 last[:, None] & (channel[None, :] < C))


def prefill_conv(qkv, weight, read, write, cu, owner, local, output):
    """Write channel-major output and fresh conv states; inputs remain intact."""
    rows, channels = qkv.shape
    assert qkv.is_cuda and qkv.is_contiguous()
    assert qkv.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert weight.shape[:2] == (channels, 1) and weight.is_contiguous()
    assert weight.device == qkv.device and weight.dtype == qkv.dtype
    assert output.shape == qkv.shape and output.stride() == (1, rows)
    assert output.device == qkv.device and output.dtype == qkv.dtype
    _prefill_conv[(tr.cdiv(rows, 16), tr.cdiv(channels, 64))](
        qkv, weight, read, write, cu, owner, local, output,
        rows, channels, weight.shape[-1], 16, 64, num_warps=4, enable_fp_fusion=False)
    return output
