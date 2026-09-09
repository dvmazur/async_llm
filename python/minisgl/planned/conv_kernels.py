"""Ragged slot conv with the existing low-product/FP32-sum rounding contract."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["LAYER"])
def ragged_conv(
    x, weight, conv_pool, prior_slots, row_active, row_request, row_time,
    pf_offsets, output, new_window, LAYER,
    BASE: tl.constexpr, REQUEST_BASE: tl.constexpr, PREFILL: tl.constexpr,
    SLOTS: tl.constexpr, C: tl.constexpr, K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    row = BASE + tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = channel < C
    if tl.load(row_active + row):
        request = tl.load(row_request + row) - REQUEST_BASE
        time = tl.load(row_time + row)
        source_slot = tl.load(prior_slots + request).to(tl.int64)
        if PREFILL:
            start = tl.load(pf_offsets + request)
            length = tl.load(pf_offsets + request + 1) - start
        else:
            start = row
            length = 1
        prior = ((LAYER.to(tl.int64) * SLOTS + source_slot) * C + channel) * K
        acc = tl.zeros((BLOCK_C,), tl.float32)
        for tap in tl.static_range(K):
            source_time = time - (K - 1) + tap
            value = tl.load(x + (start + source_time) * C + channel, mask & (source_time >= 0), other=0.)
            old = tl.load(conv_pool + prior + K + source_time,
                          mask & (source_time < 0) & (source_slot >= 0), other=0.)
            value = tl.where(source_time >= 0, value, old)
            w = tl.load(weight + channel * K + tap, mask, other=0.)
            product = (value.to(tl.float32) * w.to(tl.float32)).to(x.dtype.element_ty)
            acc += product.to(tl.float32)
        result = acc / (1. + tl.exp(-acc))
        tl.store(output + row * C + channel, result, mask)
        if time == length - 1:
            # Last token uniquely produces each raw window cell. It is saved
            # to existing window workspace, not yet over old readers' slots.
            for tap in tl.static_range(K):
                window_time = length - K + tap
                value = tl.load(x + (start + window_time) * C + channel, mask & (window_time >= 0), other=0.)
                old = tl.load(conv_pool + prior + K + window_time,
                              mask & (window_time < 0) & (source_slot >= 0), other=0.)
                value = tl.where(window_time >= 0, value, old)
                tl.store(new_window + (request * C + channel) * K + tap, value, mask)
    else:
        tl.store(output + row * C + channel, 0., mask)


@triton.jit(do_not_specialize=["LAYER"])
def publish_windows(window, pool, writes, active, LAYER, SLOTS: tl.constexpr,
                    C: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    worker = tl.program_id(0)
    if tl.load(active + worker):
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(writes + worker).to(tl.int64)
        value = tl.load(window + worker * C * K + offset, offset < C*K, other=0.)
        tl.store(pool + (LAYER.to(tl.int64)*SLOTS + slot)*C*K + offset, value, offset < C*K)
