import torch
import triton
import triton.language as tl


@triton.jit
def _conv(X, W, P, Y, SIZE, T: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
          XB: tl.constexpr, XT: tl.constexpr, XC: tl.constexpr,
          WC: tl.constexpr, WK: tl.constexpr,
          PB: tl.constexpr, PC: tl.constexpr, PK: tl.constexpr,
          HAS_PRIOR: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = index % C
    time = index // C % T
    batch = index // (T * C)
    valid = index < SIZE
    acc = tl.zeros((BLOCK,), tl.float32)
    for tap in tl.static_range(K):
        source_time = time - (K - 1) + tap
        value = tl.load(X + batch * XB + source_time * XT + channel * XC,
                        valid & (source_time >= 0), 0)
        if HAS_PRIOR:
            old = tl.load(P + batch * PB + channel * PC + (K + source_time) * PK,
                          valid & (source_time < 0), 0)
            value = tl.where(source_time >= 0, value, old)
        weight = tl.load(W + channel * WC + tap * WK, valid, 0)
        product = (value.to(tl.float32) * weight.to(tl.float32)).to(X.dtype.element_ty)
        acc = acc + product.to(tl.float32)
    out = acc / (1.0 + tl.exp(-acc))
    tl.store(Y + index, out, valid)


@triton.jit
def _window(X, P, O, SIZE, T: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
            XB: tl.constexpr, XT: tl.constexpr, XC: tl.constexpr,
            PB: tl.constexpr, PC: tl.constexpr, PK: tl.constexpr,
            HAS_PRIOR: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = index % K
    channel = index // K % C
    batch = index // (K * C)
    source_time = T - K + slot
    valid = index < SIZE
    value = tl.load(X + batch * XB + source_time * XT + channel * XC,
                    valid & (source_time >= 0), 0)
    if HAS_PRIOR:
        old = tl.load(P + batch * PB + channel * PC + (K + source_time) * PK,
                      valid & (source_time < 0), 0)
        value = tl.where(source_time >= 0, value, old)
    tl.store(O + index, value, valid)


def launch_causal_conv(x, weight, prior):
    batch, length, channels = x.shape
    kernel = weight.shape[-1]
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    window = x.new_empty(batch, channels, kernel)
    p = x if prior is None else prior
    ps = (0, 0, 0) if prior is None else prior.stride()
    _conv[(triton.cdiv(output.numel(), 256),)](
        x, weight, p, output, output.numel(), length, channels, kernel,
        *x.stride(), weight.stride(0), weight.stride(2), *ps, prior is not None, 256,
        num_warps=4)
    _window[(triton.cdiv(window.numel(), 256),)](
        x, p, window, window.numel(), length, channels, kernel,
        *x.stride(), *ps, prior is not None, 256, num_warps=4)
    return output, window
