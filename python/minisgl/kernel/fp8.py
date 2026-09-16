"""Dynamic group-128 E4M3 inputs and serialized block-128 E4M3 weights.

Only quantized projections use this module. No GDN, normalization, attention,
cache, scheduler or synchronization changes. Weight scales multiply FP8 values.
"""
import torch
import triton
import triton.language as tl

from .fp8_format import emulate_fp8, e4m3fn_decode, e4m3fn_encode_finite_sat


def validate_weight(weight, scales, *, expected_shape=None):
    if weight.dtype != torch.float8_e4m3fn or weight.ndim not in (2, 3):
        raise ValueError("Expected an E4M3 weight matrix or expert stack")
    if expected_shape is not None and weight.shape != expected_shape:
        raise ValueError(f"FP8 weight shape {weight.shape} != {expected_shape}")
    n, k = weight.shape[-2:]
    if n % 128 or k % 128 or min(n, k) == 0:
        raise ValueError("FP8 weight dimensions must be positive multiples of 128")
    shape = (*weight.shape[:-2], n // 128, k // 128)
    if scales is None or scales.shape != shape or scales.dtype != torch.float32:
        raise ValueError(f"Expected FP32 weight scales of shape {shape}")
    if weight.device != scales.device or not weight.is_contiguous() or not scales.is_contiguous():
        raise ValueError("FP8 weights/scales must be contiguous on the same device")


@triton.jit
def _quantize_groups(X, Q, S, EMULATE_FP8: tl.constexpr):
    group = tl.program_id(0)
    offsets = group * 128 + tl.arange(0, 128)
    x = tl.load(X + offsets).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 0), 1e-10)
    scale = amax * (1.0 / 448.0)
    # Match the SGLang CUDA quantizer's FP32 division, without BF16 rounding.
    multiplier = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;", "=f,f,f", [448.0, amax],
        dtype=tl.float32, is_pure=True, pack=1)
    q = tl.minimum(tl.maximum(x * multiplier, -448.0), 448.0)
    if EMULATE_FP8:
        tl.store(Q + offsets, e4m3fn_encode_finite_sat(q))
    else:
        tl.store(Q + offsets, q)
    tl.store(S + group, scale)


def quantize_fp8_groups(x):
    if (not x.is_cuda or x.ndim != 2 or not x.is_contiguous()
            or x.shape[-1] == 0 or x.shape[-1] % 128):
        raise ValueError("FP8 inputs must be contiguous CUDA [rows, K], K a positive multiple of 128")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported FP8 activation dtype: {x.dtype}")
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0], x.shape[1] // 128), device=x.device, dtype=torch.float32)
    if scales.numel():
        emulated = emulate_fp8(x.device)
        _quantize_groups[(scales.numel(),)](
            x, q.view(torch.uint8) if emulated else q, scales, emulated, num_warps=4)
    return q, scales


@triton.jit
def _block_gemm(A, B, AS, BS, C, M, N: tl.constexpr, K: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, EMULATE_FP8: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, 128)
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(K // 128):
        a = tl.load(A + m[:, None] * K + group * 128 + k[None, :], m[:, None] < M, 0.0)
        b = tl.load(B + n[None, :] * K + group * 128 + k[:, None], n[None, :] < N, 0.0)
        if EMULATE_FP8:
            a = e4m3fn_decode(a).to(tl.bfloat16)
            b = e4m3fn_decode(b).to(tl.bfloat16)
        sa = tl.load(AS + m * (K // 128) + group, m < M, 0)
        sb = tl.load(BS + (n // 128) * (K // 128) + group, n < N, 0)
        # CUTLASS association: dot * (activation_scale * weight_scale).
        acc += tl.dot(a, b) * (sa[:, None] * sb[None, :])
    tl.store(C + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def block_fp8_linear(x, weight, scales, bias=None):
    validate_weight(weight, scales)
    if weight.ndim != 2 or x.shape[-1] != weight.shape[-1] or x.device != weight.device:
        raise ValueError("FP8 Linear input/weight shape or device mismatch")
    n, k = weight.shape
    inputs, input_scales = quantize_fp8_groups(x.reshape(-1, k).contiguous())
    output = torch.empty((inputs.shape[0], n), device=x.device, dtype=x.dtype)
    if inputs.shape[0]:
        bm = 16 if inputs.shape[0] < 32 else 32
        emulated = emulate_fp8(x.device)
        _block_gemm[(triton.cdiv(inputs.shape[0], bm), triton.cdiv(n, 64))](
            inputs.view(torch.uint8) if emulated else inputs,
            weight.view(torch.uint8) if emulated else weight,
            input_scales, scales, output, inputs.shape[0], n, k, bm, 64, emulated)
    if bias is not None:
        output += bias
    return output.view(*x.shape[:-1], n)
