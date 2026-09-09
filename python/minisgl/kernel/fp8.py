"""Serialized block-FP8 weights: dynamic group-128 inputs, FP32 accumulation.

No weight quantization: weights/scales come verbatim from the HF checkpoint.
The tiled GEMM follows the standard block-scaled matmul used by SGLang/vLLM.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_groups(X, Q, S):
    group = tl.program_id(0)
    offsets = group * 128 + tl.arange(0, 128)
    x = tl.load(X + offsets).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 0), 1e-10)
    scale = amax * (1.0 / 448.0)
    # Match the CUDA group quantizer's fast FP32 division; do not round the
    # scale or the scaled input to BF16 before the E4M3 cast.
    multiplier = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;", "=f,f,f", [448.0, amax],
        dtype=tl.float32, is_pure=True, pack=1)
    q = tl.minimum(tl.maximum(x * multiplier, -448.0), 448.0)
    tl.store(Q + offsets, q)
    tl.store(S + group, scale)


def quantize_fp8_groups(x: torch.Tensor):
    if not x.is_cuda or not x.is_contiguous() or x.ndim != 2 or x.shape[-1] % 128:
        raise ValueError("FP8 inputs must be contiguous CUDA [tokens, K], K divisible by 128")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported input dtype: {x.dtype}")
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0], x.shape[1] // 128), device=x.device, dtype=torch.float32)
    if scales.numel():
        _quantize_groups[(scales.numel(),)](x, q, scales, num_warps=4)
    return q, scales


@triton.jit
def _block_gemm(A, B, AS, BS, C, M, N: tl.constexpr, K: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, 128)
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(K // 128):
        a = tl.load(A + m[:, None] * K + group * 128 + k[None, :], m[:, None] < M, 0.0)
        b = tl.load(B + n[None, :] * K + group * 128 + k[:, None], n[None, :] < N, 0.0)
        sa = tl.load(AS + m * (K // 128) + group, m < M, 0)
        sb = tl.load(BS + (n // 128) * (K // 128) + group, n < N, 0)
        # CUTLASS forms the FP32 scale product before applying it to the
        # block accumulator. Chained multiplication adds a different rounding
        # boundary; rare BF16 output differences can amplify at the next FP8
        # quantization. Keep the same association as the reference backend.
        scale = sa[:, None] * sb[None, :]
        acc += tl.dot(a, b) * scale
    tl.store(C + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def block_fp8_linear(x, weight, scales, bias=None):
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise ValueError("Expected serialized E4M3 matrix")
    n, k = weight.shape
    if x.shape[-1] != k:
        raise ValueError("Input/weight hidden dimensions differ")
    if n % 128 or k % 128 or scales.shape != (n // 128, k // 128):
        raise ValueError("Expected aligned 128x128 block scales")
    assert weight.is_contiguous() and scales.is_contiguous()
    assert scales.dtype == torch.float32 and x.device == weight.device == scales.device
    original_shape = x.shape
    inputs, input_scales = quantize_fp8_groups(x.reshape(-1, k).contiguous())
    output = torch.empty((inputs.shape[0], n), device=x.device, dtype=x.dtype)
    if inputs.shape[0]:
        bm = 16 if inputs.shape[0] < 32 else 32
        _block_gemm[(triton.cdiv(inputs.shape[0], bm), triton.cdiv(n, 64))](
            inputs, weight, input_scales, scales, output, inputs.shape[0], n, k, bm, 64)
    if bias is not None:
        output += bias
    return output.view(*original_shape[:-1], n)
