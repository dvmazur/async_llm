"""E4M3FN byte conversion for devices without native FP8 tensor cores.

Weights and activations keep the same W8A8 bytes and block scales. Only
GEMM tiles are decoded to BF16 (which represents finite E4M3 exactly).
No full-matrix BF16 allocation or alternate quantization format is used.
"""
from functools import lru_cache
import os

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=16)
def _native_e4m3(device):
    return torch.cuda.get_device_capability(device) >= (8, 9)


def emulate_fp8(device):
    # Force the Ampere path on newer devices only for numerical A/B tests.
    return os.environ.get('MINISGL_FP8_EMULATE', '0') == '1' or not _native_e4m3(device)


@triton.jit
def e4m3fn_decode(bits):
    bits = bits.to(tl.uint32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    magnitude = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
    magnitude = tl.where(exponent == 0, mantissa.to(tl.float32) * (1.0 / 512), magnitude)
    magnitude_bits = tl.where((bits & 127) == 127, 0x7fc00000,
                              magnitude.to(tl.uint32, bitcast=True))
    return (magnitude_bits | ((bits & 128) << 24)).to(tl.float32, bitcast=True)


@triton.jit
def e4m3fn_encode_finite_sat(x):
    # Round-to-nearest-even, including subnormals and signed zero.
    magnitude = tl.minimum(tl.abs(x), 448.0)
    exponent = ((magnitude.to(tl.uint32, bitcast=True) >> 23) & 255).to(tl.int32) - 127
    step_exponent = tl.maximum(exponent - 3, -9)
    inverse_step = ((127 - step_exponent).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    rounded = tl.inline_asm_elementwise(
        'cvt.rni.u32.f32 $0, $1;', constraints='=r,f', args=[magnitude * inverse_step],
        dtype=tl.uint32, is_pure=True, pack=1)
    encoded = ((step_exponent + 9).to(tl.uint32) << 3) + rounded
    sign = (x.to(tl.uint32, bitcast=True) >> 24) & 128
    return (encoded | sign).to(tl.uint8)
