"""Byte format, software rounding and native/emulated FP8 GEMM parity."""
import pytest
import torch
import triton
import triton.language as tl

from minisgl.kernel.fp8 import quantize_fp8_groups, block_fp8_linear
from minisgl.kernel.fp8_format import e4m3fn_decode, e4m3fn_encode_finite_sat
from minisgl.moe.fused import fused_experts_impl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@triton.jit
def _decode(B, O):
    i = tl.arange(0, 256)
    tl.store(O + i, e4m3fn_decode(tl.load(B + i)))


@triton.jit
def _encode(X, O, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    x = tl.load(X + i, i < N, 0)
    tl.store(O + i, e4m3fn_encode_finite_sat(x), i < N)


def test_every_e4m3_byte_matches_cpu():
    raw = torch.arange(256, dtype=torch.uint8)
    expected = raw.view(torch.float8_e4m3fn).float()
    out = torch.empty(256, device='cuda')
    _decode[(1,)](raw.cuda(), out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(out.cpu().signbit(), expected.signbit())


def test_rounding_midpoints_subnormals_and_signed_zero():
    torch.manual_seed(61)
    values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (values[:-1] + values[1:]) / 2
    edges = torch.cat([values, midpoints,
        midpoints.nextafter(torch.full_like(midpoints, float('inf'))),
        midpoints.nextafter(torch.full_like(midpoints, -float('inf')))])
    x = torch.cat([edges, -edges, torch.randn(32768) * 150]).clamp(-448, 448)
    expected = x.to(torch.float8_e4m3fn).view(torch.uint8)
    out = torch.empty_like(x, dtype=torch.uint8, device='cuda')
    _encode[(triton.cdiv(x.numel(),256),)](x.cuda(), out, x.numel())
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_quantized_bytes_identical_to_native(monkeypatch, dtype):
    if torch.cuda.get_device_capability() < (8,9):
        pytest.skip('Native E4M3 reference needs Ada or newer')
    torch.manual_seed(37)
    x = torch.randn(137, 384, device='cuda', dtype=dtype)
    monkeypatch.delenv('MINISGL_FP8_EMULATE', raising=False)
    q, scales = quantize_fp8_groups(x)
    monkeypatch.setenv('MINISGL_FP8_EMULATE','1')
    got, got_scales = quantize_fp8_groups(x)
    assert torch.equal(q.view(torch.uint8), got.view(torch.uint8))
    torch.testing.assert_close(scales, got_scales, rtol=0, atol=0)


@pytest.mark.parametrize('rows', [1,3,19,137])
@torch.inference_mode()
def test_linear_and_moe_emulation_against_native(monkeypatch, rows):
    if torch.cuda.get_device_capability() < (8,9):
        pytest.skip('Native E4M3 reference needs Ada or newer')
    torch.manual_seed(rows)
    x = torch.randn(rows,256,device='cuda',dtype=torch.bfloat16)
    w = (torch.randn(256,256,device='cuda')*20).to(torch.float8_e4m3fn)
    s = torch.rand(2,2,device='cuda')*.002
    w1 = (torch.randn(8,256,256,device='cuda')*20).to(torch.float8_e4m3fn)
    w2 = (torch.randn(8,256,128,device='cuda')*20).to(torch.float8_e4m3fn)
    s1,s2 = torch.rand(8,2,2,device='cuda')*.002,torch.rand(8,2,1,device='cuda')*.002
    ids = torch.randint(0,8,(rows,3),device='cuda',dtype=torch.int32)
    scores = torch.randn(rows,3,device='cuda').softmax(-1)
    out = []
    for emulated in ('0','1'):
        monkeypatch.setenv('MINISGL_FP8_EMULATE',emulated)
        out.append((block_fp8_linear(x,w,s), fused_experts_impl(x.clone(),w1,w2,scores,ids,
            w1_scale=s1,w2_scale=s2)))
    for actual,expected in zip(out[1],out[0]):
        torch.testing.assert_close(actual,expected,rtol=.015,atol=.001)
