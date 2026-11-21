"""H100 large-prefill tiling preserves every output bit and graph replay."""
import pytest
import torch
from minisgl.kernel.fp8 import block_fp8_linear

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(9,0),
                             reason='H100 exact tiling')

@pytest.mark.parametrize('rows',[255,256,257,513,1024])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32])
@pytest.mark.parametrize('n,k',[(256,384),(5120,6144)])
@torch.inference_mode()
def test_large_prefill_matches_original_tile_exactly(monkeypatch,rows,dtype,n,k):
    torch.manual_seed(rows)
    monkeypatch.setenv('MINISGL_FP8_BACKEND','triton')
    monkeypatch.setenv('MINISGL_FP8_EMULATE','0')
    x=torch.randn(rows,k,device='cuda',dtype=dtype)*.3
    weights=torch.randn(n,k,device='cuda').clamp(-3,3).to(torch.float8_e4m3fn)
    scales=torch.rand(n//128,k//128,device='cuda')*.03+.0001
    monkeypatch.setenv('MINISGL_FP8_TILING','reference');reference=block_fp8_linear(x,weights,scales)
    monkeypatch.setenv('MINISGL_FP8_TILING','h100_exact');actual=block_fp8_linear(x,weights,scales)
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=block_fp8_linear(x,weights,scales)
    graph.replay();torch.testing.assert_close(captured,reference,rtol=0,atol=0)

@torch.inference_mode()
def test_emulated_prefill_keeps_original_arithmetic(monkeypatch):
    monkeypatch.setenv('MINISGL_FP8_BACKEND','triton')
    monkeypatch.setenv('MINISGL_FP8_EMULATE','1');torch.manual_seed(188)
    x=torch.randn(257,384,device='cuda',dtype=torch.bfloat16)
    weights=torch.randn(256,384,device='cuda').to(torch.float8_e4m3fn)
    scales=torch.rand(2,3,device='cuda')*.01
    monkeypatch.setenv('MINISGL_FP8_TILING','reference');reference=block_fp8_linear(x,weights,scales)
    monkeypatch.setenv('MINISGL_FP8_TILING','h100_exact');actual=block_fp8_linear(x,weights,scales)
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
