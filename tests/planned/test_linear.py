from types import SimpleNamespace
import pytest
import torch

from minisgl.planned.linear import LinearWorkspace,BoundLinear
from minisgl.kernel.fp8 import block_fp8_linear,quantize_fp8_groups

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def weights(n,k):
    w=(torch.randn(n,k,device="cuda")*20).to(torch.float8_e4m3fn)
    s=torch.rand(n//128,k//128,device="cuda")*.001+.001
    return SimpleNamespace(weight=w,weight_scale_inv=s,bias=None)


@pytest.mark.parametrize("rows",[1,8,33,137])
@torch.inference_mode()
def test_bound_fp8_linear_and_quant_bytes_match_existing_with_active_holes(rows):
    torch.manual_seed(328)
    p=weights(256,384)
    active=torch.arange(rows,device="cuda")%3!=1
    workspace=LinearWorkspace(rows,512,device="cuda")
    bound=BoundLinear(p,rows,dtype=torch.bfloat16,workspace=workspace,active=active)
    x=torch.randn(rows,384,device="cuda",dtype=torch.bfloat16)
    x[~active]=float("nan")
    clean=x.clone();clean[~active]=0
    q,s=quantize_fp8_groups(clean)
    wanted=block_fp8_linear(clean,p.weight,p.weight_scale_inv)
    out=torch.empty_like(wanted)
    bound.run(x,out)
    torch.testing.assert_close(out,wanted,rtol=0,atol=0)
    assert torch.equal(bound.quant.view(torch.uint8),q.view(torch.uint8))
    torch.testing.assert_close(bound.input_scales,s,rtol=0,atol=0)
    assert torch.equal(out[~active],torch.zeros_like(out[~active]))


@torch.inference_mode()
def test_shared_fp8_arena_two_linears_and_graph_replay():
    torch.manual_seed(327)
    m=20
    a,b=weights(256,128),weights(128,256)
    workspace=LinearWorkspace(m,256,device="cuda")
    aa,bb=[BoundLinear(p,m,dtype=torch.bfloat16,workspace=workspace) for p in (a,b)]
    assert aa.quant.data_ptr()==bb.quant.data_ptr()
    assert aa.input_scales.data_ptr()==bb.input_scales.data_ptr()
    x=torch.randn(m,128,device="cuda",dtype=torch.bfloat16)
    mid=torch.empty(m,256,device="cuda",dtype=x.dtype);out=torch.empty_like(x)
    def body():aa.run(x,mid);bb.run(mid,out)
    body();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):body()
    for _ in range(4):
        x.normal_()
        want=block_fp8_linear(block_fp8_linear(x,a.weight,a.weight_scale_inv),b.weight,b.weight_scale_inv)
        graph.replay()
        torch.testing.assert_close(out,want,rtol=0,atol=0)
    torch.cuda.synchronize()
    allocated=torch.cuda.memory_allocated()
    body();torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==allocated
