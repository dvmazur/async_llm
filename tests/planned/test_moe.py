from types import SimpleNamespace

import pytest
import torch

from minisgl.planned.moe import BoundExperts
from minisgl.moe.fused import fused_experts_impl,fused_topk

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def layer(fp8=False,on_input=False):
    shape1,shape2=(8,256,128),(8,128,128)
    if fp8:
        w1,w2=[(torch.randn(shape,device="cuda")*20).to(torch.float8_e4m3fn) for shape in (shape1,shape2)]
        s1=torch.rand(8,2,1,device="cuda")*.001+.001
        s2=torch.rand(8,1,1,device="cuda")*.001+.001
    else:
        w1,w2=[torch.randn(shape,device="cuda",dtype=torch.bfloat16)*.05 for shape in (shape1,shape2)]
        s1=s2=None
    return SimpleNamespace(gate_up_proj=w1,down_proj=w2,gate_up_proj_scale_inv=s1,down_proj_scale_inv=s2,
        top_k=3,tp_size=1,activation="silu",renormalize=True,apply_router_weight_on_input=on_input)


def check_alignment(b):
    n=int(b.num_padded)
    assert n%b.bm==0 and n<=b.aligned
    sorted_ids=b.sorted[:n].cpu()
    ids=b.ids.cpu().flatten()
    expected=set(torch.nonzero(ids>=0).flatten().tolist())
    real=sorted_ids[sorted_ids<b.n].tolist()
    assert len(real)==len(set(real))==len(expected)
    assert set(real)==expected
    experts=b.expert_ids[:n//b.bm].cpu()
    for block,e in enumerate(experts):
        tokens=sorted_ids[block*b.bm:(block+1)*b.bm]
        tokens=tokens[tokens<b.n]
        assert (ids[tokens]==int(e)).all()


@pytest.mark.parametrize("fp8",[False,True])
@pytest.mark.parametrize("on_input",[False,True])
@torch.inference_mode()
def test_router_alignment_masked_experts_match_old(fp8,on_input,record_property):
    torch.manual_seed(613)
    m=17
    l=layer(fp8,on_input)
    active=torch.arange(m,device="cuda")%3!=1
    b=BoundExperts(l,m,active)
    x=torch.randn(m,128,device="cuda",dtype=torch.bfloat16)*.2
    logits=torch.randn(m,8,device="cuda",dtype=x.dtype)
    wanted_w,wanted_ids=fused_topk(x,logits,3,True)
    wanted=fused_experts_impl(x[active].clone(),l.gate_up_proj,l.down_proj,wanted_w[active],wanted_ids[active],
        apply_router_weight_on_input=on_input,w1_scale=l.gate_up_proj_scale_inv,w2_scale=l.down_proj_scale_inv)
    x[~active]=float("nan");logits[~active]=float("nan")
    b.cache.fill_(float("nan"));b.second.fill_(float("nan"))
    out=torch.empty_like(x)
    b.run(x,logits,out)
    assert torch.equal(b.ids[active],wanted_ids[active])
    torch.testing.assert_close(b.weights[active],wanted_w[active],rtol=2e-6,atol=1e-7)
    assert (b.ids[~active]==-1).all() and (b.weights[~active]==0).all()
    check_alignment(b)
    torch.testing.assert_close(out[active],wanted,rtol=.03,atol=2e-4)
    error=(out[active].float()-wanted.float()).norm()/wanted.float().norm()
    record_property("output_relative_l2",float(error))
    assert error<.006,float(error)
    assert torch.equal(out[~active],torch.zeros_like(out[~active]))
    assert torch.isfinite(b.second).all()


@torch.inference_mode()
def test_one_moe_graph_dynamic_masks_routes_and_empty_experts():
    torch.manual_seed(614)
    m=17;l=layer(True)
    active=torch.ones(m,device="cuda",dtype=torch.bool)
    b=BoundExperts(l,m,active)
    x=torch.randn(m,128,device="cuda",dtype=torch.bfloat16)
    logits=torch.randn(m,8,device="cuda",dtype=x.dtype);out=torch.empty_like(x)
    b.run(x,logits,out);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):b.run(x,logits,out)
    for mode in ("all","holes","none","skew","all"):
        active.fill_(mode!="none")
        if mode=="holes":active[1::2]=False
        x.normal_();logits.normal_()
        if mode=="skew":logits[:,0]=20
        x[~active]=float("nan");logits[~active]=float("nan")
        b.run(x,logits,out)
        wanted=out.clone()
        b.cache.fill_(float("nan"));b.second.fill_(float("nan"));b.sorted.fill_(-1)
        graph.replay()
        check_alignment(b)
        torch.testing.assert_close(out,wanted,rtol=0,atol=0)
        assert torch.equal(out[~active],torch.zeros_like(out[~active]))


@torch.inference_mode()
def test_production_expert_count_and_topk_alignment_without_weights_checkpoint():
    torch.manual_seed(615)
    m,e,k=33,256,8
    l=SimpleNamespace(gate_up_proj=torch.randn(e,256,128,device="cuda",dtype=torch.bfloat16)*.04,
        down_proj=torch.randn(e,128,128,device="cuda",dtype=torch.bfloat16)*.04,
        gate_up_proj_scale_inv=None,down_proj_scale_inv=None,top_k=k,tp_size=1,
        activation="silu",renormalize=False,apply_router_weight_on_input=False)
    active=torch.arange(m,device="cuda")%4!=2
    b=BoundExperts(l,m,active)
    x=torch.randn(m,128,device="cuda",dtype=torch.bfloat16)*.2
    logits=torch.randn(m,e,device="cuda",dtype=x.dtype)
    w,ids=fused_topk(x,logits,k,False)
    wanted=fused_experts_impl(x[active].clone(),l.gate_up_proj,l.down_proj,w[active],ids[active])
    b.run(x,logits,output:=torch.empty_like(x))
    check_alignment(b)
    assert torch.equal(b.ids[active],ids[active])
    torch.testing.assert_close(output[active],wanted,rtol=.02,atol=2e-4)


@torch.inference_mode()
def test_torch_router_compatibility_remains_available_and_capturable(monkeypatch):
    import minisgl.moe.fused as old
    monkeypatch.setattr(old,"_use_torch_moe_fallback",lambda device:True)
    monkeypatch.setattr(old,"_vllm_custom_moe_ops",lambda:None)
    torch.manual_seed(616)
    m=17;l=layer(False)
    active=torch.arange(m,device="cuda")%2==0
    b=BoundExperts(l,m,active)
    assert b.router_backend=="torch"
    x=torch.randn(m,128,device="cuda",dtype=torch.bfloat16)*.2
    logits=torch.randn(m,8,device="cuda",dtype=x.dtype)
    w,ids=old.fused_topk(x,logits,3,True)
    b.route(logits)
    assert torch.equal(b.ids[active],ids[active])
    torch.testing.assert_close(b.weights[active],w[active],rtol=2e-6,atol=1e-7)
    out=torch.empty_like(x)
    b.run(x,logits,out);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):b.run(x,logits,out)
    for _ in range(2):
        logits.normal_();b.run(x,logits,out);want=out.clone()
        graph.replay();torch.testing.assert_close(out,want,rtol=0,atol=0)
