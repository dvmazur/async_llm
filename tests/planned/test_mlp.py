from types import SimpleNamespace
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def layer(seed,monkeypatch,fp8):
    from minisgl.models.qwen3_5_moe import Qwen3_5MoeMLP
    import minisgl.distributed.info as dist
    from minisgl.distributed import DistributedInfo
    from minisgl.utils import torch_dtype
    monkeypatch.setattr(dist,"_TP_INFO",DistributedInfo(0,1))
    cfg=SimpleNamespace(num_experts=8,num_experts_per_tok=3,hidden_size=128,
        moe_intermediate_size=128,shared_expert_intermediate_size=128,norm_topk_prob=True,hidden_act="silu")
    with torch.device("meta"),torch_dtype(torch.bfloat16):l=Qwen3_5MoeMLP(cfg)
    gen=torch.Generator(device="cuda").manual_seed(seed)
    state={name:torch.randn(t.shape,device="cuda",dtype=t.dtype,generator=gen)*.05
           for name,t in l.state_dict().items()}
    l.load_state_dict(state)
    if fp8:
        for p in (l.shared_expert.gate_up_proj,l.shared_expert.down_proj):
            n,k=p.weight.shape
            p.weight=(torch.randn(n,k,device="cuda",generator=gen)*20).to(torch.float8_e4m3fn)
            p.weight_scale_inv=torch.rand(n//128,k//128,device="cuda",generator=gen)*.001+.001
        for name in ("gate_up_proj","down_proj"):
            w=getattr(l.experts,name)
            e,n,k=w.shape
            setattr(l.experts,name,(torch.randn(w.shape,device="cuda",generator=gen)*20).to(torch.float8_e4m3fn))
            setattr(l.experts,name+"_scale_inv",torch.rand(e,n//128,k//128,device="cuda",generator=gen)*.001+.001)
    return l


@pytest.mark.parametrize("fp8",[False,True])
@torch.inference_mode()
def test_full_qwen_moe_shared_and_routed_layers_reuse_workspace_and_replay(monkeypatch,fp8,record_property):
    from minisgl.planned.mlp import MLPWorkspace,BoundMoeMLP
    from minisgl.moe.fused import FusedMoe
    import minisgl.core as core
    ctx=core.Context(1);ctx.moe_backend=FusedMoe()
    monkeypatch.setattr(core,"_GLOBAL_CTX",ctx)
    layers=[layer(130+i,monkeypatch,fp8) for i in range(2)]
    m=17
    active=torch.arange(m,device="cuda")%3!=1
    workspace=MLPWorkspace(m,128,128,active,device="cuda",experts=8)
    bound=[BoundMoeMLP(l,workspace) for l in layers]
    assert bound[0].experts.cache.data_ptr()==bound[1].experts.cache.data_ptr()
    if fp8:assert bound[0].experts.xq.data_ptr()==bound[1].experts.yq.data_ptr()
    x=torch.randn(m,128,device="cuda",dtype=torch.bfloat16)*.2
    x[~active]=float("nan")
    outputs=[torch.empty_like(x) for _ in range(2)]
    wanted=layers[0].forward(x[active].clone())+x[active]
    wanted=layers[1].forward(wanted)
    def body():
        bound[0].run(x,outputs[0]);outputs[0].add_(x)
        bound[1].run(outputs[0],outputs[1])
    body()
    err=(outputs[1][active].float()-wanted.float()).norm()/wanted.float().norm()
    record_property("two_layers_relative_l2",float(err))
    assert err<.008,float(err)
    torch.testing.assert_close(outputs[1][active],wanted,rtol=.035,atol=2e-4)
    assert torch.equal(outputs[1][~active],torch.zeros_like(outputs[1][~active]))
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):body()
    for mode in ("all","holes","none","all"):
        active.fill_(mode!="none")
        if mode=="holes":active[::2]=False
        x.normal_(std=.2);x[~active]=float("nan")
        body();expected=outputs[1].clone()
        workspace.experts.tensors["cache"].fill_(float("nan"))
        workspace.shared.fill_(float("nan"));workspace.activated.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(outputs[1],expected,rtol=0,atol=0)
