from types import SimpleNamespace

import pytest
import torch

from test_shared_attention import setup
from reference_shared_attention import old_attention

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def layer(index,monkeypatch,fp8):
    import minisgl.distributed.info as dist
    from minisgl.distributed import DistributedInfo
    from minisgl.models.qwen3_5_attn import Qwen3_5Attention
    from minisgl.utils import torch_dtype
    monkeypatch.setattr(dist,"_TP_INFO",DistributedInfo(0,1))
    cfg=SimpleNamespace(num_qo_heads=4,num_kv_heads=1,head_dim=64,hidden_size=128,
        partial_rotary_factor=.5,rotary_config=SimpleNamespace(base=1e4),mrope_section=(6,5,5),rms_norm_eps=1e-6)
    with torch.device("meta"),torch_dtype(torch.bfloat16):l=Qwen3_5Attention(cfg,index)
    gen=torch.Generator(device="cuda").manual_seed(504+index)
    state={name:torch.randn(t.shape,device="cuda",dtype=t.dtype,generator=gen)*.075
           for name,t in l.state_dict().items()}
    l.load_state_dict(state)
    if fp8:
        for p in (l.qkv_proj,l.o_proj):
            n,k=p.weight.shape
            p.weight=(torch.randn(n,k,device="cuda",generator=gen)*20).to(torch.float8_e4m3fn)
            p.weight_scale_inv=torch.rand(n//128,k//128,device="cuda",generator=gen)*.001+.001
    return l


def old_layer(l,x,kv,f,old,im):
    from minisgl.models.qwen3_5_attn import _apply_output_gate
    mask=torch.tensor(f.rows.active,device=x.device)
    qkv=l.qkv_proj.forward(x[mask])
    qg,k,v=qkv.split([2*l.qo_dim,l.kv_dim,l.kv_dim],-1)
    qg=qg.view(-1,l.num_qo_heads,2*l.head_dim)
    q=l.q_norm.forward(qg[...,:l.head_dim].contiguous())
    gate=qg[...,l.head_dim:].contiguous()
    k=l.k_norm.forward(k.view(-1,l.num_kv_heads,l.head_dim))
    v=v.contiguous().view_as(k)
    physical=[]
    for value in (q,k,v):
        buf=torch.full((len(mask),*value.shape[1:]),float("nan"),device=x.device,dtype=x.dtype)
        buf[mask]=value
        physical.append(buf)
    out=old_attention(old,kv,f,*physical,l._kv_idx,image_positions=im)
    return l.o_proj.forward(_apply_output_gate(out,gate).reshape(-1,l.qo_dim))


@pytest.mark.parametrize("fp8",[False,True])
@pytest.mark.parametrize("padded",[False,True])
@torch.inference_mode()
def test_whole_attention_layer_exact_capacity_replay_and_pooled_fp8(monkeypatch,fp8,padded,record_property):
    from minisgl.planned.attention_layer import AttentionWorkspace,BoundAttentionLayer
    from minisgl.planned.gdn_device import DeviceRows
    kv,blocks,f,ap,im,kp,vp,oldk,oldv,core,old,raw,q,k,v,_=setup(padded=padded)
    layers=[layer(i,monkeypatch,fp8) for i in range(2)]
    rows=DeviceRows(f.rows,kp.device)
    workspace=AttentionWorkspace(core,rows.active)
    bound=[BoundAttentionLayer(l,workspace) for l in layers]
    if fp8:
        assert bound[0].input.quant.data_ptr()==bound[1].output.quant.data_ptr()
    x=torch.full((core.total,128),float("nan"),device=kp.device,dtype=kp.dtype)
    mask=torch.tensor(f.rows.active,device=kp.device)
    x[mask]=torch.randn(sum(f.rows.active),128,device=kp.device,dtype=kp.dtype)*.1
    out=torch.empty_like(x)
    for i,l in enumerate(layers):
        wanted=old_layer(l,x,kv,f,old,im)
        bound[i].run(x,out)
        error=(out[mask].float()-wanted.float()).norm()/wanted.float().norm()
        record_property(f"layer{i}_output_relative_l2",float(error))
        assert error<.015,float(error)
        torch.testing.assert_close(out[mask],wanted,rtol=.05,atol=5e-4)
        torch.testing.assert_close(kp,oldk,rtol=.025,atol=.004)
        torch.testing.assert_close(vp,oldv,rtol=0,atol=0)
        assert torch.equal(out[~mask],torch.zeros_like(out[~mask]))
    before=(kp.clone(),vp.clone())
    bound[0].run(x,out);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):bound[0].run(x,out)
    for _ in range(3):
        x.normal_(std=.1);x[~mask]=float("nan")
        kp.copy_(before[0]);vp.copy_(before[1]);bound[0].run(x,out)
        expected=(out.clone(),kp.clone(),vp.clone())
        kp.copy_(before[0]);vp.copy_(before[1]);workspace.raw.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out,expected[0],rtol=0,atol=0)
        torch.testing.assert_close(kp,expected[1],rtol=0,atol=0)
        torch.testing.assert_close(vp,expected[2],rtol=0,atol=0)
