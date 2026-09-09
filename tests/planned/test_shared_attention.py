from dataclasses import replace

import pytest
import torch

from minisgl.planned.forward_plan import BlockState,PlanCapacity,PrefillRequest,DecodeRequest,prepare_forward
from minisgl.planned.attention_plan import KVBlock,AttentionCapacity,prepare_attention
from reference_shared_attention import KVPool,old_attention

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def setup(heads=4,*,image=True,padded=True):
    from minisgl.planned.shared_attention import BoundSharedAttention
    from minisgl.shared_cache.attention import SharedCacheAttention
    torch.manual_seed(61)
    lengths={0:2,3:3,4:3,5:2,6:0,7:0}
    kv={b:KVBlock(b,n,n,(b,8) if b==3 else (b,)) for b,n in lengths.items()}
    blocks={b:BlockState(b,b,bool(n),bool(n)) for b,n in lengths.items()}
    pf=[PrefillRequest((0,4),6,4),PrefillRequest((0,),3,2)]
    dec=[DecodeRequest((0,6,5,4),4),DecodeRequest((0,3,4,5),5),DecodeRequest((0,6),7)]
    f=prepare_forward(blocks,capacity=PlanCapacity(3,16,4,8,16) if padded else PlanCapacity(2,6,3,8,16),
                      prefill=pf,decode=dec)
    cap=AttentionCapacity(4,64,4,32,dummy_page=31)
    im={0:((0,0,0,0),(0,0,1,1),(0,1,0,1))} if image else {}
    ap=prepare_attention(kv,f,cap,prefill_mrope=im)
    kp,vp=[torch.randn(2,32,4,1,64,device="cuda",dtype=torch.bfloat16)*.2 for _ in range(2)]
    kp[:,31].zero_();vp[:,31].zero_()
    oldk,oldv=kp.clone(),vp.clone()
    bound=BoundSharedAttention(kp,vp,f,ap,num_qo_heads=heads,rotary_dim=32,rope_base=1e4,mrope_section=(6,5,5))
    old=SharedCacheAttention(KVPool(oldk,oldv),torch.empty(0,device="cuda"),heads,1,64,4,kp.dtype,kp.device,
                            rotary_dim=32,mrope_section=(6,5,5),rope_base=1e4)
    m=f.capacity.prefill_tokens+f.capacity.decode_workers
    # Deliberately strided Q/K/V input, like slices of the projected buffer.
    active=torch.tensor(f.rows.active,device="cuda")
    raw=torch.full((2,m,(2*heads+2)*64),float("nan"),device="cuda",dtype=kp.dtype)
    raw[:,active]=torch.randn(2,sum(f.rows.active),(2*heads+2)*64,device="cuda",dtype=kp.dtype)*.2
    q=raw[:,:,:2*heads*64].view(2,m,heads,128)[...,:64]
    k=raw[:,:,2*heads*64:(2*heads+1)*64].view(2,m,1,64)
    v=raw[:,:,(2*heads+1)*64:].view(2,m,1,64)
    out=torch.empty(2,m,heads,64,device=kp.device,dtype=kp.dtype)
    return kv,blocks,f,ap,im,kp,vp,oldk,oldv,bound,old,raw,q,k,v,out


@pytest.mark.parametrize("heads",[2,4])
@pytest.mark.parametrize("padded",[False,True])
@torch.inference_mode()
def test_four_wrappers_and_fused_rope_merge_match_old_layers(heads,padded,record_property):
    kv,blocks,f,ap,im,kp,vp,oldk,oldv,bound,old,raw,q,k,v,out=setup(heads,padded=padded)
    active=torch.tensor(f.rows.active,device="cuda")
    raw[:,~active]=float("nan")
    for layer in range(2):
        wanted=old_attention(old,kv,f,q[layer],k[layer],v[layer],layer,image_positions=im)
        bound.run(layer,q[layer],k[layer],v[layer],out[layer])
        got=out[layer,active]
        torch.testing.assert_close(got,wanted,rtol=.015,atol=7e-4)
        rel=(got.float()-wanted.float()).norm()/wanted.float().norm()
        record_property(f"layer{layer}_output_relative_l2",float(rel))
        assert rel<.004,float(rel)
        torch.testing.assert_close(kp,oldk,rtol=.008,atol=1e-4)
        torch.testing.assert_close(vp,oldv,rtol=0,atol=0)
        assert torch.equal(out[layer,~active],torch.zeros_like(out[layer,~active]))


@torch.inference_mode()
def test_single_shared_attention_graph_changes_topology_images_and_phases():
    kv,blocks,f,ap,im,kp,vp,oldk,oldv,bound,old,raw,q,k,v,out=setup()
    originals=(kp.clone(),vp.clone())
    def body():
        for layer in range(2):bound.run(layer,q[layer],k[layer],v[layer],out[layer])
    body();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):body()
    addresses=(bound.queries.data_ptr(),bound.partial.data_ptr(),bound.meta.merge_sources.data_ptr())
    for np,nd,image in [(2,3,True),(1,1,False),(0,3,False),(2,0,True),(0,0,False),(2,3,False)]:
        pf=f.prefill_requests[:np]
        if np==1:pf=(replace(pf[0],length=1,context=()),)
        dec=f.decode_requests[:nd]
        now=prepare_forward(blocks,capacity=f.capacity,prefill=pf,decode=dec)
        positions=im if image else {}
        plan=prepare_attention(kv,now,ap.capacity,prefill_mrope=positions)
        bound.prepare(now,plan)
        kp.copy_(originals[0]);vp.copy_(originals[1]);raw.normal_(std=.2)
        active=torch.tensor(now.rows.active,device="cuda")
        raw[:,~active]=float("nan")
        body()
        expected=(out.clone(),kp.clone(),vp.clone())
        kp.copy_(originals[0]);vp.copy_(originals[1])
        bound.partial.fill_(float("nan"));bound.lse.fill_(float("nan"));out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out,expected[0],rtol=0,atol=0)
        torch.testing.assert_close(kp,expected[1],rtol=0,atol=0)
        torch.testing.assert_close(vp,expected[2],rtol=0,atol=0)
        assert addresses==(bound.queries.data_ptr(),bound.partial.data_ptr(),bound.meta.merge_sources.data_ptr())


@torch.inference_mode()
def test_exact_and_capacity_shared_attention_on_identical_inputs():
    exact=setup(padded=False)
    padded=setup(padded=True)
    fe,fp=exact[2],padded[2]
    me=torch.tensor(fe.rows.active,device="cuda");mp=torch.tensor(fp.rows.active,device="cuda")
    # Generator consumes only actual rows, so both physical layouts carry the
    # same logical inputs and initial KV, not two merely similar random cases.
    assert torch.equal(exact[11][:,me],padded[11][:,mp])
    assert torch.equal(exact[5],padded[5])
    for data in (exact,padded):
        core=data[9];q,k,v,out=data[12:]
        for layer in range(2):core.run(layer,q[layer],k[layer],v[layer],out[layer])
    torch.testing.assert_close(exact[-1][:,me],padded[-1][:,mp],rtol=.015,atol=7e-4)
    torch.testing.assert_close(exact[5],padded[5],rtol=0,atol=0)
    torch.testing.assert_close(exact[6],padded[6],rtol=0,atol=0)
