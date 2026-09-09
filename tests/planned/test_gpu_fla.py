"""Bounded augmented FLA parity with unmodified installed FLA, no checkpoint."""
from dataclasses import replace

import pytest
import torch

from minisgl.planned.forward_plan import BlockState,PrefillRequest,PlanCapacity,prepare_forward

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def setup(dim=32,heads=4,key_heads=2):
    from minisgl.planned.gdn_device import DeviceRows,DevicePhase,BoundGDN
    from minisgl.planned.fla_device import BoundFLA
    torch.manual_seed(981)
    cap=PlanCapacity(4,512,0,6,16)
    blocks={i:BlockState(i,i,i<7,i<7) for i in range(10)}
    requests=[PrefillRequest((0,1),6,36),PrefillRequest((0,1),7,138),PrefillRequest((0,),8,91)]
    plan=prepare_forward(blocks,capacity=cap,prefill=requests)
    pool=torch.randn(2,16,2,heads,dim,dim,device='cuda')*.025
    pool[:,:,0]+=torch.eye(dim,device='cuda')*.8
    pool[:,7:].fill_(float('nan'))
    rows,phase=DeviceRows(plan.rows,pool.device),DevicePhase(plan.prefill,pool.device)
    compose=BoundGDN(pool,phase)
    fla=BoundFLA(pool,phase,rows,cap,key_heads=key_heads)
    # Strided q/k/v match the shared projected QKV storage of a real layer.
    raw=torch.randn(1,cap.prefill_tokens,(2*key_heads+heads)*dim,device='cuda',dtype=torch.bfloat16)
    q,k,v=torch.split(raw,[key_heads*dim,key_heads*dim,heads*dim],-1)
    q=q.view(1,cap.prefill_tokens,key_heads,dim)
    k=k.view_as(q)
    v=v.view(1,cap.prefill_tokens,heads,dim)
    g=-torch.rand(1,cap.prefill_tokens,heads,device='cuda')*.1
    beta=torch.rand_like(g).to(torch.bfloat16)
    output=torch.empty_like(v,memory_format=torch.contiguous_format)
    return cap,blocks,requests,pool,rows,phase,compose,fla,raw,q,k,v,g,beta,output


def reference(pool,initial,requests,blocks,q,k,v,g,beta,layer):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    heads,d=pool.shape[3],pool.shape[-1]
    wanted=pool.clone()
    outputs=[]
    start=0
    for worker,r in enumerate(requests):
        target=blocks[r.write_to]
        if target.populated: A,B=pool[layer,target.slot]
        else:
            A=torch.eye(d,device=pool.device).expand(heads,d,d)
            B=torch.zeros_like(A)
        init=torch.cat([initial[worker],A,B],dim=-2)[None]
        val=v[:,start:start+r.length].contiguous()
        aug=torch.cat([val,torch.zeros_like(val),val],-1)
        o,ht=chunk_gated_delta_rule(q[:,start:start+r.length].contiguous(),
            k[:,start:start+r.length].contiguous(),aug,g[:,start:start+r.length].contiguous(),
            beta[:,start:start+r.length].contiguous(),initial_state=init,
            output_final_state=True,state_v_first=True,use_qk_l2norm_in_kernel=True)
        outputs.append(o[...,:d])
        wanted[layer,target.slot,0]=ht[0,:,d:2*d]
        wanted[layer,target.slot,1]=ht[0,:,2*d:]
        start+=r.length
    return torch.cat(outputs,1) if outputs else v[:,:0],wanted


@pytest.mark.parametrize("dim,heads,key_heads",[(32,4,2),(128,32,16)])
@torch.inference_mode()
def test_augmented_prefill_matches_fla_and_writes_only_targets(dim,heads,key_heads,record_property):
    cap,blocks,reqs,pool,rows,phase,compose,fla,raw,q,k,v,g,beta,out=setup(dim,heads,key_heads)
    before=pool.clone()
    compose.compose(1)
    want,want_pool=reference(before,compose.initial,reqs,blocks,q,k,v,g,beta,1)
    n=sum(r.length for r in reqs)
    raw[:,n:].fill_(float('nan'))
    fla.solved.fill_(float('nan'))  # producer must cover upper blocks too
    fla.run(1,q,k,v,g,beta,compose.initial,out)
    torch.testing.assert_close(out[:,:n],want,rtol=.035,atol=.004)
    rel=(out[:,:n].float()-want.float()).norm()/want.float().norm().clamp_min(1e-9)
    record_property('output_relative_l2',float(rel))
    record_property('output_max_abs',float((out[:,:n].float()-want.float()).abs().max()))
    written=torch.stack([pool[1,r.write_to] for r in reqs])
    wanted_written=torch.stack([want_pool[1,r.write_to] for r in reqs])
    record_property('affine_relative_l2',float((written-wanted_written).norm()/wanted_written.norm()))
    assert rel < .006, float(rel)
    torch.testing.assert_close(pool,want_pool,rtol=.025,atol=.003,equal_nan=True)
    assert torch.equal(out[:,n:],torch.zeros_like(out[:,n:]))
    assert not hasattr(fla,'final_state')


@torch.inference_mode()
def test_one_graph_with_different_chunk_counts_and_empty_requests():
    cap,blocks,reqs,pool,rows,phase,compose,fla,raw,q,k,v,g,beta,out=setup()
    original=pool.clone()
    def body():
        compose.compose(0)
        fla.run(0,q,k,v,g,beta,compose.initial,out)
    body();torch.cuda.synchronize()
    pool.copy_(original)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):body()
    addresses=(pool.data_ptr(),fla.hstates.data_ptr(),rows.chunk_indices.data_ptr())
    for lengths in [(64,64,137),(1,),(),(36,138,91),(63,64,65)]:
        current=[PrefillRequest(r.context,r.write_to,n) for r,n in zip(reqs,lengths)]
        plan=prepare_forward(blocks,capacity=cap,prefill=current)
        rows.upload(plan.rows);phase.upload(plan.prefill)
        pool.copy_(original);raw.normal_()
        compose.compose(0)
        wanted,wanted_pool=reference(original,compose.initial,current,blocks,q,k,v,g,beta,0)
        n=sum(lengths)
        raw[:,n:].fill_(float('nan'))
        graph.replay()
        torch.testing.assert_close(out[:,:n],wanted,rtol=.035,atol=.004)
        torch.testing.assert_close(pool,wanted_pool,rtol=.025,atol=.003,equal_nan=True)
        assert torch.equal(out[:,n:],torch.zeros_like(out[:,n:]))
    assert addresses==(pool.data_ptr(),fla.hstates.data_ptr(),rows.chunk_indices.data_ptr())


@torch.inference_mode()
def test_prefill_final_state_store_also_publishes_existing_conv_window():
    from minisgl.planned.conv_device import BoundConv
    cap,blocks,reqs,pool,rows,phase,compose,fla,raw,q,k,v,g,beta,out=setup()
    cvpool=torch.randn(2,16,37,4,device='cuda',dtype=torch.bfloat16)
    weight=torch.randn(37,1,4,device='cuda',dtype=cvpool.dtype)
    cv=BoundConv(cvpool,weight,phase,rows,cap,prefill=True)
    cv.new_window.normal_()
    cv.new_window[len(reqs):].fill_(float('nan'))
    original=cvpool.clone()
    expected=original.clone()
    for i,r in enumerate(reqs):expected[1,r.write_to]=cv.new_window[i]
    compose.compose(1)
    fla.run(1,q,k,v,g,beta,compose.initial,out,conv=cv)
    torch.cuda.synchronize()
    torch.testing.assert_close(cvpool,expected,rtol=0,atol=0)
    before=torch.cuda.memory_allocated()
    fla.run(1,q,k,v,g,beta,compose.initial,out,conv=cv)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==before
