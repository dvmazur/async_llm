from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from minisgl.planned.forward_plan import BlockState, PrefillRequest, DecodeRequest, PlanCapacity, prepare_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(x, weight, prior):
    n, c = x.shape
    k = weight.shape[-1]
    if prior is None: prior = torch.zeros(c,k,device=x.device,dtype=x.dtype)
    raw = torch.cat([prior, x.T], -1)
    acc = torch.zeros(c,n,device=x.device,dtype=torch.float32)
    for tap in range(k):
        acc += (raw[:,tap+1:tap+1+n] * weight[:,0,tap,None]).float()
    return F.silu(acc).to(x.dtype).T, raw[:,-k:]


@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
@pytest.mark.parametrize("lengths", [(1,), (1,2,7), (4,4,4), (0,)])
@torch.inference_mode()
def test_ragged_prefill_preserves_rounding_raw_windows_and_inactive_rows(dtype,lengths):
    from minisgl.planned.gdn_device import DevicePhase,DeviceRows
    from minisgl.planned.conv_device import BoundConv
    torch.manual_seed(953)
    cap=PlanCapacity(4,32,3,6,12)
    blocks={i:BlockState(i,i,i<3,i<3) for i in range(6)}
    reqs=[] if lengths==(0,) else [PrefillRequest((i,) if i!=1 else (),3+i,n) for i,n in enumerate(lengths)]
    plan=prepare_forward(blocks,capacity=cap,prefill=reqs)
    c,k=37,4
    pool=torch.randn(2,12,c,k,device='cuda',dtype=dtype)
    original=pool.clone()
    weight=torch.randn(c,1,k,device='cuda',dtype=dtype)
    x=torch.randn(35,c,device='cuda',dtype=dtype)
    x[sum(r.length for r in reqs):32].fill_(float('nan'))
    out=torch.full_like(x,float('nan'))
    rows,phase=DeviceRows(plan.rows,pool.device),DevicePhase(plan.prefill,pool.device)
    conv=BoundConv(pool,weight,phase,rows,cap,prefill=True)
    conv.new_window.fill_(float('nan'))
    conv.run(1,x,out)
    start=0
    for i,r in enumerate(reqs):
        slot=plan.prefill.prior_conv_slots[i]
        expected,window=reference(x[start:start+r.length],weight,None if slot<0 else original[1,slot])
        torch.testing.assert_close(out[start:start+r.length],expected,
                                   rtol=.008 if dtype==torch.bfloat16 else 2e-5,atol=1e-6)
        torch.testing.assert_close(conv.new_window[i],window,rtol=0,atol=0)
        start+=r.length
    assert torch.equal(out[start:32],torch.zeros_like(out[start:32]))
    torch.testing.assert_close(pool,original,rtol=0,atol=0)
    conv.publish(1)
    for i,r in enumerate(reqs): original[1,r.write_to]=conv.new_window[i]
    torch.testing.assert_close(pool,original,rtol=0,atol=0)


@torch.inference_mode()
def test_decode_conv_publish_is_fused_into_capture_and_replays():
    from minisgl.planned.gdn_device import DevicePhase,DeviceRows,BoundGDN
    from minisgl.planned.conv_device import BoundConv
    torch.manual_seed(954)
    cap=PlanCapacity(2,16,4,6,12)
    blocks={i:BlockState(i,i,i<6,i<6) for i in range(8)}
    reqs=[DecodeRequest((0,1,2),2),DecodeRequest((0,2,3),3),DecodeRequest((),6)]
    plan=prepare_forward(blocks,capacity=cap,decode=reqs)
    h,d,c,k=4,16,37,4
    pool=torch.randn(2,12,2,h,d,d,device='cuda')*.02
    cvpool=torch.randn(2,12,c,k,device='cuda',dtype=torch.bfloat16)
    original=cvpool.clone()
    phase,rows=DevicePhase(plan.decode,pool.device),DeviceRows(plan.rows,pool.device)
    bound=BoundGDN(pool,phase)
    weight=torch.randn(c,1,k,device='cuda',dtype=cvpool.dtype)
    conv=BoundConv(cvpool,weight,phase,rows,cap,prefill=False)
    x=torch.randn(20,c,device='cuda',dtype=cvpool.dtype)
    y=torch.empty_like(x)
    q,kk,v=[torch.randn(4,1,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    g=-torch.rand(4,1,h,device='cuda')*.2
    beta,alpha=torch.rand_like(g),g.exp()
    output=torch.empty_like(v)
    def body():
        conv.run(0,x,y)
        bound.compose(0)
        bound.decode(0,q,kk,v,g,beta,alpha,output,conv=conv)
    body();torch.cuda.synchronize()
    cvpool.copy_(original)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):body()
    for _ in range(3):
        cvpool.copy_(original);x.normal_()
        x[:16].fill_(float('nan'));x[19:].fill_(float('nan'))
        expected=original.clone()
        want=[]
        for i,r in enumerate(reqs):
            source=plan.decode.prior_conv_slots[i]
            value,window=reference(x[16+i:17+i],weight,None if source<0 else original[0,source])
            expected[0,r.write_to]=window
            want.append(value)
        graph.replay()
        torch.testing.assert_close(cvpool,expected,rtol=0,atol=0)
        torch.testing.assert_close(y[16:19],torch.cat(want),rtol=.008,atol=1e-6)
        assert torch.equal(y[19:],torch.zeros_like(y[19:]))
