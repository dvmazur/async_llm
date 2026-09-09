"""Native capacity/replan compatibility; not the complete shared-attention layer."""
import math
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def reference(q,k,v,lengths,pages,last,causal):
    outputs,lses=[],[]
    offset=0
    for n,ids,end in zip(lengths,pages,last):
        if not n:continue
        keys=torch.cat([k[i] if j<len(ids)-1 else k[i,:end] for j,i in enumerate(ids)])
        vals=torch.cat([v[i] if j<len(ids)-1 else v[i,:end] for j,i in enumerate(ids)])
        keys=keys.repeat_interleave(q.shape[1]//keys.shape[1],dim=1).float()
        vals=vals.repeat_interleave(q.shape[1]//vals.shape[1],dim=1).float()
        scores=torch.einsum("qhd,khd->hqk",q[offset:offset+n].float(),keys)*q.shape[-1]**-.5
        if causal:
            iq=torch.arange(n,device=q.device)+keys.shape[0]-n
            ik=torch.arange(keys.shape[0],device=q.device)
            scores.masked_fill_(ik[None,None,:]>iq[None,:,None],-torch.inf)
        outputs.append(torch.einsum("hqk,khd->qhd",scores.softmax(-1),vals).to(q.dtype))
        # FlashInfer cascade state uses log2(sum(exp(scores))), not natural LSE.
        lses.append(scores.logsumexp(-1).t()/math.log(2.))
        offset+=n
    return torch.cat(outputs) if outputs else q[:0],torch.cat(lses) if lses else q.new_empty(0,q.shape[1],dtype=torch.float32)


@pytest.mark.parametrize("causal",[False,True])
@torch.inference_mode()
def test_capacity_native_recipe_is_stable_with_zero_requests_and_graph(causal,record_property):
    from minisgl.planned.paged_attention import BoundPagedAttention
    torch.manual_seed(939)
    op=BoundPagedAttention(request_capacity=4,row_capacity=256,page_ref_capacity=128,
        num_qo_heads=4,num_kv_heads=1,head_dim=64,page_size=16,page_pool_capacity=32,device="cuda",causal=causal)
    q=torch.randn(256,4,64,device="cuda",dtype=torch.bfloat16)*.2
    k,v=[torch.randn(32,16,1,64,device="cuda",dtype=q.dtype)*.2 for _ in range(2)]
    out=torch.empty_like(q);lse=torch.empty(256,4,device="cuda")
    pages=[list(range(12)),[12,13,14,15],list(range(16,28)),[0]]
    last=[16,16,16,1]
    lengths=[91,64,101,0]
    op.plan(lengths,pages,last)
    op.run_capacity(q,k,v,out,lse);torch.cuda.synchronize()
    usual,usual_lse=op.wrapper.run(q,(k,v),return_lse=True)
    torch.testing.assert_close(out,usual,rtol=0,atol=0)
    torch.testing.assert_close(lse,usual_lse,rtol=0,atol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):op.run_capacity(q,k,v,out,lse)
    recipe=op._recipe
    for lengths in ([91,64,101,0],[0,0,0,0],[1,1,1,0],[36,0,138,0],[0,1,0,1],[91,64,101,0]):
        op.plan(lengths,pages,last)
        assert op._recipe==recipe
        n=sum(lengths)
        q.normal_(std=.2);q[n:]=float("nan")
        want,want_lse=reference(q,k,v,lengths,pages,last,causal)
        if n:
            # Measure the unchanged backend's own BF16 rounding against FP32.
            # The adapter may not add error; don't widen a guessed tolerance.
            usual,usual_lse=op.wrapper.run(q[:n],(k,v),return_lse=True)
        out.fill_(float("nan"));lse.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out[:n],want,rtol=.015,atol=7e-4)
        if n:
            torch.testing.assert_close(out[:n],usual,rtol=0,atol=0)
            torch.testing.assert_close(lse[:n],usual_lse,rtol=0,atol=0)
            error=(out[:n].float()-want.float()).norm()/want.float().norm()
            record_property(f"rows_{n}_relative_l2",float(error))
            baseline_error=(usual.float()-want.float()).norm()/want.float().norm()
            record_property(f"rows_{n}_upstream_relative_l2",float(baseline_error))
            assert error<=baseline_error+1e-7
            lse_error=(lse[:n]-want_lse).abs().max()
            record_property(f"rows_{n}_lse_max_abs",float(lse_error))
            assert lse_error<=(usual_lse-want_lse).abs().max()+1e-7
        assert torch.isnan(out[n:]).all()
    with pytest.raises(ValueError,match="query rows"):
        op.plan([257,0,0,0],pages,last)
    with pytest.raises(RuntimeError,match="valid capacity plan"):
        op.run_capacity(q,k,v,out,lse)
    with pytest.raises(ValueError,match="page references"):
        op.plan([1,0,0,0],[[32],[0],[0],[0]],[1,1,1,1])


@torch.inference_mode()
def test_changed_native_recipe_rejected_before_run(monkeypatch):
    from minisgl.planned.paged_attention import BoundPagedAttention,AttentionRecipeChanged
    op=BoundPagedAttention(request_capacity=2,row_capacity=8,page_ref_capacity=8,
        num_qo_heads=4,num_kv_heads=1,head_dim=64,page_size=1,page_pool_capacity=8,device="cuda")
    op.plan([4,4],[[1],[2]],[1,1])
    original=op._plan_native
    def changed(*args,**kwargs):
        original(*args,**kwargs)
        altered=list(op.wrapper._plan_info)
        altered[3]+=16
        op.wrapper._plan_info=altered
    monkeypatch.setattr(op,"_plan_native",changed)
    with pytest.raises(AttentionRecipeChanged,match="scalar recipe changed"):
        op.plan([1,0],[[1],[2]],[1,1])
    assert not op._ready


@pytest.mark.parametrize("decode,heads",[(False,2),(False,4),(True,2),(True,4)])
@torch.inference_mode()
def test_fast_plan_matches_unmodified_upstream_metadata_and_replay(decode,heads,monkeypatch):
    from minisgl.planned.paged_attention import BoundPagedAttention
    kw=dict(request_capacity=8,row_capacity=8 if decode else 256,page_ref_capacity=128,
        num_qo_heads=heads,num_kv_heads=1,head_dim=64,page_size=16,page_pool_capacity=64,
        device='cuda',decode=decode)
    fast=BoundPagedAttention(**kw)
    old=BoundPagedAttention(**kw,fast_plan=False)
    q=torch.randn(kw['row_capacity'],heads,64,device='cuda',dtype=torch.bfloat16)
    k,v=[torch.randn(64,16,1,64,device='cuda',dtype=q.dtype) for _ in range(2)]
    out=torch.empty_like(q);lse=torch.empty(q.shape[:2],device='cuda')
    graph=None
    for turn in range(8):
        lengths=([1]*3+[0]*5 if decode else [3,9,1]+[0]*5) if turn%3 else [0]*8
        pages=[tuple(range(1,1+(turn+1)*2))]+[(32+i,) for i in range(7)]
        last=[1+turn]*8
        if not decode:last[1]=16
        fast.plan(lengths,pages,last);old.plan(lengths,pages,last)
        assert fast._recipe==old._recipe
        for attr in ('qo','indptr','last','indices'):
            a,b=getattr(fast,attr),getattr(old,attr)
            if a is None:continue
            n=sum(map(len,pages)) if attr=='indices' else a.numel()
            torch.testing.assert_close(a[:n],b[:n],rtol=0,atol=0)
        if not decode or fast.tensor_cores:
            assert fast.wrapper._max_q_len==old.wrapper._max_q_len
            assert fast.wrapper._max_kv_len==old.wrapper._max_kv_len
            assert fast.wrapper._qo_indptr_last==old.wrapper._qo_indptr_last
        if graph is None:
            fast.run_capacity(q,k,v,out,lse);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fast.run_capacity(q,k,v,out,lse)
            if not decode or fast.tensor_cores:
                def forbidden(*a,**kw):raise AssertionError('generic wrapper.plan reentered')
                monkeypatch.setattr(fast.wrapper,'plan',forbidden)
        graph.replay()
        n=sum(lengths)
        if n:
            expected,expected_lse=old.wrapper.run(q[:n],(k,v),return_lse=True,enable_pdl=False)
            torch.testing.assert_close(out[:n],expected,rtol=0,atol=0)
            torch.testing.assert_close(lse[:n],expected_lse,rtol=0,atol=0)


@pytest.mark.parametrize("decode,heads",[(False,2),(False,4),(True,2),(True,4)])
@torch.inference_mode()
def test_replay_variable_kv_lengths_crossing_split_boundaries(decode,heads):
    from minisgl.planned.paged_attention import BoundPagedAttention
    qrows=4 if decode else 32
    op=BoundPagedAttention(request_capacity=4,row_capacity=qrows,page_ref_capacity=512,
        num_qo_heads=heads,num_kv_heads=1,head_dim=64,page_size=16,page_pool_capacity=257,
        device="cuda",decode=decode)
    torch.manual_seed(940)
    q=torch.randn(qrows,heads,64,device="cuda",dtype=torch.bfloat16)*.2
    k,v=[torch.randn(257,16,1,64,device="cuda",dtype=q.dtype)*.2 for _ in range(2)]
    out=torch.empty_like(q);lse=torch.empty(qrows,heads,device="cuda")
    op.plan([1]*4,[[0],[256],[256],[256]],[1]*4)
    op.run_capacity(q,k,v,out,lse);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):op.run_capacity(q,k,v,out,lse)
    recipe=op._recipe
    for pages,last in ((1,1),(8,16),(256,1),(32,7),(2,3),(1,1)):
        op.plan([1]*4,[list(range(pages)),[256],[256],[256]],[last,1,1,1])
        assert op._recipe==recipe
        q.normal_(std=.2)
        wanted,wanted_lse=op.wrapper.run(q[:4],(k,v),return_lse=True,enable_pdl=False)
        graph.replay()
        torch.testing.assert_close(out[:4],wanted,rtol=0,atol=0)
        torch.testing.assert_close(lse[:4],wanted_lse,rtol=0,atol=0)


@torch.inference_mode()
def test_real_a3b_attention_workspace_is_queried_not_assumed_32mib(record_property):
    from minisgl.planned.paged_attention import BoundPagedAttention
    op=BoundPagedAttention(request_capacity=128,row_capacity=2048,page_ref_capacity=4096,
        num_qo_heads=16,num_kv_heads=2,head_dim=256,page_size=16,page_pool_capacity=16,device="cuda")
    assert op.required_float_bytes>32*2**20
    record_property("required_float_bytes",op.required_float_bytes)
    assert op.wrapper._float_workspace_buffer.numel()>=op.required_float_bytes
    lengths=[2]+[0]*127;pages=[[1]]+[[0]]*127;last=[2]+[1]*127
    op.plan(lengths,pages,last)
    q=torch.randn(2048,16,256,device="cuda",dtype=torch.bfloat16)*.1
    k,v=[torch.randn(16,16,2,256,device="cuda",dtype=q.dtype)*.1 for _ in range(2)]
    out=torch.empty_like(q);lse=torch.empty(2048,16,device="cuda")
    op.run_capacity(q,k,v,out,lse);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):op.run_capacity(q,k,v,out,lse)
    addresses=op._addresses
    for n in (0,2,13,1):
        op.plan([n]+[0]*127,pages,[16]+[1]*127)
        assert op._addresses==addresses
        if n:expected,expected_lse=op.wrapper.run(q[:n],(k,v),return_lse=True)
        graph.replay()
        if n:
            torch.testing.assert_close(out[:n],expected,rtol=0,atol=0)
            torch.testing.assert_close(lse[:n],expected_lse,rtol=0,atol=0)
    with pytest.raises(RuntimeError,match="rebind"):op.bind_workspace(op.wrapper._float_workspace_buffer)


@torch.inference_mode()
def test_tensor_core_decode_inactive_capacity_does_not_change_split_kv():
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from minisgl.planned.paged_attention import BoundPagedAttention
    torch.manual_seed(731)
    r,h,hk,d,p=256,16,2,256,16
    op=BoundPagedAttention(request_capacity=r,row_capacity=r,page_ref_capacity=1024,
        num_qo_heads=h,num_kv_heads=hk,head_dim=d,page_size=p,page_pool_capacity=32,device="cuda",decode=True)
    # 137-token image segment crosses the 128-token split boundary; the other
    # segments do not. Dummy capacity must not count towards native Q workload.
    pages=[(1,2,3),tuple(range(4,13)),(13,14),(15,)]*2+[(0,)]*(r-8)
    last=[1,9,5,1]*2+[1]*(r-8)
    op.plan([1]*8+[0]*(r-8),pages,last)
    assert op.qo.cpu().tolist()==list(range(9))+[8]*(r-8)
    assert op.actual_rows==8
    k,v=[torch.randn(32,p,hk,d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    q=torch.randn(r,h,d,device="cuda",dtype=torch.bfloat16)
    out=torch.empty_like(q);lse=torch.empty(r,h,device="cuda")
    reference=BatchDecodeWithPagedKVCacheWrapper(torch.empty(128<<20,device="cuda",dtype=torch.uint8),
        kv_layout="NHD",use_tensor_cores=True,backend="fa2")
    ptr=[0];indices=[]
    for ids in pages[:8]:indices.extend(ids);ptr.append(len(indices))
    reference.plan(torch.tensor(ptr,dtype=torch.int32),torch.tensor(indices,dtype=torch.int32),
        torch.tensor(last[:8],dtype=torch.int32),h,hk,d,p,data_type=torch.bfloat16,
        q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,pos_encoding_mode="NONE",sm_scale=d**-.5)
    wanted,wanted_lse=reference.run(q[:8],(k,v),return_lse=True)
    op.run_capacity(q,k,v,out,lse)
    torch.testing.assert_close(out[:8],wanted,rtol=0,atol=0)
    torch.testing.assert_close(lse[:8],wanted_lse,rtol=0,atol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):op.run_capacity(q,k,v,out,lse)
    recipe=op._recipe
    op.plan([0]*r,[(0,)]*r,[1]*r);graph.replay()
    op.plan([1]*8+[0]*(r-8),pages,last);graph.replay()
    assert op._recipe==recipe
    torch.testing.assert_close(out[:8],wanted,rtol=0,atol=0)
