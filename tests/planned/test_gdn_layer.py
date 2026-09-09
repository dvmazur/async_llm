"""Whole GDN attention sublayer vs untouched old layer, not only its kernels."""
from dataclasses import replace

import pytest
import torch

from minisgl.planned.forward_plan import BlockState,DecodeRequest,PrefillRequest,PlanCapacity,prepare_forward
from reference_gdn_layer import OldState,old_mixed,raw_window_reference

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def make_layer(index,dim=32,heads=4,key_heads=2,hidden=128):
    from minisgl.models.config import ModelConfig,RotaryConfig
    from minisgl.models.qwen3_5_delta import Qwen3_5GatedDeltaNet
    from minisgl.utils import torch_dtype
    cfg=ModelConfig(num_layers=2,num_qo_heads=2,num_kv_heads=1,head_dim=dim,
        hidden_size=hidden,vocab_size=32,intermediate_size=64,rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(dim,dim,1024,1e4,None),hidden_act="silu",
        tie_word_embeddings=False,num_experts=0,num_experts_per_tok=0,moe_intermediate_size=0,
        shared_expert_intermediate_size=0,norm_topk_prob=False,model_type="qwen3_5",
        architectures=["Qwen3_5ForCausalLM"],layer_types=("linear_attention",)*2,
        linear_num_key_heads=key_heads,linear_num_value_heads=heads,
        linear_key_head_dim=dim,linear_value_head_dim=dim,linear_conv_kernel_dim=4)
    with torch.device("meta"),torch_dtype(torch.bfloat16):layer=Qwen3_5GatedDeltaNet(cfg,index)
    gen=torch.Generator(device="cuda").manual_seed(812+index)
    params={}
    for name,t in layer.state_dict().items():
        val=torch.randn(t.shape,device="cuda",dtype=t.dtype,generator=gen)*.075
        if name=="A_log":val.zero_()
        if name=="norm.weight":val.fill_(1.)
        params[name]=val
    layer.load_state_dict(params)
    return layer


def setup(padded=True,mode="mixed"):
    from minisgl.planned.pools import GDNPool,PoolShape
    from minisgl.planned.gdn_layer import GDNWorkspace,BoundGDNLayer
    torch.manual_seed(124)
    pool=GDNPool(PoolShape(2,16,4,32,256,4),device="cuda")
    pool.affine.normal_(std=.015)
    pool.affine[:,:,0]+=torch.eye(32,device="cuda")*.85
    pool.conv.normal_(std=.075)
    blocks={i:BlockState(i,i,i<6,i<6) for i in range(10)}
    pool.affine[:,6:].fill_(float("nan"));pool.conv[:,6:].fill_(float("nan"))
    # PF reads old D (block 4); D reads newly appended PF block 6 and its peer 5.
    pf=[PrefillRequest((0,1,4),6,36),PrefillRequest((0,1),3,65)] if mode!="decode" else []
    dec=[DecodeRequest((0,6,5,4),4),DecodeRequest((0,3,4,5),5),DecodeRequest((0,6),7)] if mode!="prefill" else []
    p=sum(r.length for r in pf)
    cap=PlanCapacity((4 if padded else len(pf)) if pf else 0,(192 if padded else p) if pf else 0,
                     (5 if padded else len(dec)) if dec else 0,6,16)
    plan=prepare_forward(blocks,capacity=cap,prefill=pf,decode=dec)
    ws=GDNWorkspace(pool,plan,key_heads=2)
    layers=[make_layer(i) for i in range(2)]
    bound=[BoundGDNLayer(l,ws) for l in layers]
    x=torch.randn(p+len(dec),128,device="cuda",dtype=torch.bfloat16)*.1
    actual=torch.full((ws.total,128),float("nan"),device="cuda",dtype=x.dtype)
    actual[:p]=x[:p];actual[cap.prefill_tokens:cap.prefill_tokens+len(dec)]=x[p:]
    return pool,blocks,pf,dec,plan,ws,layers,bound,x,actual


@pytest.mark.parametrize("mode",["prefill","decode","mixed"])
@pytest.mark.parametrize("padded",[False,True])
@torch.inference_mode()
def test_complete_gdn_layers_exact_and_capacity_against_old(mode,padded,record_property):
    pool,blocks,pf,dec,plan,ws,layers,bound,x,actual=setup(padded,mode)
    state=OldState(pool,blocks)
    old_pool=pool.affine.clone();old_conv=pool.conv.clone()
    n=sum(r.length for r in pf)
    out=torch.empty_like(actual)
    for i,layer in enumerate(layers):
        expected=old_mixed(layer,x,state,pf,dec)
        bound[i].run(actual,out)
        got=torch.cat((out[:n],out[plan.capacity.prefill_tokens:plan.capacity.prefill_tokens+len(dec)]))
        rel=(got.float()-expected.float()).norm()/expected.float().norm()
        record_property(f"layer{i}_output_relative_l2",float(rel))
        assert rel<.008,float(rel)
        torch.testing.assert_close(got,expected,rtol=.04,atol=2e-4)
        for r in (*pf,*dec):
            b=r.write_to
            A,B=state.affine[i,b]
            want=torch.stack((A[0],B[0]))
            err=(pool.affine[i,b]-want).norm()/want.norm()
            record_property(f"layer{i}_block{b}_affine_relative_l2",float(err))
            assert err<.006,float(err)
            torch.testing.assert_close(pool.affine[i,b],want,rtol=.035,atol=5e-4)
            torch.testing.assert_close(pool.conv[i,b],state.conv[i,b][0],rtol=0,atol=0)
        mask=torch.tensor(plan.rows.active,device=out.device)
        assert torch.equal(out[~mask],torch.zeros_like(out[~mask]))
        # Feed different weights the next layer; same scratch, separate pool slice.
        x=x+expected
        actual.add_(out)
    untouched=set(range(pool.shape.slots))-{r.write_to for r in (*pf,*dec)}
    for b in untouched:
        torch.testing.assert_close(pool.affine[:,b],old_pool[:,b],rtol=0,atol=0,equal_nan=True)
        torch.testing.assert_close(pool.conv[:,b],old_conv[:,b],rtol=0,atol=0,equal_nan=True)


@torch.inference_mode()
def test_gates_match_original_softplus_and_phase_rounding():
    import triton
    from minisgl.planned.gdn_layer_kernels import gates
    from minisgl.models.qwen3_5_delta import _gdn_gates_eager
    a=torch.linspace(-40,40,32,device="cuda").to(torch.bfloat16).view(8,4)
    b=a.flip(0).contiguous();al=torch.randn(4,device="cuda");dt=torch.randn(4,device="cuda")
    active=torch.tensor([1,1,1,0,0,1,1,0],device="cuda",dtype=torch.bool)
    g=torch.empty_like(a,dtype=torch.float32)
    bp=torch.empty(5,4,device="cuda",dtype=a.dtype)
    bd=torch.empty(3,4,device="cuda");ad=torch.empty_like(bd)
    gates[(1,)](a,b,al,dt,active,g,bp,bd,ad,P=5,D=3,H=4,BLOCK=256,enable_fp_fusion=False)
    pbet,pg=_gdn_gates_eager(a[:5],b[:5],al,dt)
    dbet,dg=_gdn_gates_eager(a[5:],b[5:],al,dt,beta_fp32=True)
    torch.testing.assert_close(bp[active[:5]],pbet[active[:5]],rtol=0,atol=0)
    torch.testing.assert_close(bd[active[5:]],dbet[active[5:]],rtol=3e-6,atol=1e-7)
    torch.testing.assert_close(g[active],torch.cat([pg,dg])[active],rtol=3e-6,atol=1e-7)
    torch.testing.assert_close(ad[active[5:]],dg.exp()[active[5:]],rtol=3e-6,atol=1e-7)


@torch.inference_mode()
def test_two_whole_layers_replay_changing_topology_and_chunks():
    pool,blocks,pf,dec,plan,ws,layers,bound,x,actual=setup(True,"mixed")
    original,conv_original=pool.affine.clone(),pool.conv.clone()
    out0,out1=torch.empty_like(actual),torch.empty_like(actual)
    def body():
        bound[0].run(actual,out0)
        out0.add_(actual)
        bound[1].run(out0,out1)
    body();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):body()
    addresses=(ws.compose_storage.data_ptr(),ws.window_storage.data_ptr(),ws.raw.data_ptr(),out1.data_ptr())
    for lengths,nd in [((64,64),3),((1,1),2),((1,),1),((1,65),2),((36,65),1),((),3),((63,65),0),((36,65),3)]:
        now_pf=[replace(r,length=n) for r,n in zip(pf,lengths)]
        now_dec=[DecodeRequest((1,0,6,4),4),DecodeRequest((1,3,4,5),5),DecodeRequest((),7)][:nd]
        current=prepare_forward(blocks,capacity=plan.capacity,prefill=now_pf,decode=now_dec)
        ws.upload(current)
        actual.normal_(std=.1)
        active=torch.tensor(current.rows.active,device=actual.device)
        actual[~active]=float("nan")
        pool.affine.copy_(original);pool.conv.copy_(conv_original)
        body()
        expected=out1.clone();expected_a=pool.affine.clone();expected_c=pool.conv.clone()
        pool.affine.copy_(original);pool.conv.copy_(conv_original)
        # Poison temporary storage: replay must produce every consumed byte.
        ws.compose_storage.fill_(float("nan"));ws.window_storage.fill_(float("nan"))
        ws.fla.solved.fill_(float("nan"));ws.core.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out1,expected,rtol=0,atol=0)
        torch.testing.assert_close(pool.affine,expected_a,rtol=0,atol=0,equal_nan=True)
        torch.testing.assert_close(pool.conv,expected_c,rtol=0,atol=0,equal_nan=True)
        assert addresses==(ws.compose_storage.data_ptr(),ws.window_storage.data_ptr(),ws.raw.data_ptr(),out1.data_ptr())


@pytest.mark.parametrize("lengths",[(1,),(1,1),(1,65)])
@torch.inference_mode()
def test_one_token_prefill_old_rounding_boundaries(lengths,record_property):
    pool,blocks,pf,dec,plan,ws,layers,bound,x,actual=setup(True,"prefill")
    pf=[replace(r,length=n) for r,n in zip(pf,lengths)]
    plan=prepare_forward(blocks,capacity=plan.capacity,prefill=pf)
    ws.upload(plan)
    n=sum(lengths)
    actual.normal_(std=.1)
    state=OldState(pool,blocks)
    expected=old_mixed(layers[0],actual[:n],state,pf,[])
    out=torch.empty_like(actual)
    bound[0].run(actual,out)
    error=(out[:n].float()-expected.float()).norm()/expected.float().norm()
    record_property("output_relative_l2",float(error))
    assert error<1e-5,float(error)
    for r in pf:
        wanted=torch.stack([x[0] for x in state.affine[0,r.write_to]])
        error=(pool.affine[0,r.write_to]-wanted).norm()/wanted.norm()
        record_property(f"block{r.write_to}_affine_relative_l2",float(error))
        assert error<2e-5,float(error)


@pytest.mark.parametrize("mode",["prefill","decode","mixed"])
@torch.inference_mode()
def test_no_fla_selects_whole_eager_layer_set_before_writes(monkeypatch,mode):
    from minisgl.models import qwen3_5_delta as delta
    from minisgl.planned import gdn_layer
    pool,blocks,pf,dec,plan,ws,layers,bound,x,actual=setup(True,mode)
    original,cv=pool.affine.clone(),pool.conv.clone()
    state=OldState(pool,blocks)
    monkeypatch.setattr(delta,"_fla_chunk",None)
    monkeypatch.setattr(delta,"_fla_recurrent",None)
    def forbidden(*a,**k):raise AssertionError("planned FLA workspace must not be constructed")
    monkeypatch.setattr(gdn_layer,"GDNWorkspace",forbidden)
    impl=gdn_layer.bind_gdn_layers(layers,pool,plan)
    assert not impl.graph_compatible and "FLA unavailable" in impl.reason
    torch.testing.assert_close(pool.affine,original,rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(pool.conv,cv,rtol=0,atol=0,equal_nan=True)
    output=torch.empty_like(actual)
    n=sum(r.length for r in pf)
    for i,l in enumerate(layers):
        expected=old_mixed(l,x,state,pf,dec)
        impl.run(i,actual,output)
        got=torch.cat([output[:n],output[plan.capacity.prefill_tokens:plan.capacity.prefill_tokens+len(dec)]])
        torch.testing.assert_close(got,expected,rtol=2e-5,atol=2e-6)
        for r in (*pf,*dec):
            want=torch.stack([v[0] for v in state.affine[i,r.write_to]])
            torch.testing.assert_close(pool.affine[i,r.write_to],want,rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(pool.conv[i,r.write_to],state.conv[i,r.write_to][0],rtol=0,atol=0)
        x=x+expected;actual.add_(output)


@torch.inference_mode()
def test_no_fla_replay_refused_before_any_state_write(monkeypatch):
    from minisgl.models import qwen3_5_delta as delta
    from minisgl.planned.gdn_layer import bind_gdn_layers
    pool,blocks,pf,dec,plan,ws,layers,bound,x,actual=setup(True,"mixed")
    monkeypatch.setattr(delta,"_fla_chunk",None)
    impl=bind_gdn_layers(layers,pool,plan)
    original=pool.affine.clone();cv=pool.conv.clone()
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:True)
    with pytest.raises(RuntimeError,match="whole forward"):
        impl.run(0,actual,torch.empty_like(actual))
    torch.testing.assert_close(pool.affine,original,rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(pool.conv,cv,rtol=0,atol=0,equal_nan=True)


@torch.inference_mode()
def test_32_stateful_mixed_replays_fresh_append_clear_and_cross_readers(record_property):
    pool,blocks,pf,dec,plan,ws,layers,bound,x,actual=setup(True,"mixed")
    state=OldState(pool,blocks)
    a0,c0=pool.affine.clone(),pool.conv.clone()
    o0,o1=torch.empty_like(actual),torch.empty_like(actual)
    def body():
        bound[0].run(actual,o0)
        o0.add_(actual)
        bound[1].run(o0,o1)
    body();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):body()
    pool.affine.copy_(a0);pool.conv.copy_(c0)
    max_output_error=max_affine_error=max_conv_error=max_projection_error=0.
    lengths_cycle=[(1,1),(1,),(36,65),(1,65),(),(63,64)]
    for step in range(32):
        if step and step%8==0:
            # A cleared logical block reuses its slot, whose stale bytes are
            # deliberately poisoned. Only metadata authorizes initializing I/0.
            blocks[7]=replace(blocks[7],populated=False,has_conv=False,revision=blocks[7].revision+1)
            pool.affine[:,7].fill_(float("nan"));pool.conv[:,7].fill_(float("nan"))
            for lin in range(2):state.affine.pop((lin,7),None);state.conv.pop((lin,7),None)
        lengths=lengths_cycle[step%len(lengths_cycle)]
        now_pf=[replace(r,length=n) for r,n in zip(pf,lengths)]
        now_dec=dec[:(1 if step%5==0 else 3)]
        current=prepare_forward(blocks,capacity=plan.capacity,prefill=now_pf,decode=now_dec)
        ws.upload(current)
        actual.normal_(std=.1)
        active=torch.tensor(current.rows.active,device=actual.device)
        actual[~active]=float("nan")
        exact=actual[active].clone()
        conv_before=pool.conv.clone()
        want0=old_mixed(layers[0],exact,state,now_pf,now_dec)
        wanted=old_mixed(layers[1],exact+want0,state,now_pf,now_dec)
        graph.replay()
        got=o1[active]
        rel=(got.float()-wanted.float()).norm()/wanted.float().norm().clamp_min(1e-12)
        max_output_error=max(max_output_error,float(rel))
        assert rel<.008,(step,float(rel))
        for lin in range(2):
            layer_input=actual if lin==0 else o0
            raw=torch.mm(layer_input,layers[lin].in_proj_qkv.weight.t())
            expected_windows=raw_window_reference(conv_before[lin],raw,current,now_pf,now_dec)
            torch.testing.assert_close(pool.conv[lin],expected_windows,rtol=0,atol=0,equal_nan=True)
            # Attribute BF16 raw-state differences to projection shape/input,
            # not to a broken window copy. Layer 1 can also receive a slightly
            # different residual because the preceding layer is capacity-shaped.
            exact_input=exact if lin==0 else exact+want0
            old_raw=torch.mm(exact_input,layers[lin].in_proj_qkv.weight.t())
            raw_error=(raw[active].float()-old_raw.float()).norm()/old_raw.float().norm().clamp_min(1e-12)
            max_projection_error=max(max_projection_error,float(raw_error))
            for r in (*now_pf,*now_dec):
                want=torch.stack([x[0] for x in state.affine[lin,r.write_to]])
                error=(pool.affine[lin,r.write_to]-want).norm()/want.norm().clamp_min(1e-12)
                max_affine_error=max(max_affine_error,float(error))
                assert error<.006,(step,lin,r.write_to,float(error))
                old_cv=state.conv[lin,r.write_to][0].float()
                cv_error=(pool.conv[lin,r.write_to].float()-old_cv).norm()/old_cv.norm().clamp_min(1e-12)
                max_conv_error=max(max_conv_error,float(cv_error))
        for r in (*now_pf,*now_dec):
            blocks[r.write_to]=replace(blocks[r.write_to],populated=True,has_conv=True,
                                       revision=blocks[r.write_to].revision+1)
    record_property("max_32_step_output_relative_l2",max_output_error)
    record_property("max_32_step_affine_relative_l2",max_affine_error)
    record_property("max_32_step_conv_relative_l2",max_conv_error)
    record_property("max_32_step_projection_relative_l2",max_projection_error)


@torch.inference_mode()
def test_fp8_gdn_layer_uses_same_bound_linear_workspace_and_matches_old(record_property):
    from minisgl.planned.gdn_layer import BoundGDNLayer,bind_gdn_layers
    pool,blocks,pf,dec,plan,ws,layers,_,x,actual=setup(True,"mixed")
    for l in layers:
        for p in (l.in_proj_qkv,l.in_proj_z,l.out_proj):
            n,k=p.weight.shape
            p.weight=(torch.randn(n,k,device="cuda")*20).to(torch.float8_e4m3fn)
            p.weight_scale_inv=torch.rand(n//128,k//128,device="cuda")*.001+.001
    bound=[BoundGDNLayer(l,ws) for l in layers]
    assert bound[0].inputs[0].quant.data_ptr()==bound[1].output.quant.data_ptr()
    assert bind_gdn_layers(layers,pool,plan).graph_compatible
    state=OldState(pool,blocks)
    out=torch.empty_like(actual)
    mask=torch.tensor(plan.rows.active,device="cuda")
    for i,l in enumerate(layers):
        wanted=old_mixed(l,x,state,pf,dec)
        bound[i].run(actual,out)
        error=(out[mask].float()-wanted.float()).norm()/wanted.float().norm()
        record_property(f"layer{i}_output_relative_l2",float(error))
        assert error<.008,float(error)
        for r in (*pf,*dec):
            expected=torch.stack([p[0] for p in state.affine[i,r.write_to]])
            torch.testing.assert_close(pool.affine[i,r.write_to],expected,rtol=.035,atol=5e-4)
