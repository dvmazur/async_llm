"""Regressions exposed only by learned, 40-layer checkpoint activations."""
import pytest
import torch
import triton

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@torch.inference_mode()
def test_gate_exp_matches_original_compiled_libdevice_lowering_exactly():
    from minisgl.models.qwen3_5_delta import _compiled_gdn_gates
    from minisgl.planned.gdn_layer_kernels import gates
    torch.manual_seed(719)
    p,d,h=64,6,32
    a=torch.linspace(-40,40,(p+d)*h,device="cuda").to(torch.bfloat16).view(p+d,h)
    b=torch.randn_like(a)*5
    al=torch.linspace(-4,6,h,device="cuda").to(torch.bfloat16)
    dt=(torch.randn(h,device="cuda")*3).to(torch.bfloat16)
    active=torch.ones(p+d,device="cuda",dtype=torch.bool)
    g=torch.empty(p+d,h,device="cuda");bp=torch.empty(p,h,device="cuda",dtype=a.dtype)
    bd=torch.empty(d,h,device="cuda");alpha=torch.empty_like(bd)
    gates[(triton.cdiv(a.numel(),256),)](a,b,al,dt,active,g,bp,bd,alpha,P=p,D=d,H=h,BLOCK=256,enable_fp_fusion=False)
    pb,pg=_compiled_gdn_gates(a[:p],b[:p],al,dt)
    db,dg=_compiled_gdn_gates(a[p:],b[p:],al,dt,beta_fp32=True)
    torch.testing.assert_close(g,torch.cat([pg,dg]),rtol=0,atol=0)
    torch.testing.assert_close(bp,pb,rtol=0,atol=0)
    torch.testing.assert_close(bd,db,rtol=0,atol=0)
    torch.testing.assert_close(alpha,dg.exp(),rtol=0,atol=0)


@pytest.mark.parametrize("rows,heads,dim",[(1,4,32),(2,32,128),(33,32,128),(64,4,128)])
@torch.inference_mode()
def test_gdn_norm_preserves_reference_reduction_recipe(rows,heads,dim):
    from minisgl.kernel.gdn_norm import gated_rmsnorm
    from minisgl.planned.gdn_layer_kernels import gated_norm
    torch.manual_seed(721)
    core=(torch.randn(rows,heads,dim,device="cuda")*.03).to(torch.bfloat16)
    z=torch.randn_like(core)*2;w=torch.randn(dim,device="cuda",dtype=core.dtype)
    active=torch.ones(rows,device="cuda",dtype=torch.bool);out=torch.empty_like(core)
    expected=gated_rmsnorm(core,w,z,1e-6)
    gated_norm[(triton.cdiv(rows*heads,4),)](core,z,w,active,out,R=rows,H=heads,D=dim,EPS=1e-6,
        BD=triton.next_power_of_2(dim),ROWS=4,num_warps=1)
    torch.testing.assert_close(out,expected,rtol=0,atol=0)


@torch.inference_mode()
def test_planned_norm_recipe_changes_on_replay_without_changing_graph():
    from minisgl.kernel.gdn_norm import gated_rmsnorm
    from minisgl.planned.gdn_layer_kernels import gated_norm
    torch.manual_seed(723)
    p,d,h,dim=128,16,32,128
    core=torch.randn(p+d,h,dim,device="cuda",dtype=torch.bfloat16)*.03
    z=torch.randn_like(core)*3;weight=torch.randn(dim,device="cuda",dtype=core.dtype)
    active=torch.zeros(p+d,device="cuda",dtype=torch.bool)
    recipe=torch.tensor([4,1],device="cuda",dtype=torch.int64)
    out=torch.empty_like(core)
    sm=torch.cuda.get_device_properties(0).multi_processor_count
    def body():
        gated_norm[(triton.cdiv((p+d)*h,4),)](core,z,weight,active,out,R=p+d,H=h,D=dim,EPS=1e-6,
            BD=dim,ROWS=4,recipes=recipe,P=p,num_warps=1)
    body();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):body()
    for np,nd in ((33,2),(1,1),(128,16),(6,4),(0,0),(33,2)):
        active.zero_();active[:np]=True;active[p:p+nd]=True
        values=[min(4,triton.next_power_of_2(max(1,triton.cdiv(n*h,2*sm)))) for n in (np,nd)]
        recipe.copy_(torch.tensor(values,device="cuda",dtype=torch.int64))
        core.normal_(std=.03);z.normal_(std=3)
        graph.replay()
        for start,n in ((0,np),(p,nd)):
            if n:
                expected=gated_rmsnorm(core[start:start+n],weight,z[start:start+n],1e-6)
                torch.testing.assert_close(out[start:start+n],expected,rtol=0,atol=0)
        assert torch.equal(out[~active],torch.zeros_like(out[~active]))


@pytest.mark.parametrize("dim",[64,128,256])
@torch.inference_mode()
def test_fused_qk_norm_matches_actual_flashinfer_cute_recipe(dim):
    from flashinfer import gemma_rmsnorm
    from minisgl.planned.attention_kernels import normalize_qk
    torch.manual_seed(724)
    rows,h,hk=33,16,2
    width=(2*h+2*hk)*dim
    raw=(torch.randn(rows,width,device="cuda")*torch.exp(torch.randn(rows,width,device="cuda"))).to(torch.bfloat16)
    qw,kw=[torch.randn(dim,device="cuda",dtype=raw.dtype)*.1 for _ in range(2)]
    active=torch.ones(rows,device="cuda",dtype=torch.bool)
    q=torch.empty(rows,h,dim,device="cuda",dtype=raw.dtype);k=torch.empty(rows,hk,dim,device="cuda",dtype=raw.dtype)
    oldq=gemma_rmsnorm(raw[:,:2*h*dim].view(rows,h,2*dim)[...,:dim].contiguous(),qw,1e-6)
    oldk=gemma_rmsnorm(raw[:,2*h*dim:(2*h+hk)*dim].contiguous().view_as(k),kw,1e-6)
    normalize_qk[(rows,h+hk)](raw,qw,kw,active,q,k,HQ=h,HK=hk,D=dim,EPS=1e-6,STRIDE=width,BD=dim,num_warps=1)
    torch.testing.assert_close(q,oldq,rtol=0,atol=0)
    torch.testing.assert_close(k,oldk,rtol=0,atol=0)


@pytest.mark.parametrize("segments",[1,3,7])
@torch.inference_mode()
def test_direct_segment_merge_matches_flashinfer_fma_and_gate(segments):
    from flashinfer import merge_states
    from minisgl.kernel.qwen_pointwise import output_gate
    from minisgl.planned.attention_kernels import merge_segments
    torch.manual_seed(727)
    rows,h,d,capacity=64,16,256,17
    partial=(torch.randn(rows*segments,h,d,device="cuda")*3).to(torch.bfloat16)
    lse=torch.randn(rows*segments,h,device="cuda")*5
    sources=torch.full((rows,capacity),-1,device="cuda",dtype=torch.int32)
    sources[:,:segments]=torch.arange(rows*segments,device="cuda",dtype=torch.int32).view(rows,segments)
    gate=torch.randn(rows,h,d,device="cuda",dtype=torch.bfloat16)*4
    expected=merge_states(partial.view(rows,segments,h,d),lse.view(rows,segments,h))[0]
    out=torch.empty_like(expected)
    for has_gate in (False,True):
        merge_segments[(rows,h)](partial,lse,sources,out,gate,BASE=0,H=h,D=d,S=capacity,BD=d,
            HAS_GATE=has_gate,GS0=gate.stride(0),GS1=gate.stride(1),num_warps=4,enable_fp_fusion=False)
        wanted=output_gate(expected,gate) if has_gate else expected
        torch.testing.assert_close(out,wanted,rtol=0,atol=0)


@torch.inference_mode()
def test_capture_row_tree_matches_original_pointer_kernel():
    from minisgl.kernel import capture_gdn_affine_pointer_update
    from minisgl.planned.gdn_kernels import recurrent_capture
    from minisgl.kernel.gdn_recurrent import recurrent_gdn_pointer
    torch.manual_seed(729)
    w,h,hk,d=2,32,16,128
    pool=torch.randn(1,w,2,h,d,d,device="cuda")*.03
    pool[:,:,0]+=torch.eye(d,device="cuda")*.7
    q,k=[torch.randn(w,1,hk,d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    v=torch.randn(w,1,h,d,device="cuda",dtype=torch.bfloat16)
    g=-torch.rand(w,1,h,device="cuda")*.4;beta=torch.rand_like(g);alpha=g.exp()
    initial=torch.randn(w,h,d,d,device="cuda")*.01
    output=torch.empty_like(v)
    active=torch.ones(w,device="cuda",dtype=torch.bool);fresh=torch.zeros_like(active)
    writes=torch.arange(w,device="cuda",dtype=torch.int64)
    old_pool=pool.clone()
    for step in range(4):
        # Dyadic activations make the pre-capture key normalization exactly
        # shared. This test isolates the matrix/vector reduction tree.
        q.random_(-16,17).mul_(.125);k.random_(-16,17).mul_(.125);v.normal_()
        expanded=k.repeat_interleave(h//hk,dim=2).float()
        kn=expanded*torch.rsqrt((expanded*expanded).sum(-1,keepdim=True)+1e-6)
        pairs=[(old_pool[0,i,0:1],old_pool[0,i,1:2]) for i in range(w)]
        capture_gdn_affine_pointer_update(pairs,kn[:,0].contiguous(),v[:,0].float(),
            alpha[:,0].contiguous(),beta[:,0].contiguous(),output_pairs=pairs)
        wanted,_=recurrent_gdn_pointer(q,k,v,g,beta,[initial[i:i+1] for i in range(w)])
        recurrent_capture[(d//8,w*h)](q,k,v,g,beta,alpha,initial,pool,writes,fresh,active,output,
            pool,output,LAYER=0,SLOTS=w,H=hk,HV=h,D=d,KD=d,ROWS=8,HAS_CONV=False,C=0,CK=0,
            QSTRIDE=q.stride(0),KSTRIDE=k.stride(0),VSTRIDE=v.stride(0),
            prefill_recipe=active,PREFILL_SPECIAL=False,num_warps=1,num_stages=3)
        torch.testing.assert_close(pool,old_pool,rtol=0,atol=0)
        torch.testing.assert_close(output,wanted,rtol=0,atol=0)
