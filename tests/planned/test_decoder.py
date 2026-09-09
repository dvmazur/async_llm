from dataclasses import replace
from contextlib import contextmanager
import pytest
import torch

from minisgl.planned.forward_plan import BlockState,PrefillRequest,DecodeRequest,PlanCapacity,prepare_forward
from minisgl.planned.attention_plan import KVBlock,AttentionCapacity,prepare_attention
from minisgl.planned.model_io import prepare_inputs
from reference_gdn_layer import OldState,old_mixed
from reference_shared_attention import KVPool
from test_attention_layer import old_layer as old_attention_layer

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def model(monkeypatch,fp8):
    from minisgl.models.config import ModelConfig,RotaryConfig
    from minisgl.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
    from minisgl.distributed import DistributedInfo
    import minisgl.distributed.info as dist
    import minisgl.core as core
    from minisgl.moe.fused import FusedMoe
    from minisgl.utils import torch_dtype
    monkeypatch.setattr(dist,"_TP_INFO",DistributedInfo(0,1))
    ctx=core.Context(4);ctx.moe_backend=FusedMoe();monkeypatch.setattr(core,"_GLOBAL_CTX",ctx)
    cfg=ModelConfig(num_layers=4,num_qo_heads=4,num_kv_heads=1,head_dim=64,hidden_size=128,
        vocab_size=128,intermediate_size=128,rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(64,32,1024,1e4,None),hidden_act="silu",tie_word_embeddings=False,
        num_experts=8,num_experts_per_tok=3,moe_intermediate_size=128,shared_expert_intermediate_size=128,
        norm_topk_prob=True,model_type="qwen3_5_moe",architectures=["Qwen3_5MoeForCausalLM"],
        layer_types=("linear_attention","full_attention")*2,linear_num_key_heads=2,linear_num_value_heads=4,
        linear_key_head_dim=32,linear_value_head_dim=32,linear_conv_kernel_dim=4,
        partial_rotary_factor=.5,mrope_section=(6,5,5))
    with torch.device("meta"),torch_dtype(torch.bfloat16):net=Qwen3_5MoeForCausalLM(cfg)
    gen=torch.Generator(device="cuda").manual_seed(209)
    state={}
    for name,t in net.state_dict().items():
        value=torch.randn(t.shape,device="cuda",dtype=t.dtype,generator=gen)*.04
        if name.endswith("A_log"):value.zero_()
        if name.endswith("linear_attn.norm.weight"):value.fill_(1.)
        state[name]=value
    net.load_state_dict(state)
    if fp8:
        # Keep the existing parameters' mathematical scale; serialize a simple
        # group-aligned test quant. This is fixture construction, not a loader.
        def quant(p):
            n,k=p.weight.shape
            if n%128 or k%128:return
            scale=torch.full((n//128,k//128),.001,device="cuda")
            p.weight=(p.weight.float()/scale.repeat_interleave(128,0).repeat_interleave(128,1)).to(torch.float8_e4m3fn)
            p.weight_scale_inv=scale
        for l in net.model.layers.op_list:
            a=l.linear_attn if l._is_linear else l.self_attn
            for name in (("in_proj_qkv","in_proj_z","in_proj_a","in_proj_b","out_proj") if l._is_linear else ("qkv_proj","o_proj")):
                quant(getattr(a,name))
            quant(l.mlp.shared_expert.gate_up_proj);quant(l.mlp.shared_expert.down_proj)
            for name in ("gate_up_proj","down_proj"):
                w=getattr(l.mlp.experts,name);e,n,k=w.shape
                setattr(l.mlp.experts,name,(w.float()/.001).to(torch.float8_e4m3fn))
                setattr(l.mlp.experts,name+"_scale_inv",torch.full((e,n//128,k//128),.001,device="cuda"))
    return net


def setup(monkeypatch,*,fp8=False,padded=True,image=True,lengths=(4,2)):
    from minisgl.planned.pools import PoolShape,GDNPool
    from minisgl.planned.decoder import DecoderProgram
    from minisgl.shared_cache.attention import SharedCacheAttention
    net=model(monkeypatch,fp8)
    torch.manual_seed(210)
    initial_lengths={0:2,3:3,4:3,5:2,6:0,7:0}
    blocks={b:BlockState(b,b,bool(n),bool(n)) for b,n in initial_lengths.items()}
    kv={b:KVBlock(b,n,n,(b,8) if b==3 else (b,)) for b,n in initial_lengths.items()}
    next_page=9
    for block,new in ((6,lengths[0]),(3,lengths[1])):
        required=(kv[block].num_tokens+new+3)//4
        extra=max(0,required-len(kv[block].pages))
        kv[block]=replace(kv[block],pages=kv[block].pages+tuple(range(next_page,next_page+extra)))
        next_page+=extra
    pages=max(32,next_page+1)
    p=sum(lengths)
    pcap=16 if p<=16 else ((p+63)//64)*64
    f=prepare_forward(blocks,capacity=PlanCapacity(3,pcap,4,8,16) if padded else PlanCapacity(2,p,3,8,16),
        prefill=[PrefillRequest((0,4),6,lengths[0]),PrefillRequest((0,),3,lengths[1])],
        decode=[DecodeRequest((0,6,5,4),4),DecodeRequest((0,3,4,5),5),DecodeRequest((0,6),7)])
    extra_pos=tuple(range(2,lengths[0]-2))
    im={0:((0,0,0,0)+extra_pos,(0,0,1,1)+extra_pos,(0,1,0,1)+extra_pos)} if image else {}
    attention=prepare_attention(kv,f,AttentionCapacity(4,512,4,pages,dummy_page=pages-1),prefill_mrope=im)
    gdn=GDNPool(PoolShape(2,16,4,32,256,4),device="cuda")
    gdn.affine.normal_(std=.02);gdn.affine[:,:,0]+=torch.eye(32,device="cuda")*.8;gdn.conv.normal_(std=.075)
    gdn.affine[:,6:].fill_(float("nan"));gdn.conv[:,6:].fill_(float("nan"))
    k,v=[torch.randn(2,pages,4,1,64,device="cuda",dtype=torch.bfloat16)*.1 for _ in range(2)]
    k[:,pages-1].zero_();v[:,pages-1].zero_()
    oldk,oldv=k.clone(),v.clone()
    old_att=SharedCacheAttention(KVPool(oldk,oldv),torch.empty(0,device="cuda"),4,1,64,4,k.dtype,k.device,
        rotary_dim=32,mrope_section=(6,5,5),rope_base=1e4)
    inputs=prepare_inputs(f,[i%128 for i in range(p+3)],vocab_size=128,image_rows=(0,1,2,3) if image else ())
    features=torch.randn(4,128,device="cuda",dtype=torch.bfloat16)*.1 if image else None
    old_gdn=OldState(gdn,blocks)
    program=DecoderProgram(net,gdn,k,v,f,attention,inputs,features=features)
    return net,blocks,kv,f,attention,inputs,features,im,gdn,k,v,old_gdn,old_att,oldk,oldv,program


def old_decoder(net,f,inputs,features,kv,im,gdn,att,*,trace=None):
    mask=torch.tensor(f.rows.active,device="cuda")
    ids=torch.tensor(inputs.token_ids,device="cuda")[mask]
    x=net.model.embed_tokens.forward(ids)
    images=torch.tensor(inputs.image_indices,device="cuda")[mask]
    if features is not None:x[images>=0]=features[images[images>=0]]
    residual=None
    for index,l in enumerate(net.model.layers.op_list):
        x,residual=l.input_layernorm.forward(x,residual)
        if trace is not None:trace[index,"input_norm"]=x.clone()
        if l._is_linear:x=old_mixed(l.linear_attn,x,gdn,f.prefill_requests,f.decode_requests)
        else:
            physical=torch.empty(len(mask),x.shape[1],device=x.device,dtype=x.dtype)
            physical[mask]=x
            x=old_attention_layer(l.self_attn,physical,kv,f,att,im)
        if trace is not None:trace[index,"attention"]=x.clone()
        x,residual=l.post_attention_layernorm.forward(x,residual)
        if trace is not None:trace[index,"post_norm"]=x.clone()
        x=l.mlp.forward(x)
        if trace is not None:trace[index,"mlp"]=x.clone()
    x=net.model.norm.forward(x,residual)[0]
    lookup={physical:logical for logical,physical in enumerate(i for i,on in enumerate(f.rows.active) if on)}
    selected=[lookup[row] for row in f.rows.output_rows if row>=0]
    head=net.lm_head.tied_embedding or net.lm_head
    return torch.nn.functional.linear(x[selected],head.weight,net.lm_head.bias)


@contextmanager
def capacity_linear_reference(f,*,exclude_layers=()):
    """Old operations at the same GEMM M, independent of planned kernels.

    Only adapt the physical row shape of unchanged old Linear operations.
    State/Attention/FLA/MoE reference logic still processes real requests. This
    separates address/math parity from near-tie routing after BF16 padding.
    """
    from minisgl.layers.linear import _LinearTPImpl
    original=_LinearTPImpl.forward
    excluded={id(layer) for layer in exclude_layers}
    mask=torch.tensor(f.rows.active,device="cuda")
    actual=sum(f.rows.active)
    def forward(layer,x):
        if id(layer) in excluded:return original(layer,x)
        assert x.shape[0]==actual,"oracle expects full-forward Linear inputs"
        physical=torch.zeros(len(mask),x.shape[1],device=x.device,dtype=x.dtype)
        physical[mask]=x
        return original(layer,physical)[mask]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_LinearTPImpl,"forward",forward)
        yield


@pytest.mark.parametrize("fp8",[False,True])
@pytest.mark.parametrize("padded",[False,True])
@pytest.mark.parametrize("lengths",[(4,2),(36,138)])
@torch.inference_mode()
def test_complete_hybrid_decoder_matches_old_operations(monkeypatch,fp8,padded,lengths,record_property):
    net,blocks,kv,f,ap,inputs,features,im,gdn,k,v,ref,att,oldk,oldv,program=setup(
        monkeypatch,fp8=fp8,padded=padded,lengths=lengths)
    with capacity_linear_reference(f):
        wanted=old_decoder(net,f,inputs,features,kv,im,ref,att)
    got=program.run()
    mask=torch.tensor([r>=0 for r in f.rows.output_rows],device="cuda")
    error=(got[mask].float()-wanted.float()).norm()/wanted.float().norm()
    record_property("logits_relative_l2",float(error))
    assert error<.025,float(error)
    torch.testing.assert_close(got[mask],wanted,rtol=.06,atol=.008)
    tv=(got[mask].float().softmax(-1)-wanted.float().softmax(-1)).abs().sum(-1)/2
    record_property("mean_tv",float(tv.mean()));assert tv.max()<.006,float(tv.max())
    torch.testing.assert_close(k,oldk,rtol=.04,atol=.015)
    torch.testing.assert_close(v,oldv,rtol=.04,atol=.015)
    for lin in range(2):
        for r in (*f.prefill_requests,*f.decode_requests):
            expected=torch.stack([x[0] for x in ref.affine[lin,r.write_to]])
            error=(gdn.affine[lin,r.write_to]-expected).norm()/expected.norm()
            record_property(f"gdn{lin}_block{r.write_to}_relative_l2",float(error))
            assert error<.025,float(error)
    assert torch.equal(got[~mask],torch.zeros_like(got[~mask]))


@pytest.mark.parametrize("fp8",[False,True])
@torch.inference_mode()
def test_decoder_padding_drift_is_separate_from_state_address_parity(monkeypatch,fp8,record_property):
    """Natural unpadded old run: quantify drift, and explain any routing flip.

    Strict per-cell KV checks live in the same-M test above. A near-tie expert
    switch may legitimately amplify one row, so here use the existing .025
    relative-norm gate, per layer, alongside logits/TV and boundary margins.
    This oracle never uses the capacity-Linear adapter.
    """
    from minisgl.moe.fused import fused_topk
    data=setup(monkeypatch,fp8=fp8,padded=True,lengths=(36,138))
    net,blocks,kv,f,ap,inputs,features,im,gdn,k,v,ref,att,oldk,oldv,program=data
    old,new={},{}
    mask=torch.tensor(f.rows.active,device="cuda")
    for index,layer in enumerate(net.model.layers.op_list):
        fn=layer.mlp.gate.forward
        def gate(x,fn=fn,index=index):
            out=fn(x);old[index,"router"]=out.clone();return out
        monkeypatch.setattr(layer.mlp.gate,"forward",gate)
        bound=program.bound_mlp[index];fn=bound.run
        def run(x,out,fn=fn,index=index,bound=bound):
            new[index,"input"]=x[mask].clone();result=fn(x,out)
            new[index,"router"]=bound.ws.router[mask].clone()
            new[index,"ids"]=bound.experts.ids[mask].clone()
            return result
        monkeypatch.setattr(bound,"run",run)
    # Isolate the first raw projection before subsequent errors/route switches.
    first=program.bound_attention[1];fn=first.run
    def attention(x,out):
        new[1,"projection_input"]=x[mask].clone()
        expected=net.model.layers.op_list[1].self_attn.qkv_proj.forward(x).clone()
        result=fn(x,out)
        torch.testing.assert_close(first.ws.raw[mask],expected[mask],rtol=0,atol=0)
        return result
    monkeypatch.setattr(first,"run",attention)
    wanted=old_decoder(net,f,inputs,features,kv,im,ref,att,trace=old)
    got=program.run()
    torch.testing.assert_close(new[1,"projection_input"],old[1,"input_norm"],rtol=0,atol=0)
    selected=torch.tensor([r>=0 for r in f.rows.output_rows],device="cuda")
    error=(got[selected].float()-wanted.float()).norm()/wanted.float().norm()
    tv=(got[selected].float().softmax(-1)-wanted.float().softmax(-1)).abs().sum(-1)/2
    record_property("logits_relative_l2",float(error));record_property("max_tv",float(tv.max()))
    assert error<.025 and tv.max()<.006
    for layer in range(2):
        for name,actual,expected in (("k",k,oldk),("v",v,oldv)):
            error=(actual[layer].float()-expected[layer].float()).norm()/expected[layer].float().norm()
            record_property(f"kv{layer}_{name}_relative_l2",float(error));assert error<.025
    for index in range(4):
        a,b=old[index,"router"].float(),new[index,"router"].float()
        _,ids=fused_topk(old[index,"post_norm"],old[index,"router"],3,True)
        changed=(ids.sort(-1).values!=new[index,"ids"].sort(-1).values).any(-1)
        sorted_scores=a.sort(-1,descending=True).values
        margin=sorted_scores[:,2]-sorted_scores[:,3]
        perturbation=(a-b).abs().amax(-1)
        # Every changed expert set must be explainable by the observed score
        # perturbation crossing the old top-k boundary, not by corrupted IDs.
        assert torch.all(margin[changed]<=2*perturbation[changed]+1e-7)
        fraction=float(changed.float().mean())
        record_property(f"layer{index}_routing_changed_fraction",fraction)
        assert fraction<=.02


@torch.inference_mode()
def test_complete_decoder_one_graph_changing_values_and_occupancy(monkeypatch):
    net,blocks,kv,f,ap,inputs,features,im,gdn,k,v,ref,att,oldk,oldv,program=setup(monkeypatch,fp8=True)
    originals=[x.clone() for x in (gdn.affine,gdn.conv,k,v)]
    program.run();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):program.run()
    for np,nd in ((2,3),(1,1),(0,3),(2,0),(0,0),(2,3)):
        now=prepare_forward(blocks,capacity=f.capacity,prefill=f.prefill_requests[:np],decode=f.decode_requests[:nd])
        pos=im if np else {}
        a=prepare_attention(kv,now,ap.capacity,prefill_mrope=pos)
        inp=prepare_inputs(now,[i%128 for i in range(sum(now.rows.active))],vocab_size=128,image_rows=(0,1,2,3) if np else ())
        feat=features if np else None
        program.prepare(now,a,inp,feat)
        for target,old in zip((gdn.affine,gdn.conv,k,v),originals):target.copy_(old)
        wanted=program.run().clone();wanted_state=[t.clone() for t in (gdn.affine,gdn.conv,k,v)]
        for target,old in zip((gdn.affine,gdn.conv,k,v),originals):target.copy_(old)
        program.hidden.fill_(float("nan"));program.residual.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(program.logits,wanted,rtol=0,atol=0)
        for target,expected in zip((gdn.affine,gdn.conv,k,v),wanted_state):
            torch.testing.assert_close(target,expected,rtol=0,atol=0,equal_nan=True)
