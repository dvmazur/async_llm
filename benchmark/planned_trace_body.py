"""Short diagnostic traces without copying the entire all-layer pools."""
import torch

GDN_TARGET=12


def instrument_old(model,patch,trace):
    import minisgl.models.qwen3_5_delta as delta
    recurrent=delta._recurrent_delta
    globals()["OLD_GDN"]={}
    def recurrent_call(q,k,v,g,beta,initial_state,**kw):
        result=recurrent(q,k,v,g,beta,initial_state,**kw)
        if globals().get("OLD_GDN_LAYER")==GDN_TARGET:
            initial=torch.cat(initial_state) if isinstance(initial_state,(list,tuple)) else initial_state
            globals()["OLD_GDN"]={name:value.clone() for name,value in
                (("q",q),("k",k),("v",v),("g",g),("beta",beta),("initial",initial),("core",result[0]))}
        return result
    patch.setattr(delta,"_recurrent_delta",recurrent_call)
    from minisgl.shared_cache.gdn import SharedCacheGDN
    capture=SharedCacheGDN.capture_token_affines
    def capture_call(op,index,key,value,alpha,beta,*args,**kw):
        watch=globals().get("OLD_GDN_LAYER")==GDN_TARGET
        def states():return torch.stack([torch.stack([t[0] for t in b.linear_affine[index]]) for b in op.write_to])
        if watch:globals()["OLD_CAPTURE_BEFORE"]=states()
        result=capture(op,index,key,value,alpha,beta,*args,**kw)
        if watch:globals()["OLD_CAPTURE_AFTER"]=states()
        return result
    patch.setattr(SharedCacheGDN,"capture_token_affines",capture_call)
    import flashinfer
    merge=flashinfer.merge_states
    def merge_call(v,s,*args,**kw):
        globals()["OLD_MERGE"][globals()["OLD_LAYER"],len(v)]=(v.clone(),s.clone())
        return merge(v,s,*args,**kw)
    patch.setattr(flashinfer,"merge_states",merge_call)
    globals()["OLD_MERGE"]={}
    from minisgl.shared_cache.attention import SharedCacheAttention
    sc_forward=SharedCacheAttention.forward
    rope=SharedCacheAttention._rope
    globals()["OLD_ROPE"]={}
    def rope_call(op,x,positions):
        result=rope(op,x,positions)
        if globals().get("OLD_LAYER")==3:
            globals()["OLD_ROPE"][len(x),x.shape[1]]=(result.clone(),positions.clone())
        return result
    patch.setattr(SharedCacheAttention,"_rope",rope_call)
    layer_map={l.self_attn._kv_idx:i for i,l in enumerate(model.model.layers.op_list) if not l._is_linear}
    def sc_call(op,q,k,v,index,batch):
        globals()["OLD_LAYER"]=layer_map[index]
        if layer_map[index]==3 and getattr(batch.attn_metadata,"phase",None)=="decode":
            globals()["OLD_NATIVE_DECODE_RECIPE"]=tuple(op.wrapper._plan_info)
        result=sc_forward(op,q,k,v,index,batch)
        trace[layer_map[index],"ungated"]=result.clone()
        return result
    patch.setattr(SharedCacheAttention,"forward",sc_call)
    for i,l in enumerate(model.model.layers.op_list):
        for name,op,pair in (("input_norm",l.input_layernorm,True),("attention",l.linear_attn if l._is_linear else l.self_attn,False),
                             ("post_norm",l.post_attention_layernorm,True),("mlp",l.mlp,False)):
            fn=op.forward
            def call(*args,fn=fn,i=i,name=name,pair=pair,**kwargs):
                if name=="attention":globals()["OLD_GDN_LAYER"]=i
                result=fn(*args,**kwargs);trace[i,name]=(result[0] if pair else result).clone();return result
            patch.setitem(op.__dict__,"forward",call)
        if not l._is_linear:
            for name,op in (("raw",l.self_attn.qkv_proj),("q",l.self_attn.q_norm),
                            ("k",l.self_attn.k_norm),("gated",l.self_attn.o_proj)):
                fn=op.forward
                def stage(x,fn=fn,i=i,name=name):
                    result=fn(x)
                    trace[i,name]=(x if name=="gated" else result).clone()
                    return result
                patch.setitem(op.__dict__,"forward",stage)


def instrument_new(session,patch,trace):
    globals()["NEW_MERGE"]={}
    from minisgl.planned.runner import ProgramRunner
    original=ProgramRunner.execute_prepared
    def execute(runner):
        p=runner.program;tx=session.store._pending;f=tx.forward
        active=torch.tensor(f.rows.active,device=session.device)
        mapping={l.linear_attn._lin_idx:i for i,l in enumerate(session.model.model.layers.op_list) if l._is_linear}
        fn=p.gdn.run
        def gdn(index,x,out,fn=fn):
            i=mapping[index];trace[i,"input_norm"]=x[active].clone()
            if i==GDN_TARGET:
                capture_slots=torch.tensor(f.decode.write_slots[:len(f.decode_requests)],device=session.device)
                globals()["NEW_CAPTURE_BEFORE"]=session.gdn_pool.affine[index].index_select(0,capture_slots)
            result=fn(index,x,out);trace[i,"attention"]=result[active].clone()
            if i==GDN_TARGET:
                globals()["NEW_CAPTURE_AFTER"]=session.gdn_pool.affine[index].index_select(0,capture_slots)
                w=p.gdn.workspace;n=sum(f.rows.active)
                globals()["NEW_GDN"]={name:value[:n].clone() for name,value in
                    (("q",w.dec_q),("k",w.dec_k),("v",w.dec_v),("g",w.dec_g),
                     ("beta",w.dec_beta),("initial",w.dec_compose.initial),("core",w.dec_core),
                     ("normed",w.normed),("z",w.z))}
            return result
        patch.setattr(p.gdn,"run",gdn)
        for i,op in enumerate(p.bound_attention):
            if op is None:continue
            fn=op.run
            def att(x,out,fn=fn,i=i,op=op):
                trace[i,"input_norm"]=x[active].clone();result=fn(x,out)
                trace[i,"attention"]=result[active].clone()
                for name,t in (("raw",op.ws.raw),("q",op.ws.q),("k",op.ws.k),("gated",op.ws.output_flat)):
                    trace[i,name]=t[active].clone()
                ungated=torch.empty_like(op.ws.output)
                op.ws.core.run(op.layer._kv_idx,op.ws.q,op.ws.k,op.ws.value,ungated)
                trace[i,"ungated"]=ungated[active].flatten(1)
                globals()["NEW_MERGE"][i]=(op.ws.core.partial.clone(),op.ws.core.lse.clone(),
                    op.ws.core.meta.merge_sources.clone(),active.clone(),op.ws.gate[active].clone())
                if i==3:
                    globals()["NEW_NATIVE_DECODE_RECIPE"]=tuple(op.ws.core.ops[2].wrapper._plan_info)
                    globals()["NEW_QUERIES"]=op.ws.core.queries.clone()
                    slots=op.ws.core.meta.write_token_slots[active]
                    globals()["NEW_KEYS"]=op.ws.core.keys[op.layer._kv_idx].flatten(0,1)[slots]
                    globals()["NEW_POSITIONS"]=[t.positions_flat.clone() for t in op.ws.core.tables]
                return result
            patch.setattr(op,"run",att)
        for i,op in enumerate(p.bound_mlp):
            fn=op.run
            def mlp(x,out,fn=fn,i=i):
                trace[i,"post_norm"]=x[active].clone();result=fn(x,out)
                trace[i,"mlp"]=result[active].clone();return result
            patch.setattr(op,"run",mlp)
        writes=torch.tensor([w.slot for w in f.writes],device=session.device)
        kvslots=torch.tensor([v for v in tx.attention.write_token_slots if v>=0],device=session.device)
        tensors=(session.gdn_pool.affine,session.gdn_pool.conv,session.k_pool.flatten(1,2),session.v_pool.flatten(1,2))
        indices=(writes,writes,kvslots,kvslots)
        before=[t.index_select(1,idx) for t,idx in zip(tensors,indices)]
        eager=p.run().clone()
        for t,idx,v in zip(tensors,indices,before):t.index_copy_(1,idx,v)
        result=original(runner)
        selected=torch.tensor([i for i,row in enumerate(f.rows.output_rows) if row>=0],device=session.device)
        torch.testing.assert_close(result.logits,eager.index_select(0,selected),rtol=0,atol=0)
        return result
    patch.setattr(ProgramRunner,"execute_prepared",execute)
