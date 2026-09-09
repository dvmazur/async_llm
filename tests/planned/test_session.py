import gc
from dataclasses import replace
import pytest
import torch

from minisgl.planned.catalogue import RuntimeProfile
from minisgl.planned.forward_plan import PlanCapacity
from minisgl.planned.attention_plan import AttentionCapacity
from minisgl.planned.session import PlannedSession
from minisgl.shared_cache.session import PrefillJob
from minisgl.shared_cache.worker_group import WorkerGroup
from test_decoder import model,old_decoder,capacity_linear_reference
from reference_gdn_layer import OldState
from reference_shared_attention import KVPool

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def session(monkeypatch,*,fp8=True,graphs=True,slots=12,pages=128):
    net=model(monkeypatch,fp8)
    att=AttentionCapacity(6,256,4,pages)
    profiles=[RuntimeProfile(PlanCapacity(r,p,d,8,slots),att) for r,p,d in ((0,0,4),(3,16,4),(3,192,4))]
    return PlannedSession(net,profiles,use_graph=graphs)


def group(*pairs):return WorkerGroup(cache_structure=[p[0] for p in pairs],write_to=[p[1] for p in pairs])


class Oracle:
    def __init__(self,s):
        from minisgl.shared_cache.attention import SharedCacheAttention
        self.s=s
        self.k,self.v=s.k_pool.clone(),s.v_pool.clone()
        a=s._first_attention
        self.att=SharedCacheAttention(KVPool(self.k,self.v),torch.empty(0,device=s.device),
            a.num_qo_heads,a.num_kv_heads,a.head_dim,s.page_size,s.dtype,s.device,
            rotary_dim=a.rotary_dim,mrope_section=a._mrope_section,rope_base=a._rope_base)
        self.max_error=0.
        self.max_kv_error=0.
        self.max_tv=0.
        self.steps=0

    def before(self,p):
        s=self.s;tx=s.store._pending;f=tx.forward
        states={b.block_id:b for b in f.blocks}
        self.ref=OldState(s.gdn_pool,states)
        self.k.copy_(s.k_pool);self.v.copy_(s.v_pool)
        kv={b:replace(record.kv(),pages=tx.after[b].pages if b in tx.after else record.pages) for b,record in tx.before.items()}
        # Reconstruct block-relative image/text coordinates from committed
        # key positions, not by calling the new Attention evaluator.
        im={};offset=0
        for i,r in enumerate(f.prefill_requests):
            base=tx.before[r.write_to].mrope_span
            im[i]=tuple(tuple(x-base for x in axis[offset:offset+r.length]) for axis in tx.attention.key_positions)
            offset+=r.length
        # The caller's immutable input plan is captured by the harness below.
        with capacity_linear_reference(f):
            self.want=old_decoder(s.model,f,self.input_plan,self.features,kv,im,self.ref,self.att,trace=getattr(self,"trace",None))
        return f

    def after(self,result,f):
        s=self.s
        err=float((result.logits.float()-self.want.float()).norm()/self.want.float().norm().clamp_min(1e-10))
        assert err<.025;self.max_error=max(self.max_error,err)
        tv=float(((result.logits.float().softmax(-1)-self.want.float().softmax(-1)).abs().sum(-1)/2).max())
        self.max_tv=max(self.max_tv,tv);assert tv<.006
        # Compare only initialized pages. Rest of the owned pool is allowed
        # uninitialized/poisoned, and must not become an oracle by coincidence.
        tx=s.store._pending
        valid=set()
        for b,record in tx.before.items():
            record=tx.after.get(b,record)
            valid.update(s._token_slots(record))
        indices=torch.tensor(sorted(valid),device=s.device)
        for actual,expected in ((s.k_pool,self.k),(s.v_pool,self.v)):
            actual=actual.flatten(1,2).index_select(1,indices).float()
            expected=expected.flatten(1,2).index_select(1,indices).float()
            for layer in range(len(actual)):
                err=float((actual[layer]-expected[layer]).norm()/expected[layer].norm().clamp_min(1e-10))
                self.max_kv_error=max(self.max_kv_error,err)
                assert err<.025
        for layer in range(2):
            for w in f.writes:
                ref=torch.stack([v[0] for v in self.ref.affine[layer,w.block_id]])
                actual=s.gdn_pool.affine[layer,w.slot]
                err=float((actual-ref).norm()/ref.norm().clamp_min(1e-10));assert err<.025
                actual=s.gdn_pool.conv[layer,w.slot].float()
                expected=self.ref.conv[layer,w.block_id][0].float()
                assert float((actual-expected).norm()/expected.norm().clamp_min(1e-10))<.025
        self.steps+=1


@pytest.mark.parametrize("fp8",[False,True])
@torch.inference_mode()
def test_session_real_decoder_stateful_replay_api_profiles_and_old_oracle(monkeypatch,fp8,record_property):
    from minisgl.planned.runner import ProgramRunner
    s=session(monkeypatch,fp8=fp8)
    s.gdn_pool.affine.fill_(float("nan"));s.gdn_pool.conv.fill_(float("nan"))
    oracle=Oracle(s)
    prepare=ProgramRunner.prepare_execution
    execute=ProgramRunner.execute_prepared
    def planning(runner,f,a,inp,features=None):
        oracle.input_plan,oracle.features=inp,features
        return prepare(runner,f,a,inp,features)
    def checked(runner):
        f=oracle.before(runner.program)
        # Strict replay/address gate: independently execute this same capacity
        # program eagerly from identical initial state, including all unused
        # pool bytes. The old-reference comparison separately allows the fixed
        # .025 relative-norm budget across four BF16 layers (tiny scan drift can
        # amplify even without a routing switch; do not tune per-cell atol).
        pools=(s.gdn_pool.affine,s.gdn_pool.conv,s.k_pool,s.v_pool)
        before=[x.clone() for x in pools]
        eager=runner.program.run().clone()
        expected=[x.clone() for x in pools]
        for x,v in zip(pools,before):x.copy_(v)
        result=execute(runner)
        indices=torch.tensor([i for i,row in enumerate(f.rows.output_rows) if row>=0],device=s.device)
        torch.testing.assert_close(result.logits,eager.index_select(0,indices),rtol=0,atol=0)
        for x,v in zip(pools,expected):torch.testing.assert_close(x,v,rtol=0,atol=0,equal_nan=True)
        oracle.after(result,f)
        return result
    monkeypatch.setattr(ProgramRunner,"prepare_execution",planning)
    monkeypatch.setattr(ProgramRunner,"execute_prepared",checked)
    common,a,b,temp=[s.create_block() for _ in range(4)]
    first=s.prefill_block(common,torch.tensor([1,2,3,4]));retained=first.clone()
    for step in range(32):
        if step%8==0:
            s.free_block(temp)
            count=36 if step%16==0 else 3
            jobs=[PrefillJob(temp,torch.arange(count)%128,[common])]
            g=group(([common,a,b],a),([common,b,a],b))
            s.mixed_step(jobs,g,torch.tensor([5+step,7+step]))
        else:
            g=group(([common,temp,b,a],a),([common,temp,a,b],b))
            s.decode_step(g,torch.tensor([5+step,7+step]))
        if step%8==3:
            merged=s.merge_blocks(temp,a)
            old_count=merged.num_tokens
            s.append_block(merged,merged)
            assert merged.num_tokens==old_count*2
            s.free_block(temp)
            s.append_block(temp,merged)
            del merged;gc.collect()
        torch.testing.assert_close(first,retained,rtol=0,atol=0)
    assert oracle.steps==33 and s.forward_count==33
    assert len(s.runners)==3
    assert sum(r.body.captures for r in s.runners.values())==3
    assert sum(r.body.replays for r in s.runners.values())==33
    assert sum(r.eager_count for r in s.runners.values())==0
    record_property("max_step_logits_relative_l2",oracle.max_error)
    record_property("max_step_kv_relative_l2",oracle.max_kv_error)
    record_property("max_step_tv",oracle.max_tv)


@torch.inference_mode()
@pytest.mark.parametrize("destination",["left","right","new"])
@pytest.mark.parametrize("self_append",[False,True])
def test_gpu_merge_ownership_aliases_and_original_math(monkeypatch,destination,self_append):
    s=session(monkeypatch,graphs=False)
    left,right=s.create_block(),s.create_block()
    s.prefill_block(left,torch.tensor([1,2,3]));s.prefill_block(right,torch.tensor([4,5]))
    if self_append:right=left
    l,r=s.store.registry[left._id].slot,s.store.registry[right._id].slot
    la,lb=s.gdn_pool.affine[:,l].clone().unbind(1)
    ra,rb=s.gdn_pool.affine[:,r].clone().unbind(1)
    expected=torch.stack((la@ra,lb@ra+rb),1)
    conv=s.gdn_pool.conv[:,r].clone()
    left_before,right_before=s.store.record(left),s.store.record(right)
    flatk,flatv=s.k_pool.flatten(1,2),s.v_pool.flatten(1,2)
    slots=torch.tensor(s._token_slots(left_before)+s._token_slots(right_before),device=s.device)
    oldk,oldv=flatk.index_select(1,slots),flatv.index_select(1,slots)
    n=left.num_tokens
    # Independent explicit partial-neox rotation for the RHS after copy.
    d=s._first_attention.rotary_dim
    freq=1./(1e4**(torch.arange(0,d,2,device=s.device,dtype=torch.float32)/d))
    angle=torch.cat((freq,freq))*left.mrope_span
    tail=oldk[:,n:,...,:d].float()
    half=torch.cat((-tail[...,d//2:],tail[...,:d//2]),-1)
    oldk[:,n:,...,:d]=(tail*angle.cos()+half*angle.sin()).to(s.dtype)
    dest=left if destination=="left" else right if destination=="right" else s.create_block()
    revision=dest.revision
    s._concatenate(left,right,dest)
    slot=s.store.registry[dest._id].slot
    torch.testing.assert_close(s.gdn_pool.affine[:,slot],expected,rtol=0,atol=0)
    torch.testing.assert_close(s.gdn_pool.conv[:,slot],conv,rtol=0,atol=0)
    outslots=torch.tensor(s._token_slots(s.store.record(dest)),device=s.device)
    torch.testing.assert_close(flatk.index_select(1,outslots),oldk,rtol=.008,atol=.002)
    torch.testing.assert_close(flatv.index_select(1,outslots),oldv,rtol=0,atol=0)
    assert dest.token_ids==list(left_before.token_ids+right_before.token_ids)
    assert dest.revision==revision+1
    for operand,before in ((left,left_before),(right,right_before)):
        if operand is not dest:assert s.store.record(operand)==before


@torch.inference_mode()
def test_session_prepare_and_partial_write_failure_are_not_fake_rollback(monkeypatch):
    from minisgl.planned.runner import ProgramRunner
    s=session(monkeypatch,graphs=False,slots=2,pages=8)
    a=s.create_block();s.prefill_block(a,torch.tensor([1,2,3]))
    old=s.store.record(a);free=s.store.free_pages
    with pytest.raises(ValueError,match="token"):
        s.prefill_block(a,torch.tensor([999,999]))
    assert s.store.record(a)==old and s.store.free_pages==free
    def partial(runner):
        target=s.store._pending.forward.writes[0].slot
        s.gdn_pool.affine[0,target].zero_()
        raise RuntimeError("injected after one layer")
    monkeypatch.setattr(ProgramRunner,"execute_prepared",partial)
    with pytest.raises(RuntimeError,match="injected"):s.prefill_block(a,torch.tensor([4,5]))
    with pytest.raises(RuntimeError,match="partial failed"):a.num_tokens
    assert s.store.free_pages==free
    s.free_block(a)
    assert a.num_tokens==0 and s.store.registry.free_slots==2


def test_existing_async_frontend_uses_planned_handles_and_whole_replays(monkeypatch):
    from minisgl.scheduler.async_engine import AsyncCacheEngine
    from minisgl.shared_cache.async_context import AsyncContext
    from minisgl.engine.sample import Sampler
    s=session(monkeypatch)
    engine=AsyncCacheEngine(session=s,sampler=Sampler(s.device,128),enable_mixed_batch=True)
    prompt,a,b,temp=[engine.create_block() for _ in range(4)]
    pf=engine.submit_prefill(torch.tensor([1,2,3]),write_to=prompt,return_logits=True)
    assert engine.tick()=="prefill" and pf.done()
    retained=pf.result().clone()
    first=AsyncContext([prompt,a,b],a)
    second=AsyncContext([prompt,b,a],b)
    for step in range(4):
        engine.free_block(temp)
        p=engine.submit_prefill(torch.tensor([4,5]),write_to=temp,cache_view=[prompt],return_logits=True)
        one=engine.submit_decode(first,5+step,return_logits=True,forbid_ids=[127])
        two=engine.submit_decode(second,6+step)
        assert engine.tick()=="mixed"
        assert one.result()[0]!=127 and isinstance(two.result(),int)
        assert p.done() and not engine.has_work
        # Returned normal tensors remain caller-mutable and don't alias graph
        # scratch. They survive subsequent executions/profile switches.
        one.result()[1][0]=1.
        torch.testing.assert_close(pf.result(),retained,rtol=0,atol=0)
    assert a.num_tokens==b.num_tokens==4
    assert s.forward_count==5 and s.decode_tokens==8 and s.prefill_input_tokens==11
    assert len(s.runners)==1
    runner=next(iter(s.runners.values()))
    assert runner.body.captures==1 and runner.body.replays==5
    # With an existing mixed profile, replay must not re-enter Python decoder
    # layers. Metadata upload remains outside; sampling/frontend still run.
    def forbidden():raise AssertionError("captured body called from Python during replay")
    runner.body.body=forbidden
    s.prefill_block(temp,torch.tensor([7,8]),context=[prompt])
    assert runner.body.replays==6


@torch.inference_mode()
def test_session_image_chunks_keep_cumulative_span_and_feature_order(monkeypatch):
    s=session(monkeypatch,graphs=True)
    image=s.create_block()
    features=torch.randn(4,128,device=s.device,dtype=s.dtype)
    s.prefill_batch([PrefillJob(image,torch.tensor([10,11]),mm_token_type_ids=torch.ones(2,dtype=torch.long),
        image_embeds=features[:2],mrope_rel=torch.tensor([[0,0],[0,0],[0,1]]),mrope_span=2)])
    assert image.num_tokens==2 and image.mrope_span==2
    s.prefill_batch([PrefillJob(image,torch.tensor([12,13]),mm_token_type_ids=torch.ones(2,dtype=torch.long),
        image_embeds=features[2:],mrope_rel=torch.tensor([[-2,-2],[-1,-1],[-2,-1]]),mrope_span=2)])
    assert image.num_tokens==4 and image.mrope_span==2
    s.prefill_block(image,torch.tensor([14,15]))
    assert image.num_tokens==6 and image.mrope_span==4
    assert image.token_ids==[10,11,12,13,14,15]
    assert len(s.runners)==1 and next(iter(s.runners.values())).body.captures==1


@torch.inference_mode()
def test_session_no_fla_is_whole_eager_with_consistent_api_commit(monkeypatch):
    from minisgl.models import qwen3_5_delta as delta
    monkeypatch.setattr(delta,"_fla_chunk",None);monkeypatch.setattr(delta,"_fla_recurrent",None)
    s=session(monkeypatch,fp8=False,graphs=True)
    a,b=s.create_block(),s.create_block()
    s.prefill_block(a,torch.tensor([1,2,3]))
    values=s.decode_step(group(([a,b],b)),torch.tensor([4]))
    assert torch.isfinite(values).all() and a.token_ids==[1,2,3] and b.token_ids==[4]
    assert all(r.body is None for r in s.runners.values())
    assert sum(r.eager_count for r in s.runners.values())==2
    assert not s.last_forward["used_graph"]


@torch.inference_mode()
def test_overflow_is_explicit_whole_eager_and_never_expands_graph_catalogue(monkeypatch):
    import weakref
    s=session(monkeypatch)
    a=s.create_block()
    first=s.prefill_block(a,torch.arange(200)%128)
    original=first.clone()
    assert len(s.runners)==0 and s.overflow_forwards==1
    assert s._overflow_runner.body is None and s.last_forward["eager_overflow"]
    previous=weakref.ref(s._overflow_runner)
    s.free_block(a)
    s.prefill_block(a,torch.arange(201)%128)
    gc.collect()
    assert previous() is None and len(s.runners)==0 and s.overflow_forwards==2
    torch.testing.assert_close(first,original,rtol=0,atol=0)
    s.free_block(a);s.prefill_block(a,torch.tensor([1,2]))
    assert len(s.runners)==1 and s.last_forward["used_graph"] and not s.last_forward["eager_overflow"]
    s.allow_eager_overflow=False
    before=s.store.record(a)
    with pytest.raises(ValueError,match="exceeds configured"):s.prefill_block(a,torch.arange(200)%128)
    assert s.store.record(a)==before


@torch.inference_mode()
def test_runtime_memory_accounting_deduplicates_views_and_stays_bounded(monkeypatch):
    from minisgl.planned.accounting import cuda_storages
    s=session(monkeypatch)
    a,b=s.create_block(),s.create_block()
    s.prefill_block(a,torch.tensor([1,2,3]))
    g=group(([a,b],b))
    s.decode_step(g,torch.tensor([4]))
    report=s.memory_report()
    assert report["gdn_live_slots"]==2
    assert report["weight_bytes"]>0 and report["explicit_owned_bytes"]>report["pool_bytes"]
    assert report["gdn_reserved_bytes"]==s.gdn_pool.reserved_bytes
    assert len(report["profiles"])==2
    one=cuda_storages(s.gdn_pool.affine)
    views=cuda_storages([s.gdn_pool.affine[:,i] for i in range(s.gdn_pool.shape.slots)])
    assert one==views
    storage=cuda_storages([r.program for r in s.runners.values()])
    for i in range(12):s.decode_step(g,torch.tensor([5+i]))
    assert cuda_storages([r.program for r in s.runners.values()])==storage
    after=s.memory_report()
    assert after["explicit_owned_bytes"]==report["explicit_owned_bytes"]
    assert sum(p["captures"] for p in after["profiles"])==2


@torch.inference_mode()
def test_session_orders_foreign_stream_inputs_and_all_mutations(monkeypatch):
    s=session(monkeypatch)
    image,copy=s.create_block(),s.create_block()
    caller=torch.cuda.Stream(device=s.device)
    original=s.store.mark_submitted
    submissions=[]
    def submitted(tx):
        assert torch.cuda.current_stream(s.device)==s.stream
        submissions.append(tx.slots.plan)
        return original(tx)
    monkeypatch.setattr(s.store,"mark_submitted",submitted)
    with torch.cuda.stream(caller):
        features=torch.randn(4,128,device=s.device,dtype=s.dtype)
        ids=torch.arange(4,device=s.device)+10
        logits=s.prefill_batch([PrefillJob(image,ids,mm_token_type_ids=torch.ones(4,dtype=torch.long),
            image_embeds=features,mrope_rel=torch.tensor([[0,0,0,0],[0,0,1,1],[0,1,0,1]]))])[0]
        assert torch.cuda.current_stream(s.device)==caller
        assert torch.isfinite(logits).all()
        s.append_block(copy,image)
    assert image.token_ids==copy.token_ids==[10,11,12,13] and len(submissions)==2
    p=next(iter(s.runners.values())).program
    torch.testing.assert_close(p.features[:4],features,rtol=0,atol=0)
    s.free_block(copy)
    with torch.cuda.stream(caller):s.decode_step(group(([image,copy],copy)),torch.tensor([14],device=s.device))
    assert copy.token_ids==[14] and len(submissions)==3
