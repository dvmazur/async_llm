"""Independent continuously evolving old/new state, not per-step reset parity."""
import pytest
import torch

from reference_legacy_session import legacy_session
from test_session import session,group
from minisgl.shared_cache.session import PrefillJob

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize("fp8",[False,True])
@torch.inference_mode()
def test_independent_legacy_and_planned_33_forward_trajectories(monkeypatch,fp8,record_property):
    import minisgl.core as core
    new=session(monkeypatch,fp8=fp8)
    old=legacy_session(new.model,page_size=new.page_size,pages=new.store.page_capacity)
    monkeypatch.setattr(core,"_GLOBAL_CTX",old.engine.ctx)
    nb=[new.create_block() for _ in range(4)]
    ob=[old.create_block() for _ in range(4)]
    maxima=dict(logits=0.,tv=0.,affine=0.,conv=0.,kv=0.)
    top1_changed=0
    def error(a,b,label):
        assert a.shape==b.shape,(label,a.shape,b.shape)
        a,b=a.float(),b.float()
        assert torch.isfinite(a).all() and torch.isfinite(b).all(),label
        value=float((a-b).norm()/b.norm().clamp_min(1e-10))
        maxima[label]=max(maxima[label],value)
        assert value<.025,(label,value)
    def check(a,b):
        nonlocal top1_changed
        error(a,b,"logits")
        tv=float(((a.float().softmax(-1)-b.float().softmax(-1)).abs().sum(-1)/2).max())
        maxima["tv"]=max(maxima["tv"],tv);assert tv<.006,tv
        top1_changed+=int((a.argmax(-1)!=b.argmax(-1)).sum())
        for current,reference in zip(nb,ob):
            assert current.token_ids==reference.token_ids
            assert current.num_tokens==reference.num_tokens and current.mrope_span==reference.mrope_span
            if not current.num_tokens:continue
            slot=new.store.registry[current._id].slot
            for layer in range(2):
                pair=torch.stack([x[0] for x in reference.linear_affine[layer]])
                error(new.gdn_pool.affine[layer,slot],pair,"affine")
                error(new.gdn_pool.conv[layer,slot],reference.linear_conv_state[layer],"conv")
                new_slots=torch.tensor(new._token_slots(new.store.record(current)),device=new.device)
                old_slots=reference.token_slots_tensor().long()
                for pool,getter in ((new.k_pool,old.kv_cache.k_cache),(new.v_pool,old.kv_cache.v_cache)):
                    error(pool[layer].flatten(0,1).index_select(0,new_slots),
                          getter(layer).flatten(0,1).index_select(0,old_slots),"kv")
    # Use the existing shared-cache prefill entry point even for the first
    # block; the minimal oracle shell intentionally has no serving backend.
    check(new.prefill_block(nb[0],torch.tensor([1,2,3,4])),
          old._prefill_batch_fused([PrefillJob(ob[0],torch.tensor([1,2,3,4]))])[0])
    for step in range(32):
        def perform(s,blocks):
            common,a,b,temp=blocks
            ids=torch.tensor([5+step,7+step])
            if step%8==0:
                s.free_block(temp)
                n=36 if step%16==0 else 3
                pf,dec=s.mixed_step([PrefillJob(temp,torch.arange(n)%128,[common])],
                                    group(([common,a,b],a),([common,b,a],b)),ids)
                return torch.cat([*pf,dec])
            return s.decode_step(group(([common,temp,b,a],a),([common,temp,a,b],b)),ids)
        check(perform(new,nb),perform(old,ob))
        if step%8==3:
            for s,blocks in ((new,nb),(old,ob)):
                merged=s.merge_blocks(blocks[3],blocks[1])
                s.append_block(merged,merged)
                s.free_block(blocks[3]);s.append_block(blocks[3],merged)
                s.free_block(merged)
    assert new.forward_count==33
    assert sum(r.body.captures for r in new.runners.values())==3
    for k,v in maxima.items():record_property("max_"+k,v)
    record_property("changed_top1_rows",top1_changed)
