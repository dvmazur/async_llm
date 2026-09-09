"""Full-body execution and failure/lifetime gates, not partial graph timing."""
import gc
import weakref
import pytest
import torch

from test_session import session,group
from minisgl.planned.paged_attention import AttentionRecipeChanged

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@torch.inference_mode()
def test_replay_does_not_reenter_any_python_layer_body(monkeypatch):
    s=session(monkeypatch)
    a,b=s.create_block(),s.create_block()
    s.prefill_block(a,torch.tensor([1,2,3]))
    workers=group(([a,b],b))
    s.decode_step(workers,torch.tensor([4]))
    runner=next(r for profile,r in s.runners.items() if profile.forward.prefill_tokens==0)
    p=runner.program
    def forbidden(*args,**kwargs):raise AssertionError("Python layer body entered during replay")
    monkeypatch.setattr(runner.body,"body",forbidden)
    monkeypatch.setattr(p.gdn,"run",forbidden)
    monkeypatch.setattr(p.attention,"run",forbidden)
    for op in p.bound_mlp:monkeypatch.setattr(op,"run",forbidden)
    for op in p.bound_attention:
        if op is not None:monkeypatch.setattr(op,"run",forbidden)
    before=runner.body.replays
    for token in range(5,13):
        logits=s.decode_step(workers,torch.tensor([token]))
        assert torch.isfinite(logits).all() and s.last_forward["used_graph"]
    assert runner.body.captures==1 and runner.body.replays==before+8
    assert runner.eager_count==0


@torch.inference_mode()
def test_native_recipe_rejection_preserves_pool_and_allows_clean_retry(monkeypatch):
    s=session(monkeypatch)
    a=s.create_block();s.prefill_block(a,torch.tensor([1,2,3]))
    runner=next(iter(s.runners.values()))
    op=runner.program.attention.ops[1]
    before=[t.clone() for t in (s.gdn_pool.affine,s.gdn_pool.conv,s.k_pool,s.v_pool)]
    record=s.store.record(a);pages=s.store.free_pages;revision=a.revision
    def reject():raise AttentionRecipeChanged("injected native recipe change")
    with monkeypatch.context() as patch:
        patch.setattr(op,"validate_recipe",reject)
        with pytest.raises(AttentionRecipeChanged):s.prefill_block(a,torch.tensor([4]))
    for actual,expected in zip((s.gdn_pool.affine,s.gdn_pool.conv,s.k_pool,s.v_pool),before):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
    assert s.store.record(a)==record and a.revision==revision and s.store.free_pages==pages
    assert s.store._pending is None and not runner.prepared
    result=s.prefill_block(a,torch.tensor([4]))
    assert torch.isfinite(result).all() and a.token_ids==[1,2,3,4]
    assert runner.body.captures==1 and runner.body.replays==2


@torch.inference_mode()
def test_repeated_runtime_destruction_keeps_outputs_not_pool_ownership(monkeypatch):
    def one_lifetime():
        s=session(monkeypatch,slots=4,pages=64)
        a,b=s.create_block(),s.create_block()
        s.prefill_block(a,torch.tensor([1,2,3]))
        first=s.decode_step(group(([a,b],b)),torch.tensor([4]))
        saved=first.clone()
        for step in range(16):
            s.decode_step(group(([a,b],b)),torch.tensor([5+step]))
            if step%4==3:
                s.append_block(a,b);s.free_block(b)
        torch.testing.assert_close(first,saved,rtol=0,atol=0)
        s.free_block(a);s.free_block(b)
        assert s.store.registry.free_slots==4 and s.store.free_pages==63
        return weakref.ref(s.gdn_pool.affine),weakref.ref(s.k_pool),first,saved
    for _ in range(5):
        affine,kv,output,expected=one_lifetime()
        torch.cuda.synchronize();gc.collect()
        assert affine() is None and kv() is None
        torch.testing.assert_close(output,expected,rtol=0,atol=0)
