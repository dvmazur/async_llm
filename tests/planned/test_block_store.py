import gc
from dataclasses import replace
import pytest

from minisgl.planned.block_store import BlockStore
from minisgl.planned.forward_plan import PlanCapacity,PrefillRequest,DecodeRequest
from minisgl.planned.attention_plan import AttentionCapacity
from test_slots import Completion


def make(slots=4,pages=16):return BlockStore(block_slots=slots,page_size=4,page_capacity=pages)


def begin(s,b,ids,**kw):
    return s.begin_forward(capacity=PlanCapacity(2,64,2,8,s.registry.capacity),
        attention_capacity=AttentionCapacity(4,64,s.page_size,s.page_capacity),
        prefill=[PrefillRequest((),s.block_id(b),len(ids))],tokens={b._id:ids},**kw)


def populate(s,b,ids=(1,2,3),**kw):
    tx=begin(s,b,ids,**kw);s.mark_submitted(tx);assert s.finish(tx,Completion(True))


def test_forward_token_kv_gdn_metadata_commits_together():
    s=make();a=s.create();b=s.create()
    tx=s.begin_forward(capacity=PlanCapacity(2,64,2,8,4),
        attention_capacity=AttentionCapacity(4,64,4,16),
        prefill=[PrefillRequest((),a._id,4)],decode=[DecodeRequest((a._id,),b._id)],
        tokens={a._id:(1,2,3,4),b._id:(5,)},prefill_mrope={0:((0,0,0,0),(0,0,1,1),(0,1,0,1))})
    assert a.num_tokens==b.num_tokens==0 and a.num_pages==b.num_pages==0
    s.mark_submitted(tx)
    assert not s.finish(tx,Completion(False))
    assert a.revision==b.revision==0
    assert s.finish(tx,Completion(True))
    assert a.token_ids==[1,2,3,4] and a.mrope_span==2
    assert b.token_ids==[5] and b.mrope_span==1
    assert a.revision==b.revision==1
    assert s.registry.token_count(a._id)==a.num_tokens
    a.token_ids.append(99)
    assert a.num_tokens==4 # public metadata snapshot, not a mutable alias


def test_page_and_slot_reservation_failures_are_atomic():
    s=make(slots=2,pages=2);a=s.create();b=s.create()
    populate(s,a)
    state=s.record(a);free_slots=s.registry.free_slots
    with pytest.raises(RuntimeError,match="KV pool exhausted"):begin(s,b,(4,5))
    assert s.registry.free_slots==free_slots and s.registry[b._id].slot is None
    assert s.record(a)==state and s.free_pages==0
    s.clear(a);assert s.free_pages==1
    with pytest.raises(ValueError,match="mRoPE"):begin(s,b,(4,),prefill_mrope={0:((-1,),)*3})
    assert s.free_pages==1 and s.registry.free_slots==2
    with pytest.raises(ValueError,match="number of committed"):
        s.begin_forward(capacity=PlanCapacity(1,4,0,4,2),attention_capacity=AttentionCapacity(2,8,4,2),
                        prefill=[PrefillRequest((),b._id,2)],tokens={b._id:(1,)})
    assert s.free_pages==1 and s.registry.free_slots==2


def test_failure_after_submission_keeps_old_metadata_and_pins_then_invalidates():
    s=make();a=s.create();populate(s,a)
    old=s.record(a);before=s.free_pages
    tx=begin(s,a,(4,5,6));s.mark_submitted(tx);s.mark_failed(tx)
    assert s.free_pages==before-1
    with pytest.raises(RuntimeError,match="partial failed"):s.record(a)
    with pytest.raises(RuntimeError,match="in flight"):a.clear()
    assert not s.finish(tx,Completion(False))
    assert s.finish(tx,Completion(True)) and s.free_pages==before
    assert s._records[a._id]==old
    with pytest.raises(RuntimeError,match="partial failed"):begin(s,a,(7,))
    a.clear();populate(s,a,(9,));assert a.token_ids==[9]


@pytest.mark.parametrize("operation",["commit","cancel","fail"])
def test_handle_gc_retires_pages_only_after_last_reader(operation):
    s=make();a=s.create();populate(s,a);b=s.create()
    aid,bid=a._id,b._id
    tx=s.begin_forward(capacity=PlanCapacity(0,0,1,4,4),attention_capacity=AttentionCapacity(2,8,4,16),
        decode=[DecodeRequest((aid,),bid)],tokens={bid:(8,)})
    del a,b;gc.collect()
    assert s.registry.live_handles==2 and s.free_pages==13
    if operation=="cancel":s.cancel(tx)
    else:
        s.mark_submitted(tx)
        if operation=="fail":s.mark_failed(tx)
        assert not s.finish(tx,Completion(False))
        assert s.finish(tx,Completion(True))
    assert s.registry.live_handles==0 and s.free_pages==15 and not s._records


@pytest.mark.parametrize("destination",["left","right","fresh"])
@pytest.mark.parametrize("self_append",[False,True])
def test_concatenation_snapshots_aliases_and_commits_once(destination,self_append):
    s=make();a=s.create();populate(s,a,(1,2,3))
    b=a if self_append else s.create()
    if b is not a:populate(s,b,(4,5),prefill_mrope={0:((0,0),)*3})
    d=a if destination=="left" else b if destination=="right" else s.create()
    expected=a.token_ids+b.token_ids;span=a.mrope_span+b.mrope_span
    before=s.record(d);revision=d.revision
    tx=s.begin_concatenation(a,b,d)
    assert s.record(d)==before
    assert tx.before[a._id].token_ids==tuple(a.token_ids)
    s.mark_submitted(tx);assert s.finish(tx,Completion(True))
    assert d.token_ids==expected and d.mrope_span==span and d.revision==revision+1
    assert s.registry.token_count(d._id)==len(expected)
    pages=[p for record in s._records.values() for p in record.pages]
    assert len(set(pages))==len(pages) and 0 not in pages


def test_mutation_overflow_and_cancel_return_new_slots_and_pages():
    s=make(slots=3,pages=3);a=s.create();b=s.create();d=s.create()
    populate(s,a,(1,2,3,4));populate(s,b,(5,6,7,8))
    with pytest.raises(RuntimeError,match="KV pool exhausted"):s.begin_concatenation(a,b,d)
    assert s.registry.free_slots==1 and d.num_tokens==0
    b.clear();tx=s.begin_concatenation(a,b,d);s.cancel(tx)
    assert s.registry.free_slots==2 and s.free_pages==1 and d.num_tokens==0


def test_foreign_handles_and_repeated_clear_do_not_release_other_owners():
    a,b=make(),make();x,y=a.create(),b.create()
    populate(a,x);populate(b,y)
    with pytest.raises(ValueError,match="another runtime"):a.clear(y)
    x.clear();x.clear();assert a.free_pages==15 and a.registry.free_slots==4
    assert y.num_tokens==3
