"""CPU Attention visibility is independent of GDN pre-update peer states."""
from dataclasses import replace

import pytest

from minisgl.planned.forward_plan import BlockState,PrefillRequest,DecodeRequest,PlanCapacity,prepare_forward
from minisgl.planned.attention_plan import KVBlock,AttentionCapacity,prepare_attention


def scenario(*,image=False):
    lengths={0:2,3:3,4:3,5:2,6:0,7:0}
    kv={b:KVBlock(b,n,n,(b,8) if b==3 else (b,)) for b,n in lengths.items()}
    states={b:BlockState(b,b,bool(n),bool(n)) for b,n in lengths.items()}
    pf=[PrefillRequest((0,4),6,4 if image else 2),PrefillRequest((0,),3,2)]
    dec=[DecodeRequest((0,6,5,4),4),DecodeRequest((0,3,4,5),5),DecodeRequest((0,6),7)]
    forward=prepare_forward(states,capacity=PlanCapacity(3,16,3,8,16),prefill=pf,decode=dec)
    cap=AttentionCapacity(4,64,4,16)
    return kv,states,forward,cap


def test_four_plans_preserve_old_pf_and_new_decode_peer_visibility():
    kv,states,f,cap=scenario()
    original=dict(kv)
    p=prepare_attention(kv,f,cap)
    assert kv==original
    assert p.prefill_context.query_lengths[:3]==(2,2,2)
    assert p.prefill_context.pages[:3]==((0,),(4,),(0,))
    assert p.prefill_context.last_page_lengths[:3]==(2,3,2)  # D4 is still old
    assert p.prefill_self.pages[:2]==((6,),(3,8))
    assert p.prefill_self.last_page_lengths[:2]==(2,1)
    # In D, all three writers' new KV are visible, unlike GDN composition.
    assert p.decode_main.pages[:4]==((0,),(6,),(5,),(4,))
    assert p.decode_main.last_page_lengths[:4]==(2,2,3,4)
    assert p.decode_main.positions[0][:4]==(10,8,6,3)
    assert p.decode_main.source_rows[:4]==(16,16,16,16)
    assert p.decode_aux.query_lengths==(1,0,0)
    assert p.decode_aux.pages[0]==(28,)  # block7/page7 -> flattened slot 28
    assert p.decode_aux.positions[0][0]==0
    assert p.decode_aux.source_rows[0]==18
    assert p.decode_aux.destinations[0]==18*5+2
    assert p.write_token_slots[:4]==(24,25,15,32)
    assert p.write_token_slots[16:]==(19,22,28)
    assert p.key_positions[0][:4]==(0,1,3,4)
    assert p.key_positions[0][16:]==(3,2,0)
    assert p.post_lengths==((6,2,2),(3,5,5),(4,4,4),(5,3,3),(7,1,1))


def test_image_mrope_uses_span_not_token_count():
    kv,states,f,cap=scenario(image=True)
    positions=((0,0,0,0),(0,0,1,1),(0,1,0,1))
    p=prepare_attention(kv,f,cap,prefill_mrope={0:positions})
    assert tuple(axis[:4] for axis in p.key_positions)==positions
    assert p.post_lengths[0]==(6,4,2)
    assert p.decode_main.positions[0][:4]==(10,8,6,3)
    assert p.prefill_context.positions[1][:4]==(5,5,6,6)
    assert p.prefill_context.positions[2][:4]==(5,6,5,6)
    assert p.prefill_self.last_page_lengths[0]==4


def test_new_decode_page_visible_only_after_pf():
    kv,states,f,cap=scenario()
    kv[4]=KVBlock(4,4,4,(4,9))
    p=prepare_attention(kv,f,cap)
    assert p.prefill_context.pages[1]==(4,)
    assert p.prefill_context.last_page_lengths[1]==4
    assert p.decode_main.pages[3]==(4,9)
    assert p.decode_main.last_page_lengths[3]==1
    assert p.write_token_slots[16]==36


@pytest.mark.parametrize("problem",["append","segments","references","population","alias","committed","coordinates"])
def test_invalid_attention_plan_rejected_before_mutation(problem):
    kv,states,f,cap=scenario()
    kwargs={}
    if problem=="append":kv[3]=replace(kv[3],pages=(3,))
    if problem=="segments":cap=replace(cap,context_segments=1)
    if problem=="references":cap=replace(cap,page_references=2)
    if problem=="population":kv[6]=KVBlock(6,1,1,(6,))
    if problem=="alias":kv[7]=replace(kv[7],pages=(6,))
    if problem=="committed":kv[7]=replace(kv[7],pages=(0,))
    if problem=="coordinates":kwargs['prefill_mrope']={0:((0,),)*3}
    original=dict(kv)
    with pytest.raises(ValueError):prepare_attention(kv,f,cap,**kwargs)
    assert kv==original


def test_empty_occupied_phases_have_same_capacity_tables_and_one_kv_lookup():
    kv,states,f,cap=scenario()
    class Counted(dict):
        count=0
        def __getitem__(self,k):self.count+=1;return super().__getitem__(k)
    source=Counted(kv)
    full=prepare_attention(source,f,cap)
    assert source.count==len(f.blocks)
    for pf,dec in ((f.prefill_requests,()),((),f.decode_requests),((),())):
        now=prepare_forward(states,capacity=f.capacity,prefill=pf,decode=dec)
        p=prepare_attention(kv,now,cap)
        assert len(p.write_token_slots)==len(full.write_token_slots)
        for name in ("prefill_context","prefill_self","decode_main","decode_aux"):
            want,got=getattr(full,name),getattr(p,name)
            assert len(got.query_lengths)==len(want.query_lengths)
            assert len(got.source_rows)==len(want.source_rows)
            assert sum(got.query_lengths)==sum(row>=0 for row in got.source_rows)
            assert all(d<0 for row,d in zip(got.source_rows,got.destinations) if row<0)


def test_image_chunk_relative_positions_can_be_negative_without_decreasing_span():
    kv,states,original,cap=scenario()
    f=prepare_forward(states,capacity=original.capacity,prefill=[PrefillRequest((),3,2)])
    p=prepare_attention(kv,f,cap,prefill_mrope={0:((-2,-2),(-2,-1),(-1,-2))})
    assert p.post_lengths==((3,5,3),)
    assert tuple(axis[:2] for axis in p.key_positions)==((1,1),(1,2),(2,1))
    with pytest.raises(ValueError,match="mRoPE"):
        prepare_attention(kv,f,cap,prefill_mrope={0:((-4,-2),(-2,-1),(-1,-2))})
