from dataclasses import replace
import random
import pytest

from minisgl.planned.catalogue import ProfileCatalogue,RuntimeProfile
from minisgl.planned.forward_plan import PlanCapacity,PrefillRequest,DecodeRequest
from minisgl.planned.attention_plan import AttentionCapacity
from minisgl.planned.block_store import BlockStore
from test_slots import Completion


def test_catalogue_selects_existing_profiles_not_topology_variants():
    att=AttentionCapacity(6,128,4,64)
    profiles=tuple(RuntimeProfile(PlanCapacity(r,p,d,8,8),att) for r,p,d in ((0,0,4),(3,16,4),(3,192,4)))
    cat=ProfileCatalogue(profiles)
    s=BlockStore(block_slots=8,page_size=4,page_capacity=64)
    blocks=[s.create() for _ in range(8)]
    for b in blocks[:4]:
        pf=(PrefillRequest((),b._id,5),)
        tx=s.begin_forward(capacity=profiles[1].forward,attention_capacity=att,prefill=pf,tokens={b._id:range(5)})
        s.mark_submitted(tx);s.finish(tx,Completion(True))
    rng=random.Random(48)
    for i in range(120):
        roots=rng.sample(range(4),rng.randrange(4))
        n=rng.choice([0,1,3,16,17,36,137])
        pf=(PrefillRequest(tuple(roots),4,n),) if n else ()
        dec=(DecodeRequest(tuple(roots)+(4,5),5),DecodeRequest((0,5,6),6))
        p=cat.select(s._records,pf,dec)
        assert p is profiles[0 if n==0 else 1 if n<=16 else 2]
        tokens={r.write_to:tuple(range(r.length)) for r in pf}
        tokens.update({r.write_to:(9,) for r in dec})
        tx=s.begin_forward(capacity=p.forward,attention_capacity=p.attention,prefill=pf,decode=dec,tokens=tokens)
        assert tx.attention # selected metadata really fits all four consumers
        s.cancel(tx)
    assert cat.profiles==profiles


def test_page_references_include_shared_repeats_and_dummy_requests():
    from minisgl.planned.block_store import BlockRecord
    att=AttentionCapacity(4,8,4,64)
    small=RuntimeProfile(PlanCapacity(0,0,4,8,8),att)
    large=replace(small,attention=replace(att,page_references=64))
    cat=ProfileCatalogue((small,large))
    records={0:BlockRecord(0,tuple(range(24)),24,tuple(range(1,7))),1:BlockRecord(1),2:BlockRecord(2)}
    dec=[DecodeRequest((0,),1),DecodeRequest((0,),2)]
    # Two references to six pages each plus fourteen padded requests. Eight
    # distinct pages being sufficient physically does not make ref budget=8 fit.
    assert cat.select(records,(),dec)==large
    with pytest.raises(ValueError,match="exceeds configured"):
        ProfileCatalogue((small,)).select(records,(),dec)


def test_catalogue_rejects_different_pool_owners_and_unbounded_duplicates():
    p=RuntimeProfile(PlanCapacity(1,16,4,8,8),AttentionCapacity(4,64,4,64))
    with pytest.raises(ValueError):ProfileCatalogue(())
    with pytest.raises(ValueError):ProfileCatalogue((p,p))
    with pytest.raises(ValueError,match="share physical"):
        ProfileCatalogue((p,replace(p,forward=replace(p.forward,block_slots=16))))


def test_eager_overflow_bounds_exact_work_without_adding_graph_profiles():
    from minisgl.planned.block_store import BlockRecord
    p=RuntimeProfile(PlanCapacity(1,4,1,1,8),AttentionCapacity(1,4,4,64))
    cat=ProfileCatalogue((p,))
    records={0:BlockRecord(0,tuple(range(20)),20,tuple(range(1,6))),1:BlockRecord(1),2:BlockRecord(2)}
    pf=[PrefillRequest((0,),1,36)];dec=[DecodeRequest((0,1,2),2)]
    with pytest.raises(ValueError):cat.select(records,pf,dec)
    overflow=cat.eager_overflow_profile(records,pf,dec)
    assert overflow.forward==PlanCapacity(1,36,1,2,8)
    assert overflow.attention.context_segments==3
    assert overflow.attention.page_references==15 # old root5 + new prefix9 + new D1
    assert cat.profiles==(p,)
