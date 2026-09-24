import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.choptree import ACTION_NAMES
from pipelines.probe import messages_for
from pipelines.choptree_target_actor import ControlMemory,TargetActorReadout,marked_view


def test_target_memory_is_bounded_and_uses_only_public_outcomes_and_pixels():
    memory=ControlMemory()
    a=np.zeros((224,224,3),dtype=np.uint8)
    b=np.full_like(a,100)
    memory.inspect_change(a,a)
    memory.observe('up',0);memory.inspect_change(a,b)
    memory.observe('down',0);memory.inspect_change(b,a)
    assert memory.visual_outcome['similar_view_actions_ago']==[2]
    memory.last_plan={'action':'left','expected':'new side trunk'}
    for _ in range(500):
        memory.observe('forward',0);memory.inspect_change(a,a)
    record=json.loads(memory.text())
    assert len(record['recent'])==len(memory._views)==12
    assert record['since_wood']==502
    assert record['previous_plan']==memory.last_plan
    assert record['visual_outcome']['almost_unchanged']
    memory.observe('dig',1)
    assert memory.since_reward==0 and memory.total_reward==1


def test_model_coordinate_marker_does_not_modify_source():
    image=np.full((224,224,3),75,dtype=np.uint8)
    out=marked_view(image,dict(x=750,y=500))
    assert out.shape==(448,448,3) and np.all(image==75)
    assert (out[224,330]==[0,220,255]).all()
    assert (out[224,214]==[255,35,35]).all()
    with pytest.raises(ValueError):marked_view(image,dict(x=1001,y=500))


def test_actor_not_python_selects_action_from_model_target(monkeypatch):
    import pipelines.choptree_target_actor as module
    async def planner(messages,**kwargs):
        return dict(kind='wood',target='right trunk',x=800,y=500,reason='close',next_goal='aim',
            proximity='close',obstacle='none'),200
    async def actor(messages,**kwargs):
        assert '800' in messages[1]['content']
        assert 'right trunk' not in messages[1]['content']
        # Deliberately disagree with the apparent geometric direction: no scripted override.
        return dict(check='target invalid',action='left',expected='find alternative'),300
    monkeypatch.setattr(module,'JsonRole',lambda b,r,n,s:planner if n=='target_planner' else actor)
    memory=ControlMemory()
    reader=TargetActorReadout(None,SimpleNamespace(log=lambda *a:None),memory)
    image=np.zeros((224,224,3),dtype=np.uint8)
    result,rows=asyncio.run(reader.generate_action(messages_for(image,image),None,
        generator=None,temperature=.7))
    assert result==ACTION_NAMES.index('left') and rows==500
    assert memory.last_plan['x']==800 and memory.last_plan['action']=='left'


@pytest.mark.parametrize('custom',[False,True])
def test_role_settings_forwarded_and_recovery_threshold_preserved(monkeypatch,custom):
    import pipelines.choptree_target_actor as module
    calls=[]
    async def planner(messages,**kwargs):
        calls.append(('planner',kwargs))
        return dict(kind='wood',x=500,y=500,proximity='close',obstacle='none'),100
    async def actor(messages,**kwargs):
        calls.append(('actor',kwargs))
        return dict(action='dig',expected='wood'),30
    monkeypatch.setattr(module,'JsonRole',lambda b,r,n,s:planner if n=='target_planner' else actor)
    options=dict(planner_temperature=.4,recovery_temperature=.8,recovery_after=3,max_role_tokens=512) if custom else {}
    memory=ControlMemory()
    reader=TargetActorReadout(None,SimpleNamespace(log=lambda *a:None),memory,**options)
    image=np.zeros((224,224,3),dtype=np.uint8)
    async def run():
        for since in [reader.recovery_after-1,reader.recovery_after]:
            memory.since_reward=since
            await reader.generate_action(messages_for(image,image),None,generator=17,temperature=.6)
    asyncio.run(run())
    assert [kw['temperature'] for role,kw in calls if role=='planner']==([.4,.8] if custom else [.5,.9])
    assert [kw['temperature'] for role,kw in calls if role=='actor']==[.6,.6]
    assert all(kw['max_tokens']==(512 if custom else 768) and kw['generator']==17 for _,kw in calls)


@pytest.mark.parametrize('non_text',[False,True])
def test_plain_text_role_modality_table_normalization(non_text):
    from experiment_runner.probe_readout import ProbeReadout
    class LLM:
        def __init__(self):self.processor=self
        def apply_chat_template(self,*args,**kwargs):
            return dict(input_ids=SimpleNamespace(numel=lambda:10),
                mm_token_type_ids=SimpleNamespace(any=lambda:non_text))
        async def __call__(self,**kwargs):
            assert 'mm_token_type_ids' not in kwargs
            return 'output'
    adapter=ProbeReadout(SimpleNamespace(llm=LLM()))
    call=adapter.prefill_action([],SimpleNamespace(raw='block'))
    if non_text:
        with pytest.raises(ValueError,match='modality'):asyncio.run(call)
    else:
        assert asyncio.run(call)==('output',10)
