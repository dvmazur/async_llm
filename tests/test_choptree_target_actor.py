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


@pytest.mark.parametrize('plan,assessment', [
    ('wood: right trunk, x=800 y=500, close, no obstacle', 'Turn right first.'),
    ('unfinished {"target":', '\t'*10576), ('', '')])
def test_readout_not_parser_selects_action_even_after_partial_or_empty_roles(monkeypatch,plan,assessment):
    import pipelines.choptree_target_actor as module
    async def planner(messages,**kwargs):
        return plan,200
    async def actor(messages,**kwargs):
        assert (plan or '[empty plan]') in messages[1]['content']
        return assessment,300
    monkeypatch.setattr(module,'TextRole',lambda b,r,n:planner if n=='target_planner' else actor)
    memory=ControlMemory()
    backend=SimpleNamespace(encode=lambda name:[ACTION_NAMES.index(name)])
    reader=TargetActorReadout(backend,SimpleNamespace(log=lambda *a:None),memory)
    async def prefill(messages,target):
        assert messages[-2]['role']=='assistant'
        assert messages[-2]['content']==(assessment or '[No assessment produced.]')
        return 'model output',100
    reader.prefill_action=prefill
    def sample(output,ids,**kwargs):
        assert output=='model output' and ids==list(range(8))
        assert kwargs['temperature']==.7
        # Deliberately disagree with prose; no parser or scripted action override.
        return ACTION_NAMES.index('left'),[1/8]*8
    reader.sample_action=sample
    image=np.zeros((224,224,3),dtype=np.uint8)
    result,rows=asyncio.run(reader.generate_action(messages_for(image,image),None,
        generator=None,temperature=.7))
    assert result==ACTION_NAMES.index('left') and rows==600
    assert memory.last_plan['plan']==plan and memory.last_plan['action']=='left'


@pytest.mark.parametrize('custom',[False,True])
def test_role_settings_forwarded_and_recovery_threshold_preserved(monkeypatch,custom):
    import pipelines.choptree_target_actor as module
    calls=[]
    async def planner(messages,**kwargs):
        calls.append(('planner',kwargs))
        return 'wood x=500 y=500 close no obstacle',100
    async def actor(messages,**kwargs):
        calls.append(('actor',kwargs))
        return 'Aligned: dig for wood.',30
    monkeypatch.setattr(module,'TextRole',lambda b,r,n:planner if n=='target_planner' else actor)
    options=dict(planner_temperature=.4,recovery_temperature=.8,recovery_after=3,max_role_tokens=512) if custom else {}
    memory=ControlMemory()
    reader=TargetActorReadout(None,SimpleNamespace(log=lambda *a:None),memory,**options)
    reader.action_ids=list(range(8))
    async def prefill(messages,target):return 'output',20
    reader.prefill_action=prefill
    reader.sample_action=lambda *args,**kw:(3,[1/8]*8)
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
