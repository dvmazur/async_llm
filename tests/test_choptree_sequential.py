import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from experiment_runner.blocks import BlockHandle
from experiment_runner.probe_readout import ProbeReadout
from pipelines.choptree import ACTION_NAMES
from pipelines.choptree_sequential import SequentialTargetActorReadout,ContextLimitError,CHAT_END
from pipelines.choptree_target_actor import ControlMemory
from pipelines.probe import messages_for,ProbePipeline


class Backend:
    def __init__(self,limit=100000):
        self.live=[];self.operations=[];self.metrics=dict(restricted_readouts=0)
        owner=self
        class LLM:
            def __init__(self):
                self.engine=SimpleNamespace(config=SimpleNamespace(max_seq_len=limit,
                    model_config=SimpleNamespace(vocab_size=100)))
                self.processor=self
                self.tokenizer=SimpleNamespace(decode=lambda ids,**kw:'x'*len(ids))
            def apply_chat_template(self,messages,**kw):
                assert kw['add_generation_prompt'] and kw['enable_thinking'] is False
                return dict(input_ids=SimpleNamespace(numel=lambda:10),messages=messages)
            async def __call__(self,input_ids,messages,cache_view):
                block=cache_view[0];block.num_tokens+=input_ids.numel()
                owner.operations.append(('messages',messages,block))
                return object()
        self.llm=LLM()
    def encode(self,text):return [ACTION_NAMES.index(text)] if text in ACTION_NAMES else [99]
    def new_generator(self,seed):return seed
    async def create_block(self):
        b=SimpleNamespace(num_tokens=0,identity=len(self.operations));self.live.append(b);return b
    async def free_block(self,block):self.live=[b for b in self.live if b is not block]
    def sample(self,output,**kw):return 90,'x',False
    async def decode(self,token,deps,target):
        assert deps==[]
        target.raw.num_tokens+=1;self.operations.append(('token',token,target.raw));return object()
    async def prefill(self,text,deps,target):
        assert text==CHAT_END and deps==[]
        target.raw.num_tokens+=1;self.operations.append(('close',text,target.raw));return object()


def sampler(self,output,ids,**kw):
    assert ids==list(range(8))
    self.backend.metrics['restricted_readouts']+=1
    return 3,[1/8]*8


@pytest.mark.parametrize('scope',['step','episode'])
def test_one_chat_has_ordered_roles_action_and_explicit_turn_boundaries(monkeypatch,scope):
    monkeypatch.setattr(ProbeReadout,'sample_action',sampler)
    b=Backend();events=[]
    r=SequentialTargetActorReadout(b,SimpleNamespace(log=lambda *x:events.append(x)),
        ControlMemory(),history_scope=scope,max_role_tokens=2)
    image=np.zeros((16,16,3),dtype=np.uint8)
    async def run():
        for i in range(3):
            async with await BlockHandle.create(b,'outer') as outer:
                idx,rows=await r.generate_action(messages_for(image,image),outer,
                    generator=17,temperature=.7)
                assert idx==3 and rows==33
                r.memory.observe('dig',1)
            assert len(b.live)==(1 if scope=='episode' else 0)
        await r.close_episode();await r.close_episode()
    asyncio.run(run())
    assert not b.live
    turns=[x for x in b.operations if x[0]=='messages']
    assert len(turns)==9
    assert sum(x[1][0]['role']=='system' for x in turns)==(1 if scope=='episode' else 3)
    assert all(x[1][0]['role']=='user' for i,x in enumerate(turns) if i%3!=0)
    for start in range(0,len(b.operations),11):
        ops=b.operations[start:start+11]
        assert [x[0] for x in ops]==['messages','token','token','close',
            'messages','token','token','close','messages','token','close']
        assert len({id(x[2]) for x in ops})==1
        assert ops[-2][1]==3  # actual sampled action, not just its probability
    counts=[e['context_tokens_after'] for kind,e in events if kind=='target_decision']
    assert counts==([38]*3 if scope=='step' else [38,76,114])


def test_model_context_limit_fails_before_appending_and_never_resets(monkeypatch):
    monkeypatch.setattr(ProbeReadout,'sample_action',sampler)
    b=Backend(limit=42)
    r=SequentialTargetActorReadout(b,SimpleNamespace(log=lambda *x:None),ControlMemory(),
        history_scope='episode',max_role_tokens=2)
    image=np.zeros((16,16,3),dtype=np.uint8)
    async def run():
        async with await BlockHandle.create(b,'outer') as outer:
            await r.generate_action(messages_for(image,image),outer,generator=0,temperature=.7)
            history=r.chat
            before=len(b.operations)
            with pytest.raises(ContextLimitError,match='NOT reset or trimmed'):
                await r.generate_action(messages_for(image,image),outer,generator=0,temperature=.7)
            assert r.chat is history and r.history_tokens==38 and len(b.operations)==before
        await r.close_episode()
    asyncio.run(run())
    assert not b.live


@pytest.mark.parametrize('fail',[None,'world','model','cancel','cleanup'])
def test_pipeline_closes_persistent_chat_on_completion_and_error(tmp_path,monkeypatch,fail):
    from experiment_runner import Recorder,EpisodeContext
    from pipelines.world import Observation
    monkeypatch.setattr(ProbeReadout,'sample_action',sampler)
    b=Backend();rec=Recorder(tmp_path)
    if fail in ('model','cancel'):
        async def broken_decode(*args):
            if fail=='cancel':raise asyncio.CancelledError()
            raise RuntimeError('model failure')
        b.decode=broken_decode
    r=SequentialTargetActorReadout(b,rec,ControlMemory(),history_scope='episode',max_role_tokens=2)
    if fail=='cleanup':
        original_close=r.close_episode
        async def broken_close():
            await original_close()
            raise RuntimeError('cleanup failure')
        r.close_episode=broken_close
    class World:
        def __init__(self):self.closed=False
        async def reset(self):return Observation(np.zeros((16,16,3),dtype=np.uint8),reward=0,info={})
        async def pass_action(self,index):
            if fail=='world':raise RuntimeError('world failure')
            return await self.reset()
        async def aclose(self):self.closed=True
    w=World()
    p=ProbePipeline(w,rec,r,context=EpisodeContext('test',0,0,tmp_path,0,0,0),
        max_actions=3,action_delay=0,action_names=ACTION_NAMES,feedback=r.memory)
    if fail=='cancel':
        with pytest.raises(asyncio.CancelledError):asyncio.run(p.run())
    elif fail:
        with pytest.raises(RuntimeError,match=fail+' failure'):asyncio.run(p.run())
    else:asyncio.run(p.run())
    assert not b.live and r.chat is None and w.closed
    import json
    assert json.loads((tmp_path/'completion.json').read_text())['status']==('failed' if fail else 'completed')


@pytest.mark.parametrize('scope,pages',[('step',4096),('episode',18432)])
def test_experiment_files_are_self_contained_and_keep_expected_settings(monkeypatch,scope,pages):
    import runpy,subprocess,sys
    from pathlib import Path
    import experiment_runner
    script=Path(__file__).resolve().parents[1]/f'experiments/choptree_1x500_r100_sequential_{scope}.py'
    captured={}
    class Runner:
        def __init__(self,venv,**kw):captured.update(kw,venv=venv)
        def set_engine_params(self,v):captured['engine']=v;return self
        def set_pipeline(self,v):captured['pipeline']=v;return self
        def set_concurrency(self,v):captured['concurrency']=v;return self
        def set_results_directory(self,v):captured['results']=v;return self
        def run(self,**kw):captured.update(kw)
    monkeypatch.setattr(experiment_runner,'Runner',Runner)
    cfg=runpy.run_path(str(script),run_name='__main__')
    assert cfg['HISTORY_SCOPE']==scope and cfg['ACTIONS']==500
    assert captured['concurrency']==1 and captured['pipeline'].repeats==100
    assert captured['engine']['engine_config']['num_page_override']==pages
    assert 'max_seq_len_override' not in captured['engine']['engine_config']
    assert captured['results'].name==script.stem
    assert cfg['TEMPERATURE']==.7 and cfg['MAX_ROLE_TOKENS']==768
    assert cfg['FRAMESKIP']==4 and cfg['PMUL']==2 and cfg['TURN_DEGREES']==7
    subprocess.run([sys.executable,'-S','-c',
        'import runpy,sys;runpy.run_path(sys.argv[1],run_name="config_only")',str(script)],check=True)
