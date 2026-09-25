import asyncio
from types import SimpleNamespace

import pytest

from experiment_runner.text_role import TextRole


def test_borrowed_conversation_retains_last_token_and_does_not_free_it():
    class Backend:
        def __init__(self):
            self.decodes=[]
            self.llm=SimpleNamespace(engine=SimpleNamespace(config=SimpleNamespace(
                model_config=SimpleNamespace(vocab_size=100))),
                tokenizer=SimpleNamespace(decode=lambda ids,**kw:'abc'))
        def sample(self,output,**kw):return len(self.decodes)+1,'x',False
        async def decode(self,token,deps,target):
            self.decodes.append((token,target));return object()
        async def create_block(self):raise AssertionError('caller owns the conversation')
        async def free_block(self,block):raise AssertionError('caller owns the conversation')
    backend=Backend();target=object()
    role=TextRole(backend,SimpleNamespace(log=lambda *a:None),'planner')
    async def prefill(messages,block):
        assert block is target
        return object(),20
    role.readout.prefill_action=prefill
    result=asyncio.run(role.on_block([dict(role='user',content='continue')],target,
        generator=17,temperature=.7,max_tokens=3))
    assert result==('abc',20)
    assert backend.decodes==[(1,target),(2,target),(3,target)]


@pytest.mark.parametrize('ending',['length','eos','boundary','error','cancel'])
def test_plain_role_length_is_normal_and_kv_always_freed(ending):
    class Backend:
        def __init__(self):
            self.live=set(); self.n=self.decodes=0
            self.llm=SimpleNamespace(engine=SimpleNamespace(config=SimpleNamespace(
                model_config=SimpleNamespace(vocab_size=100))),tokenizer=SimpleNamespace(
                    decode=lambda ids,**kw:'\t'*16*len(ids)))
        async def create_block(self):
            block=object();self.live.add(block);return block
        async def free_block(self,block):self.live.remove(block)
        def sample(self,output,**kw):
            assert kw==dict(generator=17,temperature=.7,top_k=100,top_p=1.)
            self.n+=1
            if ending=='error':raise RuntimeError('GPU failure')
            if ending=='cancel':raise asyncio.CancelledError()
            return 1, '<|im_start|>' if ending=='boundary' else '\t'*16, ending=='eos'
        async def decode(self,token,deps,target):
            assert target.raw in self.live and deps==[]
            self.decodes+=1;return object()
    b=Backend();events=[]
    r=TextRole(b,SimpleNamespace(log=lambda *x:events.append(x)),'actor')
    async def prefill(messages,target):
        assert target.raw in b.live
        return object(),50
    r.readout.prefill_action=prefill
    call=r([dict(role='user',content='align')],generator=17,temperature=.7,max_tokens=768)
    if ending in ('error','cancel'):
        with pytest.raises(RuntimeError if ending=='error' else asyncio.CancelledError):asyncio.run(call)
        assert not [x for x in events if x[0]=='stream']
    else:
        assert asyncio.run(call)==('',50)
        event=events[-1][1]
        assert event['finish_reason']==dict(length='length',eos='eos',boundary='chat_boundary')[ending]
        assert b.n==(768 if ending=='length' else 1)
        assert b.decodes==(767 if ending=='length' else 0)
        assert event['sampled_tokens']==b.n
    assert not b.live


def test_500_actions_with_role_limits_free_kv_and_count_readouts(tmp_path,monkeypatch):
    from experiment_runner.probe_readout import ProbeReadout
    from experiment_runner.logs import read_jsonl,atomic,JsonlWriter
    from experiment_runner.summary import Summary
    from pipelines.choptree import ACTION_NAMES
    from pipelines.choptree_target_actor import TargetActorReadout,ControlMemory
    from test_probe import Engine,World
    from experiment_runner import EpisodeContext
    from pipelines.probe import ProbePipeline
    class Backend(Engine):
        def __init__(self):
            super().__init__()
            self.generated=[]
            self.llm=SimpleNamespace(engine=SimpleNamespace(config=SimpleNamespace(
                model_config=SimpleNamespace(vocab_size=100))),tokenizer=SimpleNamespace(
                decode=lambda ids,**kw:'unfinished thought '+('\t'*16*len(ids))))
        def encode(self,name):return [ACTION_NAMES.index(name)]
        async def create_block(self):
            block=object();self.live.append(block);return block
        def sample(self,output,**kw):
            self.generated.append(1)
            return 1,'\t'*16,False
        async def decode(self,token,deps,target):return object()
    async def prefill(self,messages,target):return object(),10
    def sample(self,output,ids,**kw):
        assert ids==list(range(8))
        self.backend.samples+=1
        return 3,[1/8]*8
    monkeypatch.setattr(ProbeReadout,'prefill_action',prefill)
    monkeypatch.setattr(ProbeReadout,'sample_action',sample)
    b=Backend();world=World(b)
    from experiment_runner import Recorder
    ep=tmp_path/'gpu-000/slot-000/repeat-000'
    recorder=Recorder(ep)
    reader=TargetActorReadout(b,recorder,ControlMemory(),max_role_tokens=2)
    pipeline=ProbePipeline(world,recorder,reader,
        context=EpisodeContext('test',0,0,ep,0,0,0),max_actions=500,
        action_delay=0,action_names=ACTION_NAMES,feedback=reader.memory)
    asyncio.run(pipeline.run())
    assert world.i==b.samples==500 and world.closed and not b.live
    events=list(read_jsonl(ep/'events.jsonl'))
    assert all(x['mode']=='generated_with_readout' for x in events if x['kind']=='decision')
    assert sum(x['kind']=='stream' and x['finish_reason']=='length' for x in events)==1000
    atomic(ep/'context.json',dict(episode_id='test',model_seed=0,world_seed=0))
    atomic(tmp_path/'gpu-000/status.json',dict(status='completed'))
    atomic(tmp_path/'gpu-000/engine-totals.json',dict(restricted_readouts=500))
    generation=JsonlWriter(tmp_path/'gpu-000/generation.jsonl')
    for token in b.generated:generation.emit(kind='sample',token=token)
    generation.close()
    report=Summary(tmp_path).compute()
    assert report['generated_tokens']==2000
    assert report['workers'][0]['action_readouts']==500
