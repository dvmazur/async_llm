import asyncio
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from experiment_runner import EpisodeContext, Recorder
from experiment_runner.logs import read_jsonl
from experiment_runner.probe_readout import ProbeReadout
from pipelines.probe import ProbePipeline, PROMPT, messages_for
from pipelines.prompts import SPELEO
from pipelines.world import Observation


class Engine:
    def __init__(self, fail=False):
        self.live, self.frames, self.seeds = [], [], []
        self.samples = 0
        self.fail = fail

    def encode(self,name):
        return [[n for n,_ in SPELEO.actions].index(name)]

    def new_generator(self,seed):
        self.seeds.append(seed)
        return seed

    async def create_block(self):
        block = SimpleNamespace(num_tokens=0)
        self.live.append(block)
        return block

    async def free_block(self,block):
        self.live.remove(block)

    async def prefill_action(self,messages,block):
        assert len(self.live)==1 and block.raw.num_tokens==0
        assert len(messages)==2 and messages[0]['content']==PROMPT
        assert 'position' not in messages[1] and 'history' not in messages[1]
        self.frames.append([int(c['image'][0,0,0]) for c in messages[1]['content'] if c['type']=='image'])
        block.raw.num_tokens=440
        if self.fail:
            raise RuntimeError('prefill failure')
        return object(),440

    def sample_action(self,output,ids,*,generator,temperature):
        assert ids==list(range(7)) and temperature==.7
        self.samples+=1
        return 3,[.9,0,0,.1,0,0,0]  # sampled turn-right, deliberately NOT argmax


class World:
    def __init__(self,engine):
        self.engine,self.i,self.closed=engine,0,False
    def obs(self):
        return Observation(np.full((2,2,3),self.i%256,dtype=np.uint8),reward=999,
            info=dict(player_pos=[666,5-self.i,777],player_vel=[0,0,0],mt_dtime=.005))
    async def reset(self):
        return self.obs()
    async def pass_action(self,action):
        assert not self.engine.live and action==3
        self.i+=1
        return self.obs()
    async def aclose(self):
        self.closed=True


def make(path,engine,world,**kwargs):
    return ProbePipeline(world,Recorder(path),engine,
        context=EpisodeContext('probe',0,0,path,0,0,0),**kwargs)


def test_sampled_action_fresh_frames_and_delay_after_free(tmp_path,monkeypatch):
    engine=Engine();world=World(engine);delays=[]
    async def delay(seconds):
        assert seconds==.2 and not engine.live
        # The sampled action has not reached the world yet.
        assert engine.samples==world.i+1
        delays.append(seconds)
    monkeypatch.setattr(asyncio,'sleep',delay)
    asyncio.run(make(tmp_path,engine,world,max_actions=3).run())
    assert delays==[.2]*3 and engine.frames==[[0,0],[0,1],[1,2]]
    assert world.closed and world.i==3 and not engine.live
    events=list(read_jsonl(tmp_path/'events.jsonl'))
    assert not any(e['kind']=='stream' for e in events)
    decisions=[e for e in events if e['kind']=='decision']
    assert all(e['nonargmax'] and e['action']=='right' for e in decisions)
    assert len([e for e in events if e['kind']=='action_delay'])==3
    assert all('pacing_seconds' in s for s in read_jsonl(tmp_path/'steps.jsonl') if s['kind']=='action')


def test_probe_500_actions_no_retained_kv(tmp_path):
    engine=Engine();world=World(engine)
    asyncio.run(make(tmp_path,engine,world,max_actions=500,action_delay=0).run())
    assert world.i==500 and world.closed and not engine.live
    assert engine.samples==500 and len(engine.frames)==500


def test_hosted_completion_can_supply_action_without_probability_distribution(tmp_path):
    class Hosted(Engine):
        def sample_action(self,output,ids,*,generator,temperature):
            self.samples+=1
            return 3,None
    engine=Hosted();world=World(engine)
    asyncio.run(make(tmp_path,engine,world,max_actions=2,action_delay=0).run())
    events=[e for e in read_jsonl(tmp_path/'events.jsonl') if e['kind']=='decision']
    assert len(events)==2 and world.i==2 and engine.samples==2
    assert all(e['action']=='right' and e['action_probabilities'] is None and
               e['entropy_nats'] is None and e['nonargmax'] is None for e in events)


def test_probe_repeated_episodes_do_not_accumulate_state(tmp_path):
    async def run():
        engine=Engine()
        for i in range(105):
            world=World(engine)
            p=make(tmp_path/str(i),engine,world,max_actions=1,action_delay=0)
            p.context=EpisodeContext(str(i),i,i,tmp_path/str(i),0,i,0)
            await p.run()
            assert world.closed and not engine.live
        assert len(set(engine.seeds))==105
    asyncio.run(run())


def test_failed_probe_releases_state_and_world(tmp_path):
    engine=Engine(fail=True);world=World(engine)
    with pytest.raises(RuntimeError,match='prefill failure'):
        asyncio.run(make(tmp_path,engine,world,max_actions=1).run())
    assert not engine.live and world.closed


def test_cancel_during_pacing_does_not_retain_kv(tmp_path,monkeypatch):
    engine=Engine();world=World(engine)
    async def cancel(seconds):
        assert not engine.live
        raise asyncio.CancelledError()
    monkeypatch.setattr(asyncio,'sleep',cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(make(tmp_path,engine,world,max_actions=1).run())
    assert world.closed and world.i==0 and not engine.live


@pytest.mark.parametrize('kwargs',[{'temperature':0},{'temperature':float('nan')},
    {'action_delay':-1},{'action_delay':float('inf')},{'max_actions':0}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ProbePipeline(None,None,None,context=None,**kwargs)


def test_readout_uses_official_non_thinking_template_and_only_target():
    class Input:
        def numel(self): return 440
    class LLM:
        def __init__(self): self.processor=self
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs==dict(add_generation_prompt=True,enable_thinking=False,
                                tokenize=True,return_dict=True,return_tensors='pt')
            return dict(input_ids=Input(),pixel_values='images')
        async def __call__(self,**kwargs):
            assert kwargs['cache_view']==['target'] and kwargs['pixel_values']=='images'
            return 'output'
    engine=ProbeReadout(SimpleNamespace(llm=LLM()))
    assert asyncio.run(engine.prefill_action(messages_for('old','new'),SimpleNamespace(raw='target')))==('output',440)


def test_probe_config_has_no_history_and_fixed_small_capacity():
    from experiments import speleo_1x500_r105_probe_only as run
    assert (run.PIPELINES_PER_GPU,run.ACTIONS,run.REPEATS,run.ACTION_DELAY)==(1,500,105,.2)
    assert run.TEMPERATURE==.7 and run.DUMP_IMAGES is run.GIF_ON is False
    c=run.ENGINE_PARAMS['engine_config']
    assert c['num_page_override']*c['page_size']==65536
    assert c['max_seq_len_override']==2048 and c['max_prefill_rows']==512
    assert c['shared_cuda_graph_max_depth']==1
    assert c['shared_cuda_graph_prefill_rows']==[512]
    assert run.RESULTS.name=='speleo_1x500_r105_probe_only_sleep02'


def test_full_change_prompt_matches_completed_research():
    assert hashlib.sha256(PROMPT.encode()).hexdigest() == \
        '5ce89ee8bd70c115b3b5c741676a3b077d1871a54a76a5e3a9f85a4376f2a441'


def test_readout_samples_instead_of_argmax(monkeypatch):
    import sys
    from types import ModuleType
    class Tensor:
        def __init__(self,a): self.a=np.asarray(a)
        def reshape(self,*shape): return Tensor(self.a.reshape(*shape))
        def view(self,*shape): return self.reshape(*shape)
        def float(self): return self
        def __getitem__(self,index): return Tensor(self.a[index])
        def __truediv__(self,v): return Tensor(self.a/v)
        def tolist(self): return self.a.tolist()
    torch=ModuleType('torch')
    torch.softmax=lambda x,dim:Tensor(np.exp(x.a)/np.exp(x.a).sum(axis=dim,keepdims=True))
    flash=ModuleType('flashinfer');sampling=ModuleType('flashinfer.sampling');calls=[]
    def sample(probs,k,p,*,generator):
        calls.append((k,p,generator))
        return SimpleNamespace(item=lambda:2)
    sampling.top_k_top_p_sampling_from_probs=sample
    flash.sampling=sampling
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'flashinfer',flash)
    monkeypatch.setitem(sys.modules,'flashinfer.sampling',sampling)
    backend=SimpleNamespace(metrics={'restricted_readouts':0})
    output=SimpleNamespace(logits=Tensor([4.,0,0,0,0,0,0]))
    index,probs=ProbeReadout(backend).sample_action(output,list(range(7)),generator='rng',temperature=.7)
    assert index==2 and np.argmax(probs)==0
    assert calls==[(7,1.,'rng')] and backend.metrics['restricted_readouts']==1


def test_summary_counts_probe_readouts_without_generated_text(tmp_path):
    from experiment_runner.logs import atomic
    from experiment_runner.summary import Summary
    gpu=tmp_path/'gpu-000';folder=gpu/'slot-000/repeat-000'
    engine=Engine();world=World(engine)
    asyncio.run(make(folder,engine,world,max_actions=2,action_delay=0).run())
    atomic(folder/'context.json',dict(episode_id='probe',model_seed=0,world_seed=0))
    atomic(gpu/'status.json',dict(status='completed'))
    atomic(gpu/'engine-totals.json',dict(restricted_readouts=2))
    result=Summary(tmp_path).compute()
    assert result['actions']==2 and result['generated_tokens']==0
    worker=result['workers'][0]
    assert worker['action_readouts']==2
    assert worker['output_tokens_including_readouts_tps']>0
