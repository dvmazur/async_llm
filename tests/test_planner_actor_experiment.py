import asyncio
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

from experiment_runner import EpisodeContext
from experiments import choptree_1x500_r100_planner_actor_text as experiment
from pipelines.settled_world import SettledWorld


def test_self_contained_file_uses_existing_runner_api(monkeypatch):
    import experiment_runner
    captured = {}
    class Runner:
        def __init__(self, venv, **kw): captured.update(kw,venv=venv)
        def set_engine_params(self, x): captured['params']=x; return self
        def set_pipeline(self, x): captured['pipeline']=x; return self
        def set_concurrency(self, x): captured['concurrency']=x; return self
        def set_results_directory(self, x): captured['results']=x; return self
        def run(self, *, gpus): captured['gpus']=gpus
    monkeypatch.setattr(experiment_runner,'Runner',Runner)
    config=runpy.run_path(experiment.__file__,run_name='__main__')
    assert captured['concurrency']==1 and captured['pipeline'].repeats==100
    assert captured['pipeline'].factory is config['make_pipeline']
    assert config['ACTIONS']==500 and captured['gpus']==[0]
    assert captured['params'] is config['ENGINE_PARAMS']
    assert captured['params']['engine_config']['model_path']==str(config['MODEL'])
    assert captured['venv']==config['VENV']
    assert captured['results']==config['RESULTS']
    assert captured['results'].name==Path(experiment.__file__).stem
    assert captured['world_seed_start']==captured['model_seed_start']==0


def test_factory_passes_config_and_keeps_fresh_episode_memory(monkeypatch,tmp_path):
    received=[]
    def readout(engine,recorder,memory,**kwargs):
        received.append((memory,kwargs))
        return SimpleNamespace()
    monkeypatch.setattr(experiment,'TargetActorReadout',readout)
    context=EpisodeContext('test',0,0,tmp_path,0,0,0)
    pipeline=experiment.make_pipeline(None,context)
    assert pipeline.max_actions==500
    assert pipeline.temperature==.7 and pipeline.action_delay==.1
    assert pipeline.vision=='plain'
    assert pipeline.world.barrier.parties==1
    settings=pipeline.world.world.world.settings
    assert settings['max_steps']==500+SettledWorld.MAX_SETTLING_STEPS
    assert settings['gym_kwargs']['frameskip']==4
    assert settings['gym_kwargs']['pmul']==2
    assert settings['gym_kwargs']['minetest_conf']['time_speed']==0
    assert received[0][1]==dict(planner_temperature=.5,recovery_temperature=.9,
                               recovery_after=12,max_role_tokens=768)
    assert pipeline.feedback is received[0][0] and not pipeline.feedback.recent
    assert experiment.DUMP_IMAGES is False and experiment.GIF_ON is True
    asyncio.run(pipeline.recorder.finish())


def test_parent_config_import_requires_no_engine_environment():
    subprocess.run([sys.executable,'-S','-c',
        'import runpy,sys; runpy.run_path(sys.argv[1],run_name="config_only")',
        experiment.__file__],check=True)
