import asyncio
from pathlib import Path
import runpy

import pytest

from experiment_runner import EpisodeContext
from experiments import choptree_1x500_r100_probe_only as probe
from pipelines.choptree_local_policy import ChopTreeProbeFeedback, probe_prompt
from pipelines.settled_world import SettledWorld


def test_local_probe_factory_and_bounded_memory(tmp_path):
    context = EpisodeContext('test', 0, 0, tmp_path, 0, 0, 0)
    pipeline = probe.make_pipeline(None, context)
    assert pipeline.prompt == probe_prompt()
    assert '4 game frames' in pipeline.prompt and '7 degrees' in pipeline.prompt
    assert pipeline.vision == 'crop' and pipeline.include_last_action
    assert pipeline.action_names == ('wait', 'forward', 'jump', 'dig', 'right', 'left', 'up', 'down')
    assert pipeline.world.expected_spawn is None
    settings = pipeline.world.world.settings
    assert settings['max_steps'] == 500 + SettledWorld.MAX_SETTLING_STEPS
    assert settings['gym_kwargs']['pmul'] == 2
    assert settings['gym_kwargs']['frameskip'] == 4
    assert settings['gym_kwargs']['minetest_conf']['time_speed'] == 0
    assert pipeline.action_delay == .1 and pipeline.temperature == .7
    for _ in range(500):
        pipeline.feedback.observe('dig', 0)
    assert len(pipeline.feedback.recent) == 6
    assert len(pipeline.feedback.text()) < 1500
    assert len(ChopTreeProbeFeedback().recent) == 0
    assert 'upper_failure' not in vars(pipeline.feedback)
    assert probe.ENGINE_PARAMS['engine_config']['num_page_override'] * 16 == 65536
    assert settings['craftium_directory'] == str(probe.CRAFTIUM)
    asyncio.run(pipeline.recorder.finish())


@pytest.mark.parametrize('name,concurrency,repeats', [
    ('choptree_1x500_r100_probe_only', 1, 100), ('choptree_5x500_r4_assessment', 5, 4)])
def test_file_config_runs_through_existing_runner_api(monkeypatch, name, concurrency, repeats):
    import experiment_runner
    captured = {}

    class FakeRunner:
        def __init__(self, venv, **kwargs): captured.update(kwargs, venv=venv)
        def set_engine_params(self, value): captured['engine_params'] = value; return self
        def set_pipeline(self, value): captured['pipeline'] = value; return self
        def set_concurrency(self, value): captured['concurrency'] = value; return self
        def set_results_directory(self, value): captured['results'] = value; return self
        def run(self, *, gpus): captured['gpus'] = gpus

    monkeypatch.setattr(experiment_runner, 'Runner', FakeRunner)
    settings = runpy.run_path(str(Path(probe.__file__).with_name(name+'.py')), run_name='__main__')
    assert captured['concurrency'] == concurrency
    assert captured['pipeline'].repeats == repeats
    assert captured['pipeline'].factory is settings['make_pipeline']
    assert settings['ACTIONS'] == 500
    assert captured['results'].name == name
    assert captured['venv'] == settings['VENV']
    assert captured['engine_params'] is settings['ENGINE_PARAMS']
    assert captured['engine_params']['engine_config']['model_path'] == str(settings['MODEL'])
    assert captured['model_seed_start'] == captured['world_seed_start'] == 0
    assert captured['gpus'] == [0]
