import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

from experiment_runner import Runner, RepeatedPipeline
from experiment_runner.worker import run_worker
from experiment_runner.logs import atomic

EVENTS = []


class Engine:
    async def close(self):
        EVENTS.append(('close',))


class FakePipeline:
    def __init__(self, context):
        self.context = context

    async def run(self):
        c = self.context
        EVENTS.append(('start', c.slot, c.repeat, c.model_seed, c.world_seed))
        await asyncio.sleep(.01 if c.slot == 0 else .07)
        EVENTS.append(('end', c.slot, c.repeat))


def factory(engine, context):
    assert isinstance(engine, Engine)
    return FakePipeline(context)


async def create_engine(params, directory):
    EVENTS.append(('load',))
    return Engine()


class FailingPipeline:
    def __init__(self, context):
        self.context = context

    async def run(self):
        EVENTS.append(('failed_episode', self.context.slot, self.context.repeat))
        raise RuntimeError('deliberate pipeline failure')


def failing_factory(engine, context):
    return FailingPipeline(context)


def test_fail_fast_stops_repeats_and_closes_engine(tmp_path):
    EVENTS.clear()
    spec = dict(results_directory=str(tmp_path), concurrency=2, repeats=3,
        engine_params={}, model_seed_start=0, world_seed_start=0)
    with pytest.raises(RuntimeError, match='pipeline failed'):
        asyncio.run(run_worker(spec, 0, failing_factory, engine_factory=create_engine))
    assert EVENTS.count(('load',)) == 1
    assert EVENTS.count(('close',)) == 1
    assert not any(e[0] == 'failed_episode' and e[2] > 0 for e in EVENTS)
    failures = list(tmp_path.glob('gpu-*/slot-*/repeat-*/failure.json'))
    assert failures
    assert 'deliberate pipeline failure' in json.loads(failures[0].read_text())['error']


def test_independent_repeats_and_seed_assignment(tmp_path):
    EVENTS.clear()
    spec = dict(results_directory=str(tmp_path), concurrency=2, repeats=2,
        engine_params={}, model_seed_start=10, world_seed_start=20)
    asyncio.run(run_worker(spec, 0, factory, engine_factory=create_engine))
    assert EVENTS.count(('load',)) == 1
    assert EVENTS.count(('close',)) == 1
    assert EVENTS.index(('start', 0, 1, 11, 21)) < EVENTS.index(('end', 1, 0))
    assert ('start', 1, 0, 12, 22) in EVENTS
    assert ('start', 1, 1, 13, 23) in EVENTS
    assert len(list(tmp_path.glob('gpu-*/slot-*/repeat-*/context.json'))) == 4


def test_api_validation(tmp_path):
    runner = Runner(Path(sys.prefix)).set_pipeline(factory).set_results_directory(tmp_path)
    with pytest.raises(TypeError):
        runner.set_engine_params('config.json')
    with pytest.raises(ValueError):
        runner.set_concurrency(0)
    runner.set_pipeline(lambda engine, context: None)
    with pytest.raises(TypeError):
        runner.set_pipeline(42)
    runner.set_pipeline(factory)
    with pytest.raises(ValueError):
        RepeatedPipeline(factory, 0)
    params = {'engine_config': {'x': [1]}}
    runner.set_engine_params(params)
    params['engine_config']['x'].append(2)
    assert runner.params['engine_config']['x'] == [1]
    with pytest.raises(ValueError):
        runner.run(gpus=[0, 0])
    (tmp_path/'keep').write_text('user data')
    with pytest.raises(FileExistsError):
        runner.run(gpus=[0])
    assert (tmp_path/'keep').read_text() == 'user data'


def test_import_is_lightweight():
    import subprocess
    result = subprocess.check_output([sys.executable, '-c',
        'import experiment_runner, sys; assert "torch" not in sys.modules; assert "craftium" not in sys.modules'], text=True)
    assert result == ''


def test_parent_gpu_assignment_and_selected_interpreter(tmp_path, monkeypatch):
    import experiment_runner.runner as implementation
    captured = []
    class Process:
        returncode = 0
        def __init__(self, command, **kwargs):
            captured.append((command, kwargs))
            request = json.loads(kwargs['env']['_SPELEO_RUNNER_WORKER'])
            spec = json.loads(Path(request['spec']).read_text())
            ordinal = request['ordinal']
            directory = Path(spec['results_directory'])/f'gpu-{ordinal:03}'
            atomic(directory/'status.json', {'status': 'completed'})
            for slot in range(spec['concurrency']):
                for repeat in range(spec['repeats']):
                    target = directory/f'slot-{slot:03}'/f'repeat-{repeat:03}'
                    atomic(target/'completion.json', dict(status='completed', workload_start=1., workload_end=2.))
                    atomic(target/'context.json', dict(episode_id=f'{ordinal}/{slot}/{repeat}', model_seed=0, world_seed=0))
                    (target/'events.jsonl').touch()
                    (target/'steps.jsonl').touch()
        def poll(self):
            return 0
    monkeypatch.setattr(implementation.subprocess, 'Popen', Process)
    venv = tmp_path/'venv'
    (venv/'bin').mkdir(parents=True)
    (venv/'bin/python').touch()
    runner = (Runner(venv).set_engine_params({}).set_pipeline(RepeatedPipeline(factory, 3))
        .set_concurrency(5).set_results_directory(tmp_path/'results'))
    assert runner.run(gpus=[2, 7]) is None
    assert len(captured) == 2
    assert [kwargs['env']['CUDA_VISIBLE_DEVICES'] for _, kwargs in captured] == ['2', '7']
    assert all(command[0] == str(venv/'bin/python') for command, _ in captured)
    assert all(command[1:] == [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]] for command, _ in captured)
    assert all(kwargs['env']['PATH'].split(':')[0] == str(venv/'bin') for _, kwargs in captured)
    assert all(kwargs['env']['VIRTUAL_ENV'] == str(venv) for _, kwargs in captured)
    assert all(kwargs['start_new_session'] for _, kwargs in captured)
    spec = json.loads((tmp_path/'results/run.json').read_text())
    assert 'pipeline_factory' not in spec
    assert (spec['concurrency'], spec['repeats'], spec['gpus']) == (5, 3, ['2', '7'])
