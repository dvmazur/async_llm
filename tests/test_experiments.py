import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_settings_drive_the_runner(monkeypatch, tmp_path):
    from experiments import speleo_15x5_sequential as experiment
    calls = []
    class Runner:
        def __init__(self, venv, **kwargs):
            calls.append(('init', venv, kwargs))
        def __getattr__(self, name):
            def method(*args, **kwargs):
                calls.append((name, args, kwargs))
                return self
            return method
    monkeypatch.setattr(experiment, 'Runner', Runner)
    monkeypatch.setattr(experiment, 'VENV', tmp_path/'chosen-venv')
    monkeypatch.setattr(experiment, 'RESULTS', tmp_path/'chosen-results')
    monkeypatch.setattr(experiment, 'GPUS', [3, 7])
    monkeypatch.setattr(experiment, 'PIPELINES_PER_GPU', 2)
    monkeypatch.setattr(experiment, 'REPEATS', 4)
    monkeypatch.setattr(experiment, 'ENGINE_PARAMS', {'engine_config': {'model_path': 'chosen'}})
    experiment.main()
    assert calls[0][1] == tmp_path/'chosen-venv'
    assert ('set_engine_params', ({'engine_config': {'model_path': 'chosen'}},), {}) in calls
    assert ('set_concurrency', (2,), {}) in calls
    assert ('set_results_directory', (tmp_path/'chosen-results',), {}) in calls
    assert calls[-1] == ('run', (), {'gpus': [3, 7]})
    repeated = next(c[1][0] for c in calls if c[0] == 'set_pipeline')
    assert repeated.factory is experiment.make_pipeline and repeated.repeats == 4


def test_plain_file_from_other_cwd_needs_no_installed_runner(tmp_path):
    checkout = tmp_path/'checkout'
    for directory in ('experiment_runner', 'pipelines', 'experiments'):
        shutil.copytree(ROOT/directory, checkout/directory, ignore=shutil.ignore_patterns('__pycache__'))
    unrelated = tmp_path/'elsewhere'
    unrelated.mkdir()
    script = checkout/'experiments/speleo_15x5_sequential.py'
    # -I -S excludes installed packages and inherited PYTHONPATH. The source file
    # must import itself and reach the intentional missing-engine-venv guard.
    result = subprocess.run([sys.executable, '-I', '-S', str(script)], cwd=unrelated,
        text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert 'ModuleNotFoundError' not in result.stderr
    assert 'FileNotFoundError' in result.stderr
    assert str(checkout/'.venvs/minisgl/bin/python') in result.stderr


def test_dependencies_have_one_toml_source():
    data = tomllib.loads((ROOT/'pyproject.toml').read_text())
    assert 'gymnasium==0.29.1' in data['dependency-groups']['runtime']
    assert data['tool']['uv']['package'] is False
    assert data['tool']['uv']['config-settings-package']['craftium']['editable_mode'] == 'compat'
    assert not (ROOT/'environment-requirements.txt').exists()
    assert not (ROOT/'build-constraints.txt').exists()
    assert not (ROOT/'build_release.py').exists()
    assert (ROOT/'tools/build_release.py').is_file()


def test_experiment_does_not_override_backend_workspace():
    from experiments.speleo_15x5_sequential import ENGINE_PARAMS
    assert 'shared_attention_workspace_bytes' not in ENGINE_PARAMS['engine_config']
    assert 'attention_workspace_bytes' not in ENGINE_PARAMS['adapter_options']
    assert not (ROOT/'experiment_runner/workspace_config.py').exists()


def test_plain_experiment_dispatches_source_into_selected_workers(tmp_path):
    checkout = tmp_path/'checkout'
    shutil.copytree(ROOT/'experiment_runner', checkout/'experiment_runner', ignore=shutil.ignore_patterns('__pycache__'))
    # A CPU-only engine adapter isolates the dispatch protocol from model hardware.
    (checkout/'experiment_runner/engine.py').write_text('''
import sys
from experiment_runner.logs import atomic
async def create_engine(params, directory):
    assert 'torch' not in sys.modules
    assert params == {'experiment_value': 73}
    (directory/'forwards.jsonl').touch()
    (directory/'generation.jsonl').touch()
    class Engine:
        async def close(self):
            atomic(directory/'engine-totals.json', {'closed': True})
    return Engine()
''')
    scripts = checkout/'experiments'
    scripts.mkdir()
    results = tmp_path/'results'
    script = scripts/'custom_trial.py'
    script.write_text('''
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from experiment_runner import Runner, RepeatedPipeline, Recorder

class Trial:
    def __init__(self, context):
        self.context = context
    async def run(self):
        c = self.context
        r = Recorder(c.results_directory)
        await r.observation(step=0, image=None, height=float(c.world_seed))
        await r.step(step=1, image=None, action='wait', reward=0., done=True, height=-1.)
        await r.finish()

def main():
    # This closure exists only while main executes: it cannot be imported by name.
    offset = 0
    make_pipeline = lambda engine, context: Trial(context) if offset == 0 else None
    (Runner(VENV).set_engine_params({'experiment_value': 73})
        .set_pipeline(RepeatedPipeline(make_pipeline, 2)).set_concurrency(1)
        .set_results_directory(RESULTS).run(gpus=[0, 1]))
    print('PARENT_FINISHED')

if __name__ == '__main__':
    main()
'''.replace('Runner(VENV)', f'Runner({str(Path(sys.executable).parent.parent)!r})')
        .replace('.set_results_directory(RESULTS)', f'.set_results_directory({str(results)!r})'))
    result = subprocess.run([sys.executable, '-I', '-S', str(script)], cwd=tmp_path,
        text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('PARENT_FINISHED') == 1
    report = json.loads((results/'analysis/summary.json').read_text())
    assert report['status'] == 'completed' and report['actions'] == 4
    assert sorted(e['world_seed'] for e in report['episodes']) == [0, 1, 2, 3]
    assert len(list(results.glob('gpu-*/engine-ready.json'))) == 2
    for log in results.glob('gpu-*/worker.log'):
        assert 'PARENT_FINISHED' not in log.read_text()


def test_release_includes_neighbouring_configs_not_results(tmp_path):
    from tools.build_release import source_files
    for name in ('experiments/trial.py', 'experiments/engine.json', 'experiments/params.toml',
                 'experiments/results/run/summary.json', 'experiments/results/run/accidental.py'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    names = {str(p.relative_to(tmp_path)) for p in source_files(tmp_path)}
    assert {'experiments/trial.py', 'experiments/engine.json', 'experiments/params.toml'} <= names
    assert not any('/results/' in n for n in names)
