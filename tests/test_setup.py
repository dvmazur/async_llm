from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

import environment.setup as deployment


def arguments(tmp_path, **overrides):
    values = dict(engine=tmp_path/'engine', venv=tmp_path/'venv', craftium=tmp_path/'craftium',
        update_existing=False, system_deps=False, jobs=2,
        download_model=None, craftium_revision=deployment.CRAFTIUM_REVISION,
        model_revision=deployment.MODEL_REVISION)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_existing_venv_rejected_before_installation(tmp_path, monkeypatch):
    args = arguments(tmp_path, system_deps=True)
    args.venv.mkdir()
    monkeypatch.setattr(deployment, 'command', lambda *a, **k: pytest.fail('unexpected external mutation'))
    with pytest.raises(FileExistsError, match='venv exists'):
        deployment.setup(args)


def test_installs_existing_engine_without_git_mutations(tmp_path, monkeypatch):
    args = arguments(tmp_path, system_deps=True)
    args.engine.mkdir()
    (args.engine/'pyproject.toml').write_text('[project]\nname = "minisgl"\n')
    (args.engine/'uv.lock').write_text('locked')
    calls = []
    def command(*cmd, **kwargs):
        cmd = tuple(map(str, cmd))
        calls.append(cmd)
        if 'sync' in cmd:
            args.venv.mkdir()
        if cmd[:2] == ('git', 'clone') and 'craftium.git' in cmd[-2]:
            args.craftium.mkdir()
    monkeypatch.setattr(deployment, 'command', command)
    monkeypatch.setattr(deployment.shutil, 'which', lambda name: '/bin/'+name)
    monkeypatch.setattr(deployment.subprocess, 'check_output', lambda *a, **kw: '')
    deployment.setup(args)
    assert (args.engine/'uv.lock').read_text() == 'locked'
    assert any('sync' in c and str(args.engine) in c and '--locked' in c for c in calls)
    assert not any(c[0] == 'git' and str(args.engine) in c for c in calls)
    assert [c for c in calls if c[:2] == ('git', 'clone')] == [
        ('git', 'clone', '--depth', '1', 'https://github.com/mikelma/craftium.git', str(args.craftium))]
    assert (args.venv/'speleo-environment-freeze.txt').exists()
    craftium_install = next(c for c in calls if 'install' in c and '-e' in c and str(args.craftium) in c)
    assert craftium_install[craftium_install.index('--group')+1] == 'runtime'
    assert craftium_install[craftium_install.index('--constraints')+1] == '-'
    assert '--no-deps' not in craftium_install
    assert not any('-e' in c and str(deployment.Path(deployment.__file__).resolve().parent.parent) in c
                   and str(args.craftium) not in c for c in calls)
    assert not any('cuda-drivers' in c or 'nvidia-driver' in c for c in calls)


@pytest.mark.parametrize('fail_download', [False, True])
def test_hf_download_overlaps_build_and_failure_propagates(tmp_path, monkeypatch, fail_download):
    args = arguments(tmp_path, download_model=tmp_path/'model')
    args.engine.mkdir()
    (args.engine/'pyproject.toml').write_text('[project]\nname = "minisgl"\n')
    (args.engine/'uv.lock').write_text('locked')
    started, build = threading.Event(), threading.Event()
    def command(*cmd, **kwargs):
        cmd = tuple(map(str, cmd))
        if cmd[0].endswith('/hf'):
            assert cmd[1:3] == ('download', deployment.MODEL)
            assert cmd[cmd.index('--revision')+1] == deployment.MODEL_REVISION
            assert cmd[cmd.index('--local-dir')+1] == str(args.download_model)
            started.set()
            assert build.wait(3), 'download did not overlap native build'
            if fail_download:
                raise RuntimeError('download failed')
        elif cmd[:2] == ('cmake', '--build'):
            assert started.wait(3)
            build.set()
        elif 'sync' in cmd:
            args.venv.mkdir()
        elif cmd[:2] == ('git', 'clone'):
            args.craftium.mkdir()
    monkeypatch.setattr(deployment, 'command', command)
    monkeypatch.setattr(deployment.shutil, 'which', lambda name: '/bin/'+name)
    monkeypatch.setattr(deployment.subprocess, 'check_output', lambda *a, **kw: '')
    if fail_download:
        with pytest.raises(RuntimeError, match='download failed'):
            deployment.setup(args)
        assert not (args.venv/'speleo-environment-freeze.txt').exists()
    else:
        deployment.setup(args)
        assert (args.venv/'speleo-environment-freeze.txt').exists()


@pytest.mark.parametrize('present', [(), ('pyproject.toml',), ('uv.lock',)])
def test_missing_engine_metadata_fails_before_installation(tmp_path, monkeypatch, present):
    args = arguments(tmp_path, system_deps=True)
    args.engine.mkdir()
    for name in present:
        (args.engine/name).write_text('fixture')
    monkeypatch.setattr(deployment, 'command', lambda *a, **k: pytest.fail('unexpected installation'))
    with pytest.raises(FileNotFoundError, match='existing checkout'):
        deployment.setup(args)


def test_setup_cli_rejects_bundle(monkeypatch):
    monkeypatch.setattr(deployment.sys, 'argv', ['setup', '--engine', 'engine',
        '--venv', 'venv', '--craftium', 'craftium', '--engine-bundle', 'old.bundle'])
    monkeypatch.setattr(deployment, 'setup', lambda *a: pytest.fail('setup must not start'))
    with pytest.raises(SystemExit) as error:
        deployment.main()
    assert error.value.code == 2
