from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

import environment.setup as deployment


def arguments(tmp_path, **overrides):
    values = dict(engine=tmp_path/'engine', venv=tmp_path/'venv', craftium=tmp_path/'craftium',
        engine_bundle=None, update_existing=False, system_deps=False, jobs=2,
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


def test_system_git_installed_before_unpacking_bundle(tmp_path, monkeypatch):
    args = arguments(tmp_path, engine_bundle=tmp_path/'engine.bundle', system_deps=True)
    calls = []
    def command(*cmd, **kwargs):
        cmd = tuple(map(str, cmd))
        calls.append(cmd)
        if cmd[:2] == ('git', 'clone') and cmd[2] == str(args.engine_bundle):
            assert any('apt-get' in c and 'install' in c and 'git' in c for c in calls)
            args.engine.mkdir()
            (args.engine/'uv.lock').write_text('locked')
        if 'sync' in cmd:
            args.venv.mkdir()
        if cmd[:2] == ('git', 'clone') and 'craftium.git' in cmd[-2]:
            args.craftium.mkdir()
    monkeypatch.setattr(deployment, 'command', command)
    monkeypatch.setattr(deployment.shutil, 'which', lambda name: '/bin/'+name)
    monkeypatch.setattr(deployment.subprocess, 'check_output', lambda *a, **kw: '')
    deployment.setup(args)
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
