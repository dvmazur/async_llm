"""Explicit deployment only. Never invoked by Runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor


CRAFTIUM_REVISION = '8eb8707cb756df47e76131a0058ab724d2383c76'
MODEL = 'Qwen/Qwen3.6-35B-A3B-FP8'
MODEL_REVISION = '61a5771f218894aaacf97551e24a25b866750fc2'


def command(*args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(args):
    project = Path(__file__).resolve().parent.parent
    engine, venv, craftium = (Path(p).expanduser().resolve() for p in (args.engine, args.venv, args.craftium))
    if args.jobs < 1:
        raise ValueError('jobs must be positive')
    if len({engine, venv, craftium}) != 3:
        raise ValueError('engine, venv and Craftium directories must be distinct')
    if venv.exists() and not args.update_existing:
        raise FileExistsError('venv exists; pass --update-existing to authorize explicit locked sync')
    # The caller selects the checkout; setup never clones/pulls/checks out the engine.
    for name in ('pyproject.toml', 'uv.lock'):
        if not (engine/name).is_file():
            raise FileNotFoundError(f'--engine must be an existing checkout: missing {engine/name}')
    if args.system_deps:
        prefix = [] if os.geteuid() == 0 else ['sudo']
        command(*prefix, 'apt-get', 'update')
        command(*prefix, 'apt-get', 'install', '-y', 'git', 'curl', 'g++', 'make', 'cmake',
            'pkg-config', 'libpng-dev', 'libjpeg-dev', 'libgl1-mesa-dev', 'libsqlite3-dev',
            'libogg-dev', 'libvorbis-dev', 'libopenal-dev', 'libcurl4-gnutls-dev',
            'libfreetype6-dev', 'zlib1g-dev', 'libgmp-dev', 'libjsoncpp-dev', 'libzstd-dev',
            'libluajit-5.1-dev', 'gettext', 'libsdl2-dev', 'libpython3-dev', 'python3-venv',
            'xvfb', 'xauth', 'libegl1', 'libopengl0', 'libegl-mesa0', 'libglx-mesa0', 'libgl1-mesa-dri')
    uv = shutil.which('uv')
    if not uv:
        tools_dir = venv.parent/'speleo-tools'
        tools_dir.mkdir(parents=True, exist_ok=True)
        # Same pinned installer as the verified portable deployment. Do not
        # modify shell startup files or require pip in the system interpreter.
        with tempfile.TemporaryDirectory(prefix='speleo-uv-') as temporary:
            installer = Path(temporary)/'install.sh'
            urllib.request.urlretrieve('https://astral.sh/uv/0.12.14/install.sh', installer)
            command('sh', installer, env=dict(os.environ, UV_INSTALL_DIR=str(tools_dir), UV_NO_MODIFY_PATH='1'))
        uv = str(tools_dir/'uv')
        if not Path(uv).is_file():
            raise RuntimeError('uv installer did not create the expected executable')
    for name in ('git', 'cmake', 'g++', 'Xvfb', 'nvcc'):
        if not shutil.which(name):
            raise RuntimeError(f'missing {name}; install build dependencies (never GPU drivers)')
    lock_hash = digest(engine/'uv.lock')
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(venv))
    command(uv, 'sync', '--project', engine, '--locked', '--no-dev', '--no-default-groups',
            '--python', '3.12', env=env)
    python = venv/'bin/python'
    runtime_env = dict(os.environ, PATH=str(venv/'bin') + os.pathsep + os.environ.get('PATH', ''),
        VIRTUAL_ENV=str(venv), PYTHONPATH=str(project) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    if args.download_model:
        # hf is provided by the locked huggingface-hub package, no background orphan.
        download = ThreadPoolExecutor(max_workers=1)
        future = download.submit(command, venv/'bin/hf', 'download', MODEL,
            '--revision', args.model_revision, '--local-dir', Path(args.download_model).resolve())
    else:
        download = future = None
    try:
        if not craftium.exists():
            command('git', 'clone', '--depth', '1', 'https://github.com/mikelma/craftium.git', craftium)
        if subprocess.check_output(['git', '-C', str(craftium), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
            raise RuntimeError('Craftium worktree is dirty; refusing checkout')
        command('git', '-C', craftium, 'fetch', 'origin', args.craftium_revision)
        command('git', '-C', craftium, 'checkout', '--detach', args.craftium_revision)
        command('git', '-C', craftium, 'submodule', 'update', '--init', '--recursive', '--depth', '1')
        command('cmake', '-S', craftium, '-B', craftium, '-DRUN_IN_PLACE=TRUE',
            '-DCMAKE_BUILD_TYPE=Release', '-DENABLE_SOUND=OFF', '-DENABLE_GETTEXT=OFF')
        command('cmake', '--build', craftium, '-j', args.jobs)
        # Resolve the declared environment group AND Craftium together. The engine
        # lock is a hard constraint: no silent upgrades of its Torch/CUDA packages.
        # The exported constraints travel via stdin, not another maintained file.
        locked = subprocess.check_output([uv, 'export', '--project', str(engine), '--locked',
            '--no-dev', '--no-default-groups', '--no-emit-project', '--no-hashes',
            '--format', 'requirements.txt'], text=True)
        command(uv, 'pip', 'install', '--project', project, '--python', python,
            '--group', 'runtime', '--constraints', '-', '-e', craftium, input=locked, text=True)
        # Runner/experiments are source files, never installed into this venv.
        command(python, '-m', 'environment.dependencies', uv, env=runtime_env)
        if digest(engine/'uv.lock') != lock_hash:
            raise RuntimeError('setup changed the engine lock')
        command(python, '-m', 'environment.check', env=runtime_env)
        if future:
            future.result()
    finally:
        if download:
            download.shutdown(wait=True)
    freeze = subprocess.check_output([uv, 'pip', 'freeze', '--python', str(python)], text=True)
    (venv/'speleo-environment-freeze.txt').write_text(freeze)
    print(f'Setup complete: {venv}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Prepare a GPU environment from an existing engine checkout.')
    parser.add_argument('--engine', required=True, metavar='DIRECTORY',
        help='existing engine repository with pyproject.toml and uv.lock; never cloned or checked out')
    parser.add_argument('--venv', required=True, metavar='DIRECTORY',
        help='Python environment to create; use this same VENV path in the experiment')
    parser.add_argument('--craftium', required=True, metavar='DIRECTORY',
        help='Craftium source/build directory; cloned if absent, built and installed editable')
    parser.add_argument('--craftium-revision', default=CRAFTIUM_REVISION,
        help='Craftium commit to fetch/build (default: pinned validated commit)')
    parser.add_argument('--model-revision', default=MODEL_REVISION,
        help='Hugging Face revision for the optional model download')
    parser.add_argument('--download-model', metavar='DIRECTORY',
        help=f'download pinned {MODEL} weights here via hf; omit for existing weights')
    parser.add_argument('--jobs', type=int, default=8,
        help='parallel Craftium compilation jobs, not GPUs or pipeline concurrency (default: 8)')
    parser.add_argument('--system-deps', action='store_true',
        help='install build/rendering packages via apt-get/sudo; never GPU drivers or CUDA')
    parser.add_argument('--update-existing', action='store_true',
        help='allow locked synchronization of an existing venv; may change installed packages')
    setup(parser.parse_args())


if __name__ == '__main__':
    main()
