"""One independent model process per GPU; independently repeating async slots."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .logs import atomic, stamp


@dataclasses.dataclass(frozen=True)
class EpisodeContext:
    episode_id: str
    model_seed: int
    world_seed: int
    results_directory: Path
    slot: int
    repeat: int
    gpu_ordinal: int


@dataclasses.dataclass(frozen=True)
class RepeatedPipeline:
    factory: object
    repeats: int = 1

    def __post_init__(self):
        if type(self.repeats) is not int or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")


class Runner:
    def __init__(self, venv, *, model_seed_start=0, world_seed_start=0):
        self.python = Path(venv).resolve() / "bin/python"
        self.model_seed_start, self.world_seed_start = model_seed_start, world_seed_start
        self.params = None
        self.pipeline = None
        self.concurrency = 1
        self.directory = None
        self.running = False

    def _mutable(self):
        if self.running:
            raise RuntimeError("cannot configure a running Runner")

    def set_engine_params(self, params):
        self._mutable()
        if not isinstance(params, dict):
            raise TypeError("engine params must be a Python dict; parse JSON explicitly if needed")
        self.params = json.loads(json.dumps(params, allow_nan=False))
        return self

    def set_pipeline(self, pipeline):
        self._mutable()
        self.pipeline = pipeline if isinstance(pipeline, RepeatedPipeline) else RepeatedPipeline(pipeline)
        if not callable(self.pipeline.factory):
            raise TypeError('pipeline factory must be callable')
        return self

    def set_concurrency(self, pipelines_per_gpu):
        self._mutable()
        if type(pipelines_per_gpu) is not int or pipelines_per_gpu < 1:
            raise ValueError("pipelines_per_gpu must be positive")
        self.concurrency = pipelines_per_gpu
        return self

    def set_results_directory(self, directory):
        self._mutable()
        self.directory = Path(directory).resolve()
        return self

    def run(self, *, gpus):
        self._mutable()
        if self.params is None or self.pipeline is None or self.directory is None:
            raise ValueError("set engine params, pipeline and results directory first")
        worker_request = os.environ.pop('_SPELEO_RUNNER_WORKER', None)
        if worker_request is not None:
            # The SAME experiment has recreated its Python objects in the target
            # interpreter. Pass its callable directly: no import strings/pickling.
            from .worker import run
            request = json.loads(worker_request)
            run(request['spec'], request['ordinal'], self.pipeline.factory)
            raise SystemExit(0)
        if not self.python.is_file():
            raise FileNotFoundError(self.python)
        if not isinstance(gpus, (list, tuple)) or not gpus:
            raise ValueError("gpus must be a nonempty list of independent GPU IDs")
        ids = [str(g) for g in gpus]
        if len(set(ids)) != len(ids) or any(not g or "," in g or g.startswith("-") for g in ids):
            raise ValueError("GPU IDs must be unique and nonempty")
        count = len(ids) * self.concurrency * self.pipeline.repeats
        for base in (self.model_seed_start, self.world_seed_start):
            if type(base) is not int or not 0 <= base < 2**32 or base + count > 2**32:
                raise ValueError("seed range exceeds uint32")
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(f"refusing to overwrite {self.directory}")
        script = Path(sys.argv[0]).resolve()
        if not script.is_file():
            raise ValueError('Runner.run must be called from a Python experiment file')
        self.directory.mkdir(parents=True, exist_ok=True)
        spec = dict(engine_params=self.params, experiment_file=str(script),
                    experiment_args=sys.argv[1:],
                    concurrency=self.concurrency, repeats=self.pipeline.repeats,
                    model_seed_start=self.model_seed_start, world_seed_start=self.world_seed_start,
                    gpus=ids, python=str(self.python), results_directory=str(self.directory),
                    **stamp())
        sources = {str(script)}
        package_root = Path(__file__).resolve().parent.parent
        for name in ('experiment_runner', 'pipelines'):
            sources.update(str(p) for p in (package_root/name).glob('*.py'))
        spec['source_hashes'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(sources)}
        atomic(self.directory / "run.json", spec)
        processes, logs = [], []
        error = None
        self.running = True
        print(f'RUN {self.directory} | GPUs={ids} | slots/GPU={self.concurrency} | repeats/slot={self.pipeline.repeats}', flush=True)
        try:
            atomic(self.directory / "status.json", dict(status="running", **stamp()))
            for ordinal, gpu in enumerate(ids):
                target = self.directory / f"gpu-{ordinal:03}"
                target.mkdir()
                log = (target / "worker.log").open("x")
                logs.append(log)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
                # CUDA/JIT libraries spawn tools (e.g. ninja) by name. Selecting
                # just the interpreter is not enough to select the target venv.
                env['PATH'] = str(self.python.parent) + os.pathsep + env.get('PATH', '')
                env['VIRTUAL_ENV'] = str(self.python.parent.parent)
                threads = str(self.params.get('adapter_options', {}).get('cpu_threads', 4))
                env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
                roots = [str(Path(__file__).resolve().parent.parent)]
                env["PYTHONPATH"] = os.pathsep.join(roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
                env['_SPELEO_RUNNER_WORKER'] = json.dumps(dict(
                    spec=str(self.directory/'run.json'), ordinal=ordinal))
                processes.append(subprocess.Popen(
                    [str(self.python), str(script), *sys.argv[1:]],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            while any(p.poll() is None for p in processes):
                failed = [p for p in processes if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"GPU worker exited {failed[0].returncode}; see worker.log")
                time.sleep(.2)
            if any(p.returncode for p in processes):
                raise RuntimeError("GPU worker failed; see worker.log")
            for ordinal in range(len(ids)):
                gpu_directory = self.directory/f'gpu-{ordinal:03}'
                if json.loads((gpu_directory/'status.json').read_text())['status'] != 'completed':
                    raise RuntimeError(f'worker {ordinal} did not report completion')
                for slot in range(self.concurrency):
                    for repeat in range(self.pipeline.repeats):
                        completion = gpu_directory/f'slot-{slot:03}'/f'repeat-{repeat:03}'/'completion.json'
                        if not completion.exists() or json.loads(completion.read_text())['status'] != 'completed':
                            raise RuntimeError(f'missing/incomplete episode: {completion.parent}')
        except BaseException as exc:
            error = exc
            for p in processes:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
            deadline = time.monotonic() + 30
            while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
                time.sleep(.1)
            for p in processes:
                # Descendant worlds can outlive a failed worker; group is ours.
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
        finally:
            for f in logs:
                f.close()
            self.running = False
            atomic(self.directory / "status.json", dict(status="failed" if error else "completed",
                   error=repr(error) if error else None, exit_codes=[p.returncode for p in processes], **stamp()))
        from .summary import Summary
        try:
            report = Summary(self.directory).write()
            print("SUMMARY", json.dumps({k: v for k, v in report.items() if k != 'episodes'}, allow_nan=False), flush=True)
        except Exception as exc:
            atomic(self.directory / "summary-error.json", dict(error=repr(exc)))
            if not error:
                raise
        if error:
            raise error
