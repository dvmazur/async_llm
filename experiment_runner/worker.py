import asyncio
import dataclasses
import json
import os
from pathlib import Path
import signal
import traceback

from .logs import atomic, stamp
from .runner import EpisodeContext


async def run_worker(spec, ordinal, factory, *, engine_factory=None):
    directory = Path(spec["results_directory"]) / f"gpu-{ordinal:03}"
    directory.mkdir(parents=True, exist_ok=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    engine = None
    errors = []
    tasks = []
    watcher = None
    monitor = None
    real_engine = engine_factory is None
    try:
        if engine_factory is None:
            from .engine import create_engine
            engine_factory = create_engine
        started = stamp()
        atomic(directory / "status.json", dict(status="engine_initializing", **started))
        engine = await engine_factory(spec["engine_params"], directory)
        atomic(directory / "engine-ready.json", dict(start=started, end=stamp(), model_load_count=1))
        if real_engine:
            from .monitor import GPUMonitor
            monitor = GPUMonitor(directory)

        async def slot(index):
            for repeat in range(spec["repeats"]):
                if stop.is_set():
                    break
                number = (ordinal * spec["concurrency"] + index) * spec["repeats"] + repeat
                target = directory / f"slot-{index:03}" / f"repeat-{repeat:03}"
                target.mkdir(parents=True)
                context = EpisodeContext(f"gpu{ordinal}/slot{index}/repeat{repeat}",
                    spec["model_seed_start"] + number, spec["world_seed_start"] + number,
                    target, index, repeat, ordinal)
                atomic(target / "context.json", {**dataclasses.asdict(context), "results_directory": str(target)})
                try:
                    pipeline = factory(engine, context)
                    # Cooperative stop: finish current safe action, then drain roles.
                    pipeline.stop_requested = stop.is_set
                    episode_start = stamp()
                    await pipeline.run()
                    # Generic pipelines may not expose finer workload boundaries.
                    # Speleo's own record separates reset/drain/flush and is retained.
                    completion = target/'completion.json'
                    if not completion.exists():
                        atomic(completion, dict(status='stopped' if stop.is_set() else 'completed',
                            workload_start=episode_start['monotonic'], workload_end=stamp()['monotonic'],
                            boundary='pipeline.run (no finer boundaries supplied)'))
                except BaseException as exc:
                    atomic(target / "failure.json", dict(error=repr(exc), traceback=traceback.format_exc(), **stamp()))
                    errors.append(exc)
                    stop.set()
                    break

        tasks = [asyncio.create_task(slot(i)) for i in range(spec["concurrency"])]
        group = asyncio.gather(*tasks)
        watcher = asyncio.create_task(stop.wait())
        await asyncio.wait([group, watcher], return_when=asyncio.FIRST_COMPLETED)
        if stop.is_set() and not group.done():
            try:
                await asyncio.wait_for(asyncio.shield(group), 30)
            except asyncio.TimeoutError:
                atomic(directory / 'status.json', dict(status='aborted', error='graceful drain exceeded 30 seconds', **stamp()))
                # GPU futures may still own blocks: do not free underneath them.
                # Parent observes this exit and kills only our process group.
                os._exit(70)
        await group
        if errors:
            raise RuntimeError("pipeline failed") from errors[0]
        if stop.is_set():
            raise RuntimeError("worker stopped before completing assignments")
    finally:
        if watcher:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if engine is not None:
            try:
                await engine.close()
            finally:
                if monitor:
                    await monitor.close()


def run(spec_path, ordinal, factory):
    spec = json.loads(Path(spec_path).read_text())
    directory = Path(spec["results_directory"]) / f"gpu-{ordinal:03}"
    try:
        asyncio.run(run_worker(spec, ordinal, factory))
    except BaseException as exc:
        atomic(directory / "status.json", dict(status="failed", error=repr(exc), **stamp()))
        raise
    atomic(directory / "status.json", dict(status="completed", **stamp()))
