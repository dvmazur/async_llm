"""GPU smoke using the selected long-run experiment's unchanged engine settings."""
import argparse
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_runner import Runner, RepeatedPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--actions', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if args.actions < 1 or args.repeats < 1:
        parser.error('actions and repeats must be positive')
    source = args.experiment.resolve()
    spec = importlib.util.spec_from_file_location('_gpu_smoke_experiment', source)
    experiment = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = experiment
    spec.loader.exec_module(experiment)
    # Keep GPU list, concurrency, engine config, role settings and buffers exact.
    # Only shorten episode/repeat counts; the factory reads ACTIONS from this module.
    experiment.ACTIONS = args.actions
    print(f'GPU SMOKE {source}: {experiment.PIPELINES_PER_GPU} slots, '
          f'{args.actions} actions, {args.repeats} repeats; engine settings unchanged',
          flush=True)
    (Runner(experiment.VENV,
            model_seed_start=getattr(experiment, 'MODEL_SEED_START', 0),
            world_seed_start=getattr(experiment, 'WORLD_SEED_START', 0))
        .set_engine_params(experiment.ENGINE_PARAMS)
        .set_pipeline(RepeatedPipeline(experiment.make_pipeline, repeats=args.repeats))
        .set_concurrency(experiment.PIPELINES_PER_GPU)
        .set_results_directory(args.results.resolve())
        .run(gpus=experiment.GPUS))


if __name__ == '__main__':
    main()
