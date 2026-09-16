"""RTX long run: bbe7abf, FP32 GDN, larger KV pool for retained history copies."""
from pathlib import Path
from copy import deepcopy
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT.parent
sys.path[:0] = [str(ROOT), str(DEPLOY / 'engine' / 'python')]

from experiments.speleo_15x5 import ENGINE_PARAMS
from experiment_runner import Runner, RepeatedPipeline, Recorder
from pipelines.speleo import SpeleoPipeline
from pipelines.world import SpeleoWorld

VENV = Path('/home/lordvoldebug_2/runner-check-20260915/venv')
PARAMS = deepcopy(ENGINE_PARAMS)
PARAMS['engine_config'].update(
    model_path='/home/lordvoldebug_2/models/Qwen3.6-35B-A3B-FP8',
    num_page_override=24576,
    max_prefill_rows=1024,
    shared_cuda_graph_prefill_rows=[256, 1024])


def make_pipeline(engine, context):
    return SpeleoPipeline(SpeleoWorld(seed=context.world_seed, max_steps=100),
        Recorder(context.results_directory, dump_images=False, gif_on=False),
        engine, context=context, max_actions=100)


if __name__ == '__main__':
    (Runner(VENV, model_seed_start=0, world_seed_start=0)
        .set_engine_params(PARAMS)
        .set_pipeline(RepeatedPipeline(make_pipeline, repeats=1))
        .set_concurrency(15)
        .set_results_directory(DEPLOY / 'results-15x100-cap1024')
        .run(gpus=[0]))
