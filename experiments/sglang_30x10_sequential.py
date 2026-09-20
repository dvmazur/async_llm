"""Vanilla SGLang benchmark: 30 simultaneous episodes, 10 actions each."""
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from experiment_runner import Runner, RepeatedPipeline, Recorder
from pipelines.speleo import SpeleoPipeline, RoleParams
from pipelines.world import SpeleoWorld

VENV = REPOSITORY / '.venvs/sglang'
MODEL = REPOSITORY / 'models/Qwen3.6-35B-A3B-FP8'
CRAFTIUM = REPOSITORY / 'craftium'
RESULTS = REPOSITORY / 'results/sglang_30x10_sequential'
GPUS = [0]
PIPELINES_PER_GPU = 30
REPEATS = 1
ACTIONS = 10
ROLE_PARAMETERS = {
    'observer': RoleParams(18, .35, 1),
    'planner': RoleParams(60, .65, 2),
    'executor': RoleParams(16, .45, 5),
}
ENGINE_PARAMS = {
    'backend': 'sglang',
    # Native SGLang ServerArgs, not mini-sglang EngineConfig.
    'engine_config': {
        'model_path': str(MODEL), 'dtype': 'bfloat16', 'quantization': 'fp8',
        'mem_fraction_static': .8, 'context_length': 32768,
        'max_running_requests': 32, 'chunked_prefill_size': 4096,
        'cuda_graph_max_bs_decode': 32,
        'random_seed': 0,  # normal SGLang: engine-global RNG, not per-role RNG
    },
    'adapter_options': {'cpu_threads': 4},
}


def make_pipeline(engine, context):
    return SpeleoPipeline(SpeleoWorld(seed=context.world_seed, max_steps=ACTIONS,
            craftium_directory=str(CRAFTIUM)),
        Recorder(context.results_directory, dump_images=False, gif_on=False),
        engine, context=context, max_actions=ACTIONS, role_params=ROLE_PARAMETERS)


if __name__ == '__main__':
    (Runner(VENV).set_engine_params(ENGINE_PARAMS)
        .set_pipeline(RepeatedPipeline(make_pipeline, repeats=REPEATS))
        .set_concurrency(PIPELINES_PER_GPU).set_results_directory(RESULTS).run(gpus=GPUS))
