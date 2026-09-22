"""One world at a time; 105 independent episodes with the validated async policy."""
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from experiment_runner import Runner, RepeatedPipeline, Recorder
from pipelines.speleo import SpeleoPipeline, RoleParams
from pipelines.world import SpeleoWorld
from pipelines.settled_world import SettledWorld

VENV = REPOSITORY / '.venvs' / 'minisgl'
MODEL = REPOSITORY / 'models' / 'Qwen3.6-35B-A3B-FP8'
RESULTS = REPOSITORY / 'results' / 'speleo_1x500_r105_falsifier_async'
GPUS = [0]
PIPELINES_PER_GPU = 1
REPEATS = 105
ACTIONS = 500
MODEL_SEED_START = WORLD_SEED_START = 0
DUMP_IMAGES = GIF_ON = False
ROLE_PARAMETERS = {
    'observer': RoleParams(18, .35, 1, top_k=20, top_p=.9),
    'planner': RoleParams(60, .65, 2, top_k=20, top_p=.9),
    'executor': RoleParams(16, .45, 5, top_k=20, top_p=.9),
    'falsifier': RoleParams(18, .45, 4, top_k=20, top_p=.9),
}
ENGINE_PARAMS = {'engine_config': {
    'model_path': str(MODEL), 'dtype': 'bfloat16', 'quantization': 'fp8',
    # One pipeline still has concurrent role requests, not just one request.
    'max_running_req': 8, 'memory_ratio': .9, 'page_size': 16,
    'num_page_override': 16384, 'max_seq_len_override': 65536,
    'attention_backend': 'fi', 'max_prefill_rows': 4096,
    'cuda_graph_bs': [1, 2, 4, 8], 'cuda_graph_max_bs': 8,
    'shared_cuda_graph_prefill_rows': [256, 1024, 4096],
    'shared_cuda_graph_max_depth': 16,
    'generation_config': {'do_sample': True, 'temperature': .6, 'top_k': 20, 'top_p': .9},
}, 'adapter_options': {'cpu_threads': 4}}


def make_pipeline(engine, context):
    recorder = Recorder(context.results_directory, dump_images=DUMP_IMAGES, gif_on=GIF_ON)
    world = SettledWorld(SpeleoWorld(seed=context.world_seed,
        max_steps=ACTIONS + SettledWorld.MAX_SETTLING_STEPS), recorder)
    return SpeleoPipeline(world, recorder, engine, context=context,
        max_actions=ACTIONS, role_params=ROLE_PARAMETERS)


if __name__ == '__main__':
    (Runner(VENV, model_seed_start=MODEL_SEED_START, world_seed_start=WORLD_SEED_START)
        .set_engine_params(ENGINE_PARAMS)
        .set_pipeline(RepeatedPipeline(make_pipeline, repeats=REPEATS))
        .set_concurrency(PIPELINES_PER_GPU)
        .set_results_directory(RESULTS)
        .run(gpus=GPUS))
