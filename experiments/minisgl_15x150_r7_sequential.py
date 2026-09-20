"""Mini-sglang long sequential evaluation: 15 slots x 150 actions x 7 episodes."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
sys.path.insert(0, str(REPOSITORY))

from experiment_runner import Runner, RepeatedPipeline, Recorder
from pipelines.speleo import SpeleoPipeline, RoleParams
from pipelines.world import SpeleoWorld

VENV = REPOSITORY / '.venvs' / 'minisgl'
MODEL = REPOSITORY / 'models' / 'Qwen3.6-35B-A3B-FP8'
RESULTS = REPOSITORY / 'results' / 'minisgl_15x150_r7_sequential'
GPUS = [0]
PIPELINES_PER_GPU = 15
REPEATS = 7
ACTIONS = 150
MODEL_SEED_START = 0
WORLD_SEED_START = 0
DUMP_IMAGES = False
GIF_ON = False
ROLE_PARAMETERS = {
    'observer': RoleParams(budget=18, temperature=.35, seed_offset=1, top_k=20, top_p=.9),
    'planner': RoleParams(budget=60, temperature=.65, seed_offset=2, top_k=20, top_p=.9),
    'executor': RoleParams(budget=16, temperature=.45, seed_offset=5, top_k=20, top_p=.9),
    'falsifier': RoleParams(budget=18, temperature=.45, seed_offset=4, top_k=20, top_p=.9),
}

ENGINE_PARAMS = {
    'engine_config': {
        'model_path': str(MODEL), 'dtype': 'bfloat16', 'quantization': 'fp8',
        'max_running_req': 64, 'memory_ratio': .9, 'page_size': 16,
        # ~85,115 tokens/episode at full role budgets for 224x224 images.
        # 98,304 pages x 16 = 1,572,864 shared token slots (~23% spare
        # over 15 such histories). Qwen's BF16 KV costs 20 KiB/token:
        # this pool alone is 30 GiB, plus weights, GDN and graph buffers.
        # Repeats recycle private histories; do NOT multiply pool size by 7.
        'num_page_override': 98304, 'max_seq_len_override': 131072,
        'attention_backend': 'fi', 'max_prefill_rows': 4096,
        'cuda_graph_bs': [4, 16, 48, 64], 'cuda_graph_max_bs': 64,
        'shared_cuda_graph_prefill_rows': [256, 1024, 4096],
        'shared_cuda_graph_max_depth': 16,
        'generation_config': {'do_sample': True, 'temperature': .6, 'top_k': 20, 'top_p': .9},
    },
    'adapter_options': {'cpu_threads': 4},
}


def make_pipeline(engine, context):
    return SpeleoPipeline(SpeleoWorld(seed=context.world_seed, max_steps=ACTIONS),
        Recorder(context.results_directory, dump_images=DUMP_IMAGES, gif_on=GIF_ON),
        engine, context=context, max_actions=ACTIONS, role_params=ROLE_PARAMETERS)


if __name__ == '__main__':
    (Runner(VENV, model_seed_start=MODEL_SEED_START, world_seed_start=WORLD_SEED_START)
        .set_engine_params(ENGINE_PARAMS)
        .set_pipeline(RepeatedPipeline(make_pipeline, repeats=REPEATS))
        .set_concurrency(PIPELINES_PER_GPU)
        .set_results_directory(RESULTS)
        .run(gpus=GPUS))
