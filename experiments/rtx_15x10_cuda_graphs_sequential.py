"""RTX 15x10: current pipeline, cuda_graphs_minimal/07-cuda-graphs (FP32 GDN)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT.parent
sys.path[:0] = [str(ROOT), str(DEPLOY / 'engine' / 'python')]

from experiment_runner import Runner, RepeatedPipeline, Recorder
from pipelines.speleo import SpeleoPipeline, RoleParams
from pipelines.world import SpeleoWorld

VENV = Path('/home/lordvoldebug_2/runner-check-20260915/venv')
PARAMS = {
    'engine_config': {
        'model_path': '/home/lordvoldebug_2/models/Qwen3.6-35B-A3B-FP8',
        'dtype': 'bfloat16', 'quantization': 'fp8',
        'max_running_req': 64, 'memory_ratio': .9, 'page_size': 16,
        'num_page_override': 8192, 'max_seq_len_override': 32768,
        'attention_backend': 'fi', 'max_prefill_rows': 4096,
        'cuda_graph_bs': [4, 16, 48, 64], 'cuda_graph_max_bs': 64,
        'shared_cuda_graph_prefill_rows': [256, 1024, 4096],
        'shared_cuda_graph_max_depth': 16,
        'generation_config': {'do_sample': True, 'temperature': .6, 'top_k': 20, 'top_p': .9},
    },
    'adapter_options': {'cpu_threads': 4},
}

ROLE_PARAMETERS = {
    'observer': RoleParams(budget=18, temperature=.35, seed_offset=1, top_k=20, top_p=.9),
    'planner': RoleParams(budget=60, temperature=.65, seed_offset=2, top_k=20, top_p=.9),
    'executor': RoleParams(budget=16, temperature=.45, seed_offset=5, top_k=20, top_p=.9),
    'falsifier': RoleParams(budget=18, temperature=.45, seed_offset=4, top_k=20, top_p=.9),
}


def make_pipeline(engine, context):
    return SpeleoPipeline(SpeleoWorld(seed=context.world_seed, max_steps=10),
        Recorder(context.results_directory, dump_images=False, gif_on=False),
        engine, context=context, max_actions=10, role_params=ROLE_PARAMETERS)


if __name__ == '__main__':
    (Runner(VENV, model_seed_start=0, world_seed_start=0)
        .set_engine_params(PARAMS)
        .set_pipeline(RepeatedPipeline(make_pipeline, repeats=1))
        .set_concurrency(15)
        .set_results_directory(DEPLOY / 'results-15x10_sequential')
        .run(gpus=[0]))
