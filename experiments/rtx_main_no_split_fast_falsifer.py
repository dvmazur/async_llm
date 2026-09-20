"""Main + prefill split-KV opt-out; original 128MiB workspace, BF16 weights."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT.parent
p = argparse.ArgumentParser()
p.add_argument('--pipelines', type=int, choices=[1, 5, 7, 15], required=True)
args = p.parse_args()
sys.path[:0] = [str(ROOT), str(DEPLOY/'engine-main-no-split'/'python')]

from experiment_runner import Runner, RepeatedPipeline, Recorder
from pipelines.speleo import SpeleoPipeline, RoleParams
from pipelines.world import SpeleoWorld

PARAMS = {'engine_config': {
    'model_path': '/home/lordvoldebug_2/models/Qwen3.6-35B-A3B',
    'dtype': 'bfloat16', 'max_running_req': 64, 'memory_ratio': .9,
    'page_size': 16, 'num_page_override': 8192, 'max_seq_len_override': 32768,
    'attention_backend': 'fi', 'max_prefill_rows': 4096, 'cuda_graph_max_bs': 0,
    'shared_prefill_disable_split_kv': True,
    'generation_config': {'do_sample': True, 'temperature': .6, 'top_k': 20, 'top_p': .9},
}, 'adapter_options': {'cpu_threads': 4}}

ROLE_PARAMETERS = {
    'observer': RoleParams(budget=18, temperature=.35, seed_offset=1, top_k=20, top_p=.9),
    'planner': RoleParams(budget=60, temperature=.65, seed_offset=2, top_k=20, top_p=.9),
    'executor': RoleParams(budget=16, temperature=.45, seed_offset=5, top_k=20, top_p=.9),
}


def make_pipeline(engine, context):
    return SpeleoPipeline(SpeleoWorld(seed=context.world_seed, max_steps=10),
        Recorder(context.results_directory, dump_images=False, gif_on=False),
        engine, context=context, max_actions=10, role_params=ROLE_PARAMETERS)


if __name__ == '__main__':
    (Runner('/home/lordvoldebug_2/runner-check-20260915/venv', model_seed_start=0, world_seed_start=0)
        .set_engine_params(PARAMS)
        .set_pipeline(RepeatedPipeline(make_pipeline, repeats=1))
        .set_concurrency(args.pipelines)
        .set_results_directory(DEPLOY/f'main-no-split-{args.pipelines}x10_fast_falsifer')
        .run(gpus=[0]))
