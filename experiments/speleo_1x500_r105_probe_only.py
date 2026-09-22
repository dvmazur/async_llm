"""Full change prompt; one world, 105 episodes, 500 actions, 0.2s pacing."""
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPOSITORY))
from experiment_runner import Runner, RepeatedPipeline, Recorder
from experiment_runner.probe_readout import ProbeReadout
from pipelines.probe import ProbePipeline
from pipelines.world import SpeleoWorld
from pipelines.settled_world import SettledWorld

VENV = REPOSITORY / '.venvs' / 'minisgl'
MODEL = REPOSITORY / 'models' / 'Qwen3.6-35B-A3B-FP8'
RESULTS = REPOSITORY / 'results' / 'speleo_1x500_r105_probe_only_sleep02'
GPUS = [0]
PIPELINES_PER_GPU = 1
REPEATS = 105
ACTIONS = 500
MODEL_SEED_START = WORLD_SEED_START = 0
DUMP_IMAGES = GIF_ON = False
TEMPERATURE = .7
ACTION_DELAY = .2
ENGINE_PARAMS = {'engine_config': {
    'model_path': str(MODEL), 'dtype': 'bfloat16', 'quantization': 'fp8',
    'max_running_req': 4, 'memory_ratio': .9, 'page_size': 16,
    # 65536 global slots (1.25 GiB), versus ~440 live token rows per decision.
    'num_page_override': 4096, 'max_seq_len_override': 2048,
    'attention_backend': 'fi', 'max_prefill_rows': 512,
    'cuda_graph_bs': [4], 'cuda_graph_max_bs': 4,
    'shared_cuda_graph_prefill_rows': [512], 'shared_cuda_graph_max_depth': 1,
    'generation_config': {'do_sample': True, 'temperature': .7, 'top_k': 7, 'top_p': 1.},
}, 'adapter_options': {'cpu_threads': 4}}


def make_pipeline(engine, context):
    recorder = Recorder(context.results_directory,dump_images=DUMP_IMAGES,gif_on=GIF_ON)
    world = SettledWorld(SpeleoWorld(seed=context.world_seed,
        max_steps=ACTIONS+SettledWorld.MAX_SETTLING_STEPS),recorder)
    return ProbePipeline(world,recorder,ProbeReadout(engine),context=context,
        max_actions=ACTIONS,temperature=TEMPERATURE,action_delay=ACTION_DELAY)


if __name__ == '__main__':
    (Runner(VENV,model_seed_start=MODEL_SEED_START,world_seed_start=WORLD_SEED_START)
        .set_engine_params(ENGINE_PARAMS)
        .set_pipeline(RepeatedPipeline(make_pipeline,repeats=REPEATS))
        .set_concurrency(PIPELINES_PER_GPU)
        .set_results_directory(RESULTS)
        .run(gpus=GPUS))
