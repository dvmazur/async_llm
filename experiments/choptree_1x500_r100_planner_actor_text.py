"""On-device ChopTree: text Planner -> text Actor -> action readout, 1x500x100.

Edit the configuration below, then run this file without command-line arguments.
"""
from pathlib import Path
import asyncio
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from experiment_runner import Runner, RepeatedPipeline, Recorder
from pipelines.choptree_target_actor import TargetActorReadout, ControlMemory, TARGET_PROMPT
from pipelines.choptree import ACTION_NAMES
from pipelines.probe import ProbePipeline
from pipelines.world import ChopTreeWorld
from pipelines.settled_world import SettledWorld
from pipelines.synchronized_start import SynchronizedStart

VENV = REPOSITORY / '.venvs' / 'minisgl'
MODEL = REPOSITORY / 'models' / 'Qwen3.6-35B-A3B-FP8'
CRAFTIUM = REPOSITORY / 'craftium'
RESULTS = REPOSITORY / 'results' / 'choptree_1x500_r100_planner_actor_text'
GPUS = [0]
PIPELINES_PER_GPU = 1
START_BARRIER = asyncio.Barrier(PIPELINES_PER_GPU)
ACTIONS = 500
REPEATS = 100
MODEL_SEED_START = WORLD_SEED_START = 0
DUMP_IMAGES = False
GIF_ON = True
TEMPERATURE = .7  # motor actor
PLANNER_TEMPERATURE = .5
PLANNER_RECOVERY_TEMPERATURE = .9
PLANNER_RECOVERY_AFTER = 12  # actions without wood reward
MAX_ROLE_TOKENS = 768
ACTION_DELAY = .1
FRAMESKIP = 4
PMUL = 2
TURN_DEGREES = 7
VISION = 'plain'
PROMPT = TARGET_PROMPT
ENGINE_PARAMS = {'engine_config': {
    'model_path': str(MODEL), 'dtype': 'bfloat16', 'quantization': 'fp8',
    'max_running_req': 8, 'memory_ratio': .9, 'page_size': 16,
    # 65536 KV slots, 1.25 GiB on A3B. Only CURRENT decisions live in KV.
    'num_page_override': 4096, 'attention_backend': 'fi',
    'max_prefill_rows': 4096, 'cuda_graph_bs': [1, 2, 4, 8],
    'cuda_graph_max_bs': 8, 'shared_cuda_graph_prefill_rows': [1024, 2048, 4096],
    'shared_cuda_graph_max_depth': 1,
    'generation_config': {'do_sample': True, 'temperature': .7, 'top_k': 8, 'top_p': 1.},
}, 'adapter_options': {'cpu_threads': 4}}


def make_pipeline(engine, context):
    recorder = Recorder(context.results_directory, dump_images=DUMP_IMAGES, gif_on=GIF_ON)
    world = SettledWorld(ChopTreeWorld(seed=context.world_seed,
        max_steps=ACTIONS + SettledWorld.MAX_SETTLING_STEPS, frameskip=FRAMESKIP,
        pmul=PMUL, turn_degrees=TURN_DEGREES, craftium_directory=str(CRAFTIUM)),
        recorder, expected_spawn=None)
    world = SynchronizedStart(world, START_BARRIER)
    memory = ControlMemory()
    readout = TargetActorReadout(engine, recorder, memory,
        planner_temperature=PLANNER_TEMPERATURE,
        recovery_temperature=PLANNER_RECOVERY_TEMPERATURE,
        recovery_after=PLANNER_RECOVERY_AFTER, max_role_tokens=MAX_ROLE_TOKENS)
    return ProbePipeline(world, recorder, readout,
        context=context, max_actions=ACTIONS, temperature=TEMPERATURE, action_delay=ACTION_DELAY,
        prompt=PROMPT, action_names=ACTION_NAMES,
        variant='choptree-planner-actor-text-readout-v9',
        include_last_action=True, feedback=memory,
        vision=VISION)


if __name__ == '__main__':
    (Runner(VENV, model_seed_start=MODEL_SEED_START, world_seed_start=WORLD_SEED_START)
        .set_engine_params(ENGINE_PARAMS)
        .set_pipeline(RepeatedPipeline(make_pipeline, repeats=REPEATS))
        .set_concurrency(PIPELINES_PER_GPU)
        .set_results_directory(RESULTS)
        .run(gpus=GPUS))
