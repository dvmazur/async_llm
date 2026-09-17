"""Local integration: real 0.8B FP8 + real worlds, independent repeats and media."""
import argparse
from experiment_runner import Runner, RepeatedPipeline, Recorder
from pipelines.speleo import SpeleoPipeline
from pipelines.world import SpeleoWorld
from copy import deepcopy
from experiments.speleo_15x5_fast_falsifer import ENGINE_PARAMS


def make_pipeline(engine, context):
    return SpeleoPipeline(SpeleoWorld(seed=context.world_seed, max_steps=2),
        Recorder(context.results_directory, dump_images=context.repeat == 0, gif_on=context.slot == 0),
        engine, context=context, max_actions=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--venv', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--results', required=True)
    args = parser.parse_args()
    params = deepcopy(ENGINE_PARAMS)
    params['engine_config']['model_path'] = args.model
    params['engine_config'].update(max_running_req=8, num_page_override=2048,
        cuda_graph_bs=[4, 8], cuda_graph_max_bs=8, max_seq_len_override=16384)
    (Runner(args.venv).set_engine_params(params).set_pipeline(RepeatedPipeline(make_pipeline, 2))
        .set_concurrency(2).set_results_directory(args.results).run(gpus=[0]))
