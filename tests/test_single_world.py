import asyncio
from dataclasses import replace

from experiment_runner import EpisodeContext, Recorder
from experiment_runner.logs import read_jsonl
from pipelines.speleo import SpeleoPipeline, ROLE_PARAMS
from test_policy import FakeEngine, FakeWorld


def test_async_capacity_and_repeat_settings():
    from experiments import speleo_1x500_r105_falsifier_async as run
    assert (run.PIPELINES_PER_GPU, run.ACTIONS, run.REPEATS, run.GPUS) == (1,500,105,[0])
    c = run.ENGINE_PARAMS['engine_config']
    slots = c['num_page_override'] * c['page_size']
    assert slots == 262144  # 5 GiB, NOT a 256k per-request context
    # Two full 64k histories plus generous temporary-block allowance.
    assert slots >= 2*c['max_seq_len_override'] + 8192
    assert c['max_running_req'] >= len(run.ROLE_PARAMETERS)
    assert max(c['cuda_graph_bs']) == c['cuda_graph_max_bs'] == 8
    assert max(c['shared_cuda_graph_prefill_rows']) == c['max_prefill_rows'] == 4096
    assert c['shared_cuda_graph_max_depth'] >= 12
    assert run.ROLE_PARAMETERS == ROLE_PARAMS
    assert run.DUMP_IMAGES is run.GIF_ON is False


def test_factory_reserves_settling_steps(tmp_path):
    from experiments import speleo_1x500_r105_falsifier_async as run
    p = run.make_pipeline(FakeEngine(),EpisodeContext('factory',0,0,tmp_path,0,0,0))
    assert p.max_actions == 500
    assert p.world.world.settings['max_steps'] == 620
    asyncio.run(p.recorder.finish())


def test_repeats_release_all_private_blocks(tmp_path):
    async def run():
        engine = FakeEngine()
        params = {r:replace(p,budget=4) for r,p in ROLE_PARAMS.items()}
        peak = []
        for repeat in range(105):
            folder = tmp_path / str(repeat)
            world = FakeWorld()
            p = SpeleoPipeline(world, Recorder(folder), engine,
                context=EpisodeContext(f'repeat{repeat}',repeat,repeat,folder,0,repeat,0),
                max_actions=2,role_params=params)
            await p.run()
            assert world.closed and not p.jobs and all(t.done() for t in p.actors)
            assert list(engine.live_blocks.values()) == [engine.common.raw]
            assert p.history is p.common is None
            history = [e for e in read_jsonl(folder/'events.jsonl') if e['kind']=='history_size']
            assert [e['events'] for e in history] == [1,2]
            peak.append(history[0]['tokens'])
            engine.calls.clear()  # fake call history is not a production KV owner
        # New histories do not accumulate across episodes (seed-digit sizes may differ).
        assert max(peak) < 2*min(peak)
    asyncio.run(run())


def test_500_actions_complete_and_release(tmp_path):
    async def run():
        engine, world = FakeEngine(), FakeWorld()
        params = {r:replace(p,budget=4) for r,p in ROLE_PARAMS.items()}
        p = SpeleoPipeline(world,Recorder(tmp_path),engine,
            context=EpisodeContext('long',0,0,tmp_path,0,0,0),max_actions=500,role_params=params)
        await p.run()
        assert world.i == 500 and world.closed
        assert list(engine.live_blocks.values()) == [engine.common.raw]
        sizes = [e for e in read_jsonl(tmp_path/'events.jsonl') if e['kind']=='history_size']
        assert len(sizes)==500 and sizes[-1]['events']==500
    asyncio.run(run())
