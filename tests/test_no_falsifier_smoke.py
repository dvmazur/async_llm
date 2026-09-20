"""CPU smoke: real async role protocol, not a sequential replacement."""
import asyncio
from experiment_runner import EpisodeContext, Recorder
from experiment_runner.logs import read_jsonl
from pipelines.speleo import SpeleoPipeline
from test_policy import FakeEngine, FakeWorld


def test_fast_roles_without_falsifier(tmp_path):
    async def run():
        engine, world = FakeEngine(), FakeWorld()
        pipeline = SpeleoPipeline(world, Recorder(tmp_path), engine,
            context=EpisodeContext('smoke', 0, 0, tmp_path, 0, 0, 0), max_actions=2)
        await pipeline.run()
        assert world.i == 2 and world.closed
        assert len(pipeline.actors) == 2 and all(t.done() for t in pipeline.actors)
        assert not pipeline.jobs
        assert list(engine.live_blocks.values()) == [engine.common.raw]
        for call in engine.calls:
            if call[0] in ('prefill', 'system'):
                assert 'falsifier' not in call[1].lower()
                assert 'objection' not in call[1].lower()
        events = list(read_jsonl(tmp_path/'events.jsonl'))
        assert {e['role'] for e in events if e['kind'] == 'stream'} == {
            'observer', 'planner', 'executor'}
        decisions = [e for e in events if e['kind'] == 'decision']
        assert len(decisions) == 2
        assert all('critique' not in e and 'falsifier' not in e['live_inputs_at_action_submit']
                   for e in decisions)
        assert any(call[0] == 'snapshot' for call in engine.calls)
    asyncio.run(asyncio.wait_for(run(), 3))
