import asyncio
from dataclasses import replace
import json

import pytest

from experiment_runner import EpisodeContext, Recorder
from experiment_runner.logs import read_jsonl
from pipelines.speleo import SpeleoPipeline, ROLE_PARAMS
from test_policy import FakeEngine, FakeWorld


class ControlledEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.planner_release = asyncio.Event()
        self.executor_started = asyncio.Event()
        self.falsifier_started = asyncio.Event()
        self.tasks = {}
        self.live_reads = []
        self.snapshots = []

    async def merge_blocks(self, left, right):
        block = await super().merge_blocks(left, right)
        self.snapshots.append((block, block.num_tokens))
        return block

    async def prefill(self, text, deps, target):
        parts = target.name.split('/')
        if len(parts) >= 4:
            phase = parts[2]
            self.tasks.setdefault(phase, set()).add(asyncio.current_task())
            if phase in ('executor', 'falsifier'):
                observed = 'observer' if phase == 'executor' else 'executor'
                live = [b.producer for b in deps if b.producer is not None and b.producer.phase == observed]
                assert live
                self.live_reads.append((phase, len(live[0].tokens), live[0].done))
                (self.executor_started if phase == 'executor' else self.falsifier_started).set()
        return await super().prefill(text, deps, target)

    def sample(self, output, **kwargs):
        token, piece, _ = super().sample(output, **kwargs)
        return token, piece, False  # stop by each role's test budget, not early EOS

    async def decode(self, token, deps, target):
        phase = target.name.split('/')[2]
        if phase == 'planner':
            await self.planner_release.wait()
        if phase == 'observer' and target.raw.num_tokens == 1:
            await self.executor_started.wait()
        if phase == 'executor' and target.raw.num_tokens == 4:
            await self.falsifier_started.wait()
        return await super().decode(token, deps, target)


def test_live_kv_communication_and_planner_across_actions(tmp_path):
    async def run():
        engine = ControlledEngine()
        class World(FakeWorld):
            async def pass_action(self, action):
                assert not engine.planner_release.is_set()
                obs = await super().pass_action(action)
                if self.i == 2:
                    engine.planner_release.set()
                return obs
        world = World()
        params = {name: replace(p, budget={'planner': 12, 'observer': 8,
            'falsifier': 8, 'executor': 8}[name]) for name, p in ROLE_PARAMS.items()}
        pipeline = SpeleoPipeline(world, Recorder(tmp_path), engine,
            context=EpisodeContext('live', 0, 0, tmp_path, 0, 0, 0), max_actions=2, role_params=params)
        await pipeline.run()
        assert not pipeline.jobs and all(t.done() for t in pipeline.actors)
        assert list(engine.live_blocks.values()) == [engine.common.raw]
        assert all(len(engine.tasks[r]) == 1 for r in ('observer', 'planner', 'falsifier'))
        assert len(engine.tasks['executor']) == 2  # one finite generation task per action
        assert engine.live_reads[0] == ('executor', 1, False)
        assert engine.live_reads[1] == ('falsifier', 4, False)
        assert len(engine.snapshots) == 1  # same planner remained live across both steps
        assert all(block.num_tokens == original for block, original in engine.snapshots)
        events = list(read_jsonl(tmp_path/'events.jsonl'))
        decisions = [e for e in events if e['kind'] == 'decision']
        assert len(decisions) == 2
        assert all(not e['live_inputs_at_action_submit']['planner_done'] for e in decisions)
        assert decisions[0]['new_planner_started'] and not decisions[1]['new_planner_started']
        assert {e['role']: e['temperature'] for e in events if e['kind'] == 'role_parameters'} == {
            'observer': .35, 'planner': .65, 'executor': .45, 'falsifier': .45}
    asyncio.run(asyncio.wait_for(run(), 3))


def test_failed_role_unblocks_readers_and_drains_owned_blocks(tmp_path):
    class Broken(FakeEngine):
        async def prefill(self, text, deps, target):
            if '/observer/instruction' in target.name:
                raise RuntimeError('observer failed')
            return await super().prefill(text, deps, target)
    async def run():
        engine, world = Broken(), FakeWorld()
        pipeline = SpeleoPipeline(world, Recorder(tmp_path), engine,
            context=EpisodeContext('error', 0, 0, tmp_path, 0, 0, 0), max_actions=2)
        with pytest.raises(RuntimeError, match='observer failed'):
            await pipeline.run()
        assert world.closed and not pipeline.jobs
        assert all(t.done() for t in pipeline.actors)
        assert list(engine.live_blocks.values()) == [engine.common.raw]
        assert json.loads((tmp_path/'completion.json').read_text())['status'] == 'failed'
    asyncio.run(asyncio.wait_for(run(), 3))


def test_two_pipelines_share_engine_not_role_state(tmp_path):
    async def run():
        engine = FakeEngine()
        pipelines = []
        for index in range(2):
            target = tmp_path/str(index)
            target.mkdir()
            pipelines.append(SpeleoPipeline(FakeWorld(), Recorder(target), engine,
                context=EpisodeContext(str(index), index, index, target, 0, 0, 0), max_actions=2))
        await asyncio.gather(*(p.run() for p in pipelines))
        assert all(not p.jobs for p in pipelines)
        assert list(engine.live_blocks.values()) == [engine.common.raw]
        seeds = []
        for index in range(2):
            events = list(read_jsonl(tmp_path/str(index)/'events.jsonl'))
            assert len([e for e in events if e['kind'] == 'decision']) == 2
            seeds.append({e['seed'] for e in events if e['kind'] == 'role_parameters'})
        assert seeds[0].isdisjoint(seeds[1])
    asyncio.run(asyncio.wait_for(run(), 3))


def test_sampling_configuration_belongs_to_the_pipeline(tmp_path):
    async def run():
        engine = FakeEngine()
        pipeline = SpeleoPipeline(FakeWorld(), Recorder(tmp_path), engine,
            context=EpisodeContext('configured', 0, 0, tmp_path, 0, 0, 0), max_actions=1,
            role_params={'observer': replace(ROLE_PARAMS['observer'], budget=2,
                temperature=.12, top_k=7, top_p=.8)})
        await pipeline.run()
        samples = [c for c in engine.calls if c[:2] == ('sampling', 'observer')]
        assert samples == [('sampling', 'observer', .12, 7, .8)] * 2
        assert list(engine.live_blocks.values()) == [engine.common.raw]
    asyncio.run(asyncio.wait_for(run(), 3))


@pytest.mark.parametrize('replan', [False, True])
def test_roles_own_perception_refresh_publication_and_review(tmp_path, replan):
    class Engine(FakeEngine):
        async def prefill_messages(self, messages, deps, target):
            assert asyncio.current_task().get_name().endswith('/observer')
            return await super().prefill_messages(messages, deps, target)

        async def merge_blocks(self, left, right):
            assert asyncio.current_task().get_name().endswith('/planner')
            return await super().merge_blocks(left, right)

        async def prefill(self, text, deps, target):
            if '/falsifier/instruction' in target.name:
                assert asyncio.current_task().get_name().endswith('/falsifier')
            return await super().prefill(text, deps, target)

        def sample(self, output, **kwargs):
            token, piece, _ = super().sample(output, **kwargs)
            return token, piece if replan else ' evidence', False

    class Recording(Recorder):
        def log(self, kind, fields):
            if kind in ('history_snapshot', 'plan_published'):
                assert asyncio.current_task().get_name().endswith('/planner')
            return super().log(kind, fields)

    async def run():
        engine = Engine()
        params = {name: replace(p, budget=1 if name == 'planner' else 4)
                  for name, p in ROLE_PARAMS.items()}
        pipeline = SpeleoPipeline(FakeWorld(), Recording(tmp_path), engine,
            context=EpisodeContext('policy', 0, 0, tmp_path, 0, 0, 0),
            max_actions=4, role_params=params, planner_interval=2)
        await pipeline.run()
        events = list(read_jsonl(tmp_path/'events.jsonl'))
        starts = [e['observation'] for e in events if e['kind'] == 'history_snapshot']
        assert starts == [0, 2]  # REPLAN does not bypass the start interval
        publications = [e for e in events if e['kind'] == 'plan_published']
        assert [(e['based_on'], e['current_observation']) for e in publications] == (
            [(0, 1), (2, 3)])
        reviews = [e['observation'] for e in events
                   if e['kind'] == 'stream' and e['role'] == 'falsifier']
        assert reviews == [0, 1, 2, 3]
        assert list(engine.live_blocks.values()) == [engine.common.raw]
    asyncio.run(asyncio.wait_for(run(), 3))


@pytest.mark.parametrize('failure', ['images', 'snapshot', 'recent'])
def test_failure_before_role_reply_does_not_hang_or_free_live_inputs(tmp_path, failure):
    class Engine(FakeEngine):
        async def prefill_messages(self, messages, deps, target):
            await asyncio.sleep(0)
            assert not target.closed
            if failure == 'images':
                raise RuntimeError(failure)
            return await super().prefill_messages(messages, deps, target)

        async def merge_blocks(self, left, right):
            if failure == 'snapshot':
                raise RuntimeError(failure)
            return await super().merge_blocks(left, right)

        async def prefill(self, text, deps, target):
            if failure == 'recent' and '/recent:' in target.name:
                raise RuntimeError(failure)
            return await super().prefill(text, deps, target)

    async def run():
        engine, world = Engine(), FakeWorld()
        pipeline = SpeleoPipeline(world, Recorder(tmp_path), engine,
            context=EpisodeContext('failure', 0, 0, tmp_path, 0, 0, 0), max_actions=2)
        with pytest.raises(RuntimeError, match=failure):
            await pipeline.run()
        assert world.closed and not pipeline.jobs
        assert all(task.done() for task in pipeline.actors)
        assert list(engine.live_blocks.values()) == [engine.common.raw]
    asyncio.run(asyncio.wait_for(run(), 3))
