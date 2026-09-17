"""V9's policy contract, independent of GPU throughput or sampling numerics."""
import asyncio
from collections import Counter
import hashlib
import json

import pytest

from experiment_runner import EpisodeContext, Recorder
from experiment_runner.logs import read_jsonl
from pipelines import prompts
from pipelines.speleo import SpeleoPipeline, ROLE_PARAMS
from test_policy import FakeEngine, FakeWorld


def test_prompt_payloads_match_historical_v9():
    # Golden computed from shared_control_t60_v9/prompts.py in the preserved
    # -13.6276 run, not from the previous Falsifier prompts. Whitespace matters
    # in model input; source formatting/comments do not affect this contract.
    texts = [prompts.SPELEO.system()]
    for step in (0, 1, 10):
        for role in ('observer', 'planner', 'executor', 'falsifier'):
            for close in (False, True):
                texts.append(prompts.role_request(role, step, 'left', close))
        texts.append(prompts.action_request(prompts.SPELEO, step))
        event = prompts.history_event(step, 'left', 'report', {'reward': -5.5, 'episode_done': False})
        texts.extend([prompts.event_text(event), prompts.recent_text([event], 'plan', 0, step)])
    assert hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest() == (
        '346de0cd2a90c2cdd361b6720fb7b217d72105a88b411ebd657dce49717d0a9a')


@pytest.mark.parametrize('empty_plans', [False, True])
def test_v9_budget_cadence_rng_and_readout_chain(tmp_path, empty_plans):
    class Engine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.readouts = 0

        def sample(self, output, **kwargs):
            token, text, _ = super().sample(output, **kwargs)
            terminal = empty_plans and kwargs['generator'] % 1_000_003 == 2
            return token, text, terminal  # executor continually requests REPLAN

        async def prefill(self, text, deps, target):
            if '/action:' in target.name:
                self.readouts += 1
                # Historical V9 reads each block exactly once, with executor
                # BEFORE falsifier; no hidden refinement or duplicate tail.
                expected = ['system', 'recent', 'images', 'observer/instruction', 'observer/draft']
                if any('/planner/' in b.name for b in deps):
                    expected += ['planner/instruction', 'planner/draft']
                expected += ['executor/instruction', 'executor/draft', 'falsifier/instruction', 'falsifier/draft']
                names = [b.name.split(':')[0] for b in deps]
                assert len({id(b.raw) for b in deps}) == len(deps)
                assert len(names) == len(expected)
                assert all(n.endswith('/'+e) for n, e in zip(names, expected))
                assert all(b.producer.done for b in deps if b.producer and
                           b.producer.phase in ('executor', 'falsifier'))
            return await super().prefill(text, deps, target)

    async def run():
        engine = Engine()
        context = EpisodeContext('v9', 7, 9, tmp_path, 3, 0, 0)
        pipeline = SpeleoPipeline(FakeWorld(), Recorder(tmp_path), engine,
                                 context=context, max_actions=100)
        await pipeline.run()
        rows = list(read_jsonl(tmp_path/'events.jsonl'))
        streams = [r for r in rows if r['kind'] == 'stream']
        assert Counter(r['role'] for r in streams) == {
            'observer': 100, 'executor': 100, 'falsifier': 100, 'planner': 10}
        starts = [r['observation'] for r in rows if r['kind'] == 'history_snapshot']
        assert starts == list(range(0, 100, 10))  # also after EOS-only plans
        assert sum(r['sampled_tokens'] for r in streams) == 5200 + (10 if empty_plans else 600)
        assert engine.readouts == 100  # <=5900 total generated+action tokens / 100 actions
        parameters = {r['role']: r for r in rows if r['kind'] == 'role_parameters'}
        assert {r: p['seed'] for r, p in parameters.items()} == {
            'observer': 1_000_003*8+1, 'planner': 1_000_003*8+2,
            'executor': 1_000_003*8+5, 'falsifier': 1_000_003*8+4}
        assert {r: (p['budget'], p['temperature'], p['top_k'], p['top_p'])
                for r, p in parameters.items()} == {
            'observer': (18, .35, 20, .9), 'planner': (60, .65, 20, .9),
            'executor': (16, .45, 20, .9), 'falsifier': (18, .45, 20, .9)}
        assert not pipeline.jobs and all(t.done() for t in pipeline.actors)
        assert list(engine.live_blocks.values()) == [engine.common.raw]
    asyncio.run(asyncio.wait_for(run(), 10))


@pytest.mark.parametrize('phase', ['executor', 'falsifier'])
def test_live_executor_is_drained_after_generation_failure(tmp_path, phase):
    class Broken(FakeEngine):
        async def decode(self, token, deps, target):
            if f'/{phase}/draft:' in target.name:
                raise RuntimeError(phase)
            return await super().decode(token, deps, target)

    async def run():
        engine, world = Broken(), FakeWorld()
        pipeline = SpeleoPipeline(world, Recorder(tmp_path), engine,
            context=EpisodeContext('failure', 0, 0, tmp_path, 0, 0, 0), max_actions=2)
        with pytest.raises(RuntimeError, match=phase):
            await pipeline.run()
        assert world.closed and not pipeline.jobs
        assert all(t.done() for t in pipeline.actors)
        assert list(engine.live_blocks.values()) == [engine.common.raw]
    asyncio.run(asyncio.wait_for(run(), 3))
