"""CPU contract tests for the single append-only conversation (no model mocks run GPU)."""
import asyncio
from collections import Counter
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiment_runner import EpisodeContext, Recorder
from experiment_runner.blocks import BlockHandle
from experiment_runner.logs import read_jsonl
from pipelines.prompts import SPELEO, role_request
from pipelines.speleo import SpeleoPipeline, ROLE_ORDER, ROLE_PARAMS
from pipelines.world import Observation


def test_role_limits_and_sequential_order():
    assert ROLE_ORDER == ('observer', 'planner', 'falsifier', 'executor')
    assert {r: (p.budget, p.temperature, p.seed_offset, p.top_k, p.top_p)
            for r, p in ROLE_PARAMS.items()} == {
        'observer': (18, .35, 1, 20, .9), 'planner': (60, .65, 2, 20, .9),
        'falsifier': (18, .45, 4, 20, .9), 'executor': (16, .45, 5, 20, .9)}


class Engine:
    from experiment_runner.generation import generate
    def __init__(self, *, fail=None, eos=False):
        self.common = None
        self.live = {}
        self.calls = []
        self.active = set()
        self.fail, self.eos = fail, eos
        self.samples = Counter()
        self.created = 0

    async def create_block(self):
        raw = SimpleNamespace(num_tokens=0, data=[])
        self.live[id(raw)] = raw
        self.created += 1
        return raw

    async def free_block(self, raw):
        assert id(raw) not in self.active
        del self.live[id(raw)]

    async def cached_prefix(self, text):
        if self.common is None:
            self.common = await BlockHandle.create(self, 'shared/system')
            self.common.raw.data.append(('system', text))
        return self.common.share()

    async def append(self, kind, value, deps, target):
        raw = target.raw
        assert [d.raw for d in deps] == [self.common.raw]
        assert raw is not self.common.raw
        assert id(raw) not in self.active  # no concurrent role requests per conversation
        self.active.add(id(raw))
        try:
            before = list(raw.data)
            self.calls.append((kind, value, raw, before, asyncio.current_task()))
            await asyncio.sleep(0)
            assert list(raw.data) == before and id(raw) in self.live
            raw.data.append((kind, value))
            raw.num_tokens += 1
        finally:
            self.active.remove(id(raw))

    async def prefill(self, text, deps, target):
        role = next((r for r in ROLE_ORDER if f'Role: {r.upper()}.' in text), None)
        if self.fail is not None and (self.fail == role or
                (self.fail == 'readout' and 'choose the actual next action' in text)):
            raise RuntimeError(self.fail)
        await self.append('text', text, deps, target)

    async def prefill_messages(self, messages, deps, target):
        if self.fail == 'images':
            raise RuntimeError('images')
        contents = messages[0]['content']
        value = (contents[0]['text'], [int(x['image'][0, 0, 0]) for x in contents[1:]])
        await self.append('images', value, deps, target)

    async def decode(self, token, deps, target):
        await self.append('token', token, deps, target)

    def new_generator(self, seed):
        return seed

    def sample(self, output, *, generator, temperature, top_k, top_p):
        role = {1: 'observer', 2: 'planner', 4: 'falsifier', 5: 'executor'}[generator % 1_000_003]
        self.samples[role] += 1
        return generator, role+' answer ', self.eos

    def encode(self, text):
        return [[name for name, _ in SPELEO.actions].index(text)]

    async def score_tokens(self, output, ids):
        assert ids == list(range(7))
        return [1., 0., 0., 0., 0., 0., 0.]


class World:
    def __init__(self, *, fail=False, done_after=None):
        self.i, self.closed = 0, False
        self.fail, self.done_after = fail, done_after

    async def reset(self):
        return Observation(np.zeros((2, 2, 3), dtype=np.uint8), info={'player_pos': [99, 10, 99]})

    async def pass_action(self, action):
        assert action == 0
        if self.fail:
            raise RuntimeError('world')
        self.i += 1
        return Observation(np.full((2, 2, 3), self.i, dtype=np.uint8), float(self.i),
            self.i == self.done_after, {'player_pos': [99, 10-self.i, 99]})

    async def aclose(self):
        self.closed = True


def pipeline(tmp_path, engine, world, **kwargs):
    return SpeleoPipeline(world, Recorder(tmp_path), engine,
        context=EpisodeContext('test', 7, 9, tmp_path, 3, 0, 0), **kwargs)


@pytest.mark.parametrize('eos', [False, True])
def test_order_whole_history_and_chosen_actions_are_retained(tmp_path, eos):
    engine, world = Engine(eos=eos), World()
    p = pipeline(tmp_path, engine, world, max_actions=3)
    asyncio.run(p.run())
    # One shared system block, exactly one private conversation; no per-role
    # allocations, snapshots, merges, recent-history truncation or rolling images.
    assert engine.created == 2
    assert len({id(c[2]) for c in engine.calls}) == 1
    assert len({id(c[4]) for c in engine.calls}) == 1
    transcript = engine.calls[-1][2].data
    assert all(c[3] == transcript[:i] for i, c in enumerate(engine.calls))
    role_calls = [c for c in engine.calls if c[0] == 'text' and c[1].startswith(('<|im_start|>user\nRole:', '<|im_end|>\n<|im_start|>user\nRole:'))]
    expected = [(step, role) for step in range(3) for role in ROLE_ORDER
                if role != 'planner' or step % 10 == 0]
    assert [r for c in role_calls for r in ROLE_ORDER if f'Role: {r.upper()}.' in c[1]] == [r for _, r in expected]
    for call, (step, role) in zip(role_calls, expected):
        assert call[1] == role_request(role, step,
            'none' if step == 0 else 'wait', close_previous=role != 'observer')
    images = [c for c in engine.calls if c[0] == 'images']
    assert [c[1][1] for c in images] == [[0, 0], [0, 1], [1, 2]]
    assert '"reward": 1.0' in images[1][1][0]
    assert '"reward": 2.0' in images[2][1][0]
    for call in images[1:]:
        assert call[3][-2:] == [('token', 0), ('text', '<|im_end|>\n')]
        assert any(kind == 'images' and value[1] == [0, 0] for kind, value in call[3])
    assert engine.samples == {r: (1 if r == 'planner' else 3)*(1 if eos else p.budget)
                              for r, p in ROLE_PARAMS.items()}
    rows = list(read_jsonl(tmp_path/'events.jsonl'))
    assert [r['role'] for r in rows if r['kind'] == 'stream'] == [r for _, r in expected]
    assert len([r for r in rows if r['kind'] == 'decision']) == 3
    assert [r['feedback'] for r in rows if r['kind'] == 'event'] == [None,
        {'reward': 1., 'episode_done': False}, {'reward': 2., 'episode_done': False}]
    assert all('player_pos' not in c[1] for c in engine.calls if c[0] == 'text')
    assert world.closed and p.history is p.common is None
    assert list(engine.live.values()) == [engine.common.raw]
    assert json.loads((tmp_path/'completion.json').read_text())['status'] == 'completed'


@pytest.mark.parametrize('failure', ['images', *ROLE_ORDER, 'readout', 'world'])
def test_failure_releases_conversation(tmp_path, failure):
    engine, world = Engine(fail=failure), World(fail=failure == 'world')
    p = pipeline(tmp_path, engine, world, max_actions=2)
    with pytest.raises(RuntimeError, match=failure):
        asyncio.run(p.run())
    assert world.closed and not engine.active
    assert list(engine.live.values()) == [engine.common.raw]
    assert json.loads((tmp_path/'completion.json').read_text())['status'] == 'failed'


def test_two_episodes_share_model_not_conversations(tmp_path):
    async def run():
        engine = Engine(eos=True)
        ps = []
        for i in range(2):
            out = tmp_path/str(i)
            ps.append(SpeleoPipeline(World(done_after=2), Recorder(out), engine,
                context=EpisodeContext(str(i), i, i, out, i, 0, 0), max_actions=5))
        await asyncio.gather(*(p.run() for p in ps))
        assert engine.created == 3
        assert len({id(c[2]) for c in engine.calls}) == 2
        assert all(p.world.i == 2 and p.world.closed for p in ps)
        assert list(engine.live.values()) == [engine.common.raw]
    asyncio.run(run())


def test_stop_waits_for_complete_action_without_starting_another(tmp_path):
    engine, world = Engine(eos=True), World()
    p = pipeline(tmp_path, engine, world, max_actions=5)
    p.stop_requested = lambda: world.i >= 1
    asyncio.run(p.run())
    assert world.i == 1 and world.closed
    assert engine.samples == dict.fromkeys(ROLE_ORDER, 1)
    assert list(engine.live.values()) == [engine.common.raw]
    assert json.loads((tmp_path/'completion.json').read_text())['status'] == 'stopped'


@pytest.mark.parametrize('eos', [False, True])
def test_planner_cadence_including_empty_answers(tmp_path, eos):
    engine, world = Engine(eos=eos), World()
    p = pipeline(tmp_path, engine, world, max_actions=21)
    asyncio.run(p.run())
    rows = list(read_jsonl(tmp_path/'events.jsonl'))
    assert [r['observation'] for r in rows if r['kind'] == 'stream' and r['role'] == 'planner'] == [0, 10, 20]
    decisions = [r for r in rows if r['kind'] == 'decision']
    assert [r['observation'] for r in decisions if r['new_planner_started']] == [0, 10, 20]
    assert [r['published_plan_based_on'] for r in decisions] == (
        [-1]*21 if eos else [0]*10 + [10]*10 + [20])
    assert engine.samples['planner'] == 3*(1 if eos else 60)
