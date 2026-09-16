import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiment_runner import EpisodeContext, Recorder
from experiment_runner.logs import read_jsonl
from experiment_runner.blocks import BlockHandle
from pipelines.speleo import SpeleoPipeline
from pipelines.world import Observation


class FakeEngine:
    def __init__(self):
        self.live_blocks = {}
        self.common = None
        self.calls = []
        self.role_counts = {}

    async def cached_prefix(self, text):
        self.calls.append(('system', text))
        if self.common is None:
            self.common = await BlockHandle.create(self, 'shared/system')
            await self.prefill(text, [], self.common)
        return self.common.share()

    async def create_block(self):
        block = SimpleNamespace(num_tokens=0)
        self.live_blocks[id(block)] = block
        return block

    async def free_block(self, block):
        del self.live_blocks[id(block)]

    async def merge_blocks(self, left, right):
        self.calls.append(('snapshot', left.num_tokens, right.num_tokens))
        block = await self.create_block()
        block.num_tokens = left.num_tokens+right.num_tokens
        return block

    async def prefill(self, text, deps, target):
        self.calls.append(('prefill', text, [b.name for b in deps], target.name))
        target.raw.num_tokens += len(text)
        await asyncio.sleep(0)
        return None

    async def decode(self, token, deps, target):
        target.raw.num_tokens += 1
        await asyncio.sleep(0)
        return None

    async def prefill_messages(self, messages, deps, target):
        content = messages[0]['content']
        self.calls.append(('images', content[0]['text'], [int(item['image'][0, 0, 0]) for item in content[1:]]))
        target.raw.num_tokens += 2
        await asyncio.sleep(0)

    def new_generator(self, seed):
        return seed

    def encode(self, text):
        from pipelines.prompts import SPELEO
        return [[name for name, _ in SPELEO.actions].index(text)]

    def sample(self, output, *, generator, temperature, top_k, top_p):
        role = {1: 'observer', 2: 'planner', 3: 'executor_draft', 4: 'falsifier', 5: 'executor_refine'}[(generator % 1_000_000_007) % 1_000_003]
        self.calls.append(('sampling', role, temperature, top_k, top_p))
        self.role_counts[generator] = self.role_counts.get(generator, 0) + 1
        n = self.role_counts[generator]
        # Short real role streams; exercise EOS before wait_tokens(4).
        return n, ' REPLAN' if role == 'executor_draft' else ' evidence', n % 3 == 0

    def score_tokens(self, output, token_ids):
        return [float(i == 0) for i in range(len(token_ids))]


class FakeWorld:
    closed = False
    def __init__(self, fail=False):
        self.i = 0
        self.fail = fail

    async def reset(self):
        return Observation(np.zeros((2, 2, 3), dtype=np.uint8), info={'player_pos': [99, 10, 99]})

    async def pass_action(self, action):
        if self.fail:
            raise RuntimeError('world failed')
        assert action == 0
        self.i += 1
        return Observation(np.full((2, 2, 3), self.i, dtype=np.uint8), float(self.i), False,
                           {'player_pos': [99, 10-self.i, 99]})

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize('fail', [False, True])
def test_policy_feedback_frames_drain_and_ownership(tmp_path, fail):
    engine, world = FakeEngine(), FakeWorld(fail)
    context = EpisodeContext('test', 0, 0, tmp_path, 0, 0, 0)
    async def run():
        recorder = Recorder(tmp_path)
        pipeline = SpeleoPipeline(world, recorder, engine, context=context, max_actions=2)
        await pipeline.run()
    if fail:
        with pytest.raises(RuntimeError, match='world failed'):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert world.closed
    assert list(engine.live_blocks.values()) == [engine.common.raw]
    result = json.loads((tmp_path/'completion.json').read_text())
    assert result['status'] == ('failed' if fail else 'completed')
    events = list(read_jsonl(tmp_path/'events.jsonl'))
    decisions = [e for e in events if e['kind'] == 'decision']
    # A readout is recorded even when World subsequently fails to execute it.
    assert len(decisions) == (1 if fail else 2)
    assert all(e['action'] == 'wait' for e in decisions)
    if not fail:
        history = [e for e in events if e['kind'] == 'event']
        assert history[0]['feedback'] is None
        assert history[1]['feedback'] == {'reward': 1., 'episode_done': False}
        assert history[1]['after_action'] == 'wait'
        assert [c for c in engine.calls if c[0] == 'images'] == [
            ('images', 'Observation 0. Last action: none. First image previous; second image current.', [0, 0]),
            ('images', 'Observation 1. Last action: wait. First image previous; second image current.', [0, 1])]
        assert any(c[0] == 'snapshot' for c in engine.calls)
        assert {e['role'] for e in events if e['kind'] == 'stream'} == {
            'observer', 'planner', 'executor_draft', 'falsifier', 'executor_refine'}
        for call in engine.calls:
            if call[0] == 'prefill':
                assert 'player_pos' not in call[1]
                assert '[99,' not in call[1]
        assert [r['height'] for r in read_jsonl(tmp_path/'steps.jsonl')] == [10., 9., 8.]
