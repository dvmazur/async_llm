import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiment_runner import EpisodeContext, Recorder
from experiment_runner.logs import read_jsonl
from pipelines.choptree import ACTION_NAMES, PROMPT, WOOD_ONLY_PROMPT, ChopTreeFeedback
from pipelines.probe import ProbePipeline, messages_for
from pipelines.settled_world import SettledWorld
from pipelines.world import ChopTreeWorld, Observation, _make_environment


class Engine:
    def __init__(self):
        self.live, self.frames, self.last_actions = [], [], []
    def encode(self, name):
        return [ACTION_NAMES.index(name)]
    def new_generator(self, seed):
        return seed
    async def create_block(self):
        b = SimpleNamespace(num_tokens=0)
        self.live.append(b)
        return b
    async def free_block(self, b):
        self.live.remove(b)
    async def prefill_action(self, messages, block):
        assert len(self.live) == 1 and block.raw.num_tokens == 0
        assert len(messages) == 2 and messages[0]['content'] == PROMPT
        assert set(messages[1]) == {'role', 'content'}
        self.last_actions.append([x['text'] for x in messages[1]['content']
                                  if x['type'] == 'text' and x['text'].startswith('Action between')])
        self.frames.append([int(x['image'][0, 0, 0]) for x in messages[1]['content'] if x['type'] == 'image'])
        block.raw.num_tokens = 500
        return object(), 500
    def sample_action(self, output, ids, *, generator, temperature):
        assert ids == list(range(8)) and temperature == .7
        return 3, [.9, 0., 0., .1, 0., 0., 0., 0.]  # dig, not argmax


class World:
    metadata = {'environment': 'Craftium/ChopTree-v0'}
    def __init__(self, engine):
        self.engine, self.n, self.closed = engine, 0, False
    async def reset(self):
        return Observation(np.zeros((2, 2, 3), dtype=np.uint8), info={})
    async def pass_action(self, action):
        assert action == 3 and not self.engine.live
        self.n += 1
        return Observation(np.full((2, 2, 3), self.n, dtype=np.uint8), 1., self.n == 2,
                           {'terminated': True, 'truncated': False})
    async def aclose(self):
        self.closed = True


def test_choptree_fresh_prefill_sampled_dig_reward_and_termination(tmp_path):
    engine = Engine()
    world = World(engine)
    context = EpisodeContext('chop', 0, 0, tmp_path, 0, 0, 0)
    pipeline = ProbePipeline(world, Recorder(tmp_path), engine, context=context,
        max_actions=10, action_delay=0, prompt=PROMPT, action_names=ACTION_NAMES,
        variant='choptree-change-224-fs8', include_last_action=True)
    asyncio.run(pipeline.run())
    assert engine.frames == [[0, 0], [0, 1]] and world.closed and not engine.live
    assert engine.last_actions == [['Action between these observations: none (episode start).'],
                                   ['Action between these observations: dig.']]
    steps = [s for s in read_jsonl(tmp_path/'steps.jsonl') if s['kind'] == 'action']
    assert len(steps) == 2 and sum(s['reward'] for s in steps) == 2
    assert all(s['action'] == 'dig' and s['action_index'] == 3 for s in steps)
    assert steps[-1]['done'] and steps[-1]['info']['terminated']
    events = list(read_jsonl(tmp_path/'events.jsonl'))
    assert all(e['nonargmax'] for e in events if e['kind'] == 'decision')
    assert events[0]['action_names'] == list(ACTION_NAMES)
    assert events[0]['last_action_input'] is True
    assert json.loads((tmp_path/'completion.json').read_text())['status'] == 'completed'


@pytest.mark.parametrize('correct', [True, False])
def test_registered_environment_not_reconstructed(monkeypatch, correct):
    calls, closed = [], []
    actions = ['forward', 'jump', 'dig', 'mouse x+', 'mouse x-', 'mouse y+', 'mouse y-']
    env = SimpleNamespace(action_space=SimpleNamespace(n=8), actions=actions if correct else actions[::-1],
                          close=lambda: closed.append(True))
    def make(*args, **kwargs):
        calls.append((args, kwargs))
        return env
    monkeypatch.setitem(sys.modules, 'gymnasium', SimpleNamespace(make=make))
    settings = ChopTreeWorld(seed=0, max_steps=2120).settings
    if correct:
        assert _make_environment(settings, Path('/craftium')) is env
        assert env.mouse_mov == .5*64/224
    else:
        with pytest.raises(RuntimeError, match='mapping'):
            _make_environment(settings, Path('/craftium'))
        assert closed
    assert calls == [(('Craftium/ChopTree-v0',), dict(minetest_dir='/craftium',
                      env_dir='/craftium/craftium-envs/chop-tree', max_timesteps=2120,
                      offscreen_sdl=False, render_mode='rgb_array',
                      obs_width=224, obs_height=224, frameskip=8, pmul=20,
                      minetest_conf={'mouse_sensitivity': .2*10/35.84, 'time_speed': 0}))]


@pytest.mark.parametrize('frameskip', [1, 4, 8])
def test_choptree_freezes_noon_without_changing_movement(frameskip):
    from pipelines.world import SpeleoWorld
    settings = ChopTreeWorld(seed=0, frameskip=frameskip).settings['gym_kwargs']
    assert settings['minetest_conf']['time_speed'] == 0
    assert settings['minetest_conf']['mouse_sensitivity'] == .2*10/(4.48*frameskip)
    assert settings['frameskip'] == frameskip and settings['pmul'] == 20
    assert 'sync_mode' not in settings
    assert 'gym_kwargs' not in SpeleoWorld(seed=0).settings


def test_procedural_spawn_waits_for_stationary_player_not_speleo_coordinates():
    class MovingWorld:
        metadata = {}
        n = 0
        async def reset(self):
            return self.obs()
        def obs(self):
            return Observation(np.arange(12).reshape(2, 2, 3), info={
                'player_pos': [100., float(max(0, 12-self.n)), 100.],
                'player_vel': [0., -1. if self.n < 12 else 0., 0.]})
        async def pass_action(self, a):
            assert a == 0
            self.n += 1
            return self.obs()
    world = MovingWorld()
    logs = []
    ready = SettledWorld(world, SimpleNamespace(log=lambda k, v: logs.append((k, v))),
                         interval=0, expected_spawn=None)
    obs = asyncio.run(ready.reset())
    assert world.n == 17 and obs.info['player_pos'] == [100., 0., 100.]
    assert logs[-1][0] == 'reset_ready'


def test_last_action_is_opt_in_and_only_one_step_not_a_chat_history():
    plain = messages_for('previous', 'current')
    informed = messages_for('previous', 'current', PROMPT, last_action='left')
    assert len(plain) == len(informed) == 2
    assert len(plain[1]['content']) == 5
    assert len(informed[1]['content']) == 6
    assert informed[1]['content'][-2] == dict(type='text', text='Action between these observations: left.')


def test_frameskip_preserves_camera_angle_and_movement_multiplier():
    four = ChopTreeWorld(seed=0, frameskip=4).settings['gym_kwargs']
    eight = ChopTreeWorld(seed=0, frameskip=8).settings['gym_kwargs']
    assert four['pmul'] == eight['pmul'] == 20
    assert four['minetest_conf']['mouse_sensitivity']*4 == eight['minetest_conf']['mouse_sensitivity']*8
    with pytest.raises(ValueError):
        ChopTreeWorld(seed=0, frameskip=0)


@pytest.mark.parametrize('kwargs', [{'pmul': 0}, {'pmul': float('inf')},
    {'turn_degrees': -1}, {'turn_degrees': float('nan')}])
def test_invalid_native_control_parameters(kwargs):
    with pytest.raises(ValueError):
        ChopTreeWorld(seed=0, **kwargs)


def test_game_start_port_skips_live_udp_sockets_and_ephemeral_range(tmp_path, monkeypatch):
    import socket
    import tempfile
    from pipelines.world import _game_start_port
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path))
    low, high = map(int, Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as live:
        with _game_start_port() as first:
            assert not low <= first <= high
            live.bind(('0.0.0.0', first))
        with _game_start_port() as second:
            assert second != first and not low <= second <= high
