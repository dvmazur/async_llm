
import _self_evolving_bootstrap  # noqa: F401
import asyncio
import json
import os
import time
import unittest
from unittest.mock import patch

from tasks.health_gathering_env import HealthGatheringEnv
from tasks.runner import run_episodes


class SynchronousTests(unittest.TestCase):
    def test_configured_horizons_and_explicit_override(self):
        from tasks.doom_env import DoomEnv
        with patch.dict(os.environ, SEA_GAME_MODE='synchronous', SEA_DOOM_TIC_LIMIT='20',
                        SEA_HEALTH_GATHERING_TIC_LIMIT='40'):
            for cls, expected in [(DoomEnv, 20), (HealthGatheringEnv, 40)]:
                env = cls(seed=123)
                try:
                    self.assertEqual(env.env.game_tic_limit, expected)
                finally:
                    env.close()
            env = HealthGatheringEnv(episode_timeout=8, seed=123)
            try:
                self.assertEqual(env.env.game_tic_limit, 8)
            finally:
                env.close()

    def test_inference_pauses_native_clock_and_steps_reach_exact_timeout(self):
        with patch.dict(os.environ, SEA_GAME_MODE='synchronous'):
            env = HealthGatheringEnv(episode_timeout=10, seed=123)
        try:
            env.reset()
            time.sleep(.2)
            self.assertEqual(env.env.game.get_episode_time() - env.env._initial_tic, 0)
            self.assertEqual(env.poll()[3]['game_tics'], 0)
            for tics in (4, 8, 10):
                _, reward, done, info = env.step('wait')
                self.assertEqual(info['game_tics'], tics)
                self.assertEqual(reward, 2 if tics == 10 else 4)
                self.assertEqual(done, tics == 10)
                json.dumps(info)
            self.assertTrue(info['timeout'])
            self.assertEqual(info['episode_total_reward'], 10)
        finally:
            env.close()

    def test_runner_ignores_decision_cap_for_native_sync_horizon(self):
        with patch.dict(os.environ, SEA_GAME_MODE='synchronous'):
            env = HealthGatheringEnv(episode_timeout=12, seed=123)
        env.max_episodes = 1
        env.max_steps_per_episode = 1
        class Engine:
            async def act(self, obs, on_token=None):
                before = env.env.game.get_episode_time()
                await asyncio.sleep(.1)
                self_test.assertEqual(env.env.game.get_episode_time(), before)
                return 'wait'
        self_test = self
        result = asyncio.run(run_episodes(env, Engine()))['episodes'][0]
        self.assertNotIn('error', result['info'])
        self.assertEqual(result['steps'], 3)
        self.assertEqual(result['reward'], 12)
        self.assertTrue(result['info']['timeout'])
        self.assertFalse(result['info']['real_time'])
