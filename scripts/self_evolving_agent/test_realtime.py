"""Run with the project Python; native tests require ViZDoom runtime access."""
import asyncio
import time
import unittest

from tasks.realtime_vizdoom import RealtimeVizdoom
from tasks.runner import run_episodes


class NativeRealtimeTests(unittest.TestCase):
    def test_game_advances_during_sleep_and_rewards_include_autonomous_tics(self):
        env = RealtimeVizdoom("VizdoomHealthGathering-v1", frame_skip=4, ticrate=35,
                             game_tic_limit=35, action_policy="hold_last")
        try:
            env.reset(seed=123)
            time.sleep(.35)
            _, first_reward, done, first = env.poll()
            self.assertFalse(done)
            self.assertGreaterEqual(first["game_tics"], 8)
            self.assertEqual(first_reward, first["game_tics"])
            time.sleep(1)
            _, rest_reward, done, end = env.poll()
            self.assertTrue(done)
            self.assertEqual(end["game_tics"], 35)
            self.assertEqual(first_reward + rest_reward, 35)
            self.assertEqual(env.poll()[1], 0)  # Polling must not double-count.
            env.reset(seed=123)
            self.assertLess(env.poll()[3]["game_tics"], 5)
        finally:
            env.close()

    def test_slow_action_is_cancelled_at_terminal_time(self):
        from types import SimpleNamespace
        adapter = RealtimeVizdoom("VizdoomDefendLine-v1", frame_skip=4, ticrate=35,
                                 game_tic_limit=35, action_policy="hold_last")
        env = SimpleNamespace(real_time=True, max_episodes=1, name="test",
                              reset=lambda: adapter.reset(seed=123)[0],
                              poll=adapter.poll, step=adapter.step, close=adapter.close)
        class SlowEngine:
            cancelled = False
            async def act(self, obs, on_token=None):
                try:
                    await asyncio.sleep(10)
                    return 1
                finally:
                    self.cancelled = True
        engine = SlowEngine()
        start = time.monotonic()
        result = asyncio.run(run_episodes(env, engine))
        self.assertLess(time.monotonic() - start, 4)
        self.assertTrue(engine.cancelled)
        ep = result["episodes"][0]
        self.assertEqual(ep["steps"], 0)
        self.assertTrue(ep["info"]["decision_cancelled"])
        self.assertNotIn("error", ep["info"])

    def test_all_public_game_wrappers_are_realtime(self):
        from tasks.doom_env import DoomEnv
        from tasks.health_gathering_env import HealthGatheringEnv
        from tasks.my_way_home_env import MyWayHomeEnv
        for cls in [DoomEnv, HealthGatheringEnv, MyWayHomeEnv]:
            env = cls(episode_timeout=14, seed=123)
            try:
                env.reset()
                time.sleep(.6)
                _, _, done, info = env.poll()
                self.assertTrue(env.real_time)
                self.assertTrue(done)
                self.assertEqual(info["game_tics"], 14)
                self.assertEqual(info["ticrate"], 35)
                self.assertEqual(info["action_policy"], "wait")
            finally:
                env.close()



class CancellationTests(unittest.TestCase):
    def test_inflight_gpu_future_is_drained_before_caller_cleanup(self):
        from minisgl.llm.async_llm import _await_inflight
        async def check():
            future = asyncio.get_running_loop().create_future()
            cleaned = []
            async def consumer():
                try:
                    await _await_inflight(future)
                finally:
                    cleaned.append(True)
            task = asyncio.create_task(consumer())
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(future.cancelled())
            self.assertFalse(cleaned)
            task.cancel()  # Repeated cancellation must not free a live block.
            await asyncio.sleep(0)
            self.assertFalse(cleaned)
            future.set_result(None)
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(cleaned)
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
