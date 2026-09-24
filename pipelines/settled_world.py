"""Observed reset stability; does not modify the simulation clock."""
import asyncio
import time


def physics_info(info):
    info = info or {}
    result = {}
    for key in ('player_pos', 'player_vel'):
        if key in info:
            result[key] = [float(x) for x in info[key]]
    for key in ('player_pitch', 'player_yaw', 'mt_dtime'):
        if key in info:
            result[key] = float(info[key])
    for key in ('terminated', 'truncated'):
        if key in info:
            result[key] = bool(info[key])
    return result


class SettledWorld:
    # Caller reserves these extra world steps so settling cannot shorten rollout.
    MAX_SETTLING_STEPS = 120

    def __init__(self, world, recorder, *, timeout=120., interval=.1, minimum=10, stable=5,
                 expected_spawn=(24.3, 5.5, -36.3)):
        self.world, self.recorder = world, recorder
        self.timeout, self.interval, self.minimum, self.stable = timeout, interval, minimum, stable
        self.expected_spawn = expected_spawn

    @property
    def metadata(self):
        return self.world.metadata

    async def reset(self):
        import numpy as np  # only the selected engine venv needs runtime dependencies
        started = time.monotonic()
        consecutive = 0
        async with asyncio.timeout(self.timeout):
            obs = await self.world.reset()
            self.recorder.log('reset_raw', physics_info(obs.info))
            previous_pos = None
            for count in range(1, self.MAX_SETTLING_STEPS + 1):
                await asyncio.sleep(self.interval)
                obs = await self.world.pass_action(0)
                info = physics_info(obs.info)
                pos, vel = info.get('player_pos', []), info.get('player_vel', [])
                # Procedural ChopTree has no fixed Speleo spawn coordinates.
                target = self.expected_spawn if self.expected_spawn is not None else previous_pos
                ready = (len(pos) == len(vel) == 3
                    and target is not None and np.all(np.isfinite(pos))
                    and np.allclose(pos, target, rtol=0, atol=1e-3)
                    and np.all(np.isfinite(vel)) and np.linalg.norm(vel) < 1e-3
                    and obs.image is not None and np.std(obs.image) > 1)
                previous_pos = pos if len(pos) == 3 else None
                if obs.done:
                    raise RuntimeError('World terminated during reset settling')
                consecutive = consecutive + 1 if ready else 0
                self.recorder.log('reset_settling', dict(nop=count, ready=bool(ready), **info))
                if count >= self.minimum and consecutive >= self.stable:
                    self.recorder.log('reset_ready', dict(nops=count, seconds=time.monotonic()-started,
                        stable_observations=consecutive, **info))
                    return obs
        raise RuntimeError('World spawn did not become stable; refusing to start episode')

    async def pass_action(self, action):
        return await self.world.pass_action(action)

    async def aclose(self):
        await self.world.aclose()
