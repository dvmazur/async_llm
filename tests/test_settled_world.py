import asyncio
from types import SimpleNamespace
import numpy as np
import pytest
from pipelines.settled_world import SettledWorld, physics_info


class FakeWorld:
    metadata = {'test': True}
    def __init__(self, ready_at=0):
        self.n = 0
        self.ready_at = ready_at
    def obs(self):
        return SimpleNamespace(done=False, image=np.arange(12).reshape(2,2,3),
            info=dict(player_pos=[24.3,5.5,-36.3] if self.n >= self.ready_at else [0,0,0],
                      player_vel=[0,0,0]))
    async def reset(self):
        return self.obs()
    async def pass_action(self, action):
        assert action == 0
        self.n += 1
        return self.obs()


@pytest.mark.parametrize('ready_at,expected', [(0,10), (12,16)])
def test_wait_for_spawn_and_stability(ready_at, expected):
    w = FakeWorld(ready_at)
    logs = []
    ready = SettledWorld(w, SimpleNamespace(log=lambda k,v:logs.append((k,v))), interval=0)
    asyncio.run(ready.reset())
    assert w.n == expected
    assert logs[-1][0] == 'reset_ready'


def test_missing_spawn_fails_instead_of_running():
    ready = SettledWorld(FakeWorld(10000), SimpleNamespace(log=lambda *a:None), interval=0)
    with pytest.raises(RuntimeError, match='spawn'):
        asyncio.run(ready.reset())


def test_physics_fields_are_json_compatible():
    assert physics_info({'player_vel':np.zeros(3), 'mt_dtime':np.float32(.5)}) == {
        'player_vel':[0.,0.,0.], 'mt_dtime':.5}
