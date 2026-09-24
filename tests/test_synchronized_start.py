import asyncio
from types import SimpleNamespace

from pipelines.synchronized_start import SynchronizedStart


def test_all_native_worlds_ready_and_nopped_before_policy_start():
    async def run():
        barrier=asyncio.Barrier(5)
        ready=[];nops=[]
        class World:
            metadata={}
            def __init__(self,i):self.i=i
            async def reset(self):
                await asyncio.sleep(self.i*.001)
                ready.append(self.i)
            async def pass_action(self,action):
                assert len(ready)==5 and action==0
                nops.append(self.i)
                return SimpleNamespace(done=False)
        async def worker(i):
            await SynchronizedStart(World(i),barrier).reset()
            assert len(nops)==5
        await asyncio.gather(*(worker(i) for i in range(5)))
    asyncio.run(run())


def test_native_failure_aborts_gate_for_peers():
    async def run():
        barrier=asyncio.Barrier(2)
        class World:
            def __init__(self,fail):self.fail=fail
            async def reset(self):
                if self.fail:raise RuntimeError('native failure')
        values=await asyncio.wait_for(asyncio.gather(
            SynchronizedStart(World(False),barrier).reset(),
            SynchronizedStart(World(True),barrier).reset(),return_exceptions=True),1.)
        assert all(isinstance(x,Exception) for x in values)
    asyncio.run(run())
