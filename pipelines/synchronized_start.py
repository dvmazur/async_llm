"""Optional experimental start gate; running episodes are not synchronized."""


class SynchronizedStart:
    def __init__(self, world, barrier):
        self.world, self.barrier = world, barrier

    @property
    def metadata(self):
        return self.world.metadata

    async def reset(self):
        try:
            await self.world.reset()
            await self.barrier.wait()
            # Consume waiting-time dtime with a NOP, not the first policy action.
            observation = await self.world.pass_action(0)
            if observation.done:
                raise RuntimeError('world terminated during start synchronization')
            await self.barrier.wait()
            return observation
        except BaseException:
            await self.barrier.abort()
            raise

    async def pass_action(self, action):
        return await self.world.pass_action(action)

    async def aclose(self):
        await self.world.aclose()
