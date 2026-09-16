"""Real native IPC/ownership heartbeat check. Run as python -m tools.check_world."""
import argparse
import asyncio
import hashlib
import json
import time
import os
from pathlib import Path
from pipelines.world import SpeleoWorld


async def check(directory):
    world = SpeleoWorld(seed=0, max_steps=3, craftium_directory=directory)
    ticks = []
    async def heartbeat():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(.01)
    task = asyncio.create_task(heartbeat())
    try:
        initial = await world.reset()
        saved = hashlib.sha256(initial.image.tobytes()).hexdigest()
        assert initial.image.shape == (224, 224, 3)
        assert initial.image.std() > 3, 'blank rendering'
        children = Path(f'/proc/{world.process.pid}/task/{world.process.pid}/children').read_text().split()
        assert children, 'native children not found'
        assert all(os.getpgid(int(pid)) == os.getpgid(world.process.pid) for pid in children), 'native process escaped worker group'
        heights = [float(initial.info['player_pos'][1])]
        for action in (0, 1, 2):
            observation = await world.pass_action(action)
            heights.append(float(observation.info['player_pos'][1]))
        assert hashlib.sha256(initial.image.tobytes()).hexdigest() == saved, 'frame alias'
    finally:
        await world.aclose()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    gap = max(b-a for a, b in zip(ticks, ticks[1:]))
    assert gap < .5, f'event loop blocked for {gap:.3f}s'
    result = dict(heartbeat_max_gap=gap, ticks=len(ticks), initial_sha256=saved, heights=heights)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--craftium')
    args = p.parse_args()
    asyncio.run(check(args.craftium))
