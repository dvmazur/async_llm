"""Two frames, fresh prefill, one sampled action. No history or role generations."""
import asyncio
import hashlib
import json
import math
import time

from experiment_runner.blocks import BlockHandle
from experiment_runner.logs import atomic
from .prompts import SPELEO
from .settled_world import physics_info

PROMPT = '''You control a first-person agent in a cave. Your goal is to reach lower elevations.
You receive exactly two images: the previous observation, then the current observation.
At the start they are identical. Decide from these images alone.
Controls (one action lasts eight game frames):
wait: no input; forward: move forward; jump: jump in place;
right: turn right about 36 degrees; left: turn left about 36 degrees;
up: look up about 36 degrees; down: look down about 36 degrees.
The thin black rectangle outlines the targeted solid block. It is not evidence of a doorway.
Looking down changes the camera direction, not the player's elevation.
Use visible geometry: move into genuinely open space leading downhill.
If a wall blocks forward movement, turn to inspect another direction. Look down when
needed to locate a ledge or a descending route. Do not mistake dark rock for empty space.
Compare the two frames. An unchanged view is not evidence of progress through a passage.
When the view is unchanged and the way ahead is uncertain or blocked, favor an exploratory
turn or camera adjustment over assuming that forward movement is working.
Return exactly one of: wait, forward, jump, right, left, up, down.
Return only the action, without explanation.'''


def messages_for(previous, current):
    return [dict(role='system', content=PROMPT), dict(role='user', content=[
        dict(type='text', text='Previous observation:'), dict(type='image', image=previous),
        dict(type='text', text='Current observation:'), dict(type='image', image=current),
        dict(type='text', text='Choose the next action.')])]


class ProbePipeline:
    def __init__(self, world, recorder, engine, *, context, max_actions=500,
                 temperature=.7, action_delay=.2):
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError('max_actions must be a positive integer')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if not math.isfinite(action_delay) or action_delay < 0:
            raise ValueError('action_delay must be finite and nonnegative')
        self.world, self.recorder, self.engine, self.context = world, recorder, engine, context
        self.max_actions, self.temperature, self.action_delay = max_actions, temperature, action_delay
        self.stop_requested = lambda: False

    async def run(self):
        status, start = 'failed', None
        try:
            names = [name for name, _ in SPELEO.actions]
            ids = [self.engine.encode(name) for name in names]
            if not all(len(x)==1 for x in ids) or len({x[0] for x in ids}) != len(names):
                raise ValueError('actions must be distinct single tokens')
            ids = [x[0] for x in ids]
            seed = 1_000_003*(self.context.model_seed+1)+5
            rng = self.engine.new_generator(seed)
            self.recorder.log('policy',dict(variant='change',prompt=PROMPT,seed=seed,
                temperature=self.temperature,top_p=1.,top_k=len(names),action_names=names,
                action_ids=ids,history=False,feedback_input=False,action_delay=self.action_delay))
            await self.recorder.event('reset_started')
            initial = await self.world.reset()
            self.recorder.log('world_metadata',getattr(self.world,'metadata',{}))
            info = physics_info(initial.info)
            await self.recorder.observation(step=0,image=initial.image,info=info,
                height=info.get('player_pos',[None,None,None])[1])
            await self.recorder.event('reset_finished')
            previous = current = initial.image
            start = time.monotonic()
            await self.recorder.event('workload_started')
            for step in range(self.max_actions):
                if self.stop_requested():
                    break
                before = time.perf_counter()
                messages = messages_for(previous,current)
                self.recorder.log('model_image_input',dict(observation=step,
                    frame_steps=[max(0,step-1),step],conversation_tokens_before=0,
                    sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in (previous,current)]))
                async with await BlockHandle.create(self.engine,'probe/action') as block:
                    output, input_rows = await self.engine.prefill_action(messages,block)
                    index, probs = self.engine.sample_action(output,ids,generator=rng,
                                                           temperature=self.temperature)
                del output  # no reusable logits/state references held during the pause
                decision_seconds = time.perf_counter()-before
                self.recorder.log('decision',dict(observation=step,action=names[index],action_index=index,
                    action_probabilities=dict(zip(names,probs)),probabilities_temperature=self.temperature,
                    input_rows=input_rows,decision_seconds=decision_seconds,
                    entropy_nats=-sum(p*math.log(p) for p in probs if p>0),
                    nonargmax=index != max(range(len(probs)),key=probs.__getitem__)))
                before = time.perf_counter()
                if self.action_delay:
                    await asyncio.sleep(self.action_delay)
                pacing_seconds = time.perf_counter()-before
                self.recorder.log('action_delay',dict(observation=step,requested_seconds=self.action_delay,
                                                     actual_seconds=pacing_seconds))
                before = time.perf_counter()
                observation = await self.world.pass_action(index)
                world_seconds = time.perf_counter()-before
                info = physics_info(observation.info)
                height = info.get('player_pos',[None,None,None])[1]
                await self.recorder.step(step=step+1,image=observation.image,action=names[index],
                    action_index=index,reward=observation.reward,done=observation.done,info=info,height=height,
                    decision_seconds=decision_seconds,pacing_seconds=pacing_seconds,world_seconds=world_seconds)
                print('ACTION',json.dumps(dict(episode=self.context.episode_id,step=step+1,
                                              action=names[index],height=height)),flush=True)
                previous, current = current, observation.image
                if observation.done:
                    break
            status = 'stopped' if self.stop_requested() else 'completed'
        finally:
            end = time.monotonic()
            try:
                try:
                    await self.recorder.event('workload_finished',status=status)
                finally:
                    await self.world.aclose()
            finally:
                try:
                    await self.recorder.finish()
                finally:
                    atomic(self.context.results_directory/'completion.json',dict(status=status,
                        workload_start=start,workload_end=end))
