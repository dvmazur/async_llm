"""Fresh visual decision: one-token probe or one assessment/action completion."""
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


def messages_for(previous, current, prompt=PROMPT, *, last_action=None, feedback=None):
    content = [
        dict(type='text', text='Previous observation:'), dict(type='image', image=previous),
        dict(type='text', text='Current observation:'), dict(type='image', image=current)]
    if last_action is not None:
        content.append(dict(type='text', text=f'Action between these observations: {last_action}.'))
    if feedback is not None:
        content.append(dict(type='text', text='Recent interaction feedback: ' + feedback))
    content.append(dict(type='text', text='Choose the next action.'))
    return [dict(role='system', content=prompt), dict(role='user', content=content)]


class ProbePipeline:
    def __init__(self, world, recorder, engine, *, context, max_actions=500,
                 temperature=.7, action_delay=.2, prompt=PROMPT,
                 action_names=None, variant='change', include_last_action=False, feedback=None,
                 vision='plain', demonstrations=None):
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError('max_actions must be a positive integer')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if not math.isfinite(action_delay) or action_delay < 0:
            raise ValueError('action_delay must be finite and nonnegative')
        self.world, self.recorder, self.engine, self.context = world, recorder, engine, context
        self.max_actions, self.temperature, self.action_delay = max_actions, temperature, action_delay
        self.prompt, self.variant = prompt, variant
        self.include_last_action = include_last_action
        self.feedback = feedback
        if vision not in ('plain', 'aim', 'crop', 'current-large'):
            raise ValueError('unknown vision presentation')
        self.vision = vision
        self.demonstrations = list(demonstrations or [])
        self.action_names = tuple(action_names) if action_names is not None else tuple(n for n, _ in SPELEO.actions)
        if not self.action_names or len(set(self.action_names)) != len(self.action_names):
            raise ValueError('action_names must be nonempty and distinct, in world action-index order')
        self.stop_requested = lambda: False

    async def run(self):
        status, start = 'failed', None
        try:
            names = list(self.action_names)
            ids = [self.engine.encode(name) for name in names]
            if not all(len(x)==1 for x in ids) or len({x[0] for x in ids}) != len(names):
                raise ValueError('actions must be distinct single tokens')
            ids = [x[0] for x in ids]
            seed = 1_000_003*(self.context.model_seed+1)+5
            rng = self.engine.new_generator(seed)
            self.recorder.log('policy',dict(variant=self.variant,prompt=self.prompt,seed=seed,
                temperature=self.temperature,top_p=1.,
                top_k=None if hasattr(self.engine, 'generate_action') else len(names),action_names=names,
                action_ids=ids,history=getattr(self.engine,'has_history',False),
                feedback_input=self.feedback is not None,action_delay=self.action_delay,
                last_action_input=self.include_last_action, vision=self.vision,
                demonstration_messages=len(self.demonstrations)))
            if hasattr(self.engine, 'policy_metadata'):
                self.recorder.log('decision_backend', self.engine.policy_metadata)
            await self.recorder.event('reset_started')
            initial = await self.world.reset()
            self.recorder.log('world_metadata',getattr(self.world,'metadata',{}))
            info = physics_info(initial.info)
            await self.recorder.observation(step=0,image=initial.image,info=info,
                height=info.get('player_pos',[None,None,None])[1])
            await self.recorder.event('reset_finished')
            previous = current = initial.image
            last_action = 'none (episode start)'
            start = time.monotonic()
            await self.recorder.event('workload_started')
            for step in range(self.max_actions):
                if self.stop_requested():
                    break
                before = time.perf_counter()
                feedback = self.feedback.text() if self.feedback is not None else None
                messages = messages_for(previous,current,self.prompt,
                    last_action=last_action if self.include_last_action else None, feedback=feedback)
                if self.vision != 'plain':
                    from .probe_vision import aim_messages
                    messages = aim_messages(messages, current, self.vision)
                if self.demonstrations:
                    from .probe_demonstrations import add_demonstrations
                    messages = add_demonstrations(messages, self.demonstrations)
                model_frames=[part['image'] for msg in messages if isinstance(msg['content'],list)
                              for part in msg['content'] if part['type']=='image']
                self.recorder.log('model_image_input',dict(observation=step,
                    frame_steps=[step] if self.vision=='current-large' else [max(0,step-1),step],
                    conversation_tokens_before=getattr(self.engine,'history_tokens',0),
                    last_action=last_action if self.include_last_action else None,
                    feedback=feedback,vision=self.vision,
                    model_image_shapes=[list(x.shape) for x in model_frames],
                    model_image_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in model_frames],
                    sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in (previous,current)]))
                async with await BlockHandle.create(self.engine,'probe/action') as block:
                    if hasattr(self.engine, 'generate_action'):
                        index, input_rows = await self.engine.generate_action(
                            messages, block, generator=rng, temperature=self.temperature)
                        probs = None  # a sampled completion is not a first-token readout
                    else:
                        output, input_rows = await self.engine.prefill_action(messages,block)
                        index, probs = self.engine.sample_action(output,ids,generator=rng,
                                                               temperature=self.temperature)
                        del output  # no logits/scratch references held during the pause
                decision_seconds = time.perf_counter()-before
                self.recorder.log('decision',dict(observation=step,action=names[index],action_index=index,
                    mode=getattr(self.engine, 'decision_mode',
                        'generated' if hasattr(self.engine, 'generate_action') else 'readout'),
                    action_probabilities=dict(zip(names,probs)) if probs is not None else None,
                    probabilities_temperature=self.temperature,
                    input_rows=input_rows,decision_seconds=decision_seconds,
                    entropy_nats=-sum(p*math.log(p) for p in probs if p>0) if probs is not None else None,
                    nonargmax=index != max(range(len(probs)),key=probs.__getitem__) if probs is not None else None))
                before = time.perf_counter()
                if self.action_delay:
                    await asyncio.sleep(self.action_delay)
                pacing_seconds = time.perf_counter()-before
                self.recorder.log('action_delay',dict(observation=step,requested_seconds=self.action_delay,
                                                     actual_seconds=pacing_seconds))
                before = time.perf_counter()
                observation = await self.world.pass_action(index)
                world_seconds = time.perf_counter()-before
                if self.feedback is not None:
                    self.feedback.observe(names[index], observation.reward)
                info = physics_info(observation.info)
                height = info.get('player_pos',[None,None,None])[1]
                await self.recorder.step(step=step+1,image=observation.image,action=names[index],
                    action_index=index,reward=observation.reward,done=observation.done,info=info,height=height,
                    decision_seconds=decision_seconds,pacing_seconds=pacing_seconds,world_seconds=world_seconds)
                print('ACTION',json.dumps(dict(episode=self.context.episode_id,step=step+1,
                                              action=names[index],height=height)),flush=True)
                previous, current = current, observation.image
                last_action = names[index]
                if observation.done:
                    break
            status = 'stopped' if self.stop_requested() else 'completed'
        finally:
            end = time.monotonic()
            try:
                try:
                    await self.recorder.event('workload_finished',status=status)
                finally:
                    try:
                        close_episode=getattr(self.engine,'close_episode',None)
                        if close_episode is not None:
                            await close_episode()
                    finally:
                        await self.world.aclose()
            except BaseException:
                status = 'failed'
                raise
            finally:
                try:
                    await self.recorder.finish()
                finally:
                    atomic(self.context.results_directory/'completion.json',dict(status=status,
                        workload_start=start,workload_end=end))
