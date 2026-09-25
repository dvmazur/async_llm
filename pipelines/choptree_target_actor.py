"""Explicit visual target -> model-selected motor action; no scripted aiming."""
import json
import math
from collections import deque

from experiment_runner.text_role import TextRole
from experiment_runner.probe_readout import ProbeReadout
from .choptree import ACTION_NAMES


class ControlMemory:
    def __init__(self):
        self.recent = deque(maxlen=12)
        self.step = self.since_reward = 0
        self.total_reward = 0.
        self.last_plan = None
        self._views = deque(maxlen=12)
        self.visual_outcome = None

    def inspect_change(self, previous, current):
        import numpy as np
        from PIL import Image
        def thumbnail(frame):
            return np.asarray(Image.fromarray(frame).resize((64,64))).astype('float32')
        old, new = thumbnail(previous), thumbnail(current)
        mask = np.ones((64,64),dtype=bool)
        mask[32:,40:] = False  # omit most held-axe animation, no semantic sensor
        difference = lambda a,b: float(abs(a-b)[mask].mean()/255.)
        delta = difference(old,new)
        # Position in the finite observation queue is actions ago, not elapsed time.
        matches = [(i+1,difference(view,new)) for i,view in enumerate(reversed(self._views))]
        revisits = [age for age,diff in matches if age>=2 and diff<.01]
        self.visual_outcome = dict(last_action=self.recent[-1]['action'] if self.recent else None,
            normalized_rgb_change=round(delta,4), almost_unchanged=delta<.01,
            similar_view_actions_ago=revisits,
            caveat='Pixel difference outside axe region, not a position/material sensor.')
        self._views.append(new)

    def observe(self, action, reward):
        self.step += 1
        self.total_reward += float(reward)
        self.since_reward = 0 if reward > 0 else self.since_reward+1
        self.recent.append(dict(action=action, reward=float(reward)))

    def text(self):
        return json.dumps(dict(step=self.step, total_wood=self.total_reward,
            since_wood=self.since_reward, recent=list(self.recent), previous_plan=self.last_plan,
            visual_outcome=self.visual_outcome),
            separators=(',', ':'))


TARGET_PROMPT = '''You are the target planner in ChopTree, harvesting wooden trunk cubes.
Choose the next useful target in the CURRENT FULL image. There is no center crop.
Do not merely describe whatever happens to lie at image center. The big NEARBY trunk
at the side is usually preferable to a skinny DISTANT trunk seen through a central
gap. A huge close tree on the right can be reached by turning RIGHT, not by swinging
at a far tree in the center. Prioritize reachable solid wood over distant canopy.

Pick a point in the middle of a specific exposed WOOD FACE, not the silhouette,
leaves, dirt, snow or an empty hole. Brown vertical bark AND pale concentric-ring cut
faces are wood. Flat ringed squares flush with snow are remaining log TOPS: harvest
them too. They are not loose planks or dropped items. Upper cubes remain suspended.
After a positive reward, the old cube is gone: relocalize a remaining stump or upper
face in the NEW image. Coordinates from your old plan are stale after any movement.

Output x,y normalized to 0..1000 across the FULL picture: left/top=0, right/bottom=1000.
The actual aiming point is (500,500). The chosen point must be inside the target
face, not at the player's aim unless that aim already intersects your chosen target.
If no accessible wood is visible, choose a visible route to inspect ('route') or
new direction to search ('search'), instead of inventing a wood face. Dirt/snow banks
are BODY barriers: camera tilt cannot climb them. A low step needs jump then forward;
a tall wall needs a side route. A point on dirt is a route proposal, NEVER wood.
Public feedback includes previous controls/rewards, a last plan, and measured image
change. Nearly unchanged after forward indicates the previous approach may be blocked.
Report proximity of the chosen target and the actual BODY obstacle, separately.
If many actions passed without wood, actively reconsider the failed target/route.
Repeating the same description is not new evidence. Prefer a different nearby solid
wood face or a new open direction. Do not confuse a skinny distant face with a close
large trunk that occupies a substantial part of the full picture.
Do not choose the same failed distant wood indefinitely. Describe the new target and
one short next goal. Be concise. The actor will choose the next control, not you.
Start with the target type/name, x and y, proximity, and body obstacle. Then give
at most two short sentences explaining the target and the expected next change.
Use ordinary text, not JSON. Put the actionable visual facts before explanations.'''

MOTOR_PROMPT = '''Aim is x=500,y=500. First classify target alignment:
x<425: left. x>575: right. Otherwise horizontal=aligned.
y<425: above. y>575: below. Otherwise vertical=aligned.
For wood choose the FIRST needed operation: horizontal left/right -> left/right;
otherwise vertical above/below -> up/down; only when both aligned and close -> dig.
Aligned but distant with no obstacle -> forward. A low_step requires jump then
forward (check last_action). A wall/water requires horizontal search, not digging.
For kind=search choose a horizontal turn, not up/down or dig.
Allowed: wait,forward,jump,dig,right,left,up,down. Choose ONE immediate action.
Examples: (800,300) -> horizontal right, vertical above, action right.
(500,250) -> aligned,above,up. (500,750) -> aligned,below,down.
(500,500), wood close -> aligned,aligned,dig.
Briefly assess alignment, the first necessary operation and its expected visible
effect. Use ordinary text, not JSON. No simultaneous actions. Target coordinates
and the current visual facts take priority over the planner's proposed future goal;
'dig after turning' does not mean dig now. The plan may be incomplete. Do not invent
missing visual facts or copy an action merely because the planner mentioned it.'''


def marked_view(frame, point=None):
    import numpy as np
    from PIL import Image,ImageDraw
    image=Image.fromarray(frame).resize((448,448),Image.Resampling.NEAREST)
    draw=ImageDraw.Draw(image)
    def cross(x,y,color):
        draw.line((x-12,y,x-5,y),fill=color,width=3)
        draw.line((x+5,y,x+12,y),fill=color,width=3)
        draw.line((x,y-12,x,y-5),fill=color,width=3)
        draw.line((x,y+5,x,y+12),fill=color,width=3)
    cross(224,224,(255,35,35))
    if point is not None:
        if not all(type(point[k]) is int and 0<=point[k]<=1000 for k in ('x','y')):
            raise ValueError('invalid model target coordinates')
        x,y=(round(point[k]*447/1000) for k in ('x','y'))
        cross(x,y,(0,220,255))
    return np.asarray(image).copy()


class TargetActorReadout(ProbeReadout):
    decision_mode = 'generated_with_readout'

    def __init__(self,backend,recorder,memory,*,planner_temperature=.5,
                 recovery_temperature=.9,recovery_after=12,max_role_tokens=768):
        super().__init__(backend)
        if any(not math.isfinite(t) or t<=0 for t in (planner_temperature,recovery_temperature)):
            raise ValueError('planner temperatures must be finite and positive')
        if any(type(n) is not int or n<1 for n in (recovery_after,max_role_tokens)):
            raise ValueError('recovery threshold and token budget must be positive integers')
        self.planner_temperature=planner_temperature
        self.recovery_temperature=recovery_temperature
        self.recovery_after=recovery_after
        self.max_role_tokens=max_role_tokens
        self.recorder,self.memory=recorder,memory
        self.planner=TextRole(backend,recorder,'target_planner')
        self.actor=TextRole(backend,recorder,'motor_actor')
        self.action_ids=None
        self.policy_metadata=dict(mode='planner-actor-text-readout-v9',on_device=True,
            action_overrides=False, planner_temperature=planner_temperature,
            recovery_temperature=recovery_temperature,recovery_after=recovery_after,
            max_role_tokens=max_role_tokens,motor_temperature='pipeline temperature',
            vision='planner: full frame448; motor: text plan only',
            role_format='plain text', length_limit='finish role, continue to action readout',
            action_selection='sample one legal action token')

    async def generate_action(self,messages,target,*,generator,temperature):
        frames=[p['image'] for p in messages[1]['content'] if p['type']=='image']
        previous,current=frames[0],frames[-1]
        self.memory.inspect_change(previous,current)
        feedback=self.memory.text()
        plan,rows1=await self.planner([
            dict(role='system',content=TARGET_PROMPT),dict(role='user',content=[
                dict(type='text',text='CURRENT full scene. Red=actual aim, choose target anywhere in scene.'),
                dict(type='image',image=marked_view(current)),
                dict(type='text',text='Public outcomes and previous model plan: '+feedback)])],
            generator=generator,max_tokens=self.max_role_tokens,
            temperature=self.recovery_temperature if self.memory.since_reward>=self.recovery_after
            else self.planner_temperature)
        actor_messages=[dict(role='system',content=MOTOR_PROMPT),
            dict(role='user',content='Current target plan (may be incomplete):\n'+
                (plan or '[empty plan]')+'\nLast executed action: '+
                (self.memory.recent[-1]['action'] if self.memory.recent else 'none'))]
        assessment,rows2=await self.actor(actor_messages,
            generator=generator,temperature=temperature,max_tokens=self.max_role_tokens)
        # A separate model readout chooses the button. No parsing of either role's text.
        readout_messages=actor_messages+[
            dict(role='assistant',content=assessment or '[No assessment produced.]'),
            dict(role='user',content='Choose the ONE immediate action now, using the target plan '
                'and assessment above. An incomplete assessment is not an instruction to wait. '
                'Return only one of: '+', '.join(ACTION_NAMES)+'.')]
        output,rows3=await self.prefill_action(readout_messages,target)
        if self.action_ids is None:
            ids=[self.encode(name) for name in ACTION_NAMES]
            if not all(len(x)==1 for x in ids) or len({x[0] for x in ids})!=len(ids):
                raise ValueError('actions must be distinct single tokens')
            self.action_ids=[x[0] for x in ids]
        index,probs=self.sample_action(output,self.action_ids,
            generator=generator,temperature=temperature)
        action=ACTION_NAMES[index]
        self.memory.last_plan=dict(plan=plan,assessment=assessment,action=action)
        self.recorder.log('target_decision',dict(step=self.memory.step,plan=plan,
            assessment=assessment,decision=dict(action=action),
            action_probabilities=dict(zip(ACTION_NAMES,probs))))
        return index,rows1+rows2+rows3
