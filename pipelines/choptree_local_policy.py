"""Bounded single-probe memory, derived from issued controls/rewards only."""
from .choptree import ChopTreeTextFeedback


class ChopTreeProbeFeedback(ChopTreeTextFeedback):
    def __init__(self, turn_degrees=7):
        super().__init__()
        self.turn_degrees=turn_degrees
        self.pitch_estimate=0.
        self.last_reward_pitch=None
        self.digs_since_reward=0

    def observe(self, action, reward):
        super().observe(action,reward)
        if action in ('up','down'):
            delta=self.turn_degrees*(1 if action=='down' else -1)
            self.pitch_estimate=max(-90.,min(90.,self.pitch_estimate+delta))
        if reward>0:
            self.last_reward_pitch=self.pitch_estimate
            self.digs_since_reward=0
        elif action=='dig':
            self.digs_since_reward+=1

    def text(self):
        text=super().text()
        p=self.pitch_estimate
        direction='DOWN' if p>0 else 'UP' if p<0 else 'at the horizon'
        text+=f'\nCamera estimate from your own controls: {abs(p):g} degrees {direction}. '
        text+='This is an estimate assuming a level reset view, not a material/raycast sensor.'
        if p>=28:
            text+='\nYou have tilted well DOWN. If the center is snow/soil, use up to recover the trunk. '
            text+='Do NOT continue down or dig ground. A visible solid wooden stump at the center is an exception: dig it.'
        elif p<=-28:
            text+='\nYou have tilted well UP. If the center is sky/leaves, use down to recover reachable trunk. '
            text+='Do NOT continue up into canopy. A close solid wooden face at the center is an exception: dig it.'
        if self.last_reward_pitch is not None and self.digs_since_reward>=3:
            text+=f'\nLast successful cut was at estimated tilt {self.last_reward_pitch:g} degrees '
            text+='(negative=up, positive=down). That cube is gone. '
            text+=f'{self.digs_since_reward} subsequent digs gave no wood. '
            text+='If still aimed through that hole, shift aim to its upper or lower wood, not more swings through empty space.'
        return text


PROBE_PROMPT = '''You hold an axe in a snowy forest. Collect wood reward by chopping trunk cubes.
Use the two images and recent action/reward feedback to choose ONE next control.
Your aim is the EXACT CENTER of the CURRENT image, not the axe and not the largest tree.

VISUAL SERVO: decide which of these applies NOW, in this order.
1. Looking at sky/canopy with reachable trunks below: down. Looking at ground/snow
   with trunks above: up. Stop tilting once reachable wood comes to center.
2. Nearest reachable wood is right of center: right; left of center: left.
   If vertically displaced: wood above center -> up; below center -> down.
   Do not chop merely because a large tree occupies the SIDE of the picture.
3. Center is a HOLE between a stump below and floating wood above: aim up toward
   the bottom face of the upper log, or down toward the stump. The hole is EMPTY.
   Pale ringed cut surfaces are wood. Once that solid face reaches center, stop turning.
4. A solid wood block is centered and close enough: dig. A black outline on WOOD
   is useful evidence of the targeted block. Continue briefly while that block
   remains and cracks show progress. Reward means that cube is now GONE; re-aim.
5. Centered wood but several digs without cracks/reward: approach on clear ground,
   or choose another reachable wood block. More camera tilt does not extend reach.
6. If blocked, turn to find an open approach; do not repeatedly walk into an obstacle.

Never dig snow, soil, stone or leaves. Never keep digging into a removed cube's hole.
Finish reachable wood on the same tree when practical, but do not stare at distant
canopy to finish a plan. Interpret the newest image, not what used to be there.

Controls, held for eight game frames:
up/down: camera tilt up/down 10 degrees; left/right: camera turn left/right 10 degrees;
forward: walk forward; jump: jump in place; dig: chop at image center; wait: release.
Return one word only: wait, forward, jump, dig, right, left, up, down.
'''



def probe_prompt():
    return PROBE_PROMPT.replace('eight game frames', '4 game frames').replace('10 degrees', '7 degrees')
