"""Bounded single-probe memory, derived from issued controls/rewards only."""
from .choptree import ChopTreeTextFeedback


class ChopTreeAimFeedback(ChopTreeTextFeedback):
    VERSION = 'control-memory-v2-upper-recovery'
    def __init__(self, turn_degrees=7, recovery_advice=True):
        super().__init__()
        self.turn_degrees=turn_degrees
        self.pitch_estimate=0.
        self.last_reward_pitch=None
        self.digs_since_reward=0
        self.upper_failure=None
        self.recovery_advice=recovery_advice

    def observe(self, action, reward):
        super().observe(action,reward)
        if action in ('up','down'):
            delta=self.turn_degrees*(1 if action=='down' else -1)
            self.pitch_estimate=max(-90.,min(90.,self.pitch_estimate+delta))
        if reward>0:
            self.last_reward_pitch=self.pitch_estimate
            self.digs_since_reward=0
            self.upper_failure=None
        elif action=='dig':
            self.digs_since_reward+=1
            if self.pitch_estimate<0 and self.failed_digs>=3 and self.last_reward_pitch is not None:
                self.upper_failure=dict(step=self.steps,pitch=self.pitch_estimate)

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
        if self.upper_failure is not None and not self.recovery_advice:
            text+=f'\nPast failed attempt: repeated upward digs at step {self.upper_failure["step"]} '
            text+='gave no wood. This is a past observation, NOT an instruction to keep aiming down. '
            text+='Choose the nearest useful CURRENT target, including a different trunk at the side.'
        if self.upper_failure is not None and self.recovery_advice:
            text+=f'\nUNPRODUCTIVE UPPER TARGET: at step {self.upper_failure["step"]} repeated digs '
            text+='while looking upward gave NO wood after earlier successful cuts. '
            text+='Try the remaining LOWER part of this trunk now: aim down toward the stump until '
            text+='a solid wooden face is centered, then dig. Do not aim down into soil; stop at WOOD. '
            text+='If no reachable lower wood remains, acquire another nearby tree. '
            text+='This is a remembered failed attempt, not a claim that every upper block is unreachable.'
        return text


class ChopTreeLocalAimFeedback(ChopTreeAimFeedback):
    """Remember failures for estimated aim directions, not every future upper target.

    Directions are relative to reset, inferred from commands only. Translation,
    jumping or a successful cut invalidates the small cache. It is advisory, not
    a geometric raycast or an action override.
    """
    VERSION='control-memory-v3-local-aim'

    def __init__(self,turn_degrees=7):
        from collections import OrderedDict
        super().__init__(turn_degrees)
        self.yaw_estimate=0.
        self.failed_at_aim=OrderedDict()

    def observe(self,action,reward):
        super().observe(action,reward)
        if action in ('left','right'):
            self.yaw_estimate=(self.yaw_estimate+self.turn_degrees*(1 if action=='left' else -1))%360
        key=(round(self.yaw_estimate,3),round(self.pitch_estimate,3))
        if reward>0 or action in ('forward','jump'):
            self.failed_at_aim.clear()
        elif action=='dig':
            self.failed_at_aim[key]=self.failed_at_aim.get(key,0)+1
            self.failed_at_aim.move_to_end(key)
            while len(self.failed_at_aim)>12:self.failed_at_aim.popitem(last=False)
        failures=self.failed_at_aim.get(key,0)
        self.upper_failure=(dict(step=self.steps,pitch=self.pitch_estimate)
            if self.pitch_estimate<0 and failures>=3 and self.last_reward_pitch is not None else None)

    def text(self):
        key=(round(self.yaw_estimate,3),round(self.pitch_estimate,3))
        count=self.failed_at_aim.get(key,0)
        return (super().text()+f'\nAt the CURRENT estimated aim direction: {count} digs without wood '
            'since the last move/jump/successful cut. Failures at other directions do not mean '
            'this target is bad. A newly aimed solid wood block should still be cut.')


class ChopTreeEvidenceFeedback(ChopTreeLocalAimFeedback):
    """Public outcomes take priority over the model's earlier material guesses.

    This is textual advice, NOT an action mask or scripted control override.
    Counterfactuals, raycasts and future rewards are never available here.
    """
    VERSION='control-memory-v5-evidence-reassess'
    SUCCESS_MESSAGE=('SUCCESS: one wood cube was removed. Reassess the CURRENT center: '
        'it may now show a gap OR a different reachable wood cube behind it. '
        'If new reachable wood is centered, dig is still useful; otherwise re-aim at remaining wood.')

    def text(self):
        key=(round(self.yaw_estimate,3),round(self.pitch_estimate,3))
        failures=self.failed_at_aim.get(key,0)
        text=ChopTreeTextFeedback.text(self)
        p=self.pitch_estimate
        text+=f'\nCamera tilt estimated from issued controls: {p:g} degrees (negative=up).'
        text+=' This estimate is not a material or distance sensor.'
        if self.recent and self.recent[-1]['reward']>0:
            text+='\nDECISION PRIORITY: the old cube was just removed. Do not assume another close cube is at this aim. Reacquire remaining solid wood.'
        if failures>=3:
            text+=f'\nDECISION PRIORITY: {failures} digs at THIS aim after the last move/cut returned ZERO wood.'
            text+=' The claim "this is reachable wood" has failed its test.'
            text+=' Choose one of left, right, up, down, forward, jump for this decision; do not repeat dig or wait.'
            text+=' Choose the correction from the full current scene. Remaining stump is below; upper cut face is above; another trunk can be at the side.'
        if p<=-28:
            text+='\nYou have turned far upward. If center is canopy or sky, choose down, not further up or dig.'
        if p>=28:
            text+='\nYou have turned far downward. If center is ground rather than a wooden stump, choose up, not further down or dig.'
        return text


ASSESSMENT_PROMPT = "You play ChopTree, a first-person block forest game, holding an axe.\nScore comes from removing wooden trunk cubes. No crafting or item collection.\nLook at the CURRENT frame: what exactly is under the center point, and where is\nthe nearest remaining wooden face? Do not confuse the axe at the side with aim.\nUse the previous frame and actual action/reward feedback to check your guess.\nIf a dig earned reward, that cube is gone. The gap is empty; distant scenery seen\nthrough it is not the old cube. The upper wood floats; the stump remains below.\nLight ringed cut faces are wood, like bark. Snow, dirt, leaves and stone are not.\nThe extra close-up, if present, is magnified current-center detail, NOT evidence\nthat background wood is within reach. Use the full frame for distance/location.\n\nChoose one button for 4 game frames: dig = axe at center; forward = walk;\nleft/right/up/down = turn camera 7 degrees in that direction; jump = jump in place;\nwait = release. Target left/right/above/below center needs left/right/up/down.\nWhen center is sky or canopy, aim down toward reachable trunk. When center is\nground, aim up toward wood. If close wood is centered, dig. If wood is distant,\napproach then re-aim. Repeated zero rewards contradict a claim of reachable wood.\nDo not endlessly repeat the same failed approach or keep aiming higher after\nupper wood becomes unreachable: try the stump or another nearby tree.\n\nGive a short factual assessment (at most 60 words) and choose ONE next action.\nReturn JSON with only two fields, in order:\n{\"observation\":\"Your brief assessment\", \"action\":\"one button\"}.\nAllowed buttons: wait, forward, jump, dig, right, left, up, down.\nDo not propose simultaneous controls or a list of future actions.\n\nHARVEST WHAT IS ALREADY NEAR YOU\nAfter cutting the middle of a trunk, the big ringed stump face below the center\nis still valuable WOOD. If it fills a large part of the lower foreground, you are\nALREADY close enough: tilt DOWN to put the face at the aim; do not walk forward\ninto it while chasing a thin background tree. Camera aiming is not body movement.\nTry nearby remaining stump/upper wood before wandering to another tree. A wood\ncube directly behind the one just cut can also be chopped if still within reach.\nIf no close solid wood remains, select another nearby trunk and approach it."
