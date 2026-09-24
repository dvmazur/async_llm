"""ChopTree policy only; native controls remain the registered Discrete(8)."""

# Index 0 is the wrapper's NOP; 3 is dig, not Speleo's turn-right.
ACTION_NAMES = ('wait', 'forward', 'jump', 'dig', 'right', 'left', 'up', 'down')

PROMPT = '''You control a first-person agent in Craftium ChopTree, a block-based forest.
You already hold a steel axe. Your goal is to chop as many wooden tree trunk blocks
as possible. Each chopped trunk block earns reward; leaves and ground are not the goal.
You do not need to craft a tool, collect dropped items, or fell an entire tree at once.

INPUT
You receive the previous image, then the current image, and the action taken between
them. At the start the images are identical and the action is none. There is no older
conversation or plan. Decide from this local evidence; do not invent unseen progress.

CONTROLS
One action holds a single control for eight game frames:
wait: release controls; forward: walk in the direction you face; jump: jump in place;
dig: hold the axe's dig button on the block aimed at in the center of the image;
right/left: turn the camera right/left about 10 degrees;
up/down: tilt the camera up/down about 10 degrees, without moving the player.
Use these turns to bring the trunk toward the center. You cannot turn,
walk and dig simultaneously. Consecutive dig choices continue holding the dig button.

APPROACH AND AIM
Look for solid wood/bark trunk blocks, often forming a vertical column under leaves.
There may be no crosshair: use the exact center of the image as the aiming point.
A thin selection outline, when visible, identifies the block currently targeted.
Approach a reachable trunk and center a wood block, not leaves, ground, the sky or
a gap beside the trunk. Being near a tree is not enough if you are aiming past it.
When a trunk is left/right of center, turn toward it; recheck the new image because
one turn can overshoot. Use up/down only when the target is above/below your aim.
Do not walk toward a distant tree if a reachable trunk is already correctly targeted.

CHOP AND CHECK
When nearby wood is centered, choose dig. Breaking a block can require consecutive
dig actions; an unchanged image after dig is not by itself evidence of failure.
Visible cracks indicate digging progress. Avoid interrupting a correctly aimed chop
with unnecessary turns or walking. If clearly out of reach, approach before digging.
After a wood block disappears, reassess what is now under the center: another trunk
block can be chopped, but empty space, leaves or ground call for retargeting.
Remaining upper trunk blocks can stay suspended; do not wait for the whole tree to fall.
Compare the frames in light of the last action: forward should approach the target,
turning should move it across the image, and dig may damage or remove the target.
If forward leaves the scene unchanged, check for a blocking trunk or other obstacle:
chop a correctly aimed trunk, otherwise adjust direction. Prefer purposeful actions
over idle waiting, jumping without an obstacle, or assuming that a missing tree was cut.

Return exactly one of: wait, forward, jump, dig, right, left, up, down.
Return only the action, without explanation.'''


FEEDBACK_PROMPT = '''You control a first-person agent in a block forest, holding a steel axe.
Goal: chop wood trunk blocks. You see two frames, the last action and a compact
record of recent actions and rewards. Positive reward means wood was actually removed;
zero reward means no wood was removed. Do not assume that swinging the axe is progress.

Choose ONE control held for eight game frames:
wait: release inputs; forward: walk forward; jump: jump in place;
dig: chop the block at the EXACT CENTER of the image;
right/left: turn camera right/left 10 degrees; up/down: aim up/down 10 degrees.
There is no crosshair. Aim is the center, not whichever tree looks biggest.

Use the feedback to avoid repeating failed actions:
- A successful dig removes the targeted block. The new hole is EMPTY. Retarget
  the remaining wood above or below it using up or down; upper blocks do not fall.
- A few consecutive digs may finish breaking a block. But 3 or more digs with no
  reward mean this attempt is not working: choose an aiming or approach action,
  NOT another dig. This applies even if a tree is large and nearby in the image.
- If many actions passed without reward, change your approach rather than idling
  or oscillating between the same two directions. Use the recent actions to tell.
- If nearby wood is right/left of center, turn right/left. If above/below, aim up/down.
- If wood is centered but too distant to reach, walk forward. Do not chop empty space,
  leaves or ground. Do not use wait as a substitute for choosing a new target.
- Dig when centered on reachable wood; continue briefly only while that block remains.
  You need neither to collect drops nor to craft tools. Each wood block counts.

Compare the frames, check the feedback, and choose the next useful action.
Return exactly one word: wait, forward, jump, dig, right, left, up, down.'''


WOOD_ONLY_PROMPT = '''You control a first-person agent in a snowy block forest, holding a steel axe.
Your ONLY objective is to remove WOODEN TREE TRUNK blocks. This is NOT a mining,
digging, tunneling or terrain-clearing task. A destroyed block is NOT automatically
progress: only positive wood reward confirms that a trunk block was chopped.
You receive two images, the last action and a compact record of recent actions/rewards.

IDENTIFY THE MATERIAL BEFORE DIGGING
A tree trunk is a column of wood/bark, usually with long vertical grain and leaves
above it. Gray mottled or layered blocks are stone. Brown soil capped with white
snow is ground, NOT a tree, even though both soil and bark can be brown.
Do not dig stone, dirt, snow or leaves, including when they block the way to a tree.
Do not excavate a route or deepen a hole. Find a route around the obstacle instead.

CONTROLS (one control for eight game frames)
wait: release inputs; forward: walk forward; jump: jump in place;
dig: use the axe on the block at the EXACT CENTER of the image;
right/left: turn the camera right/left 10 degrees;
up/down: tilt the camera up/down 10 degrees, without moving the player.
There is no crosshair. A trunk elsewhere in the image is not the center target.

CHOOSE A USEFUL NEXT ACTION
If the center shows ground or stone and trunks are higher in the image, choose UP
to aim toward the wood. Do not dig the wall at eye level. If looking up shows the
tree is on higher terrain, find an open approach or a low step; looking up alone
does not lift your body. Jump can help with a low obstacle, not a tall vertical wall.
If no trunk is visible, look up or turn to find one before deciding where to walk.
Approach visible trunks along open ground. When wood is to the left/right of the
center, aim left/right; when above/below the center, aim up/down.
Choose dig ONLY when the exact center points at nearby solid WOOD.

USE THE RECENT FEEDBACK
Positive reward means the aimed wood was removed: re-aim at remaining wood above
or below the hole. Upper trunk blocks remain suspended; do not wait for a tree to fall.
After 3 consecutive digs with zero wood reward, stop this attempt. Recheck MATERIAL,
aim and distance. Visible changes to stone or soil with zero reward are still failure.
Do not alternate the same failed turns forever; choose a different approach to a tree.
There is no need to collect drops or craft a tool. Do not attack random scenery.

Before answering, check: will this action target or approach WOOD, rather than dig ground?
Return exactly one word: wait, forward, jump, dig, right, left, up, down.'''


class ChopTreeFeedback:
    """Fixed-size memory of public action/reward observations, never hidden world state."""
    def __init__(self):
        from collections import deque
        self.recent = deque(maxlen=6)
        self.steps = self.since_reward = self.failed_digs = 0
        self.total_reward = 0.

    def observe(self, action, reward):
        self.steps += 1
        self.total_reward += float(reward)
        self.since_reward = 0 if reward > 0 else self.since_reward + 1
        self.failed_digs = self.failed_digs + 1 if action == 'dig' and reward <= 0 else 0
        self.recent.append(dict(action=action, reward=float(reward)))

    def text(self):
        import json
        return json.dumps(dict(step=self.steps, total_wood_reward=self.total_reward,
            actions_since_reward=self.since_reward,
            consecutive_digs_without_reward=self.failed_digs,
            recent=list(self.recent)), separators=(',', ':'))


class ChopTreeTextFeedback(ChopTreeFeedback):
    """Same public observations, explicit textual status and soft recovery advice."""
    SUCCESS_MESSAGE=('SUCCESS: the last dig removed a wood block. Its old position is now a hole. '
        'Re-aim at remaining wood ABOVE or BELOW the hole; do not keep digging through it.')
    def text(self):
        recent = ', '.join(f"{x['action']} (reward {x['reward']:g})" for x in self.recent) or 'none'
        lines = [f'Step {self.steps}. Wood reward so far: {self.total_reward:g}.',
                 f'Last actions, oldest first: {recent}.',
                 f'Actions since last wood reward: {self.since_reward}.']
        if self.recent and self.recent[-1]['reward'] > 0:
            lines.append(self.SUCCESS_MESSAGE)
        elif self.failed_digs >= 3:
            lines.append(f'STALLED: {self.failed_digs} consecutive digs removed NO wood. '
                         'Do not choose dig again at this aim. Turn toward wood, adjust up/down, '
                         'or approach if the wood is out of reach.')
        elif len(self.recent) >= 4 and sum(x['action'] in ('left','right') for x in self.recent) >= 4 \
                and not any(x['action']=='forward' for x in self.recent):
            lines.append('TURNING LOOP: several recent turns, no approach. '
                         'If a tree is ahead and the route is clear, walk forward rather than turning back again.')
        elif self.failed_digs:
            lines.append(f'{self.failed_digs} digs so far without wood reward; check that the CENTER targets wood.')
        return '\n'.join(lines)
