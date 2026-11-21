"""Wire contract and nominal geometry for bounded multi-action plans."""

import math

from .direct_control import validate_response as validate_direct_response
from .direct_control import LIVE_ACTION_SPEC

PROTOCOL = "action_trajectory_v1"
MAX_STEPS = 32
LABELS = {"F": "forward", "L": "left", "R": "right", "S": "stop", "G": "creep_forward", "B": "backward", "C": "creep_backward"}
LABELS.update(Q='left_11_25', E='right_11_25', A='left_45', D='right_45')
ACTION_LABELS = {action: label for label, action in LABELS.items()}

# Feedback still stops at the same measured distance/angle. Give the physical
# rover time to reach them before judging the sequence against its rollout.
ACTION_PLAN_SPEC = {action: dict(spec) for action, spec in LIVE_ACTION_SPEC.items()}
ACTION_PLAN_SPEC['forward']['max_sim_s'] = 6.0
ACTION_PLAN_SPEC['forward']['max_distance_m'] = .10
ACTION_PLAN_SPEC['creep_forward'] = dict(ACTION_PLAN_SPEC['forward'], max_distance_m=.05)
ACTION_PLAN_SPEC['backward'] = dict(ACTION_PLAN_SPEC['forward'], linear_m_s=-.06)
ACTION_PLAN_SPEC['creep_backward'] = dict(ACTION_PLAN_SPEC['backward'], max_distance_m=.05)
ACTION_PLAN_SPEC['left']['angle_rad'] = math.pi / 8
ACTION_PLAN_SPEC['right']['angle_rad'] = math.pi / 8
ACTION_PLAN_SPEC['left']['max_sim_s'] = 4.0
ACTION_PLAN_SPEC['right']['max_sim_s'] = 4.0
for direction in ('left', 'right'):
    for suffix, angle in (('11_25', math.pi/16), ('45', math.pi/4)):
        ACTION_PLAN_SPEC[direction+'_'+suffix] = dict(ACTION_PLAN_SPEC[direction], angle_rad=angle)
        if suffix == '11_25':
            ACTION_PLAN_SPEC[direction+'_'+suffix]['angular_rad_s'] *= .5


def turn_delta(action, spec):
    return math.copysign(spec[action]['angle_rad'], spec[action]['angular_rad_s'])


def turn_actions_for_angle(angle, spec):
    """Shortest exact sequence of available same-direction turn primitives."""
    if abs(angle)<1e-8:
        return []
    options=sorted((name for name,item in spec.items()
                    if 'angle_rad' in item and item['angular_rad_s']*angle>0),
                   key=lambda name:spec[name]['angle_rad'], reverse=True)
    remaining=abs(angle);actions=[]
    for name in options:
        count=int((remaining+1e-8)/spec[name]['angle_rad'])
        actions.extend([name]*count)
        remaining-=count*spec[name]['angle_rad']
    if remaining>1e-6:
        raise ValueError('Heading is not representable by the supplied turn primitives')
    return actions


def trajectory_steps(value):
    count = int(value)
    if str(count) != str(value) or not 1 <= count <= MAX_STEPS:
        raise ValueError(f"Trajectory steps must be an integer from 1 to {MAX_STEPS}")
    return count


def validate_response(response, observation):
    limit = trajectory_steps(observation["trajectory_steps"])
    actions = response.get("actions")
    if (response.get("protocol") != PROTOCOL
            or type(response.get("trajectory_steps")) is not int
            or response["trajectory_steps"] != limit):
        raise ValueError("Mismatched trajectory protocol or horizon")
    if not isinstance(actions, list) or not 1 <= len(actions) <= limit:
        raise ValueError("Invalid trajectory length")
    if any(not isinstance(action, str) or action not in LABELS.values() for action in actions):
        raise ValueError("Unknown trajectory action")
    if "stop" in actions[:-1]:
        raise ValueError("Stop must end a trajectory")
    if type(response.get("task_complete")) is not bool:
        raise ValueError("Invalid trajectory completion flag")
    if response.get("action") != actions[0]:
        raise ValueError("First trajectory action disagrees with its action label")
    if response.get('goal_id') not in LABELS and response.get('goal_id') != 'Z':
        raise ValueError('Unknown trajectory label')
    if ({**LABELS, 'Z':'stop'}.get(response.get('goal_id')) != response['action']
            or response['task_complete'] != (response.get('goal_id') == 'Z')):
        raise ValueError('Inconsistent trajectory label')
    if response['action'] in LIVE_ACTION_SPEC:
        validate_direct_response({**response, "protocol": "direct_navigation_v1"}, observation)
    committed=response.get('selected_plan')
    if committed is not None and (committed.get('version')!=1 or committed.get('actions')!=actions
            or committed.get('snapshot_id')!=observation.get('snapshot_id')
            or committed.get('map_version')!=observation.get('map_version')):
        raise ValueError('Committed plan differs from trajectory or snapshot')
    if response["task_complete"] and actions != ["stop"]:
        raise ValueError("Completion requires a stop-only trajectory")


def rollout(pose, actions, spec):
    """Nominal robot-centre poses for display and deviation checks, not a planner."""
    x, y, heading = pose
    poses = [(x, y, heading)]
    for action in actions:
        if "max_distance_m" in spec[action]:
            distance = math.copysign(spec[action]["max_distance_m"], spec[action]["linear_m_s"])
            x += distance * math.cos(heading)
            y += distance * math.sin(heading)
        elif 'angle_rad' in spec[action]:
            heading += turn_delta(action, spec)
            heading = math.atan2(math.sin(heading), math.cos(heading))
        poses.append((x, y, heading))
    return poses


def pose_matches(actual, expected, angular_tolerance=0.20):
    if actual is None:
        return False
    angle = actual[2] - expected[2]
    return (
        math.dist(actual[:2], expected[:2]) <= 0.12
        and abs(math.atan2(math.sin(angle), math.cos(angle))) <= angular_tolerance
    )


def translation_finished_at_brake(start, end, target, spec):
    """Confirm the settled pose after a predictive stop on a bounded translation."""
    if start is None or end is None or target is None or 'max_distance_m' not in spec:
        return False
    if not all(math.isfinite(n) for pose in (start, end, target) for n in pose):
        return False
    dx, dy = target[0] - end[0], target[1] - end[1]
    c, s = math.cos(target[2]), math.sin(target[2])
    sign = math.copysign(1, spec['linear_m_s'])
    remaining = sign * (dx*c + dy*s)
    lateral = abs(dy*c - dx*s)
    travel = sign * ((end[0]-start[0])*c + (end[1]-start[1])*s)
    yaw_error = math.remainder(target[2] - end[2], math.tau)
    nominal = spec['max_distance_m']
    return (travel >= nominal * .6 and -min(.02, nominal/4) <= remaining <= .008
            and lateral <= .02 and abs(yaw_error) <= .20)
