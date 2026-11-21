"""Bounded, measured execution memory for the live action-sequence policy."""

import math
from .action_plan import ACTION_PLAN_SPEC, ACTION_LABELS, turn_actions_for_angle

PLANNING_MARGIN_M = 0.002


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def compact_pose(pose):
    if pose is None or len(pose) < 3 or not all(math.isfinite(v) for v in pose[:3]):
        return None
    return [round(float(v), 3) for v in pose[:3]]


def model_outcome(plan):
    if plan is None:
        return None
    # Full precision and per-step poses remain in events.jsonl. Avoid duplicating
    # them in the prompt: blocked headings are explicitly represented below.
    result = {k: v for k, v in plan.items() if k != 'step_results'}
    for name in ('start_pose', 'end_pose'):
        result[name] = compact_pose(plan.get(name))
    result['step_results'] = [
        {k: round(v, 3) if isinstance(v, float) else v
         for k, v in step.items() if k in {'action', 'outcome', 'distance_m', 'turn_rad','signed_distance_m','control_frame'}}
        for step in plan.get('step_results', [])]
    return result


def maneuver_memory(history, current_pose):
    """Report evidence, not a prescribed turn direction or a completion decision."""
    current = compact_pose(current_pose)
    recent = list(history)[-8:]
    local = [p for p in recent if compact_pose(p.get('end_pose')) is not None
             and current is not None and math.dist(p['end_pose'][:2], current[:2]) <= .30]
    turns, blocked, forward_m, translation_m = [], [], 0.0, 0.0
    for plan in local:
        for step in plan.get('step_results', []):
            if 'angle_rad' in ACTION_PLAN_SPEC.get(step['action'], {}):
                angle = step.get('turn_rad')
                noise_limit=min(.1,ACTION_PLAN_SPEC[step['action']]['angle_rad']/3)
                if angle is not None and math.isfinite(angle) and abs(angle) > noise_limit:
                    turns.append(float(angle))
            if step['action'] in {'forward','creep_forward','backward','creep_backward'}:
                translation_m += max(0.0, float(step.get('distance_m',0)))
            if step['action'] in {'forward','creep_forward'}:
                forward_m += max(0.0, float(step.get('distance_m', 0.0)))
                if step['outcome'] in {'trajectory_collision', 'lidar_brake'}:
                    pose = compact_pose(step.get('start_pose'))
                    if pose is not None:
                        blocked.append({'plan_id': plan['plan_id'], 'yaw_rad': pose[2],
                                        'relative_yaw_rad': round(wrap(pose[2]-current[2]), 3),
                                        'outcome': step['outcome'],
                                        'distance_m': round(float(step.get('distance_m', 0.0)), 3)})
    reversals = sum(a*b < 0 for a,b in zip(turns, turns[1:]))
    first = compact_pose(local[0].get('start_pose')) if local else None
    displacement = math.dist(first[:2], current[:2]) if first is not None else None
    summaries = []
    for plan in recent[-4:]:
        steps = plan.get('step_results', [])
        summaries.append({
            'plan_id': plan['plan_id'], 'planned': plan.get('planned_actions', []),
            'start_pose': compact_pose(plan.get('start_pose')),
            'end_pose': compact_pose(plan.get('end_pose')), 'outcome': plan['outcome'],
            'executed_count': len(steps),
            'measured_turn_rad': round(sum(s.get('turn_rad', 0) for s in steps), 3),
            'executed': [{k: step[k] for k in ('action', 'outcome', 'distance_m', 'turn_rad')
                          if k in step} for step in steps[-6:]],
            'discarded': plan.get('discarded_actions', []),
        })
    return {'coordinate_frame': 'map', 'current_pose': current,
            'recent_plans': summaries, 'local_blocked_forward_attempts': blocked[-6:],
            'local_measured_turn_rad': round(sum(turns), 3),
            'local_absolute_turn_rad': round(sum(map(abs, turns)), 3),
            'last_measured_turn_direction': ('left' if turns[-1] > 0 else 'right') if turns else None,
            'local_forward_progress_m': round(forward_m, 3), 'local_translation_progress_m':round(translation_m,3),
            'local_displacement_m': round(displacement, 3) if displacement is not None else None,
            'turn_reversals': reversals,
            'oscillation_detected': displacement is not None and displacement < .15
                                    and ((len(blocked) >= 2 and reversals > 0)
                                         or (len(turns) >= 4 and reversals > 0 and translation_m < .05)
                                         or (sum(map(abs, turns)) >= 2*math.pi and translation_m < .05))}


def heading_options(grid, pose, spec):
    """Measured-map affordances for turns at the current position, not a route choice.

    Every direction is treated symmetrically. Unknown cells are not free; the
    live swept-scan guard still independently checks every executed command.
    """
    from .grid_motion import GridMotion
    from .live_trajectory import HALF_LENGTH, HALF_WIDTH

    # Map discretization uses a small body padding. The live lidar braking
    # margin is independent and is still applied when a command executes.
    geometry = GridMotion(grid, pose, HALF_LENGTH, HALF_WIDTH, gap=PLANNING_MARGIN_M)
    result = []
    turn_step = min(item['angle_rad'] for item in spec.values() if 'angle_rad' in item)
    max_turns = round(math.pi / turn_step)
    for count in range(-max_turns, max_turns+1):
        delta = count * turn_step
        turns = turn_actions_for_angle(delta, spec)
        heading = pose[2] + delta
        contacts = geometry.turn(delta)
        sweep = contacts is not None
        distance = geometry.distance(heading, contacts, 1) if sweep else 0.0
        backward = geometry.distance(heading, contacts, -1) if sweep else 0.0
        result.append({'turn_prefix':''.join(ACTION_LABELS[name] for name in turns),
                       'turn_actions':turns,
                       'turn_degrees':round(math.degrees(delta),2),
                       'map_yaw_degrees':round(math.degrees(wrap(heading)),2),
                       'image_heading_degrees':round(math.degrees(wrap(heading-grid.yaw)),2),
                       'turn_sweep_known_free':sweep,
                       'initial_overlap_cells':len(geometry.initial),
                       'initial_overlap_kind':('deep' if geometry.deep_overlap else
                                               'edge' if geometry.initial else 'none'),
                       'known_forward_m':round(distance,3),
                       'full_forward_steps':int((distance+1e-6)/spec['forward']['max_distance_m']),
                       'short_forward_steps':int((distance+1e-6)/.05),
                       'known_backward_m':round(backward,3),
                       'full_backward_steps':int((backward+1e-6)/.10),
                       'short_backward_steps':int((backward+1e-6)/.05)})
    return result
