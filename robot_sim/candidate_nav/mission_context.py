"""Grounded route candidates and events for strategy 4, sub-strategy 2."""
import math

from .action_plan import ACTION_LABELS, LABELS, rollout
from .committed_plan import first_action_labels
from .trajectory_policy import translation_options

MEMORY = "mission_memory_v1"
SUBGOAL = "mission_subgoal_v1"
PROTOCOLS = (MEMORY, SUBGOAL)
ROUTE_LABELS = "ABCDEF"
BLOCKED = {"trajectory_collision", "lidar_brake", "insufficient_progress_or_deviation"}


def anchor(scope):
    if scope.get("tracking_valid") and scope.get("map_session_id") and scope.get("anchor_epoch"):
        return (scope["map_session_id"], scope["anchor_epoch"])
    return None


def measured_event(current, earlier, scope):
    """Use executed poses and distances. A command or discarded step proves nothing."""
    reference = anchor(scope)
    if not current or not reference or anchor(current.get("memory_scope", {})) != reference:
        return None
    end = current.get("end_pose")
    start = current.get("start_pose")
    if not start or not end:
        return None
    event = {"plan_id": current["plan_id"], "pose": end[:3], "outcome": current["outcome"]}
    if current["outcome"] in BLOCKED:
        step = current["step_results"][-1] if current["step_results"] else {}
        attempt = step.get("start_pose") or start
        return {**event, "kind": "blocked", "action": step.get("action"),
                "attempt_heading_deg": round(math.degrees(attempt[2]), 1)}
    if (not earlier or anchor(earlier.get("memory_scope", {})) != reference
            or not earlier.get("start_pose") or not current.get("control_end_pose")
            or not earlier.get("control_start_pose")):
        return None
    walked = sum(s.get("distance_m", 0) for plan in (earlier, current)
                 for s in plan["step_results"] if s["action"] in ("forward", "backward", "creep_forward", "creep_backward"))
    net = math.dist(earlier["control_start_pose"][:2], current["control_end_pose"][:2])
    if walked >= .18 and net <= .06:
        return {**event, "kind": "returned", "walked_m": round(walked, 3), "net_m": round(net, 3)}
    return None


def local_routes(observation, endpoints):
    """Use the same translation menu as the primitive selector."""
    pose = observation["pose"]
    spec = observation["action_spec"]
    horizon = observation["trajectory_steps"]
    routes = []
    for row in translation_options(observation["heading_options"]):
        turns = row["turn_actions"]
        if observation.get("collision_protection") != "disabled" and any(
                turn in observation.get("recent_turn_overshoots", []) for turn in turns):
            continue
        first = ACTION_LABELS[turns[0]] if turns else row["translation_label"]
        if not first_action_labels([first], observation):
            continue
        # A route with more turns than one horizon can still retain a subgoal
        # while each bounded sequence makes progress toward its alignment.
        move = LABELS[row["translation_label"]]
        unit = spec[move]["max_distance_m"]
        count = min(row["then_forward_steps"], max(1, int(.45 / unit)))
        target = rollout(pose, turns + [move] * count, spec)[-1]
        visits = sum(math.dist(point[:2], target[:2]) <= .10 for point in endpoints)
        routes.append(dict(turn_actions=turns, translation_label=row["translation_label"],
                           then_forward_steps=count, goal_xy=[round(x, 3) for x in target[:2]],
                           body_heading_deg=row["destination_map_yaw_degrees"],
                           bearing_deg=round(math.degrees(math.atan2(target[1]-pose[1], target[0]-pose[0])), 2),
                           distance_m=round(count * unit, 3), recent_visits=visits,
                           can_translate_this_plan=len(turns)<horizon))
    return routes


def diverse_routes(routes):
    ranked = sorted(routes, key=lambda x: (x["recent_visits"], -x["distance_m"],
                                          len(x["turn_actions"]), x["translation_label"] in ("B", "C")))
    selected = []
    for row in ranked:
        if all(abs((row["bearing_deg"] - other["bearing_deg"] + 180) % 360 - 180) >= 22
               for other in selected):
            selected.append(row)
        if len(selected) == len(ROUTE_LABELS):
            break
    return {key: row for key, row in zip(ROUTE_LABELS, selected)}


def route_to_goal(observation, routes, goal):
    """Revalidate the first leg of a retained target in the current measured map."""
    origin = observation["pose"][:2]
    target = goal["goal_xy"]
    remaining = math.dist(origin, target)
    if remaining <= .06:
        return None
    # Compare direction as well as distance. A long oblique route can otherwise
    # appear to approach a point while taking a different branch.
    bearing = math.degrees(math.atan2(target[1]-origin[1], target[0]-origin[0]))
    candidates = []
    spec = observation["action_spec"]
    for row in routes:
        if goal.get('kind') == 'passage' and row['translation_label'] not in ('F', 'G'):
            continue
        delta = abs((row["bearing_deg"] - bearing + 180) % 360 - 180)
        if delta > 23:
            continue
        step = spec[LABELS[row["translation_label"]]]["max_distance_m"]
        count = min(row["then_forward_steps"], max(1, round(remaining / step)))
        end = rollout(observation["pose"], row["turn_actions"] + [LABELS[row["translation_label"]]] * count, spec)[-1]
        if math.dist(end[:2], target) < remaining:
            heading_change = abs((row["body_heading_deg"] - goal["body_heading_deg"] + 180) % 360 - 180)
            candidates.append((round(math.dist(end[:2], target), 3), delta,
                               heading_change, len(row["turn_actions"]), count, row))
    if not candidates:
        return None
    *_, count, row = min(candidates, key=lambda item: item[:4])
    return {**row, "then_forward_steps": count}



def passage_goal(route, snapshot_id):
    return dict(kind='passage', goal_xy=route['goal_xy'], body_heading_deg=route['body_heading_deg'],
                travel_bearing_deg=route['bearing_deg'], source_snapshot=snapshot_id)


def choose_passage(observation, lanes):
    """Select measured map progress, consulting the current image for the entrance."""
    mission = observation.get('mission_context') or {}
    candidates = observation.get('passage_candidates', [])
    pose = observation['pose']
    if not candidates or observation.get('collision_protection') == 'disabled':
        return None, None

    def direction(row):
        return (row['body_heading_deg'] - math.degrees(pose[2]) + 180) % 360 - 180

    def visible_block(row):
        # Only an almost straight corridor shares the current camera's center ray.
        return lanes.get('center') == 'B' and abs(direction(row)) < 8

    retained = mission.get('subgoal') or {}
    leg = mission.get('route')
    if retained.get('kind') == 'passage' and leg and not visible_block(leg):
        return leg, retained

    forward = [row for row in candidates if row['translation_label'] in ('F', 'G')
               and not visible_block(row)]
    event = mission.get('latest_measured_event') or {}
    if (event.get('kind') == 'blocked' and event.get('action') in ('forward', 'creep_forward')
            and math.dist(event['pose'][:2], pose[:2]) < .12):
        different = [row for row in forward if abs((row['bearing_deg']
                      - event['attempt_heading_deg'] + 180) % 360 - 180) >= 18]
        if different:
            forward = different

    if forward:
        def score(row):
            angle = direction(row)
            lane = 'left' if angle >= 8 else 'right' if angle <= -8 else 'center'
            visual = {'O': .07, 'B': -.07}.get(lanes.get(lane), 0.) if abs(angle) <= 50 else 0.
            return (row['distance_m'] + visual - .14 * abs(angle) / 90
                    - .06 * min(row['recent_visits'], 4))

        chosen = max(forward, key=score)
        return chosen, passage_goal(chosen, observation['snapshot_id'])

    # A back-off is one short straight step. The next observation either offers
    # a forward passage or justifies another recovery for the newly measured pose.
    for row in candidates:
        if row['translation_label'] in ('B', 'C') and not row['turn_actions']:
            if first_action_labels(['C'], observation):
                return {**row, 'translation_label': 'C', 'then_forward_steps': 1}, None
    return None, None


def passage_labels(prefix, route, spec, allowed):
    checked = [letter for letter in allowed if letter in route_labels(prefix, route, spec)]
    return [letter for letter in checked if letter != 'S'] or ['S']

def allowed_reply(packet, response):
    if response.get("protocol") != packet["protocol"]:
        return False
    choices = ("K", "S") if packet["protocol"] == MEMORY else (*packet["routes"], "S")
    return response.get("choice") in choices


def route_labels(prefix, route, spec):
    """Primitives that continue one checked route within its metric budget."""
    turns = route["turn_actions"]
    if len(prefix) < len(turns):
        return {ACTION_LABELS[turns[len(prefix)]], "S"}
    label = route["translation_label"]
    candidates = (label, {"F": "G", "B": "C"}.get(label))
    used = sum(spec[LABELS[action]].get("max_distance_m", 0) for action in prefix)
    budget = spec[LABELS[label]]["max_distance_m"] * route["then_forward_steps"]
    return {"S"} | {code for code in candidates if code
                    and used + spec[LABELS[code]]["max_distance_m"] <= budget + 1e-8}


def repeated_return(event, notes):
    """Treat nearby measured returns as the same observed problem."""
    return any(note.get("kind") == "returned" and math.dist(
        event["pose"][:2], note["pose"][:2]) <= .12 for note in notes)
