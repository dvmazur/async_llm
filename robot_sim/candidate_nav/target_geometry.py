"""Task-local camera bearings and bounded strategy-4 trajectory preference."""

import itertools
import math

from .action_plan import ACTION_LABELS, LABELS, rollout
from .committed_plan import admissible, wrap

MAX_SIGHTINGS = 6
# The RTSP image comes without intrinsics. Bound a generic centred front camera's
# projection until a stream-specific calibration is supplied. This is a prior.
HALF_FOV_RANGE = tuple(map(math.radians, (22.5, 50)))


def _anchor(frame):
    if frame.get('tracking_valid') is False:
        return None
    session, epoch = frame.get('map_session_id'), frame.get('anchor_epoch')
    return (session, epoch) if session and epoch else None


def _arc(angles):
    """Smallest circular interval containing the input bearings."""
    angles = sorted(angle % math.tau for angle in angles)
    gaps = [next_angle - angle for angle, next_angle in
            zip(angles, angles[1:] + [angles[0] + math.tau])]
    cut = max(range(len(gaps)), key=gaps.__getitem__)
    half = (math.tau - gaps[cut]) / 2
    return wrap(angles[(cut + 1) % len(angles)] + half), half


def _ray(sighting):
    x0, _, x1, _ = sighting['bbox_pct']
    # Include horizontal box width, pixel/report error and unknown lens yaw.
    angles = [math.atan((1 - x / 50) * math.tan(fov))
              for x, fov in itertools.product((x0, x1), HALF_FOV_RANGE)]
    clipped = x0 <= 2 or x1 >= 98
    padding = math.radians(7)
    lo = min(angles) - padding - (math.radians(18) if x1 >= 98 else 0)
    hi = max(angles) + padding + (math.radians(18) if x0 <= 2 else 0)
    x, y, yaw = sighting['pose']
    return (x, y), yaw + lo, yaw + hi, clipped


def remember(state, sighting):
    if not _anchor(sighting) or not sighting.get('pose'):
        return
    old = state.get('sightings', [])
    samples = [item for item in old if _anchor(item) == _anchor(sighting)
               and 0 < sighting['stamp'] - item['stamp'] <= 120]
    state['sightings'] = samples[-(MAX_SIGHTINGS - 1):] + [sighting]


def _intersection(a, angle_a, b, angle_b):
    denominator = math.sin(angle_b - angle_a)
    if abs(denominator) < .12:
        return None
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = (dx * math.sin(angle_b) - dy * math.cos(angle_b)) / denominator
    u = (dx * math.sin(angle_a) - dy * math.cos(angle_a)) / denominator
    if min(t, u) <= 0 or max(t, u) > 4:
        return None
    return (a[0] + t * math.cos(angle_a), a[1] + t * math.sin(angle_a))


def _region(rays):
    """Only report a bounded region if both complete bearing wedges constrain it."""
    b, blo, bhi, clipped = rays[-1]
    if clipped:
        return ()
    for a, alo, ahi, old_clipped in reversed(rays[:-1]):
        if old_clipped or math.dist(a, b) < .10:
            continue
        points = [_intersection(a, alpha, b, beta) for alpha, beta in
                  itertools.product((alo, ahi), (blo, bhi))]
        if all(points) and max(math.dist(x, y) for x, y in itertools.combinations(points, 2)) <= .35:
            return tuple(points)
    return ()


class TargetEstimate:
    def __init__(self, rays, status):
        self.origin, self.low, self.high, self.clipped = rays[-1]
        self.region = _region(rays)
        self.visible = status == 'visible'

    def bearing(self, pose):
        x, y, yaw = pose
        origin_distance = math.dist((x, y), self.origin)
        if self.region:
            # At a position inside or close to the uncertainty region, bearing
            # ceases to be useful. A fresh observation/measurement is needed.
            xmin, xmax = min(p[0] for p in self.region), max(p[0] for p in self.region)
            ymin, ymax = min(p[1] for p in self.region), max(p[1] for p in self.region)
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return 0.0, math.pi
            angles = [math.atan2(py - y, px - x) for px, py in self.region]
        else:
            # With no range, a translated viewpoint sees an arc spanning from
            # the original camera position to the far end of the bearing wedge.
            angles = [self.low, self.high]
            if origin_distance > .02:
                back = math.atan2(self.origin[1] - y, self.origin[0] - x)
                center = (self.low + self.high) / 2
                # A pose in the original wedge could be on either side of the
                # target. Its direction cannot be inferred from that sighting.
                if abs(wrap(back + math.pi - center)) <= (self.high - self.low) / 2:
                    return 0.0, math.pi
                angles.append(back)
        center, half = _arc(angles)
        return wrap(center - yaw), half

    def describe(self, pose):
        center, half = self.bearing(pose)
        return dict(body_bearing_deg=round(math.degrees(center), 1),
                    uncertainty_deg=round(math.degrees(half), 1),
                    camera_report='seen' if self.visible else 'historical',
                    position_region_xy=[[round(v, 3) for v in p] for p in self.region],
                    source='uncalibrated_front_camera_bearing_prior')


def estimate(state, observation):
    scope = observation.get('memory_context') or {}
    anchor = _anchor(scope)
    if not anchor or not scope.get('tracking_valid'):
        return None
    target = state['target']
    latest = target.get('last_seen')
    if not latest:
        return None
    # last_seen is retained for native visual identity even when its timestamp
    # could not be paired with TF. Older, correctly paired observations may still
    # provide bearings within the current epoch.
    samples = {s['stamp']: s for s in [*state.get('sightings', []), latest]
               if s.get('pose') and _anchor(s) == anchor
               and 0 <= observation['sim_time'] - s['stamp'] <= 120}
    if not samples:
        return None
    paired = samples[max(samples)]
    status = target['status'] if paired is latest else 'uncertain'
    result = TargetEstimate([_ray(samples[t]) for t in sorted(samples)], status)
    if not result.region and math.dist(paired['pose'][:2], observation['pose'][:2]) > .65:
        return None
    return result


class TrajectoryPreference:
    """Small soft preference based on checked trajectories and their final poses."""

    def __init__(self, target, observation, translations, labels, turns):
        self.scores = {}
        options = observation.get('heading_options', [])
        pose, spec = observation['pose'], observation['action_spec']
        horizon = observation['trajectory_steps']
        center, half = target.bearing(pose)
        if half >= math.pi / 2:
            return
        nearest = max(0., abs(center) - half)
        if target.visible and target.clipped:
            # Recent image evidence gives a preferred direction within the broad
            # edge sector. Near a blocked forward corridor, leave detours to Qwen.
            forward = next((row['known_forward_m'] for row in options
                            if not row['turn_actions']), 0.)
            if forward < .10:
                return
            aim = center
        elif not target.visible and nearest >= math.radians(8):
            # Scan toward the nearer boundary of a wide historical sector first.
            aim = math.copysign(min(abs(center), nearest + math.radians(18)), center)
        else:
            aim = center if target.region else 0.
        for row in options:
            prefix = tuple(ACTION_LABELS[a] for a in row['turn_actions'])
            delta = math.radians(row['turn_degrees'])
            if prefix and (delta * aim <= 0 or abs(delta) > math.radians(45)):
                continue
            if (not row['turn_sweep_known_free'] or len(prefix) > horizon
                    or any(letter not in turns for letter in prefix)):
                continue
            candidates = [prefix] if abs(aim) > math.radians(10) else []
            if target.visible:
                for option in translations:
                    if option['turn_actions'] != row['turn_actions']:
                        continue
                    step = option['translation_label']
                    limit = min(option['then_forward_steps'], horizon - len(prefix))
                    candidates.extend(prefix + (step,) * count for count in range(1, limit + 1))
            for path in candidates:
                if not path or path[0] not in labels:
                    continue
                if any(code not in admissible(path[:i], translations, spec, turns)
                       for i, code in enumerate(path)):
                    continue
                gain = abs(aim) - abs(wrap(aim - delta))
                value = 0.
                if gain > math.radians(5):
                    value = .9 + gain / math.radians(6.5) - .14 * len(prefix)
                    # When depth is unknown, the alignment advantage stops at
                    # the turn. Re-evaluate the target using the new viewpoint.
                    if not target.region and len(path) > len(prefix):
                        continue
                if target.region and len(path) > len(prefix):
                    end = rollout(pose, [LABELS[code] for code in path], spec)[-1]
                    bearing, spread = target.bearing(end)
                    old_range = min(math.dist(pose[:2], p) for p in target.region)
                    new_range = min(math.dist(end[:2], p) for p in target.region)
                    # Prefer reductions in range only well outside the uncertain region.
                    if (abs(bearing) + spread < math.radians(38)
                            and .08 < new_range < old_range):
                        value += min(1.0, 7 * (old_range - new_range))
                if value <= 0:
                    continue
                for i, letter in enumerate(path):
                    key = (path[:i], letter)
                    self.scores[key] = max(self.scores.get(key, 0.), min(3.6, value))
                # The checked route may be a turn-only observation.
                if len(path) < horizon:
                    self.scores[(path, 'S')] = max(self.scores.get((path, 'S'), 0.), min(3.6, value))

    def choose(self, prefix, labels, probabilities):
        key = tuple(prefix)
        return max(range(len(labels)), key=lambda i: math.log(max(1e-38, probabilities[i]))
                   + self.scores.get((key, labels[i]), 0.))
