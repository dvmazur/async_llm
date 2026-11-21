import math

from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC
from robot_sim.candidate_nav.target_geometry import estimate, TrajectoryPreference
from robot_sim.candidate_nav.target_vision import apply_report, initial_state


SCOPE = dict(map_session_id='mapping', anchor_epoch='first', tracking_valid=True)


def record(state, stamp, pose, box=None, scope=SCOPE):
    frame = dict(frame_id=stamp, stamp=stamp, pose=pose, **scope)
    report = dict(status='not_visible', scene='stand')
    if box:
        report.update(status='visible', appearance='green book', bbox_pct=box)
    apply_report(state, report, frame)


def observation(pose, stamp=2, scope=SCOPE, forward=.3, turn_clear=True):
    return dict(
        pose=pose, sim_time=stamp, memory_context=scope,
        action_spec=ACTION_PLAN_SPEC, trajectory_steps=4,
        heading_options=[
            dict(turn_actions=[], turn_degrees=0, known_forward_m=forward, turn_sweep_known_free=True),
            dict(turn_actions=['left_11_25'], turn_degrees=11.25, turn_sweep_known_free=turn_clear),
            dict(turn_actions=['left'], turn_degrees=22.5, turn_sweep_known_free=turn_clear),
            dict(turn_actions=['right'], turn_degrees=-22.5, turn_sweep_known_free=True),
        ],
    )


def test_visible_book_on_left_prefers_checked_alignment_over_straight():
    state = initial_state('Approach the book')
    record(state, 1., [0, 0, 0], [0, 15, 38, 95])
    obs = observation([0, 0, 0])
    target = estimate(state, obs)
    assert target.describe(obs['pose'])['body_bearing_deg'] > 25
    assert not target.region
    labels = ['G', 'L', 'R', 'Q', 'S']
    scores = [.75, .08, .02, .14, .01]
    pref = TrajectoryPreference(target, obs, [], labels, ['L', 'R', 'Q'])
    assert labels[pref.choose((), labels, scores)] in ('L', 'Q')
    assert pref.choose(('L',), ['R', 'G', 'S'], [.05, .90, .05]) == 2
    for obs in [observation([0, 0, 0], forward=.05), observation([0, 0, 0], turn_clear=False)]:
        pref = TrajectoryPreference(target, obs, [], labels, ['L', 'R', 'Q'])
        assert labels[pref.choose((), labels, scores)] == 'G'
    excluded = TrajectoryPreference(target, observation([0, 0, 0]), [], ['G', 'R', 'S'], ['R'])
    assert not excluded.scores


def test_historical_bearing_rotates_with_body_and_expires_on_anchor_change():
    state = initial_state('Approach the book')
    record(state, 1., [0, 0, 0], [0, 10, 15, 100])
    record(state, 2., [.1, 0, 0])
    straight = observation([.1, 0, 0])
    turned = observation([.1, 0, math.pi / 8])
    estimate_before_turn = estimate(state, straight)
    assert estimate_before_turn.describe(straight['pose'])['camera_report'] == 'historical'
    assert abs(estimate_before_turn.bearing(turned['pose'])[0]) < abs(estimate_before_turn.bearing(straight['pose'])[0])
    assert estimate(state, observation([.1, 0, 0], scope=dict(SCOPE, anchor_epoch='new'))) is None
    assert estimate(state, observation([.1, 0, 0], stamp=123)) is None
    assert state['target']['last_seen']['appearance'] == 'green book'


def test_crossing_bearings_can_bound_a_region_without_claiming_range():
    state = initial_state('Approach the book')
    record(state, 1., [0, 0, 0], [49, 5, 51, 85])
    record(state, 2., [.19, -.21, math.pi / 2], [49, 5, 51, 85])
    target = estimate(state, observation([.19, -.21, math.pi / 2]))
    assert len(target.region) == 4
    assert all(math.dist(p, (.19, 0.)) < .07 for p in target.region)
    record(state, 3., [.19, -.21, math.pi / 2], [0, 5, 35, 85])
    assert not estimate(state, observation([.19, -.21, math.pi / 2], stamp=3)).region


def test_bad_camera_pose_is_not_a_spatial_sighting():
    state = initial_state('Approach the book')
    record(state, 1., [0, 0, 0], [20, 5, 36, 85], dict(SCOPE, tracking_valid=False))
    assert estimate(state, observation([0, 0, 0])) is None
    assert state['target']['last_seen']['appearance'] == 'green book'
