"""Camera framing and measured sensor provenance; no steering or action vetoes."""

import math

CAMERA_RULES = """The front RGB camera is monocular; field of view and depth are uncalibrated.
Apparent size, blur or an object filling the frame alone do not establish metric distance
or physical contact. Perspective and object size matter. Do not overrule an associated
measured range just because an object looks large. Describe visual obstacles/gaps
qualitatively, without invented centimetres.
"""

SENSOR_RULES = (
    CAMERA_RULES
    + """Lidar measures a horizontal 2-D scan plane, not the whole 3-D body volume.
It may miss objects below/above that plane, overhangs, thin legs, transparent/reflective
surfaces and objects in blind/invalid regions. No return or a free mapped cell is not proof
that the full body path is empty. A sector minimum is the nearest returned beam in a sector,
not an associated depth of the camera target or guaranteed body clearance. Keep these distinct.
Use fresh camera evidence to identify visible path obstructions even when the lidar map is
free; use lidar geometry for distances it actually measures. Neither sensor universally wins.
When they seem to disagree, check bearing, height, age, association and sensor/body offsets;
keep unmeasured object depth unknown rather than invent a reconciliation.
Read sensor_grounding.execution: a selected/planned action or software stop is not measured
travel or proof of physical contact. Undertravel despite a translation attempt can reflect an
unseen obstruction, wheel slip, low battery, drivetrain/control trouble or localisation error.
Fresh camera evidence and measured outcomes help distinguish these; do not assert a cause
without evidence. Do not repeatedly push in a heading that produces no measured progress merely
because lidar looks clear. Reassess and choose a justified route around/away from the obstruction,
short repositioning or observation. A turn-only/stop plan did not attempt translation.
Do not drive into a visible surface to 'clear' that surface. Normal useful multi-step travel
remains appropriate when camera, measured clearance and execution are consistent.
"""
)


def frame_size(width, height, max_width=320, max_height=240):
    """Fit the entire received frame, preserving aspect ratio and never enlarging."""
    if min(width, height, max_width, max_height) <= 0:
        raise ValueError("Camera dimensions must be positive")
    scale = min(1.0, max_width / width, max_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def camera_image(frame):
    from PIL import Image

    width, height = frame_size(frame.width, frame.height)
    rgb = frame.reformat(width=width, height=height, format="rgb24")
    return Image.fromarray(rgb.to_ndarray())


def sensor_context(observation, camera_size):
    """Expose executed evidence without diagnosing collision or fabricating depth."""
    plan = observation.get("previous_plan_outcome")
    steps = (plan or {}).get("step_results", [])
    attempts = []
    for step in steps:
        action = step.get("action")
        spec = observation.get("action_spec", {}).get(action, {})
        if "max_distance_m" not in spec:
            continue
        distance = step.get("distance_m")
        if not isinstance(distance, (int, float)) or not math.isfinite(distance):
            distance = None
        outcome = step.get("outcome")
        kind = (
            "software_geometry_stop"
            if outcome in ("lidar_brake", "trajectory_collision")
            else (
                "undertravel_or_deviation_cause_unknown"
                if outcome == "insufficient_progress_or_deviation"
                else (
                    "measured_step"
                    if outcome == "duration_complete"
                    else "other_or_unknown"
                )
            )
        )
        attempts.append(
            dict(
                action=action,
                nominal_target_m=spec["max_distance_m"],
                measured_displacement_m=distance,
                controller_outcome=outcome,
                interpretation=kind,
            )
        )
    return dict(
        camera=dict(
            image_size_px=list(camera_size),
            intrinsics_and_depth="uncalibrated",
            source_field_of_view="unknown; upstream crop not verified",
        ),
        lidar=dict(
            coverage="horizontal scan plane", sector_minimum_is_target_range=False
        ),
        execution=dict(
            plan_id=(plan or {}).get("plan_id"),
            last_plan_status=(
                "unavailable"
                if plan is None
                else (
                    "translation_was_attempted"
                    if attempts
                    else "no_translation_attempted"
                )
            ),
            translation_attempts=attempts[-3:],
        ),
    )
