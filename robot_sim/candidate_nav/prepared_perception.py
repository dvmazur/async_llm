"""Validate exact-frame perception reuse without importing ROS, Torch or the GPU."""
import hashlib
import math

from .temporal_vision import task_key
from . import target_vision


def image_digest(image):
    return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def prepared_pose_matches(frame, pose, now):
    if pose is None or not 0 <= now-frame['stamp'] <= 8:
        return False
    old = frame['pose']
    yaw = math.atan2(math.sin(old[2]-pose[2]), math.cos(old[2]-pose[2]))
    return math.dist(old[:2], pose[:2]) <= .03 and abs(yaw) <= .08


def state_usable(state, split=False):
    if split:
        return target_vision.is_split_state(state)
    return state.get('relevant_visible') in (True, False)


def observer_text(observation, camera):
    prepared = observation.get('prepared_perception') or {}
    state = prepared.get('visual_state') or {}
    frame = state.get('latest_frame') or {}
    instruction = observation.get('instruction', '')
    if (prepared.get('task_key') != task_key(instruction)
            or state.get('task_key') != task_key(instruction)
            or not state_usable(state, target_vision.tracking_enabled(observation))
            or frame.get('stamp') != observation.get('camera_stamp')
            or prepared.get('camera_stamp') != observation.get('camera_stamp')
            or prepared.get('camera_sha256') != image_digest(camera)
            or not frame.get('pose')
            or not prepared_pose_matches(frame, observation['pose'], observation['sim_time'])):
        return None
    if target_vision.tracking_enabled(observation):
        return target_vision.describe(state) if 'scene_directions' in state else None
    visible = ('task-relevant evidence is visible' if state['relevant_visible'] else
               'task-relevant evidence is not visible')
    return (f"Current (already analysed exact IMAGE 1, frame {frame['frame_id']}): {visible}. "
            f"{state.get('description', 'unknown')[:600]}\n"
            f"Change: {prepared.get('evidence', 'nothing new')[:600]}\n"
            "Recorded camera pose/time apply; visibility alone is not arrival or measured range.")
