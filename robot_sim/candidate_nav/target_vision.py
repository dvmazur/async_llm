"""Independent visual target and scene evidence for live strategy 4."""

import base64
import copy
import io
import math
import re

from .temporal_vision import task_key
from .target_geometry import remember

MODE = 'target_scene_v2'
STATUSES = {'none', 'visible', 'not_visible', 'uncertain'}
FRAME_FIELDS = ('frame_id', 'stamp', 'pose', 'plan_id', 'map_session_id', 'anchor_epoch', 'tracking_valid')

RULES = """Target and scene are independent. A target is an entity the instruction asks to
find, inspect, follow, avoid as a named entity, or otherwise act with reference to.
A relative observation can be relevant to the task without being its target. Use NONE
for tasks with no visual target, for example rotate or leave a featureless maze.
Do not invent a target for a motion-only instruction. Respect the complete user task.
Match a prior target using its distinctive appearance and any historical reference crop.
An unrelated item, obstacle or body part never confirms that target.
Use UNSURE if identity or location in the image cannot be confirmed. If a requested
target is absent from this frame, use LOST. Obstructions and passages belong in SCENE.
Never infer metric distance, safe clearance, contact, arrival, or completion from this report."""

SCENE_DIRECTIONS = ('left', 'center', 'right')
REPORT = """Describe only the CURRENT camera image. Reply with exactly three lines.
TARGET NONE, TARGET LOST, TARGET UNSURE, or TARGET SEEN|short distinctive appearance|x1,y1,x2,y2
For SEEN use integer percentages 0..100 relative to the full current image (left,top,right,bottom).
Box only the visible target. Use UNSURE if you cannot localize the specified entity.
SCENE followed by one short clause giving the location of foreground obstacles
and visible gaps or open floor. Give relative directions from this image.
PATH followed by three letters separated by spaces, for the left, center, and
right thirds of the image in that order. Use O for continuous visible floor
through the foreground toward open space, B when a foreground obstacle crosses
that direction, and U if blurred, uncertain, or the floor is obscured.
Use directions in the image only. Do not infer if the rover fits.
Examples of the format:
TARGET LOST
SCENE The stand blocks the right; the carpet opens across the left and center.
PATH O O B
TARGET SEEN|ribbed red cylinder|12,40,28,85
SCENE Narrow opening at the left; cylinder blocks the center and right.
PATH O B B"""


def tracking_enabled(packet):
    return packet.get('physical') is True and packet.get('visual_tracking') == MODE


def initial_state(instruction, previous=None):
    key = task_key(instruction)
    if previous and previous.get('schema') == MODE and previous.get('task_key') == key:
        return copy.deepcopy(previous)
    return dict(schema=MODE, task_key=key, target=dict(status='uncertain', last_seen=None),
                scene='unknown', latest_frame=None)


def is_split_state(state):
    return (state.get('schema') == MODE and isinstance(state.get('target'), dict)
            and state['target'].get('status') in STATUSES)


def valid_box(box):
    return (isinstance(box, (tuple, list)) and len(box) == 4
            and all(type(n) is int and 0 <= n <= 100 for n in box)
            and box[0] < box[2] and box[1] < box[3])


def parse_report(text):
    """Invalid target text cannot promote a scene observation to a sighting."""
    targets = re.findall(r'^TARGET\s+([^\r\n]+)$', text, re.MULTILINE)
    scenes = re.findall(r'^SCENE\s+([^\r\n]+)$', text, re.MULTILINE)
    scene = scenes[0].strip()[:180] if len(scenes) == 1 else 'unknown'
    paths = re.findall(r'^PATH[ \t]+([OBU])[ \t]+([OBU])[ \t]+([OBU])[ \t]*$', text, re.MULTILINE)
    report = dict(status='uncertain', scene=scene or 'unknown')
    if paths:
        report['scene_directions'] = dict(zip(SCENE_DIRECTIONS, paths[0] if len(paths) == 1 else 'UUU'))
    if len(targets) != 1:
        return report
    target = targets[0].strip()
    if target in ('NONE', 'LOST', 'UNSURE'):
        report['status'] = {'NONE': 'none', 'LOST': 'not_visible', 'UNSURE': 'uncertain'}[target]
        return report
    match = re.fullmatch(r'SEEN\|([^|]{1,100})\|\s*(\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})', target)
    if match:
        box = [int(n) for n in match.groups()[1:]]
        appearance = match[1].strip()
        if appearance and valid_box(box):
            report.update(status='visible', appearance=appearance, bbox_pct=box)
    return report


def frame_record(frame):
    return {key: frame[key] for key in FRAME_FIELDS if key in frame}


def apply_report(state, report, frame):
    frame = frame_record(frame)
    previous = state.get('latest_frame')
    if previous and previous['stamp'] >= frame['stamp']:
        return
    target = state['target']
    target['status'] = report['status']
    if report['status'] == 'visible':
        target['last_seen'] = dict(frame, appearance=report['appearance'],
                                   bbox_pct=report['bbox_pct'])
        remember(state, target['last_seen'])
    elif report['status'] == 'none':
        target['last_seen'] = None
        state.pop('sightings', None)
    state.update(latest_frame=frame, scene=report['scene'])
    state.pop('scene_directions', None)
    if 'scene_directions' in report:
        state['scene_directions'] = report['scene_directions']


def describe(state):
    target = state['target']
    status = target['status']
    last = target.get('last_seen')
    text = {'none': 'This task has no visual target.',
            'visible': 'Target seen in this camera image.',
            'not_visible': 'Requested target is not visible in this camera image.',
            'uncertain': 'Target visibility or identity is uncertain.'}[status]
    if last:
        if status == 'visible':
            text += f" {last['appearance']}. Image box (%): {last['bbox_pct']}."
        else:
            text += f" Last reported target sighting: {last['appearance']} (frame {last['frame_id']})."
    text += f"\nScene: {state['scene']}"
    if 'scene_directions' in state:
        labels = {'O': 'visible open floor', 'B': 'foreground obstruction', 'U': 'uncertain'}
        text += "\nImage thirds: " + ', '.join(
            f"{direction} {labels.get(state['scene_directions'].get(direction), 'uncertain')}"
            for direction in SCENE_DIRECTIONS)
    return text


def remember_reference(current, state, images):
    """Keep a small historical crop from the exact frame of the last sighting."""
    if not is_split_state(state):
        return None
    sighting = state['target'].get('last_seen') or {}
    if not valid_box(sighting.get('bbox_pct')):
        return None
    frame_id = sighting.get('frame_id')
    stamp = sighting.get('stamp')
    if current and current['frame_id'] == frame_id and current['stamp'] == stamp:
        return current
    if frame_id not in images:
        return None
    from PIL import Image
    image = images[frame_id]
    if isinstance(image, str):
        with Image.open(io.BytesIO(base64.b64decode(image))) as decoded:
            image = decoded.convert('RGB')
    x1, y1, x2, y2 = sighting['bbox_pct']
    box = (math.floor(image.width*x1/100), math.floor(image.height*y1/100),
           math.ceil(image.width*x2/100), math.ceil(image.height*y2/100))
    crop = image.crop(box).convert('RGB')
    crop.thumbnail((112, 112))
    data = io.BytesIO()
    crop.save(data, format='JPEG', quality=80)
    return dict(frame_id=frame_id, stamp=stamp, image=base64.b64encode(data.getvalue()).decode())


def reference_content(packet, state, decode_image):
    ref = packet.get('target_reference') or {}
    last = state['target'].get('last_seen') or {}
    if (not last or ref.get('frame_id') != last['frame_id'] or ref.get('stamp') != last['stamp']
            or not isinstance(ref.get('image'), str) or len(ref['image']) > 80000
            or state['target']['status'] == 'none'):
        return []
    try:
        image = decode_image(ref['image'])
    except (ValueError, OSError):
        return []
    return [dict(type='text', text='HISTORICAL target identity crop. Never use this as CURRENT visibility.'),
            dict(type='image', image=image)]


def snapshot_frame(packet):
    scope = packet.get('memory_context') or {}
    return dict(tracking_valid=bool(scope.get('tracking_valid')) and packet.get('camera_pose_paired') is not False,
                frame_id=f"snapshot:{packet['snapshot_id']}", stamp=packet['camera_stamp'],
                pose=packet.get('camera_pose', packet['pose']), plan_id=None,
                map_session_id=scope.get('map_session_id'), anchor_epoch=scope.get('anchor_epoch'))
