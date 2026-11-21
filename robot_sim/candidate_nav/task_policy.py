"""Instruction-driven live policy; simulation retains its own evaluation rules."""

from .body_model import BODY_RULES
from .map_memory import MAP_MEMORY_RULES
from .sensor_grounding import SENSOR_RULES

import hashlib
from pathlib import Path

MODE = 'task_completion'

MOTION_REALITY_RULES = """
Physical motion is imperfect. Distances and turn angles in action_spec are NOMINAL targets,
not guaranteed exact outcomes. Wheel slip, inertia, motor response and localisation error
can cause undertravel, overshoot and heading drift. Expected poses are predictions; measured
start/end poses, distance_m and turn_rad are the evidence of what actually happened.
Plan the next sequence from the measured current pose. Do not infer progress by counting
commands or multiplying their count by nominal distance/angle. Retain useful measured progress
after a partial or interrupted action; discarded actions were never executed. Small errors are
normal: do not treat them as complete failure or repeatedly undo progress to match an ideal
pose. Correct residual errors when they matter for the remaining instruction. Control tolerances
alone do not establish that a precise final task requirement has been satisfied.
"""

TASK_RULES = """Carry out the user's entire instruction using the available rover controls.
Read every requested step, condition, quantity, direction, sequence, timing and distance.
Track completed and remaining steps using task_context, odometry, history, camera and map.
For compound instructions, keep the remaining requirements active as each step is achieved.
Base decisions on the observations and execution history. State uncertainty when the inputs
cannot establish a requested fact. Choose actions that advance the remaining requirements.
Respect explicit approximation words using available measurement uncertainty; do not require
exact arithmetic equality for an approximate request. Preserve exact requirements when specified.
Completion means that the whole instruction has been fulfilled. Z reports completion and stops.
Only executed actions and measured outcomes count as progress. A planned action is still pending.
Use odometry for distances and turns. Read sensor availability and uncertainty before using a value.
When a command is blocked or makes little progress, use the map and execution history to revise it.
"""

TASK_RULES_BASE = TASK_RULES
TASK_RULES += BODY_RULES + MAP_MEMORY_RULES + MOTION_REALITY_RULES + SENSOR_RULES

WAYPOINT_SYSTEM = """You control a physical mobile rover through candidate waypoints.
IMAGE 1 is the front RGB camera. IMAGE 2 is the measured online map.
White is measured free space, black is occupied, gray is unknown. The cyan arrow shows
position and heading, the blue trail shows previous travel, and letters mark candidate poses.
The selected candidate is executed by the local navigation controller. W waits for observations.
Use measured geometry and action outcomes to plan the next part of the supplied instruction.
Concurrent camera and spatial streams share the current observation and their analysis.
""" + TASK_RULES

DIRECT_SYSTEM = """You control a physical mobile rover through bounded direction commands.
IMAGE 1 is the front RGB camera. IMAGE 2 is the measured online map.
White is free space, black is occupied, gray is unknown. The cyan arrow shows heading.
Choose F=forward, L=turn left, R=turn right, S=stop or wait. Read action_spec for speeds,
distance, angle and duration limits. Each action finishes before the next observation.
Use task_context and the action history to track the results of your previous commands.
Concurrent camera and spatial streams share the current observation and their analysis.
""" + TASK_RULES

GENERATION_SYSTEM = """Generate candidate poses that advance the supplied rover instruction.
IMAGE 1 is the front RGB camera. IMAGE 2 is a 512x512 measured map with pixel coordinates.
White pixels are reachable free space for the robot centre, black is occupied, gray is unknown
or unreachable. The cyan arrow shows the rover, blue shows its trail. Use task_context and
history to select poses for the remaining steps. Provide useful positions or orientations
in measured reachable space using the supplied pixel coordinates and heading convention.
There are no precomputed candidates: author new positions yourself. Use generation_map.rover_pixel
and the labeled pixel grid as references. Distinct movement poses need distinct positions;
changing only the heading at the same pixel does not provide a different route.
Read proposal_feedback: it records your previous coordinates and the validator's decisions.
Correct rejected positions using the current map instead of repeating them. A look/rotation
uses the rover's actual pixel, not the image centre. Stop proposing when no useful pose remains.
""" + TASK_RULES

TRAJECTORY_SYSTEM = """You control a physical rover by planning a short sequence of direction actions.
IMAGE 1 is current front RGB. IMAGE 2 is the measured online map. White is free, black occupied,
gray unknown. The cyan arrow gives current position and heading; blue shows previous travel.
Return up to trajectory_steps actions. F=forward, L=left turn in place, R=right turn in place,
S=stop this sequence and request a fresh observation. S must be last. Read action_spec for the
actual distance, angle, speed and time limits of each primitive. One F is one bounded step.
Simulate the sequence from the current pose: each turn changes the frame of later forward steps.
Choose a concrete next subgoal of the full instruction and a short sequence toward it. Check
every segment against measured free space and the rover footprint. End with S before needing
an unseen area, a new visual confirmation or an uncertain manoeuvre. Prefer fewer justified
actions to filling the horizon. A safety interruption discards all unexecuted actions.
Read recent_actions and previous_analysis. Avoid repeating blocked forward moves or alternating
left/right at the same place without a new purpose. Preserve unfinished instruction requirements.
Read previous_plan_outcome.step_results for measured progress of every attempted plan step;
discarded_actions were never executed. A blocked or partial step does not fulfil its target distance.
Use maneuver_memory for measured start/end headings, executed turns, blocked forward headings
and progress across multiple plans. Completed turns remain completed when a later forward step
is blocked. Keep the resulting heading: the forward veto did not undo those turns.
Read action_geometry in degrees. Add the actual primitive angles when planning a larger turn;
do not assume a left/right command is a quarter-turn. Compare world headings, not just labels.
Map yaw follows mathematical/ROS angles: 0 degrees is +X, +90 is +Y, 180 is -X, -90 is -Y.
L increases yaw; R decreases yaw. The map image may be rotated relative to map coordinates.
Read available_translations: each entry gives literal turn_actions from the CURRENT pose,
followed by how many F steps fit in measured free space. These are the geometry-supported
translation options. prefixes_without_known_forward_step cannot justify an F step yet.
Use the literal turn_actions rather than relabeling a right turn as left from an image guess.
To make forward progress, choose an available option relevant to the task. For rotation-only
instructions or a new observation, turns followed by S remain valid without a translation.
These are geometry facts, not preselected goals; use them for the actual user instruction.
When rotation toward an opening is unfinished, continue toward that opening using the map.
Before reversing a recent rotation, explain which new evidence or instruction justifies it and
check that it does not merely return to a previously blocked heading. Opposite turns are valid
when needed by the instruction or a measured alternative route; no turn direction is mandatory.
If oscillation_detected is true, choose a concrete exit/viewpoint heading and retain progress
toward it across replans. End with S to inspect uncertainty instead of appending unsupported F.
The proposed sequence is a prediction, not proof of arrival or completion. Z is available only
in the separate completion check using observations from actions already executed.
Concurrent camera and spatial streams share the observation and their developing analysis.
""" + TASK_RULES

TRAJECTORY_SPATIAL_PROMPT = (
    'Read maneuver_memory before choosing a plan. Give three concise plain-text lines, no '
    'preamble or repetition of the task: Progress: measured turns/translation and what remains; '
    'Heading: an available_translations option using its literal turn_actions and destination yaw, checking prior blocked headings; '
    'Plan: F/L/R/S sequence and when to request another observation. Use action_geometry degrees '
    'and action_spec for each step. Completed turns persist after a blocked F. If proposing an '
    'opposite turn, explain why it is useful instead of undoing progress. Verify free space after '
    'every predicted turn before adding F; use available_translations.then_forward_steps rather than '
    'guessing the next direction from the image. Respect the whole user instruction.'
)

GENERATION_SPATIAL_PROMPT = (
    'Use IMAGE 2 and generation_map to identify white reachable regions by pixel coordinates. '
    'Suggest distinct new poses for the unfinished instruction. Read proposal_feedback and '
    'correct previous rejections. State useful coordinates directly; be concise.'
)

GENERATION_FORMAT_PROMPT = (
    'Generate up to eight useful, distinct next poses for the remaining instruction. '
    'Each row is u,v,h,k. u and v are three-digit zero-padded image pixel coordinates '
    '000..511 (x right, y down). h is an image heading 0..7: 0 right,2 down,4 left,6 up. '
    'k is F for a new viewpoint, I for an approach, L for a look/rotation. '
    'For L use generation_map.rover_pixel and a different heading. '
    'Choose positions in WHITE reachable space. Use proposal_feedback to avoid repeating '
    'invalid coordinates. For movement, do not repeat one position with different headings. '
    'Reply with W instead of a row when no more useful poses remain. '
    'Use only rows, no prose. First row:\n'
)

OBSERVER_PROMPT = (
    'Describe current camera evidence relevant to the full user instruction and its remaining '
    'requirements. Include observations needed to assess progress. Be concise.'
)
SPATIAL_PROMPT = (
    'Assess progress against every requirement of the user instruction using task_context, '
    'pose, measured geometry and execution history. Identify the next unfinished step and '
    'recommend an available action or candidate that advances it. Be concise.'
)
COMPLETION_PROMPT = (
    "Has the rover completed the user's entire instruction as written? Assess every requested "
    'step and condition, including any movement, final position, orientation, distance, timing '
    'or sequence. Use current observations and task_context execution history. Answer Yes when '
    'all requested requirements are fulfilled, otherwise No. Answer exactly Yes or No:'
)


def is_task_mode(observation):
    return observation.get('physical') is True and observation.get('completion_mode') == MODE


def source_hash():
    """Invalidate a warm service when inference, its protocol or prompts change."""
    root = Path(__file__).resolve().parent
    return hashlib.sha256(b''.join((root / name).read_bytes()
                                  for name in ('llm.py', 'task_policy.py', 'action_plan.py', 'mission_context.py', 'mission_policy.py', 'system_prefix.py', 'prompt_state.py', 'trajectory_policy.py', 'motion_vision_policy.py', 'motion_vision.py', 'temporal_vision.py', 'target_vision.py', 'target_vision_policy.py', 'body_model.py', 'map_memory.py', 'hardware_profile.py', 'parallel_planning.py', 'streaming_perception.py', 'prepared_perception.py', 'inference_cache.py', 'inference_tuning.py', 'inference_decisions.py', 'action_tokens.py', 'context_budget.py', 'model_catalog.py', 'long_term_memory.py', 'memory_scope.py', 'sensor_grounding.py', 'committed_plan.py', 'local_odometry.py', 'model_transport.py', 'artifact_writer.py', 'wire_images.py')) +
                          b''.join((root.parents[1] / name).read_bytes() for name in (
                              'python/minisgl/kernel/fp8.py',
                              'python/minisgl/llm/async_llm.py',
                              'python/minisgl/scheduler/async_engine.py',
                              'python/minisgl/engine/config.py',
                              'python/minisgl/engine/engine.py',
                              'python/minisgl/engine/graph.py',
                              'python/minisgl/shared_cache/session.py',
                              'python/minisgl/shared_cache/vision_cache.py',
                              'python/minisgl/models/qwen3_5.py',
                              'python/minisgl/shared_cache/attention.py',
                              'python/minisgl/shared_cache/graph.py',
                              'python/minisgl/shared_cache/gdn_prefill.py'))).hexdigest()


TRAJECTORY_EVIDENCE_SYSTEM = """Control a physical rover to fulfil the entire user's instruction.
IMAGE 1 is the current front camera; IMAGE 2 is a measured map: white free, black occupied,
gray unknown. Current measured facts override previous plans. Initial_pose defines the frame
of spatial words in the original instruction unless the instruction explicitly says otherwise.
Use measured execution totals and maneuver_memory. Local maneuver sums are not lifetime totals.
Poses are [x metres, y metres, yaw RADIANS]. A historical camera pose locates the rover
when an image was captured, not the observed object. It is not a target position or bearing.
F/L/R/S are bounded primitives in action_spec. L increases map yaw; R decreases it.
Available_translations gives literal turn prefixes and how many F steps fit from this pose.
Keep completed turns when a later step was blocked. Stop early at a meaningful evidence boundary.
Only executed, measured actions count as progress. Discarded steps did not happen.
A visible target alone does not fulfil an instruction requiring a final distance or pose.
A lidar sector minimum is not a range measurement of a particular visual object. Unknown
measurements cannot establish precise completion. Carry out arbitrary compound instructions;
report completion only after all conditions have observed evidence. motion_visual_memory contains historical sightings at recorded poses/times: use them to
remember where a target was seen and orient back toward that world bearing when relevant.
Do not assume a historical sighting means the object is visible now or the task is done.
Current camera evidence
is supplied by a dedicated observer before you choose the action sequence.
While an action sequence executes, its camera observer updates perception only; it cannot
replace or shorten that committed sequence. Choose the next sequence at the measured end pose.
Visual memory contains VLM interpretations, not measured goal progress. A past phrase such as
"approach completed" is not evidence of arrival; stopping a trajectory is not completing a task.
prepared_perception, when validated, is the completed analysis of the exact current camera image.
Use it with fresh measured sensors and maps; it is not a forecast and does not imply completion.
"""
ADAPTIVE_MOTION_RULES = """
Plan a useful metric subgoal, then encode it into primitives. In a measured clear corridor,
prefer consecutive normal steps F/B toward that subgoal. G/C need an evidenced reason why a
normal step would overshoot the subgoal or lack clearance: tight fit, alignment, nearby body
obstacle or small residual displacement. Do not select them merely because the task contains
a small number. For an object-approach request, the desired FINAL separation is not a travel
increment. Explicit relative travel instructions still limit the TOTAL distance travelled.
For an approach, distinguish measured object range from UNKNOWN object range. A description
such as "closer", historical poses and unassociated lidar minima cannot establish remaining
distance to the object. Do not assert that one short step will achieve the requested gap.
Unknown object range prevents claiming precise arrival, not useful travel through confirmed
free space; visible low obstacles and actual body clearance still limit that travel.
Each extra observation costs time. Choose S for a concrete new decision boundary or required
wait/verification, not automatically after one primitive. Motion frames are already processed
during the sequence. Use several slots when justified, but never pass a subgoal or clearance
boundary merely to fill the horizon. Disabling collision enforcement does not make occupied
space free: still check the literal turn prefix AND subsequent translation budget for the
whole proposed route, not just a visually open heading.
"""
TRAJECTORY_EVIDENCE_SYSTEM += ADAPTIVE_MOTION_RULES
TRAJECTORY_EVIDENCE_SYSTEM += BODY_RULES + MAP_MEMORY_RULES + MOTION_REALITY_RULES + SENSOR_RULES

TRAJECTORY_EVIDENCE_PROMPT = (
    'Give two short clauses: Current: task-relevant evidence visible in IMAGE 1 now; '
    'Change: only NEW task-relevant information versus the PREVIOUS CAMERA identified by image_roles and its recorded pose/time. '
    'If no previous image, establish a brief baseline. If no meaningful change, write Change: nothing new. '
    'Use temporal_visual_state and historical motion events to retain where relevant evidence was last seen; '
    'historical sightings are not current visibility. Account for ego-motion: a camera turn does not mean '
    'an object moved. Mention appearance, loss from view, newly revealed passage or changed relevant obstruction. '
    'Include a task-relevant path obstruction even if it is not the target and lidar misses its height. '
    'A large close-up alone is not metric range/contact. A visible gap is not proof of body clearance. '
    'Do not repeat scenery, history or actions, infer target distance from lidar minima, or invent changes. '
    'Finish within 48 tokens.'
)

TRAJECTORY_EVIDENCE_SYSTEM += """
Read action_spec: F/G move forward 10/5 cm, B/C move backward 10/5 cm when available.
Nominal turns: Q/E=left/right 11.25 degrees, L/R=left/right 22.5 degrees,
A/D=left/right 45 degrees. Each letter is ONE action and uses one trajectory slot.
Choose an angle that advances the remaining instruction: a small alignment adjustment or
a larger heading change. Use the actual angle in action_spec, not an assumed quarter-turn.
Translation options carry translation_label; follow that exact label and its step budget.
A short measured translation can reposition a rover in a tight space; a full long step is not required.
When oscillation_detected and no progress, change the maneuver rather than repeat inspection.
Retain escape_heading_commitment when still useful; a new measured obstacle or instruction may change it.
If collision_protection is disabled, geometry is advisory: actions are not removed by obstacle filtering.
You remain responsible for the instruction, scene interpretation and selecting collision-free actions.
"""


GROUNDED_SPATIAL_PROMPT = (
    'Assess the next metric subgoal in at most 28 words: Fact: decisive evidence; '
    'Next: TOTAL distance/angle and its stopping boundary. Read the finished camera observer, '
    'sensors, rover_body, maps, available_translations, immediate_mobility and measured history. '
    'For an object approach, explicitly distinguish measured object range from UNKNOWN object range. '
    'A closer-looking target or a lidar sector minimum does not establish its range or remaining gap. '
    'When a normal step fits the task and swept-body clearance, prefer a multi-step normal translation. '
    'Choose creep only for an evidenced short residual, nearby obstacle, tight fit or alignment; '
    'a small requested FINAL separation alone is insufficient. '
    'Check the exact turn prefix and its subsequent distance budget before proposing a corridor. '
    'Use recent geometry for immediate clearance and accumulated SLAM for routes/history. '
    'Keep useful heading progress; consider translation before another scan when it is justified. '
    'Name a specific reason to end early; otherwise plan useful progress within the horizon. '
    'Read sensor_grounding.execution: distinguish no translation attempted, software guard and measured undertravel. '
    'A clear scan can miss a visible low obstacle; image size alone cannot establish range. '
    'No scenery, task repetition or list of action letters.'
)


PIPELINE_FORECAST_RULES = """
When pipeline.speculative=true, prepare a CONTINUATION for the predicted boundary of the
currently executing sequence. pose and available_translations describe that FUTURE planning
origin, not a measured arrival. The RGB camera, map arrows, perception_pose, camera_pose,
measured_task_context are measured at capture time. task_context.current_pose and execution
totals are a CONDITIONAL forecast assuming the committed sequence finishes successfully.
Use that forecast to decide what remains AFTER it; do not repeat work already pending. Interpret camera
bearings from camera_pose, then reason in map coordinates about planned_start_pose. Do not
repeat committed_remaining_actions in the continuation; they belong to the current sequence.
pipeline.projected_execution_totals predicts nominal progress at the boundary; use it only
to avoid repeating pending work in the continuation, never as observed completion evidence.
Pending actions have not happened and cannot prove completion. Completion is checked separately
on a fresh measured observation. The executor may discard this proposal after any interruption,
pose mismatch, changed local geometry or meaningful new camera evidence.
"""

PIPELINE_SPATIAL_PROMPT = (
    'Reason ONLY about what to do AFTER committed_remaining_actions finish successfully. '
    'task_context is the conditional future state; measured_task_context is the actual past. '
    'Compare projected totals and planned_start_pose with ALL instruction requirements. '
    'If committed actions would finish them, propose S to obtain final measured verification; '
    'do not repeat the same distance or turn. Otherwise name the next remaining maneuver '
    'from the forecast origin using available_translations and maps. Camera evidence is from '
    'camera_pose, not the forecast pose. Give Fact and Next in at most 22 words.'
)


TRAJECTORY_EVIDENCE_SYSTEM += """
When long_term_memory is supplied, use its selected evidence and visited viewpoints across
older decisions. Event records are fallible visual interpretations with source IDs;
separate sightings may be different objects. Preserve uncertainty and check fresh camera
evidence when returning. camera_viewpoint is where the camera was, NOT the object's
position. Never invent an object's coordinates or range from a remembered camera pose.
Only current_anchor records have usable spatial anchors; previous-run/session records
provide semantic context only. Place edges record measured traversal, not a promise that
a straight segment is clear now. A blocked attempt need not be a permanent dead end.
Completion in another run, a proposed action or an old visual claim is not current progress.
"""


COMMITTED_COMPLETION_RULE = """Complete the entire arbitrary instruction, including requested movement, final
pose/range, timing and sequence. Count only measured executed progress. Preserve remaining
requirements and explicit approximation words. Unknown measurements, planned/discarded
steps and historical claims cannot establish precise arrival or completion."""

COMMITTED_RATIONALE_PROMPT = (
    'Explain the ALREADY COMMITTED plan in at most 16 words: Fact: decisive camera/sensor '
    'or measured-history evidence; Why: how the committed actions advance the unfinished task '
    'or resolve uncertainty. This is a rationale, not a new plan: do not propose alternative '
    'actions, modify its direction, length or stopping point, or narrate completed future travel. '
    'Read orientation.forward_xy/backward_xy. Map axes and image edges are not body directions. '
    'For reverse travel, the camera still faces forward and body heading stays unchanged. '
    'If geometry permits no translation, explain that limitation rather than recommending travel.'
)

TRAJECTORY_EVIDENCE_SYSTEM += """
Choose one authoritative primitive sequence from the currently valid choices. Its fixed
structured selected_plan drives both execution and display. A following spatial rationale
explains that committed plan; it cannot replace it. valid_choices lists admissible FIRST
primitives, not every capability in action_spec; admissibility is updated after each primitive.
Orientation.forward_xy/backward_xy are body travel directions in map coordinates. Image
coordinates increase right/down. body_heading_map_degrees is where the body faces;
travel_map_bearing_degrees is displacement direction, opposite for reverse without turning.
Translation options describe measured straight travel after alignment, including backward;
then_forward_steps is a legacy field name for the selected translation_label's step budget.
A correction to map->odom is not physical travel. Local execution/progress uses odom;
map poses describe global layout. Do not treat changed map coordinates alone as measured motion.
"""

RATIONALE_CONSISTENCY_PROMPT = (
    'RATIONALE FAITHFULNESS CHECK: Does the literal rationale just written faithfully describe '
    'the already committed plan above? Check forward/reverse, turn direction/angle, length and '
    'whether future movement is falsely described as completed. Do not judge whether a different '
    'plan would be better; only whether this explanation contradicts the fixed actions. '
    'A concise statement of uncertainty is allowed. Answer exactly Yes or No:'
)

COMMITTED_ASSESSMENT_PROMPT = (
    'In at most 16 words give the decisive measured spatial fact and uncertainty for the '
    'unfinished task. Read orientation, current valid_choices, available_translations and '
    'measured execution. No task repetition, scene narration, future action sequence or '
    'new maneuver recommendation: the following structured plan owns that decision.'
)
