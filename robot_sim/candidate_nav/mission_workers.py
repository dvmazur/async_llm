"""Run memory and subgoal selection alongside execution on the existing model link."""
import asyncio
import math
from collections import deque

from .mission_context import MEMORY, SUBGOAL, anchor, diverse_routes, local_routes, measured_event
from .mission_context import passage_goal, repeated_return, route_to_goal
from .model_transport import request_model


class MissionWorkers:
    def __init__(self, agent):
        self.agent = agent
        self.reference = None
        self.notes = deque(maxlen=8)
        self.goal = None
        self.minimum_goal_snapshot = 0
        self.last_checked_event = None
        self.jobs = {kind: asyncio.Queue(maxsize=1) for kind in (MEMORY, SUBGOAL)}

    def decorate(self, observation):
        """Expose the latest valid state without waiting for either worker."""
        reference = anchor(observation["memory_context"])
        if reference != self.reference:
            self.reference = reference
            self.notes.clear()
            self.goal = None
            self.last_checked_event = None
        measured = [p for p in self.agent.plan_history
                    if reference and anchor(p.get("memory_scope", {})) == reference]
        event = measured_event(measured[-1] if measured else None,
                               measured[-2] if len(measured) > 1 else None, observation["memory_context"])
        endpoints = [s["end_pose"] for p in measured for s in p["step_results"] if s.get("end_pose")]
        routes = local_routes(observation, endpoints) if reference else []
        step = None
        if self.goal:
            step = route_to_goal(observation, routes, self.goal)
            if step is None or event:
                self.agent.log("events", dict(event="subgoal_released", goal=self.goal,
                                             reason=event["kind"] if event else "reached_or_no_clear_route"))
                self.goal = None
                self.minimum_goal_snapshot = observation["snapshot_id"]
                step = None
        observation['passage_candidates'] = routes
        notes = sorted(self.notes, key=lambda n: math.dist(n["pose"][:2], observation["pose"][:2]))[:3]
        observation["mission_context"] = dict(sub_strategy=2, subgoal=self.goal, route=step,
                                              memory=notes, latest_measured_event=event)
        return (reference, event, diverse_routes(routes), notes)

    def submit(self, observation, reply, pending):
        if pending is None:
            return
        reference, event, routes, notes = pending
        if not reference or reference != self.reference:
            return
        base = {k: observation[k] for k in ("snapshot_id", "map_version", "instruction", "rover_id",
                                             "episode_id", "memory_context")}
        base.update(physical=True, notes=notes)
        new_event = event and self.last_checked_event != event["plan_id"]
        if new_event and not (event["kind"] == "returned" and repeated_return(event, self.notes)):
            self.enqueue(MEMORY, reference, dict(base, event=event))
        if reply.get("visual_task_scope") == "M":
            self.record_passage(observation, reply)
        elif self.goal is None and routes:
            self.enqueue(SUBGOAL, reference, dict(base, routes=routes,
                         view=reply.get("observer", "")[:280],
                         progress=dict(task_complete=False,
                                       executed=observation["task_context"].get("execution_totals", {}),
                                       latest_measured_event=event)))

    def record_passage(self, observation, reply):
        if reply.get('task_complete') or observation.get('collision_protection') == 'disabled':
            return
        goal = reply.get('passage_goal')
        current_leg = observation['mission_context'].get('route') or {}
        allowed = [passage_goal(route, observation['snapshot_id'])
                   for route in observation.get('passage_candidates', []) if route['translation_label'] in ('F', 'G')]
        if goal and (goal in allowed or goal == self.goal):
            if goal != self.goal:
                self.goal = goal
                self.agent.log('events', dict(event='mission_subgoal_selected', **goal))
        elif self.goal and (self.goal.get('kind') != 'passage' or current_leg):
            self.agent.log('events', dict(event='subgoal_released', goal=self.goal,
                                         reason='passage_not_selected'))
            self.goal = None
            self.minimum_goal_snapshot = observation['snapshot_id'] + 1

    def enqueue(self, protocol, reference, packet):
        queue = self.jobs[protocol]
        if queue.full():
            queue.get_nowait()
        queue.put_nowait((reference, dict(packet, protocol=protocol)))

    async def run_worker(self, protocol):
        agent = self.agent
        queue = self.jobs[protocol]
        while not agent.finished.is_set():
            reference, packet = await queue.get()
            # A request is admitted only during execution. Camera/plan requests
            # already in flight take precedence. Never queue behind their lock.
            started = asyncio.get_running_loop().time()
            while (reference == self.reference and not agent.finished.is_set()
                   and queue.empty() and asyncio.get_running_loop().time() - started < 4):
                if (agent.executing_sequence and agent.trajectory_progress
                        and agent.trajectory_progress["plan_id"] == packet["snapshot_id"]
                        and not agent.plan_request_pending and not agent.model_rpc_lock.locked()):
                    try:
                        if protocol == SUBGOAL:
                            # Include any memory review completed during this same trajectory.
                            nearby = packet["notes"]
                            recent = list(self.notes)[-2:]
                            packet["notes"] = (recent + [note for note in nearby if note not in recent])[:3]
                        async with agent.model_rpc_lock:
                            reply = await request_model(packet, 18766, timeout=8, socket_path=agent.args.model_socket)
                        self.accept(reference, packet, reply)
                    except Exception as error:
                        agent.log("events", dict(event="mission_worker_error", protocol=protocol,
                                                 snapshot_id=packet["snapshot_id"], error=str(error)))
                    break
                await asyncio.sleep(.04)

    def accept(self, reference, packet, reply):
        if (reference != self.reference or anchor(self.agent.memory_scope.export()) != reference
                or self.agent.finished.is_set()):
            return
        choice = reply["choice"]
        if packet["protocol"] == MEMORY:
            note = packet["event"]
            self.last_checked_event = note["plan_id"]
            if choice == "K" and note not in self.notes:
                self.notes.append(note)
            self.agent.log("events", dict(event="mission_memory_review", choice=choice, evidence=note,
                                         source=reply.get("source", "qwen")))
        elif (choice != "S" and self.goal is None
              and packet["snapshot_id"] >= self.minimum_goal_snapshot):
            route = packet["routes"][choice]
            self.goal = dict(goal_xy=route["goal_xy"], travel_bearing_deg=route["bearing_deg"],
                             body_heading_deg=route["body_heading_deg"], source_snapshot=packet["snapshot_id"])
            self.agent.log("events", dict(event="mission_subgoal_selected", **self.goal))
        else:
            self.agent.log("events", dict(event="mission_subgoal_deferred", snapshot_id=packet["snapshot_id"]))
