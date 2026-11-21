"""Persistent, evidence-backed CPU memory for the physical strategy-4 agent.

No Torch, motor API, autonomous steering, extra model call, or inferred target
coordinates. SQLite keeps the full event journal; bounded retrieval is model input.
"""

import asyncio, base64, hashlib, json, math, os, re, sqlite3, threading, time, uuid
from collections import deque
from pathlib import Path

from . import target_vision
from .mission_context import MEMORY

VERSION = 1
RULES = (
    "Historical visual records are fallible VLM claims. Viewpoint poses locate the CAMERA, "
    "not the object. Current visibility/range/completion need current evidence. "
    "Only current_anchor records may be used spatially; old-session/run anchors are unverified. "
    "Traversed edges record actual motion, not a currently clear straight path. "
    "Blocked attempts can be temporary. Plans and model completion claims are not measured execution."
)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def pose(value):
    return (
        [round(float(x), 3) for x in value[:3]]
        if isinstance(value, (list, tuple))
        and len(value) >= 3
        and all(isinstance(x, (int, float)) and math.isfinite(x) for x in value[:3])
        else None
    )


def thin(value):
    if isinstance(value, dict):
        return {
            k: thin(v)
            for k, v in value.items()
            if k not in ("image", "camera", "map_image", "accumulated_map_image")
        }
    if isinstance(value, list):
        return [thin(v) for v in value]
    return value


def terms(text):
    return set(re.findall(r"[^\W_]{3,}", text.casefold(), re.UNICODE))


def applicable(packet):
    scope = packet.get("memory_context", {})
    return (
        scope.get("version") == VERSION
        and scope.get("enabled", True)
        and packet.get("physical") is True
        and packet.get("protocol")
        in ("action_trajectory_v1", "motion_visual_memory_v1", MEMORY)
        and packet.get("rover_id") in (80, 107)
        and bool(packet.get("episode_id"))
    )


class EpisodeMemory:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.images = self.directory / "keyframes"
        self.images.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(
            self.directory / "episodes.sqlite3", timeout=1, check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        if self.db.execute("PRAGMA user_version").fetchone()[0] not in (0, VERSION):
            raise RuntimeError("Unsupported memory schema")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs(
          id TEXT PRIMARY KEY, rover INTEGER, namespace TEXT, task TEXT, instruction TEXT,
          session TEXT, anchor TEXT, updated REAL, state TEXT, current_place INTEGER);
        CREATE TABLE IF NOT EXISTS journal(
          key TEXT PRIMARY KEY, run TEXT, protocol TEXT, packet TEXT, reply TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS frames(
          key TEXT PRIMARY KEY, run TEXT, frame TEXT, stamp REAL, session TEXT, anchor TEXT,
          pose TEXT, image TEXT);
        CREATE TABLE IF NOT EXISTS events(
          id INTEGER PRIMARY KEY, key TEXT UNIQUE, rover INTEGER, namespace TEXT, run TEXT,
          task TEXT, kind TEXT, text TEXT, stamp REAL, session TEXT, anchor TEXT,
          pose TEXT, image TEXT, source TEXT, payload TEXT);
        CREATE INDEX IF NOT EXISTS memory_task ON events(rover,namespace,task,kind,id DESC);
        CREATE TABLE IF NOT EXISTS event_terms(term TEXT,event INTEGER,PRIMARY KEY(term,event));
        CREATE INDEX IF NOT EXISTS memory_recent ON events(rover,namespace,id DESC);
        CREATE TABLE IF NOT EXISTS plans(key TEXT PRIMARY KEY,run TEXT,plan TEXT);
        CREATE TABLE IF NOT EXISTS places(
          id INTEGER PRIMARY KEY,rover INTEGER,namespace TEXT,anchor TEXT,
          x REAL,y REAL,yaw REAL,visits INTEGER,last_event INTEGER);
        CREATE INDEX IF NOT EXISTS memory_places ON places(rover,namespace,anchor,x,y);
        CREATE TABLE IF NOT EXISTS edges(
          a INTEGER,b INTEGER,traversals INTEGER,source INTEGER,PRIMARY KEY(a,b));
        PRAGMA user_version=1;
        """)
        self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()

    def identity(self, p):
        s = p["memory_context"]
        namespace = "replay" if s.get("replay") else "live"
        run = f"{namespace}:{p['rover_id']}:{p['episode_id']}"
        task = hashlib.sha256(p["instruction"].encode()).hexdigest()
        session = s.get("map_session_id") or "unknown"
        anchor = (
            digest([p["rover_id"], session, s.get("anchor_epoch"), run])
            if s.get("tracking_valid")
            and s.get("map_session_id")
            and s.get("anchor_epoch")
            else None
        )
        return dict(
            run=run,
            rover=p["rover_id"],
            namespace=namespace,
            task=task,
            session=session,
            anchor=anchor,
        )

    def image(self, encoded):
        if not isinstance(encoded, str) or len(encoded) > 3_000_000:
            return None
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            return None
        extension = (
            ".png"
            if data.startswith(b"\x89PNG\r\n\x1a\n")
            else ".jpg" if data.startswith(b"\xff\xd8") else None
        )
        if not extension:
            return None
        name = hashlib.sha256(data).hexdigest() + extension
        target = self.images / name
        if not target.exists():
            stage = target.with_suffix(extension + "." + uuid.uuid4().hex + ".tmp")
            stage.write_bytes(data)
            os.replace(stage, target)
        return name

    def event(
        self,
        ident,
        kind,
        text,
        stamp,
        viewpoint=None,
        image=None,
        source=None,
        payload=None,
        key=None,
    ):
        payload = payload or {}
        key = key or digest([ident["run"], kind, stamp, source, payload])
        inserted = self.db.execute(
            "INSERT OR IGNORE INTO events(key,rover,namespace,run,task,kind,text,stamp,session,anchor,pose,image,source,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                ident["rover"],
                ident["namespace"],
                ident["run"],
                ident["task"],
                kind,
                str(text)[:4000],
                float(stamp),
                ident["session"],
                ident["anchor"],
                dumps(pose(viewpoint)),
                image,
                dumps(source),
                dumps(payload),
            ),
        )
        event_id = self.db.execute(
            "SELECT id FROM events WHERE key=?", (key,)
        ).fetchone()[0]
        if inserted.rowcount:
            self.db.executemany(
                "INSERT OR IGNORE INTO event_terms VALUES(?,?)",
                [(t, event_id) for t in sorted(terms(str(text)))[:128]],
            )
        return event_id

    def place(self, ident, position, last_event=None):
        position = pose(position)
        if not ident["anchor"] or position is None:
            return None
        x, y, yaw = position
        rows = self.db.execute(
            "SELECT * FROM places WHERE rover=? AND namespace=? AND anchor=? AND x BETWEEN ? AND ? AND y BETWEEN ? AND ?",
            (
                ident["rover"],
                ident["namespace"],
                ident["anchor"],
                x - 0.12,
                x + 0.12,
                y - 0.12,
                y + 0.12,
            ),
        ).fetchall()
        nearby = [r for r in rows if math.hypot(x - r["x"], y - r["y"]) <= 0.12]
        if nearby:
            return min(nearby, key=lambda r: math.hypot(x - r["x"], y - r["y"]))["id"]
        c = self.db.execute(
            "INSERT INTO places(rover,namespace,anchor,x,y,yaw,visits,last_event) VALUES(?,?,?,?,?,?,0,?)",
            (
                ident["rover"],
                ident["namespace"],
                ident["anchor"],
                x,
                y,
                yaw,
                last_event,
            ),
        )
        return c.lastrowid

    def visit(self, ident, position, event=None):
        node = self.place(ident, position, event)
        if node is None:
            return
        run = self.db.execute(
            "SELECT current_place FROM runs WHERE id=?", (ident["run"],)
        ).fetchone()
        old = run[0] if run else None
        if old != node:
            self.db.execute(
                "UPDATE places SET visits=visits+1,last_event=? WHERE id=?",
                (event, node),
            )
            if old is not None:
                prior = self.db.execute(
                    "SELECT anchor FROM places WHERE id=?", (old,)
                ).fetchone()
                if prior and prior[0] == ident["anchor"]:
                    self.db.execute(
                        "INSERT INTO edges VALUES(?,?,1,?) ON CONFLICT(a,b) DO UPDATE SET traversals=traversals+1,source=excluded.source",
                        (old, node, event),
                    )
            self.db.execute(
                "UPDATE runs SET current_place=? WHERE id=?", (node, ident["run"])
            )

    def ingest(self, p, ident):
        state = dict(
            measured_totals=p.get("task_context", {}).get("execution_totals", {}),
            initial_pose=p.get("task_context", {}).get("initial_pose"),
            current_pose=p.get("pose"),
            visibility=(p.get('temporal_visual_state') or {}).get('target', {}).get('status')
            if target_vision.tracking_enabled(p) else p.get("temporal_visual_state", {}).get("relevant_visible"),
        )
        conflict = (
            "DO NOTHING"
            if p["protocol"] == "motion_visual_memory_v1"
            else "DO UPDATE SET updated=excluded.updated,state=excluded.state,session=excluded.session,anchor=excluded.anchor"
        )
        self.db.execute(
            "INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,NULL) ON CONFLICT(id) "
            + conflict,
            (
                ident["run"],
                ident["rover"],
                ident["namespace"],
                ident["task"],
                p["instruction"],
                ident["session"],
                ident["anchor"],
                time.time(),
                dumps(state),
            ),
        )
        if p["protocol"] != "action_trajectory_v1":
            return
        plan = p.get("memory_update", {}).get("previous_plan") or p.get(
            "previous_plan_outcome"
        )
        if plan and plan.get("plan_id") is not None:
            plan_scope = plan.get("memory_scope")
            pi = (
                self.identity(
                    {
                        **p,
                        "memory_context": {
                            **plan_scope,
                            "replay": p["memory_context"].get("replay", False),
                        },
                    }
                )
                if plan_scope
                else {**ident, "anchor": None}
            )
            key = digest([ident["run"], plan["plan_id"]])
            new = self.db.execute(
                "INSERT OR IGNORE INTO plans VALUES(?,?,?)",
                (key, ident["run"], dumps(plan)),
            ).rowcount
            if new:
                event = self.event(
                    pi,
                    "execution",
                    "Measured trajectory outcome",
                    p.get("sim_time", 0),
                    plan.get("end_pose"),
                    source={"plan_id": plan["plan_id"]},
                    payload=plan,
                    key=key,
                )
                if pi["anchor"] and pi["anchor"] == ident["anchor"]:
                    self.visit(pi, plan.get("start_pose"), event)
                steps = plan.get("step_results", [])
                for step in steps:
                    if step.get("outcome") in (
                        "trajectory_collision",
                        "lidar_brake",
                        "insufficient_progress_or_deviation",
                    ):
                        self.event(
                            pi,
                            "blocked_attempt",
                            "A measured attempt was blocked or made insufficient progress; not proof of a permanent dead end",
                            p.get("sim_time", 0),
                            step.get("start_pose") or plan.get("start_pose"),
                            source={"plan_id": plan["plan_id"]},
                            payload=step,
                        )
                    # Only measured step endpoints create graph edges. Planned or
                    # discarded actions, and inferred straight paths, never do.
                    if (
                        pi["anchor"]
                        and pi["anchor"] == ident["anchor"]
                        and pose(step.get("end_pose"))
                    ):
                        self.visit(pi, step["end_pose"], event)
        current = self.db.execute(
            "SELECT p.anchor FROM runs r LEFT JOIN places p ON r.current_place=p.id WHERE r.id=?",
            (ident["run"],),
        ).fetchone()
        if not current or current[0] != ident["anchor"]:
            self.visit(ident, p.get("pose"))

    def journal_key(self, p, ident):
        return digest([ident["run"], p["protocol"], p.get("snapshot_id")])

    def record_request(self, p):
        """Commit the complete text ledger before optional prompt retrieval/inference."""
        if not applicable(p):
            return
        with self.lock, self.db:
            ident = self.identity(p)
            if self.db.execute(
                "SELECT 1 FROM journal WHERE key=?", (self.journal_key(p, ident),)
            ).fetchone():
                return
            self.ingest(p, ident)
            ledger = thin(p)
            if p.get("camera"):
                ledger["camera_keyframe"] = self.image(p["camera"])
            self.db.execute(
                "INSERT OR IGNORE INTO journal VALUES(?,?,?,?,NULL,?)",
                (
                    self.journal_key(p, ident),
                    ident["run"],
                    p["protocol"],
                    dumps(ledger),
                    time.time(),
                ),
            )

    def record_mission_note(self, packet):
        note = packet["event"]
        with self.lock, self.db:
            ident = self.identity(packet)
            if ident["anchor"]:
                description = ("Measured return to a recently visited position" if note["kind"] == "returned"
                               else "Measured obstruction or insufficient progress")
                self.event(ident, "mission_note", description, time.time(), note["pose"],
                           source={"plan_id": note["plan_id"]}, payload=note,
                           key=digest([ident["run"], "mission_note", note["plan_id"]]))

    def record_reply(self, p, r):
        if not applicable(p):
            return
        self.record_request(p)
        with self.lock, self.db:
            ident = self.identity(p)
            self.db.execute(
                "UPDATE journal SET reply=? WHERE key=?",
                (dumps(thin(r)), self.journal_key(p, ident)),
            )
            if r.get("error"):
                return
            if p["protocol"] == "motion_visual_memory_v1":
                self.record_motion(p, r, ident)
            elif target_vision.tracking_enabled(p):
                state = r.get('target_visual_state') or {}
                if target_vision.is_split_state(state) and not r.get('observer_reused'):
                    frame = target_vision.snapshot_frame(p)
                    target = state['target']
                    report = dict(status=target['status'], scene=state['scene'])
                    sighting = target.get('last_seen') or {}
                    if sighting.get('frame_id') == frame['frame_id'] and sighting.get('stamp') == frame['stamp']:
                        report.update(appearance=sighting['appearance'], bbox_pct=sighting['bbox_pct'])
                    self.record_visual_channels(ident, frame, report, p.get('camera'))
            elif r.get("observer") and not r.get("observer_reused"):
                self.event(
                    ident,
                    "visual_report",
                    r["observer"],
                    p.get("camera_stamp", p.get("sim_time", 0)),
                    p.get("camera_pose", p.get("pose")),
                    image=self.image(p.get("camera")),
                    source={
                        "snapshot_id": p.get("snapshot_id"),
                        "type": "camera_observer",
                    },
                    payload={"interpretation": "unverified VLM report"},
                    key=digest([ident["run"], "camera_report", p.get("snapshot_id")]),
                )

    def record_visual_channels(self, ident, frame, report, image):
        key = [ident['run'], 'target_scene_v2', frame['frame_id']]
        source = dict(frame_id=frame['frame_id'], plan_id=frame.get('plan_id'))
        if report['status'] == 'visible' and target_vision.valid_box(report.get('bbox_pct')):
            self.event(ident, 'target_sighting', report['appearance'], frame['stamp'],
                       frame.get('pose'), self.image(image), source=source,
                       payload=dict(bbox_pct=report['bbox_pct'], object_position='not_measured'),
                       key=digest(key + ['target']))
        if report['scene'] != 'unknown':
            self.event(ident, 'scene_observation', report['scene'], frame['stamp'],
                       frame.get('pose'), source=source, key=digest(key + ['scene']))

    def record_motion(self, p, r, ident):
        if target_vision.tracking_enabled(p):
            frames = {frame['frame_id']: frame for frame in p['frames']}
            for report in r.get('reports', []):
                frame = frames[report['frame']['frame_id']]
                scope = frame.get('memory_scope')
                fi = self.identity({**p, 'memory_context': {
                    **scope, 'replay': p['memory_context'].get('replay', False)}}) if scope else ident
                self.record_visual_channels(fi, frame, report, frame.get('image'))
            return
        events = {e["frame"]["frame_id"]: e for e in r.get("events", [])}
        probes = {e["frame_id"]: e.get("event") for e in r.get("probes", [])}
        previous = p.get("visual_state", {})
        visible = previous.get("relevant_visible")
        description = previous.get("description", "unknown")
        for frame in p.get("frames", []):
            # Frames can be queued across an epoch change. Use capture provenance,
            # not the current request's scope, for each historical camera pose.
            fi = dict(ident)
            if frame.get("memory_scope"):
                fi = self.identity(
                    {
                        **p,
                        "memory_context": {
                            **frame["memory_scope"],
                            "replay": p["memory_context"].get("replay", False),
                        },
                    }
                )
            label = probes.get(frame["frame_id"], "U")
            event = events.get(frame["frame_id"], {})
            if label == "A":
                visible = True
                description = event.get("description", "uncertain appearance")
            elif label == "D":
                visible = False
            elif label == "U":
                visible = None
            elif label == "C":
                description = event.get("description", description)
            picture = (
                self.image(frame.get("image"))
                if visible is True or label in ("A", "D", "C")
                else None
            )
            key = digest([ident["run"], "frame", frame["frame_id"]])
            self.db.execute(
                "INSERT OR IGNORE INTO frames VALUES(?,?,?,?,?,?,?,?)",
                (
                    key,
                    ident["run"],
                    str(frame["frame_id"]),
                    frame["stamp"],
                    fi["session"],
                    fi["anchor"],
                    dumps(pose(frame.get("pose"))),
                    picture,
                ),
            )
            if label in ("A", "D", "C", "U") or visible is True:
                kind = (
                    "sighting"
                    if visible is True
                    else "disappearance" if label == "D" else "visual_change"
                )
                text = (
                    description
                    if kind == "sighting"
                    else event.get("description", "Visual evidence uncertain")
                )
                self.event(
                    fi,
                    kind,
                    text,
                    frame["stamp"],
                    frame.get("pose"),
                    picture,
                    {
                        "frame_id": frame["frame_id"],
                        "plan_id": frame.get("plan_id"),
                        "probe": label,
                    },
                    {
                        "status": "model_reported",
                        "visible_after": visible,
                        "object_position": "not_measured",
                    },
                    key=digest([ident["run"], "vision", frame["frame_id"], kind]),
                )

    def selected(self, p, ident, count):
        base = (ident["rover"], ident["namespace"])
        # Pinned earliest and latest sightings preserve initial appearance as
        # well as recency; inconsistent candidates remain separate hypotheses.
        split = target_vision.tracking_enabled(p)
        sighting_kind = 'target_sighting' if split else 'sighting'
        pinned = []
        for order in ("ASC", "DESC"):
            row = self.db.execute(
                f'SELECT * FROM events WHERE rover=? AND namespace=? AND task=? AND kind=? ORDER BY id {order} LIMIT 1',
                (*base, ident["task"], sighting_kind),
            ).fetchone()
            if row and row["id"] not in [x["id"] for x in pinned]:
                pinned.append(row)
        if not pinned and not split:
            # A stationary initial camera observer can discover the task object
            # before any inter-step motion probe. Keep that report too, without
            # pretending it is confirmed target recognition.
            for order in ("ASC", "DESC"):
                row = self.db.execute(
                    f'SELECT * FROM events WHERE rover=? AND namespace=? AND task=? AND kind="visual_report" ORDER BY id {order} LIMIT 1',
                    (*base, ident["task"]),
                ).fetchone()
                if row and row["id"] not in [x["id"] for x in pinned]:
                    pinned.append(row)
        kinds = ('target_sighting', 'scene_observation', 'blocked_attempt') if split else (
            'sighting', 'visual_report', 'disappearance', 'visual_change', 'blocked_attempt')
        if split and p.get('mission_context'):
            kinds = (*kinds, 'mission_note')
        placeholders = ','.join('?' for _ in kinds)
        candidates = self.db.execute(
            f'SELECT * FROM events WHERE rover=? AND namespace=? AND kind IN ({placeholders}) ORDER BY id DESC LIMIT 160',
            (*base, *kinds),
        ).fetchall()
        if split:
            candidates = [row for row in candidates
                          if (row['kind'] != 'target_sighting' or row['task'] == ident['task'])
                          and (row['kind'] not in ('scene_observation', 'mission_note') or row['run'] == ident['run'])]
        query = terms(p["instruction"])
        position = pose(p.get("pose"))

        def score(row):
            match = len(query & terms(row["text"]))
            rank = 8 * (row["task"] == ident["task"]) + 2 * match
            if row["anchor"] and row["anchor"] == ident["anchor"] and position:
                point = json.loads(row["pose"])
                if point:
                    rank += 3 / (1 + math.dist(position[:2], point[:2]))
            return rank, row["id"]

        lexical = []
        if query and not split:
            tokens = sorted(query, key=len, reverse=True)[:8]
            placeholders = ",".join("?" for _ in tokens)
            lexical = self.db.execute(
                "SELECT e.*,count(*) AS matches FROM event_terms t JOIN events e ON e.id=t.event "
                "WHERE e.rover=? AND e.namespace=? AND t.term IN ("
                + placeholders
                + ") "
                'AND e.kind IN ("sighting","visual_report","visual_change") '
                "GROUP BY e.id ORDER BY matches DESC,e.id DESC LIMIT 4",
                (*base, *tokens),
            ).fetchall()
        candidates = sorted(candidates, key=score, reverse=True)
        chosen = list(pinned)
        # Reserve retrieval capacity for old lexical matches, not just the tail.
        for row in list(lexical[:2]) + candidates:
            if len(chosen) >= count:
                break
            if row["id"] not in {x["id"] for x in chosen}:
                chosen.append(row)
        result = []
        for row in chosen[:count]:
            trusted = bool(
                ident["anchor"]
                and row["anchor"] == ident["anchor"]
                and row["session"] == ident["session"]
            )
            item = dict(
                event_id=row["id"],
                kind=row["kind"],
                report=row["text"][:500],
                source=json.loads(row["source"]),
                same_task=row["task"] == ident["task"],
                same_run=row["run"] == ident["run"],
                spatial_status=(
                    "current_anchor" if trusted else "unverified_previous_scope"
                ),
                has_keyframe=bool(row["image"]),
                report_is_model_interpretation=row["kind"] not in ("blocked_attempt", "mission_note"),
            )
            if len(row["text"]) > 500:
                item["report_is_excerpt"] = True
            if row["kind"] == "mission_note" and trusted:
                item["measured_event"] = json.loads(row["payload"])
            elif trusted and json.loads(row["pose"]):
                item["camera_viewpoint"] = json.loads(row["pose"])
            result.append(item)
        return result

    def places_context(self, p, ident):
        if not ident["anchor"]:
            return dict(available=False, reason="spatial_scope_unverified")
        position = pose(p.get("pose"))
        if not position:
            return dict(available=False, reason="pose_unavailable")
        rows = self.db.execute(
            "SELECT * FROM places WHERE rover=? AND namespace=? AND anchor=? ORDER BY id DESC LIMIT 400",
            (ident["rover"], ident["namespace"], ident["anchor"]),
        ).fetchall()
        nearest = sorted(
            rows, key=lambda r: math.hypot(position[0] - r["x"], position[1] - r["y"])
        )[:4]
        nodes = [
            dict(
                place_id=r["id"],
                viewpoint=[round(r["x"], 3), round(r["y"], 3), round(r["yaw"], 3)],
                arrivals=r["visits"],
            )
            for r in nearest
        ]
        ids = {r["id"] for r in nearest}
        edges = []
        for edge in self.db.execute(
            "SELECT e.* FROM edges e JOIN places p ON e.a=p.id WHERE p.anchor=? ORDER BY e.rowid DESC LIMIT 24",
            (ident["anchor"],),
        ):
            if edge["a"] in ids and edge["b"] in ids:
                edges.append(
                    dict(
                        from_place=edge["a"],
                        to_place=edge["b"],
                        measured_traversals=edge["traversals"],
                    )
                )
        route = None
        last = self.db.execute(
            'SELECT pose FROM events WHERE rover=? AND namespace=? AND anchor=? AND task=? AND kind=? ORDER BY id DESC LIMIT 1',
            (ident["rover"], ident["namespace"], ident["anchor"], ident["task"],
             'target_sighting' if target_vision.tracking_enabled(p) else 'sighting'),
        ).fetchone()
        if last and json.loads(last[0]) and rows:
            target = json.loads(last[0])
            goal = min(rows, key=lambda r: math.dist(target[:2], [r["x"], r["y"]]))
            if math.dist(target[:2], [goal["x"], goal["y"]]) <= 0.15:
                start = nearest[0]["id"]
                known = {r["id"]: r for r in rows}
                adj = {k: [] for k in known}
                for edge in self.db.execute(
                    "SELECT e.* FROM edges e JOIN places p ON e.a=p.id WHERE p.anchor=?",
                    (ident["anchor"],),
                ):
                    if edge["a"] in known and edge["b"] in known:
                        adj[edge["a"]].append(edge["b"])
                queue = deque([start])
                parents = {start: None}
                while queue and goal["id"] not in parents:
                    current = queue.popleft()
                    for node in adj[current]:
                        if node not in parents:
                            parents[node] = current
                            queue.append(node)
                if goal["id"] in parents:
                    path = []
                    node = goal["id"]
                    while node is not None:
                        path.append(node)
                        node = parents[node]
                    path.reverse()
                    route = dict(
                        destination="historical camera viewpoint, not object location",
                        measured_directed_route_points=len(path),
                        prefix_viewpoints=[
                            [known[k]["x"], known[k]["y"]] for k in path[:8]
                        ],
                    )
        return dict(
            available=True,
            nodes=nodes,
            edges=edges[:6],
            route_to_last_sighting=route,
            meaning="camera/rover viewpoint clusters, not recognized intersections or target locations",
        )

    def prepare(self, p, count_tokens, budget=1000):
        if not applicable(p) or p["protocol"] != "action_trajectory_v1":
            return p
        self.record_request(p)
        started = time.perf_counter()
        with self.lock, self.db:
            ident = self.identity(p)
            self.ingest(p, ident)
            records = self.selected(p, ident, 8)
            places = self.places_context(p, ident)
            mission = dict(
                progress_source="task_context.execution_totals; current run only",
                completion_status="requires current evidence; historical claims do not complete this run",
                last_measured_outcome=(p.get("previous_plan_outcome") or {}).get(
                    "outcome"
                ),
            )
            memory = dict(
                version=VERSION,
                rules=RULES,
                mission=mission,
                evidence=records,
                places=places,
            )
            # Bound actual tokenizer cost; oldest optional retrieval goes first.
            # The first two sightings are protected until an essential-budget failure.
            while count_tokens(dumps(memory)) > budget and len(memory["evidence"]) > 2:
                memory["evidence"].pop()
            if count_tokens(dumps(memory)) > budget:
                memory["places"] = dict(available=False, reason="retrieval_budget")
            while count_tokens(dumps(memory)) > budget and memory["evidence"]:
                record = memory["evidence"][-1]
                if len(record["report"]) > 120:
                    record["report"] = record["report"][:120]
                    record["report_is_excerpt"] = True
                else:
                    memory["evidence"].pop()
            if count_tokens(dumps(memory)) > budget:
                # Never duplicate/truncate a long arbitrary instruction: it is
                # already present in the main, mandatory user message.
                memory["mission"].pop("instruction", None)
            if count_tokens(dumps(memory)) > budget:
                memory = dict(version=VERSION, rules=RULES, evidence=[])
            result = dict(p)
            result["long_term_memory"] = memory
            context = dict(p.get("task_context") or {})
            maneuver = dict(p.get("maneuver_memory") or {})
            selection = dict(
                full_visual_events=len(p.get("motion_visual_memory", [])),
                full_recent_plans=len(maneuver.get("recent_plans", [])),
                full_recent_actions=len(context.get("recent_actions", [])),
            )
            # Detail is moved to durable storage/retrieval, never discarded from
            # the journal. Preserve current safety facts and execution totals.
            result["motion_visual_memory"] = list(p.get("motion_visual_memory", []))[
                -2:
            ]
            if "recent_plans" in maneuver:
                maneuver["recent_plans"] = maneuver["recent_plans"][-2:]
            result["maneuver_memory"] = maneuver
            if "recent_actions" in context:
                context["recent_actions"] = context["recent_actions"][-4:]
            result["task_context"] = context
            if p.get("recent_outcomes") == p.get("task_context", {}).get(
                "recent_actions"
            ):
                result["recent_outcomes"] = context.get("recent_actions", [])
            selection.update(
                selected_evidence=len(memory["evidence"]),
                memory_tokens=count_tokens(dumps(memory)),
                stored_events=self.db.execute(
                    "SELECT count(*) FROM events WHERE rover=? AND namespace=?",
                    (ident["rover"], ident["namespace"]),
                ).fetchone()[0],
                prepare_ms=round((time.perf_counter() - started) * 1000, 3),
                status="ready",
                storage="sqlite_wal",
                spatial_scope="current_run_and_tracking_epoch",
                extra_llm_calls=0,
            )
            result["_memory_status"] = selection
            return result


class MemoryCoordinator:
    """Rank-zero admission and persistence. Storage failures never stop control.

    All filesystem/SQLite work runs on a CPU worker outside the GPU schedule.
    Prepared packets are broadcast once by the existing TP dispatcher.
    """

    def __init__(self, directory, count_tokens, budget=1000, factory=EpisodeMemory):
        self.directory = directory
        self.count_tokens = count_tokens
        self.budget = budget
        self.factory = factory
        self.store = None
        self.retry_after = 0

    async def prepare(self, packet):
        # Only the server may supply retrieved memory. Reject stale/client copies.
        clean = {
            k: v
            for k, v in packet.items()
            if k not in ("long_term_memory", "_memory_status")
        }
        if not applicable(clean):
            return clean

        def work():
            started = time.perf_counter()
            if self.store is None:
                if time.monotonic() < self.retry_after:
                    raise OSError("memory retry backoff")
                self.store = self.factory(self.directory)
            if clean["protocol"] == "motion_visual_memory_v1":
                self.store.record_request(clean)
            prepared = self.store.prepare(clean, self.count_tokens, self.budget)
            status = prepared.get(
                "_memory_status", dict(status="recorded", extra_llm_calls=0)
            )
            status["prepare_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return {**prepared, "_memory_status": status}

        try:
            return await asyncio.to_thread(work)
        except Exception as error:
            self.retry_after = time.monotonic() + 30
            print("MEMORY_WARNING " + str(error), flush=True)
            return {
                **clean,
                "_memory_status": dict(
                    status="unavailable", error=str(error), extra_llm_calls=0
                ),
            }

    async def record(self, packet, reply):
        if packet.get("protocol") == MEMORY:
            if applicable(packet) and self.store is not None and reply.get("choice") == "K":
                try:
                    await asyncio.to_thread(self.store.record_mission_note, packet)
                except Exception as error:
                    print("MEMORY_WARNING mission note: " + str(error), flush=True)
            return
        if not applicable(packet) or self.store is None:
            return
        try:
            await asyncio.to_thread(self.store.record_reply, packet, reply)
        except Exception as error:
            print("MEMORY_WARNING record: " + str(error), flush=True)

    async def close(self):
        if self.store is not None:
            try:
                await asyncio.to_thread(self.store.close)
            except Exception as error:
                print("MEMORY_WARNING close: " + str(error), flush=True)
