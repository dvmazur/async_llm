"""Continuous local odometry for physical primitive execution; no motor interface."""

import math
import time
import uuid
import threading


class LocalOdometry:
    def __init__(self, clock=time.monotonic):
        self.lock = threading.RLock()
        self.clock = clock
        self.epoch = uuid.uuid4().hex
        self.last = None
        self.valid = False
        self.reason = "waiting_for_odometry"
        self.session = None
        self.path_m = 0.0
        self.path_anchor = None

    def invalidate(self, reason):
        if self.valid:
            self.epoch = uuid.uuid4().hex
        self.valid = False
        self.reason = reason
        self.path_anchor = None

    def observe(self, *args, **kwargs):
        with self.lock:
            return self._observe(*args, **kwargs)

    def _observe(
        self, stamp, pose, valid=True, frame="odom", child="base_link", session=None
    ):
        now = self.clock()
        if (
            not valid
            or frame.lstrip("/") != "odom"
            or child.lstrip("/") != "base_link"
            or not all(math.isfinite(x) for x in (stamp, *pose))
        ):
            self.invalidate("invalid_local_odometry")
            return
        if self.last:
            old_stamp, old_pose, received = self.last
            dt = stamp - old_stamp
            angle = abs(
                math.atan2(
                    math.sin(pose[2] - old_pose[2]), math.cos(pose[2] - old_pose[2])
                )
            )
            changed = (
                session is not None
                and self.session is not None
                and session != self.session
            )
            if (
                changed
                or dt < 0
                or dt > 2
                or now - received > 2
                or (
                    dt <= 1.25
                    and (
                        math.dist(pose[:2], old_pose[:2]) > 0.16 + 0.35 * max(dt, 0)
                        or angle > 0.4 + 1.2 * max(dt, 0)
                    )
                )
            ):
                self.epoch = uuid.uuid4().hex
                self.path_anchor = None
        if self.path_anchor is None:
            self.path_anchor = list(pose)
        elif math.dist(pose[:2], self.path_anchor[:2]) >= 0.02:
            self.path_m += math.dist(pose[:2], self.path_anchor[:2])
            self.path_anchor = list(pose)
        if session is not None:self.session = session
        self.last = (stamp, list(pose), now)
        self.valid = True
        self.reason = "tracking"

    def snapshot(self, ros_now=None):
        with self.lock:
            return self._snapshot(ros_now)

    def _snapshot(self, ros_now=None):
        if self.last and self.clock() - self.last[2] > 0.5:
            self.invalidate("stale_local_odometry")
        if (
            self.last
            and ros_now is not None
            and not -0.15 <= ros_now - self.last[0] < 1.0
        ):
            self.invalidate("stale_local_odometry_stamp")
        return dict(
            frame="odom",
            epoch=self.epoch,
            valid=self.valid,
            status=self.reason,
            stamp=self.last[0] if self.last else None,
            pose=list(self.last[1]) if self.valid else None,
        )
