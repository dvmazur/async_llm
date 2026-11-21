"""CPU-only provenance for memory; never publishes motion or resets odometry."""

import hashlib, math, time, uuid
from pathlib import Path


def mapping_instance():
    """Identify the boot and actual ICP/SLAM processes, without calling ROS services."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = "unknown-boot"
    found = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            command = (
                (p / "cmdline").read_bytes().split(b"\0")[0].decode(errors="replace")
            )
            if Path(command).name in (
                "icp_odometry",
                "async_slam_toolbox_node",
                "sync_slam_toolbox_node",
            ):
                fields = (p / "stat").read_text().split(") ", 1)[1].split()
                found.append((Path(command).name, p.name, fields[19]))
        except (OSError, IndexError):
            continue
    names = {x[0] for x in found}
    if "icp_odometry" not in names or not names.intersection(
        ("async_slam_toolbox_node", "sync_slam_toolbox_node")
    ):
        return None
    return hashlib.sha256(repr((boot, sorted(found))).encode()).hexdigest()[:24]


class MemoryScope:
    def __init__(
        self,
        rover_id,
        episode,
        enabled=True,
        identity=mapping_instance,
        clock=time.monotonic,
    ):
        self.rover_id = rover_id
        self.episode = episode
        self.enabled = enabled
        self.identity = identity
        self.clock = clock
        self.session = None
        self.checked = -math.inf
        # Across-run anchors are intentionally unverified: a reset/lift may happen
        # while no agent is observing. Semantic evidence still survives in SQLite.
        self.epoch = uuid.uuid4().hex[:16]
        self.last = None
        self.map_transform = None
        self.valid = False
        self.reason = "waiting_for_odometry"

    def invalidate(self, reason):
        if self.valid:
            self.epoch = uuid.uuid4().hex[:16]
        self.valid = False
        self.reason = reason

    def observe(self, stamp, pose, valid):
        now = self.clock()
        if now - self.checked >= 2:
            session = self.identity()
            self.checked = now
            if session != self.session:
                self.invalidate("mapping_process_changed")
                self.session = session
                self.epoch = uuid.uuid4().hex[:16]
                self.last = None
        if not valid or not all(math.isfinite(v) for v in (stamp, *pose)):
            self.invalidate("invalid_odometry")
            return
        if self.last:
            old_stamp, old_pose, received = self.last
            dt = stamp - old_stamp
            angle = abs(
                math.atan2(
                    math.sin(pose[2] - old_pose[2]), math.cos(pose[2] - old_pose[2])
                )
            )
            if (
                dt < 0
                or now - received > 2
                or (
                    dt <= 1.25
                    and (
                        math.dist(pose[:2], old_pose[:2]) > 0.16 + 0.35 * max(dt, 0)
                        or angle > 0.40 + 1.2 * max(dt, 0)
                    )
                )
            ):
                self.invalidate("tracking_discontinuity")
                self.epoch = uuid.uuid4().hex[:16]
        self.last = (stamp, list(pose), now)
        self.valid = self.session is not None
        self.reason = "tracking" if self.valid else "mapping_identity_unknown"

    def observe_map_transform(self, transform):
        if not isinstance(transform, (list, tuple)) or len(transform) != 3:
            return
        if not all(math.isfinite(v) for v in transform):
            return
        if self.map_transform is not None:
            old = self.map_transform
            angle = abs(
                math.atan2(
                    math.sin(transform[2] - old[2]), math.cos(transform[2] - old[2])
                )
            )
            # Compare to the epoch baseline, so gradual SLAM corrections also
            # revoke anchors before historical poses become misleading.
            if math.dist(transform[:2], old[:2]) > 0.08 or angle > math.radians(5):
                self.epoch = uuid.uuid4().hex[:16]
                self.map_transform = list(transform)
        else:
            self.map_transform = list(transform)

    def export(self):
        if self.last and self.clock() - self.last[2] > 1:
            self.invalidate("stale_odometry")
        return dict(
            version=1,
            enabled=self.enabled,
            rover_id=self.rover_id,
            map_session_id=self.session,
            anchor_epoch=self.epoch,
            tracking_valid=self.valid,
            tracking_status=self.reason,
        )
