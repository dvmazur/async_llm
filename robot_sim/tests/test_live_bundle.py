"""Smoke-test the packaged rover agent without the source checkout or ROS."""
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def test_packaged_transport_and_mission_workers_import():
    bundle = Path(__file__).resolve().parents[1] / "scripts" / "rover_live_bundle.sh"
    archive = subprocess.check_output(["bash", str(bundle)])
    with tempfile.TemporaryDirectory() as root:
        with tarfile.open(fileobj=io.BytesIO(archive)) as package:
            package.extractall(root, filter="data")
        subprocess.run(
            [
                sys.executable, "-c",
                "from robot_sim.candidate_nav import model_transport, mission_workers",
            ],
            cwd=root,
            env={**os.environ, "PYTHONPATH": root},
            check=True,
            capture_output=True,
            text=True,
        )
