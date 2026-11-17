"""Fetch one ProactiveVideoQA ZIP member and decode a few genuine video frames.

CPU-only optional preparation; the notebook uses the bundled PNG frames.
Requires remotezip and ffmpeg (only for rerunning preparation, not training).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO = "wangyueqian/ProactiveVideoQA"
REVISION = "7f59bce2a6cdf40b8a895767aac7a52a92136c5f"
VIDEO = "Explosion016_x264.mp4"
TIMES = (2.0, 5.0, 12.0, 14.0)
ROOT = Path(__file__).resolve().parents[4]
ASSETS = Path(__file__).with_name("assets")


def prepare():
    import urllib.request

    from remotezip import RemoteZip

    work = ROOT / "training_runs/video_understanding/source"
    work.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/VAD/"
    annotation = json.load(urllib.request.urlopen(base + "anno.json"))
    row = next(r for r in annotation if r["video"] == VIDEO)
    video = work / VIDEO
    member = None
    if not video.exists():
        with RemoteZip(base + "videos.zip") as archive:
            member = next(i for i in archive.infolist() if Path(i.filename).name == VIDEO)
            print("Selected ZIP member:", member.filename, "bytes:", member.file_size, flush=True)
            if member.file_size > 25_000_000:
                raise ValueError("Unexpectedly large clip; do not download it blindly")
            video.write_bytes(archive.read(member))
    frames = []
    for i, seconds in enumerate(TIMES):
        output = ASSETS / f"frame_{i:02d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(seconds),
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=448:-2",
                "-update",
                "1",
                str(output),
            ],
            check=True,
        )
        frames.append(
            {
                "time": seconds,
                "path": output.name,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "dataset": REPO,
        "revision": REVISION,
        "subset": "VAD",
        "annotation": row,
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "video_bytes": video.stat().st_size,
        "archive_url": base + "videos.zip",
        "zip_member": member.filename if member else VIDEO,
        "frames": frames,
        "expected_first_reply_frame": 2,
        "description": "A short sampled-frame stream from the original video, not a full benchmark run.",
    }
    (ASSETS / "task.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
