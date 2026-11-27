"""Download a pinned ProactiveVideoQA clip and decode frames into the HF cache."""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

REPO = "wangyueqian/ProactiveVideoQA"
REVISION = "7f59bce2a6cdf40b8a895767aac7a52a92136c5f"
VIDEO = "Explosion016_x264.mp4"
TIMES = (2.0, 5.0, 12.0, 14.0)
VIDEO_SHA256 = "0672b58c87ec43c15e03f5a543f1b86f81852bc34756a6eb6dabf9080b7de42c"


def cache_dir():
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / "video-understanding" / REVISION


def frame_path(frame):
    return cache_dir() / frame["path"]


@lru_cache(maxsize=1)
def prepare():
    import imageio_ffmpeg
    from filelock import FileLock
    from huggingface_hub import hf_hub_download
    from remotezip import RemoteZip

    work = cache_dir()
    work.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/VAD/"
    annotation = json.loads(
        Path(
            hf_hub_download(
                repo_id=REPO, repo_type="dataset", revision=REVISION, filename="VAD/anno.json"
            )
        ).read_text()
    )
    row = next(r for r in annotation if r["video"] == VIDEO)
    video = work / VIDEO
    member = None
    with FileLock(str(video) + ".lock"):
        if not video.exists():
            with RemoteZip(base + "videos.zip") as archive:
                member = next(i for i in archive.infolist() if Path(i.filename).name == VIDEO)
                if member.file_size > 25_000_000:
                    raise ValueError("Unexpectedly large video clip")
                payload = archive.read(member)
                if hashlib.sha256(payload).hexdigest() != VIDEO_SHA256:
                    raise ValueError("Source video SHA256 mismatch")
                video.write_bytes(payload)
        if hashlib.sha256(video.read_bytes()).hexdigest() != VIDEO_SHA256:
            raise ValueError("Cached video SHA256 mismatch")
    frames = []
    for i, seconds in enumerate(TIMES):
        output = work / f"frame_{i:02d}.png"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
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
        "zip_member": "videos/" + VIDEO,
        "frames": frames,
        "expected_first_reply_frame": 2,
        "description": "A short sampled-frame stream from the original video, not a full benchmark run.",
    }
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
