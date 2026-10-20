"""CPU regressions for Tutor cache ownership and reproducible MP4 timelines."""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from scripts.async_tutor.async_tutor.demo import Snapshot, TutorSession
from scripts.async_tutor.async_tutor.record import FONT, render


async def eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


class Backend:
    def __init__(self):
        self.observations, self.freed = [], []
        self.started, self.resume = asyncio.Event(), asyncio.Event()
        self.reply_closed = False

    async def inspect(self, image, revision, previous_notes, stopped, **kwargs):
        self.observations.append((revision, image))
        return Snapshot(revision, [len(self.observations)], f"work {revision}")

    async def free(self, blocks):
        for block in blocks:
            assert block not in self.freed, "double free"
            self.freed.append(block)

    async def reply(self, snapshot, history, question, stopped):
        self.started.set()
        try:
            yield "Check"
            await self.resume.wait()
            assert all(block not in self.freed for block in snapshot.blocks)
            if not stopped():
                yield "Check both sides."
        finally:
            self.reply_closed = True


@pytest.mark.parametrize("ending", ["completed", "interrupted", "disconnected"])
def test_latest_frame_and_snapshot_lifetime(ending):
    async def run():
        backend, events = Backend(), []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit)
        session.auto = False
        closing = None
        try:
            # A question waits for the camera; queued obsolete frames are dropped.
            session.submit_text("Help")
            session.submit_frame("obsolete")
            session.submit_frame("first")
            session.start()
            await asyncio.wait_for(backend.started.wait(), 2)
            assert backend.observations == [(2, "first")]
            assert any(e == {"type": "text", "text": "Check"} for e in events)
            session.submit_frame("second")
            await eventually(lambda: session.latest.revision == 3)
            assert backend.observations == [(2, "first"), (3, "second")]
            assert not backend.freed  # The reply still reads the old snapshot.
            if ending == "interrupted":
                session.interrupt()
            elif ending == "disconnected":
                closing = asyncio.create_task(session.close())
                await asyncio.sleep(0)
                assert not closing.done()  # Drain inference before freeing either cache.
                assert not backend.freed
            backend.resume.set()
            await eventually(lambda: any(e["type"] == "done" for e in events))
            assert events[-1] == {"type": "done", "interrupted": ending != "completed"}
            assert backend.reply_closed
            if ending == "completed":
                assert list(session.history) == [
                    {"role": "user", "content": "Help"},
                    {"role": "assistant", "content": "Check both sides."},
                ]
            else:
                assert not session.history
                assert [e["text"] for e in events if e["type"] == "text"] == ["Check"]
            if ending != "disconnected":
                assert backend.freed == [1]
        finally:
            backend.resume.set()
            await asyncio.wait_for(closing or session.close(), 2)
        for task in session.tasks:
            task.result()
        assert sorted(backend.freed) == [1, 2]

    asyncio.run(run())


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")) or not Path(FONT).is_file(),
    reason="FFmpeg, ffprobe, and DejaVu Sans are required for MP4 rendering",
)
@pytest.mark.parametrize("kind", ["real-video", "simulated-student"])
def test_recording_preserves_source_and_wall_clock_timeline(tmp_path, kind):
    # A leading green frame detects wrong seeks; red at 0s changes to blue at 0.5s.
    frames = [tmp_path / f"source-{i}.png" for i in range(3)]
    for path, color in zip(frames, ("lime", "red", "blue")):
        Image.new("RGB", (960, 640), color).save(path)
    source = tmp_path / "source.mkv"
    if kind == "real-video":
        subprocess.run(
            "ffmpeg -v error -nostdin -y -framerate 2 -i".split()
            + [str(tmp_path / "source-%d.png"), "-c:v", "ffv1", str(source)],
            check=True,
            timeout=30,
        )
    events = [
        {"t": 0, "type": "frame", "revision": 1, "file": str(frames[1])},
        {"t": 0.25, "type": "start", "revision": 1},
        {"t": 0.5, "type": "frame", "revision": 2, "file": str(frames[2])},
        {"t": 0.5, "type": "text", "text": "Check"},
        {"t": 0.75, "type": "text", "text": "Check both sides."},
        {"t": 1, "type": "observation", "revision": 2, "notes": "New work"},
        {"t": 1.5, "type": "done", "interrupted": True},
    ]
    event_file = tmp_path / "events.json"
    event_file.write_text(
        json.dumps(
            {
                "kind": kind,
                "title": "Timeline test",
                "duration": 2,
                "events": events,
                "source_file": str(source),
                "source_start": 0.5,
                "clip_duration": 1,
            }
        )
    )
    video = render(event_file, fps=2)
    probe = subprocess.run(
        "ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames -of json".split()
        + [str(video)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(probe.stdout)["streams"][0] == {
        "codec_name": "h264",
        "width": 1440,
        "height": 900,
        "r_frame_rate": "2/1",
        "nb_frames": "12",  # Two seconds of events plus four seconds of reading time.
    }
    decoded = subprocess.run(
        "ffmpeg -v error -nostdin -i".split()
        + [str(video)]
        + "-f rawvideo -pix_fmt rgb24 pipe:1".split(),
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    size = 1440 * 900 * 3
    assert len(decoded) == 12 * size
    images = [
        Image.frombytes("RGB", (1440, 900), decoded[i : i + size])
        for i in range(0, len(decoded), size)
    ]
    assert images[0].getpixel((200, 200))[0] > 230
    assert all(image.getpixel((200, 200))[2] > 230 for image in images[1:])
    # Streaming, cumulative text, and interruption occur at successive 0.5s frames;
    # the final source image is held after the video ends.
    chats = [image.crop((954, 116, 1406, 250)).tobytes() for image in images[:4]]
    assert all(before != after for before, after in zip(chats, chats[1:]))
