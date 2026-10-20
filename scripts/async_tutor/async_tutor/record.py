"""Record Tutor sessions from lesson clips or a text-only simulated student.

MP4s preserve wall-clock timing and include timestamped JSON transcripts.
Use --download for lesson sources, --render for saved events, or --report for HTML.
"""

import argparse
import asyncio
import hashlib
import html
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .demo import AsyncLLMBackend, FrameGate, TutorSession, create_llm

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WIDTH, HEIGHT = 1440, 900

SAMPLES = [
    {
        "name": "equations",
        "title": "Solving equations",
        "guid": "rzYX1f1U",
        "source": "https://corbettmaths.com/2012/08/24/solving-equations/",
    },
    {
        "name": "fractions",
        "title": "Adding and subtracting fractions",
        "guid": "WOp1z2mV",
        "source": "https://corbettmaths.com/2012/08/21/fractions-addition-and-subtraction/",
    },
    {
        "name": "quadratics",
        "title": "Solving quadratics by factorising",
        "guid": "KwrW1N0M",
        "source": "https://corbettmaths.com/2013/05/03/solving-quadratics-by-factorising/",
    },
]

SCENARIOS = [
    {
        "name": "sign-error",
        "task": "Solve: 2x + 3 = 11",
        "work": ["2x = 11 + 3", "2x = 14", "x = 7"],
        "misconception": "You added 3 instead of subtracting it from both sides.",
        "expected": ["x=4"],
    },
    {
        "name": "fraction-error",
        "task": "Calculate: 1/2 + 1/3",
        "work": ["1/2 + 1/3 = 2/5"],
        "misconception": "You added the numerators and denominators separately.",
        "expected": ["5/6"],
    },
    {
        "name": "distribution-error",
        "task": "Solve: 3(x + 2) = 15",
        "work": ["3x + 2 = 15", "3x = 13", "x = 13/3"],
        "misconception": "You multiplied x by 3 but forgot to multiply the 2.",
        "expected": ["x=3"],
    },
]

SOURCES = Path("artifacts/tutor-eval/sources")


def paper(task, work):
    """An explicitly typeset simulator input, not claimed to be real handwriting."""
    image = Image.new("RGB", (960, 640), "#fffdf5")
    draw = ImageDraw.Draw(image)
    draw.text(
        (35, 25),
        "SIMULATED STUDENT / TYPESET WORK",
        font=ImageFont.truetype(FONT, 20),
        fill="#59616b",
    )
    draw.text((35, 75), task, font=ImageFont.truetype(FONT, 34), fill="black")
    for i, line in enumerate(work):
        draw.text((45, 150 + 53 * i), line, font=ImageFont.truetype(FONT, 29), fill="#243e73")
    return image


def parse_student(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)[:-3].strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Student output must be a JSON object")
    work, message, finished = data.get("work"), data.get("message"), data.get("finished")
    if not isinstance(work, list) or not 1 <= len(work) <= 8:
        raise ValueError("Student must return 1–8 work lines")
    if any(not isinstance(line, str) or not 0 < len(line) <= 55 for line in work):
        raise ValueError("Invalid or overlong student work line")
    if not isinstance(message, str) or not 0 < len(message) <= 500 or type(finished) is not bool:
        raise ValueError("Invalid student message or finished flag")
    return data


def final_answer_matches(work, expected):
    # Deliberately narrow, transparent check; never ask either model to grade itself.
    # Only the final line counts: a correct value elsewhere cannot mask a wrong answer.
    value = work[-1].casefold().replace(" ", "").rstrip(".")
    value = value.replace("\\frac{5}{6}", "5/6")
    return any(value == answer or value.endswith("=" + answer) for answer in expected)


class Journal:
    def __init__(self, path, metadata):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.events = []
        self.started = time.monotonic()
        self.answers = []
        self.current = None

    def add(self, kind, /, **fields):
        event = {"t": round(time.monotonic() - self.started, 4), "type": kind, **fields}
        self.events.append(event)
        if event["type"] == "start":
            self.current = {"revision": event["revision"], "start": event["t"], "text": ""}
        elif event["type"] == "text" and self.current is not None:
            self.current.setdefault("first_token", event["t"])
            self.current["text"] = event["text"]
        elif event["type"] == "done" and self.current is not None:
            self.current.update(end=event["t"], interrupted=event["interrupted"])
            self.answers.append(self.current)
            print(f"[{self.path.name} {event['t']:.1f}s] TUTOR: {self.current['text']}", flush=True)
            self.current = None

    async def emit(self, event):
        self.add(event["type"], **event)

    def save(self, **result):
        duration = time.monotonic() - self.started
        document = {
            **self.metadata,
            "duration": duration,
            "events": self.events,
            "answers": self.answers,
            "result": result,
        }
        (self.path / "events.json").write_text(json.dumps(document, indent=2, ensure_ascii=False))
        return document


class ObservedBackend(AsyncLLMBackend):
    def __init__(self, llm, journal):
        super().__init__(llm, observation_tokens=160, reply_tokens=160)
        self.journal = journal

    async def observe(self, image, revision, previous_notes, stopped, **kwargs):
        self.journal.add("observe_begin", revision=revision)
        snapshot = await super().observe(image, revision, previous_notes, stopped, **kwargs)
        self.journal.add("observation", revision=revision, notes=snapshot.notes)
        return snapshot


async def wait_session(session, predicate, timeout=120):
    deadline = time.monotonic() + timeout
    while not predicate():
        for task in session.tasks:
            if task.done():
                task.result()
                raise RuntimeError("Tutor worker stopped before the test finished")
        if time.monotonic() > deadline:
            raise TimeoutError("Tutor did not reach the expected state before the deadline")
        await asyncio.sleep(0.05)


async def student_turn(backend, scenario, work, dialogue):
    """No image inputs, observer notes, tutor cache blocks, or answer key are passed."""
    system = f"""Role-play a student learning mathematics. Your problem: {scenario["task"]}
Your initial misconception: {scenario["misconception"]}
Use only the tutor's visible feedback to improve your current work. Make at most
ONE new mathematical transformation this turn. If correcting an earlier mistake,
replace that line and remove its dependent incorrect lines. Do not jump directly
to a final answer. Ask a short natural question if confused. When the work is
complete, set finished=true. Do not impersonate the tutor.
Return ONLY JSON: {{"work": ["current line", "next line"], "message": "your reply",
"finished": false}}. Use ASCII math, at most 8 lines, each under 55 characters.
"""
    messages = [
        {"role": "system", "content": system},
        *dialogue[-8:],
        {
            "role": "user",
            "content": "Current work:\n"
            + "\n".join(work)
            + "\nRespond to the tutor and update your work by one step.",
        },
    ]
    text = backend.llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    block = await backend.llm.create_block()
    try:
        # Seed the JSON opening brace, so a conversational student cannot start
        # with prose instead of the requested transport object.
        out = await backend.llm.forward(backend.encode(text + "{"), write_to=block)
        answer = ""
        async for answer in backend.generate([block], out, 384, lambda: False):
            pass
        answer = "{" + answer
        if hasattr(backend, "journal"):
            backend.journal.add("student_raw", raw=answer)
        return parse_student(answer), answer
    finally:
        await backend.free([block])


def prepare_frames(source, directory, start, duration, interval):
    directory.mkdir(parents=True, exist_ok=True)
    # Explicit seek per timestamp avoids fps-filter midpoint sampling (which
    # could otherwise show the model writing from a future point in the clip).
    for index in range(math.ceil(duration / interval)):
        subprocess.run(
            "ffmpeg -v error -nostdin -y -threads 2 -ss".split()
            + [str(start + index * interval), "-i", str(source)]
            + "-frames:v 1 -vf scale=960:-2 -threads 2".split()
            + [str(directory / f"{index:05d}.jpg")],
            check=True,
        )
    paths = sorted(directory.glob("*.jpg"))
    if not paths:
        raise ValueError(f"No video frames extracted from {source}")
    return paths


async def replay(llm, sample, source, output, duration, interval, start=0):
    path = output / ("video-" + sample["name"])
    frames = prepare_frames(source, path / "frames", start, duration, interval)
    journal = Journal(
        path,
        {
            "kind": "real-video",
            "title": sample["title"],
            "source": sample["source"],
            "source_file": str(source.resolve()),
            "source_start": start,
            "clip_duration": duration,
            "interval": interval,
            "model": "Qwen/Qwen3.8-27B",
            "audio_used": False,
        },
    )
    session = TutorSession(ObservedBackend(llm, journal), journal.emit, frame_gate=FrameGate())
    session.start()
    error = None
    try:
        last = -1
        while time.monotonic() - journal.started < duration:
            elapsed = time.monotonic() - journal.started
            index = min(int(elapsed / interval), len(frames) - 1)
            if index != last:
                with Image.open(frames[index]) as image:
                    session.submit_frame(image.convert("RGB"))
                journal.add(
                    "frame",
                    revision=session.revision,
                    source_time=round(start + index * interval, 3),
                    file=str(frames[index].resolve()),
                )
                last = index
            for task in session.tasks:
                if task.done():
                    task.result()
                    raise RuntimeError("Tutor worker stopped")
            await asyncio.sleep(0.05)
        journal.add("source_finished")
        final_revision = session.revision
        await wait_session(
            session,
            lambda: (
                session.processed_revision >= final_revision
                and session.latest is not None
                and session.auto_revision >= session.latest.revision
                and journal.current is None
            ),
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        journal.add("error", text=error)
    finally:
        await session.close()
    return journal.save(
        error=error,
        frames_submitted=session.revision,
        frames_observed=sum(e["type"] == "observation" for e in journal.events),
        completed_replies=len(journal.answers),
        codec_skipped=sum(e["type"] == "frame_gate" and not e["accepted"] for e in journal.events),
        event_skipped=sum(e["type"] == "event_probe" and not e["accepted"] for e in journal.events),
        output_probes=sum(e["type"] == "output_probe" for e in journal.events),
    )


async def simulate(llm, scenario, output, turns):
    path = output / ("student-" + scenario["name"])
    journal = Journal(
        path,
        {
            "kind": "simulated-student",
            "title": scenario["task"],
            "model": "Qwen/Qwen3.8-27B",
            "student_input": "text only",
            "student_cache": "isolated",
            "seed_work": scenario["work"],
            "misconception": scenario["misconception"],
        },
    )
    backend = ObservedBackend(llm, journal)
    session = TutorSession(backend, journal.emit)
    session.auto = False
    session.start()
    work = list(scenario["work"])
    message = "Is my solution right? Please help me understand my mistake."
    dialogue = []
    error = None
    matched = False
    try:
        for turn in range(turns + 1):
            image = paper(scenario["task"], work)
            image_path = path / f"paper-{turn}.jpg"
            image.save(image_path)
            session.submit_frame(image)
            journal.add(
                "frame",
                revision=session.revision,
                file=str(image_path.resolve()),
                work=work,
                origin="scripted misconception" if turn == 0 else "model-generated",
            )
            revision = session.revision
            await wait_session(
                session, lambda: session.latest is not None and session.latest.revision == revision
            )
            journal.add("student", text=message)
            before = len(journal.answers)
            session.submit_text(message)
            await wait_session(session, lambda: len(journal.answers) > before)
            tutor = journal.answers[-1]["text"]
            dialogue.extend(
                [
                    {"role": "assistant", "content": message},
                    {"role": "user", "content": "Tutor: " + tutor},
                ]
            )
            matched = final_answer_matches(work, scenario["expected"])
            if matched or turn == turns:
                break
            journal.add("student_thinking")
            update, raw = await student_turn(backend, scenario, work, dialogue)
            work, message = update["work"], update["message"]
            journal.add("student_generated", **{"raw": raw, **update})
            print(f"[{path.name}] STUDENT: {work} / {message}", flush=True)
            # The model's finished flag does not determine success; check independently.
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        journal.add("error", text=error)
    finally:
        await session.close()
    return journal.save(
        error=error,
        final_work=work,
        final_answer_matches=matched,
        expected_final=scenario["expected"],
        tutor_turns=len(journal.answers),
    )


async def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    llm = create_llm(distributed_port=args.distributed_port)
    results = []
    try:
        # Warm up outside the recorded timeline: video playback must not skip past
        # the clip while first-use vision/sampling kernels compile.
        print("Warming vision and generation before starting recorded timelines…", flush=True)
        backend = AsyncLLMBackend(llm, observation_tokens=32, reply_tokens=32)
        warm = await backend.observe(paper("Solve x + 1 = 2", ["x = 1"]), 0, "None", lambda: False)
        try:
            async for _ in backend.reply(warm, [], "Check this work briefly.", lambda: False):
                pass
        finally:
            await backend.free(warm.blocks)
        for mode, cases in (("video", SAMPLES), ("student", SCENARIOS)):
            if args.mode not in {"all", mode}:
                continue
            for case in cases:
                if args.case and case["name"] not in args.case:
                    continue
                if mode == "video":
                    print("REPLAY:", case["name"], flush=True)
                    result = await replay(
                        llm,
                        case,
                        args.sources / (case["name"] + ".mp4"),
                        args.output,
                        args.duration,
                        args.interval,
                        args.start,
                    )
                else:
                    print("SIMULATE:", case["name"], flush=True)
                    result = await simulate(llm, case, args.output, args.turns)
                results.append(result)
    finally:
        await llm.close()
    (args.output / "summary.json").write_text(
        json.dumps(
            [
                {"title": r["title"], "kind": r["kind"], "duration": r["duration"], **r["result"]}
                for r in results
            ],
            indent=2,
        )
    )
    return any(r["result"]["error"] for r in results)


def download_sources(output):
    """Download public lessons and record provenance alongside the media."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample in SAMPLES:
        with urllib.request.urlopen(
            "https://public-api.wordpress.com/rest/v1.1/videos/" + sample["guid"], timeout=30
        ) as response:
            metadata = json.load(response)
        url = metadata["file_url_base"]["https"] + metadata["files"]["hd"]["mp4"]
        target = output / (sample["name"] + ".mp4")
        if not target.exists():
            temporary = target.with_suffix(".part")
            try:
                urllib.request.urlretrieve(url, temporary)
                temporary.rename(target)
            finally:
                temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        with target.open("rb") as video:
            for chunk in iter(lambda: video.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append(
            {
                **sample,
                "media_url": url,
                "file": str(target.resolve()),
                "duration_seconds": metadata["duration"] / 1000,
                "sha256": digest.hexdigest(),
                "bytes": target.stat().st_size,
            }
        )
        print(target, flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


@lru_cache(maxsize=2048)
def wrap(text, font, width):
    lines = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            candidate = (line + " " + word).strip()
            if line and font.getlength(candidate) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def render(event_path, *, fps=8):
    event_path = Path(event_path)
    data = json.loads(event_path.read_text())
    events = data["events"]
    destination = event_path.with_name("review.mp4")
    regular, small, heading = [ImageFont.truetype(FONT, size) for size in (22, 18, 27)]
    length = data["duration"] + 4  # readable final hold; everything before it stays at 1x
    is_video = data["kind"] == "real-video"
    label = (
        "RECORDED LESSON / REAL HANDWRITING" if is_video else "SIMULATED STUDENT / TYPESET PAPER"
    )
    y = 633 if is_video else 700
    decoder = None
    if is_video:
        decoder = subprocess.Popen(
            "ffmpeg -v error -nostdin -threads 2 -ss".split()
            + [
                str(data["source_start"]),
                "-i",
                data["source_file"],
                "-t",
                str(data["clip_duration"]),
            ]
            + [
                "-frames:v",
                str(min(math.ceil(length * fps), math.ceil(data["clip_duration"] * fps))),
            ]
            + [
                "-an",
                "-vf",
                f"fps={fps},scale=900:506:force_original_aspect_ratio=decrease,pad=900:506:(ow-iw)/2:(oh-ih)/2",
            ]
            + "-threads 2 -f rawvideo -pix_fmt rgb24 pipe:1".split(),
            stdout=subprocess.PIPE,
        )
    encoder = subprocess.Popen(
        "ffmpeg -v error -nostdin -y -f rawvideo -pix_fmt rgb24 -s".split()
        + [f"{WIDTH}x{HEIGHT}", "-r", str(fps)]
        + "-i pipe:0 -an -c:v libx264 -threads 2 -preset veryfast -crf 23".split()
        + "-pix_fmt yuv420p -movflags +faststart".split()
        + [str(destination)],
        stdin=subprocess.PIPE,
    )
    index = 0
    source_image = Image.new("RGB", (900, 506), "#192531")
    chats = []
    active = None
    notes = "Waiting for the first observation."
    status = "Starting"
    submitted = observed = 0
    submitted_at = {}
    replying_revision = None

    def text(x, y, value, font=small, color="#e9eef5"):
        draw.text((x, y), value, font=font, fill=color)

    try:
        for frame_index in range(math.ceil(length * fps)):
            t = frame_index / fps
            if decoder and t < data["clip_duration"]:
                raw = decoder.stdout.read(900 * 506 * 3)
                if len(raw) == 900 * 506 * 3:
                    source_image = Image.frombytes("RGB", (900, 506), raw)
            while index < len(events) and events[index]["t"] <= t:
                event = events[index]
                kind = event["type"]
                if kind == "frame":
                    submitted = event["revision"]
                    submitted_at[submitted] = event.get("source_time", event["t"])
                    if not is_video:
                        with Image.open(event["file"]) as image:
                            source_image = ImageOps.contain(image.convert("RGB"), (900, 580))
                elif kind == "observation":
                    observed, notes = event["revision"], event["notes"]
                elif kind == "status":
                    status = event["text"]
                elif kind == "observed":
                    status = "Following work" if event["changed"] else "No change detected"
                elif kind == "student":
                    chats.append({"role": "Student", "t": event["t"], "text": event["text"]})
                elif kind == "student_thinking":
                    status = "Text-only student is responding"
                elif kind == "start":
                    replying_revision = event["revision"]
                    active = {
                        "role": f"Tutor / snapshot {replying_revision}",
                        "t": event["t"],
                        "text": "…",
                    }
                    chats.append(active)
                elif kind == "text" and active:
                    active["text"] = event["text"]
                elif kind == "done":
                    if active and event["interrupted"]:
                        active["text"] += " [interrupted]"
                    active = None
                elif kind == "error":
                    status = "ERROR: " + event["text"]
                index += 1
            canvas = Image.new("RGB", (WIDTH, HEIGHT), "#101821")
            draw = ImageDraw.Draw(canvas)
            text(24, 17, data["title"], heading)
            text(24, 57, label, color="#9cdfcc")
            text(1090, 23, f"Elapsed {t:06.1f}s / 1x", regular)
            canvas.paste(source_image, (20, 105))
            text(
                24,
                y,
                f"Submitted {submitted}   Observed {observed}   {status[:60]}",
                color="#9cdfcc",
            )
            if replying_revision is not None:
                ts = submitted_at.get(replying_revision, 0)
                text(
                    24,
                    y + 28,
                    f"Latest reply refers to snapshot {replying_revision} ({ts:.1f}s)",
                    color="#acbccd",
                )
            text(24, y + 61, "Observer notes (review only):", color="#acbccd")
            for i, line in enumerate(wrap(notes, small, 880)[: 5 if is_video else 3]):
                text(24, y + 89 + 23 * i, line)
            draw.rounded_rectangle((940, 100, 1420, 840), radius=16, fill="#192531")
            # Most recent complete messages + streaming message; explicitly mark
            # clipping in this view. The JSON transcript always retains full text.
            cards = []
            for chat in reversed(chats):
                lines = wrap(chat["text"], regular, 425)
                if len(lines) > 15:
                    lines = lines[:14] + ["… (full reply in transcript)"]
                height = 52 + 29 * len(lines)
                if sum(card[2] for card in cards) + height > 710:
                    break
                cards.append((chat, lines, height))
            cy = 116
            for chat, lines, height in reversed(cards):
                color = "#24463f" if chat["role"] == "Student" else "#243646"
                draw.rounded_rectangle((954, cy, 1406, cy + height - 10), radius=9, fill=color)
                text(967, cy + 9, f"{chat['t']:.1f}s  {chat['role']}", color="#9cdfcc")
                for i, line in enumerate(lines):
                    text(967, cy + 36 + 29 * i, line, regular)
                cy += height
            footer = "Qwen3.8-27B · Actual inference timestamps · Silent review recording"
            if is_video:
                footer += " · Source: Corbettmaths"
                if t >= data["clip_duration"]:
                    footer += " · End of clip (held)"
            text(24, 861, footer, color="#acbccd")
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("MP4 encoding failed")
        if decoder:
            decoder.stdout.close()
            if decoder.wait() != 0:
                raise RuntimeError("Source video decoding failed")
    finally:
        for process in (decoder, encoder):
            if process and process.poll() is None:
                process.terminate()
                process.wait()
    print("RECORDING:", destination, flush=True)
    return destination


def metrics(data):
    frames = {e["revision"]: e for e in data["events"] if e["type"] == "frame"}
    latencies = [
        a["first_token"] - frames[a["revision"]]["t"]
        for a in data["answers"]
        if "first_token" in a and a["revision"] in frames
    ]
    return {
        "median_frame_to_first_token": statistics.median(latencies) if latencies else None,
        "max_frame_to_first_token": max(latencies) if latencies else None,
        "replies": len(data["answers"]),
    }


def build_report(output, runs, notes=None):
    # Later runs replace earlier cases with the same name; originals stay on disk.
    cases = {path.parent.name: path for run in runs for path in sorted(run.glob("*/events.json"))}
    cards = []
    summaries = []
    for name, path in cases.items():
        data = json.loads(path.read_text())
        measured = metrics(data)
        result = data["result"]
        is_video = data["kind"] == "real-video"
        label = "Real recorded lesson" if is_video else "Simulated student; typeset paper"
        summary = [f"{len(data['answers'])} tutor replies", f"{data['duration']:.1f}s recorded"]
        if measured["median_frame_to_first_token"] is not None:
            summary.append(
                f"{measured['median_frame_to_first_token']:.1f}s median frame → first reply token"
            )
        if is_video:
            summary.append(
                f"{result['frames_observed']}/{result['frames_submitted']} submitted frames processed"
            )
            summary.append(
                f"Source excerpt: {data['source_start']:.0f}–{data['source_start'] + data['clip_duration']:.0f}s"
            )
            if "codec_skipped" in result:
                summary.append(
                    f"Codec skipped {result['codec_skipped']} candidates; "
                    f"event probe skipped {result['event_skipped']}; "
                    f"{result['output_probes']} output-probe decisions"
                )
        else:
            match = "matches" if result["final_answer_matches"] else "does not match"
            summary.append(f"Final answer {match} fixed key")
        if result.get("error"):
            summary.append("EXECUTION ERROR: " + result["error"])
        transcript = [
            (e["t"], "Student", e["text"]) for e in data["events"] if e["type"] == "student"
        ]
        transcript.extend(
            (answer["start"], f"Tutor / snapshot {answer['revision']}", answer["text"])
            for answer in data["answers"]
        )
        transcript.sort()
        text = "".join(
            f"<p><b>{t:.1f}s · {html.escape(role)}</b><br>{html.escape(message)}</p>"
            for t, role, message in transcript
        )
        media = html.escape(
            os.path.relpath(path.with_name("review.mp4"), output.parent), quote=True
        )
        evidence = html.escape(os.path.relpath(path, output.parent), quote=True)
        source = (
            f'<a href="{html.escape(data["source"], quote=True)}">Original source / Corbettmaths</a>'
            if "source" in data
            else "Initial error is scripted; subsequent student turns are model-generated."
        )
        cards.append(f"""<section><h2>{html.escape(data["title"])}</h2><p>{label}</p>
<video controls preload="metadata" src="{media}"></video>
<ul>{"".join("<li>" + html.escape(item) + "</li>" for item in summary)}</ul>
<p><a href="{media}">Open MP4</a> · <a href="{evidence}">Full timestamped JSON</a></p><p>{source}</p>
<details><summary>Conversation transcript</summary>{text}</details></section>""")
        summaries.append({"case": name, "run": path.parent.parent.name, **measured, **result})
    output.write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Tutor evaluation recordings</title>
<style>body{font:17px/1.5 system-ui;max-width:1100px;margin:35px auto;padding:0 20px;background:#101821;color:#e9eef5}section{margin:30px 0;padding:24px;background:#192531;border-radius:15px}video{width:100%}a{color:#9cdfcc}summary{cursor:pointer}p{white-space:pre-wrap}li{margin:4px 0}</style>
<h1>Camera tutor: evaluation recordings</h1>
<p>Qwen3.8-27B · Real inference, silent H.264 MP4s, preserved wall-clock timing.</p>
<p>Recorded instructor lessons test perception and tracking. Text-only simulated students test feedback on controlled errors. The tutor sees synthetic paper in student cases. This is a small functional evaluation, not a validated educational benchmark. A final-answer match does not establish that every intermediate step or hint was correct.</p>
<p>Observation notes in the recordings are reviewer diagnostics. They were not sent to the student. MP4 views may show only recent messages; full output is retained in the transcripts.</p>
"""
        + (
            "<section><h2>Manual review findings</h2><ul>"
            + "".join("<li>" + html.escape(note) + "</li>" for note in notes)
            + "</ul></section>"
            if notes
            else ""
        )
        + "\n".join(cards)
        + "</html>"
    )
    output.with_suffix(".json").write_text(json.dumps(summaries, indent=2))
    print(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true", help="Download lesson sources only")
    action.add_argument(
        "--render", type=Path, nargs="+", metavar="EVENTS", help="Render saved events"
    )
    action.add_argument(
        "--report", type=Path, nargs="+", metavar="RUN", help="Build an HTML report"
    )
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--output", type=Path, help="New recording directory or HTML report path")
    parser.add_argument("--notes", type=Path, help="JSON list of manual review notes for --report")
    parser.add_argument("--mode", choices=["all", "video", "student"], default="all")
    parser.add_argument("--duration", type=float, default=90)
    parser.add_argument("--start", type=float, default=40, help="Source clip start in seconds")
    parser.add_argument("--interval", type=float, default=2.5)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--distributed-port", type=int, default=0)
    parser.add_argument(
        "--case", action="append", choices=[item["name"] for item in SAMPLES + SCENARIOS]
    )
    parser.add_argument("--failed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.notes and not args.report:
        parser.error("--notes requires --report")
    if args.failed and not args.render:
        parser.error("--failed requires --render")
    if args.download:
        return download_sources(args.output or args.sources)
    if args.render:
        if args.output:
            parser.error("--render writes review.mp4 beside each event log")
        for path in args.render:
            render(path)
        if args.failed:
            raise SystemExit("Some evaluation cases failed; see their events.json files")
        return
    if args.output is None:
        parser.error("--output is required for recording or --report")
    if args.report:
        return build_report(
            args.output, args.report, json.loads(args.notes.read_text()) if args.notes else None
        )
    if not all(math.isfinite(v) and v > 0 for v in (args.duration, args.interval)):
        parser.error("duration and interval must be finite and positive")
    if args.turns < 1:
        parser.error("turns must be positive")
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("start must be finite and nonnegative")
    if not 0 <= args.distributed_port <= 65535:
        parser.error("distributed-port must be in 0..65535")
    cases = {"video": SAMPLES, "student": SCENARIOS, "all": SAMPLES + SCENARIOS}[args.mode]
    if args.case and not set(args.case) <= {item["name"] for item in cases}:
        parser.error("Selected case is incompatible with --mode")
    failed = asyncio.run(evaluate(args))
    print("Inference finished. Replacing the model process with the CPU renderer…", flush=True)
    # Replacing the process releases model/KV tensors before CPU-only rendering.
    command = [sys.executable, "-m", "async_tutor.record", "--render"]
    command.extend(str(path) for path in sorted(args.output.glob("*/events.json")))
    if failed:
        command.append("--failed")
    os.execve(sys.executable, command, {**os.environ, "CUDA_VISIBLE_DEVICES": ""})


if __name__ == "__main__":
    main()
