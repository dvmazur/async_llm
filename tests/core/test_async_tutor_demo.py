"""CPU regressions for Tutor cache ownership and reproducible MP4 timelines."""

import asyncio
import io
import json
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from scripts.async_tutor.async_tutor import demo
from scripts.async_tutor.async_tutor.demo import AsyncLLMBackend, EventGate, Snapshot, TutorSession
from scripts.async_tutor.async_tutor.record import FONT, render
from scripts.async_tutor.async_tutor.trace import Trace


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
            epoch_before_frame_change = session.epoch
            session.submit_frame("second")
            await eventually(lambda: session.latest.revision == 3)
            assert session.epoch == epoch_before_frame_change
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
                assert session.last_reply_assessment == "work 2"
                assert list(session.history) == [
                    {"role": "user", "content": "Help"},
                    {"role": "assistant", "content": "Check both sides."},
                ]
            else:
                assert session.last_reply_assessment == "None."
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


def test_paused_frame_is_inspected_once_after_inflight_work_and_force_is_not_sticky():
    class InspectBackend(Backend):
        def __init__(self):
            super().__init__()
            self.inspect_started, self.inspect_resume = asyncio.Event(), asyncio.Event()

        async def inspect(self, image, revision, previous_notes, stopped, *, force, **kwargs):
            self.observations.append((revision, image, force))
            if revision == 1:
                self.inspect_started.set()
                await self.inspect_resume.wait()
            if not force:
                return None
            return Snapshot(revision, [revision], "Same work")

    class ForceOnlyGate:
        def __init__(self):
            self.calls = []

        def check(self, image, *, force=False):
            self.calls.append((image, force))
            return demo.GateDecision(force, None, "forced" if force else "unchanged")

        def close(self):
            pass

    async def run():
        backend, gate, events = InspectBackend(), ForceOnlyGate(), []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, frame_gate=gate)
        session.auto = False
        session.submit_frame("first")
        session.start()
        try:
            await asyncio.wait_for(backend.inspect_started.wait(), 2)
            session.submit_frame("paused", force=True)
            assert backend.observations == [(1, "first", True)]
            backend.inspect_resume.set()
            await eventually(lambda: session.latest is not None and session.latest.revision == 2)
            assert backend.observations == [(1, "first", True), (2, "paused", True)]
            assert gate.calls == [("first", True), ("paused", True)]
            assert any(e["type"] == "observed" and e["revision"] == 2 for e in events)
            # A paused picture has no producer after its one delivery.
            for _ in range(10):
                await asyncio.sleep(0)
            assert len(backend.observations) == 2
            assert not session.frames_ready.is_set()

            session.submit_frame("normal")
            await eventually(lambda: session.processed_revision == 3)
            assert gate.calls[-1] == ("normal", False)
            assert session.latest.revision == 2
            assert len(backend.observations) == 2
        finally:
            backend.inspect_resume.set()
            await asyncio.wait_for(session.close(), 2)
        for task in session.tasks:
            task.result()
        assert sorted(backend.freed) == [1, 2]

    asyncio.run(run())


class ProbeBackend(Backend):
    def __init__(self, notes):
        super().__init__()
        self.notes = notes
        self.probes = asyncio.Queue()
        self.probe_gates = []
        self.active_probes = 0
        self.unblock_probes = False
        self.replies = []
        self.reply_assessments = []
        self.probe_assessments = []

    async def inspect(self, image, revision, previous_notes, stopped, **kwargs):
        self.observations.append((revision, image))
        notes = self.notes[revision]
        facts = demo.clean_observe_notes(notes, include_check=False).strip()
        return Snapshot(
            revision,
            [revision],
            notes,
            observation_valid=bool(facts) and facts.upper().rstrip(".") != "WAIT",
        )

    async def output_probability(self, snapshot, history, student_context, **kwargs):
        gate = asyncio.Event()
        self.probe_gates.append(gate)
        self.probe_assessments.append(
            (snapshot.current_assessment, kwargs.get("replied_assessment"))
        )
        self.probes.put_nowait((snapshot, gate))
        self.active_probes += 1
        if self.unblock_probes:
            gate.set()
        try:
            await gate.wait()
            return 1.0
        finally:
            self.active_probes -= 1

    async def reply(self, snapshot, history, question, stopped):
        self.replies.append(snapshot.revision)
        self.reply_assessments.append(snapshot.current_assessment)
        yield "An explanation of the current work."

    def release_probes(self):
        self.unblock_probes = True
        for gate in self.probe_gates:
            gate.set()


@pytest.mark.parametrize("draft_header", ["Reply", "Analysis", "Drafted answer"])
def test_structured_observe_cleanup_preserves_multiline_requests_and_mathematics(draft_header):
    request = 'REQUEST:\n{\n  "text": "Check this",\n  "ready": true\n}'
    facts = """TASK: Solve the equation.
COMMENTS: A handwritten comment.
WORK: x: real
x² + 2x - 1 = 0
Case 1: x > 0
CHECK: The equation is legible.
READABILITY: Clear."""
    raw = (
        "Internal preamble.\nAnalysis:\nA drafted explanation.\n"
        + request
        + "\n"
        + facts.replace("CHECK:", f"WAIT\n{draft_header}:\nA hidden draft.\nCHECK:")
        + "\n\nAn unlabelled drafted reply.\nReply:\nx = -1 ± √2\n"
        + "Analysis:\nDo not copy this.\nWAIT"
    )
    cleaned = demo.clean_observe_notes(raw)
    assert cleaned == request + "\n" + facts
    assert demo.parse_page_request(cleaned) == (True, demo.PageRequest("Check this"))
    assert Snapshot(1, [], raw).current_assessment == facts


def test_modern_fact_cleanup_removes_the_entire_internal_check_without_rewriting_work():
    prefix = "REQUEST: null\nREVIEW: work\nTASK: x² + 2x − 4 = 3\n"
    work = "WORK: x² + 2x − 4 = 3\nD = 4 + 16 = 20"
    check = "CHECK: Move 3 to the left: c = -7.\nD = 32; recompute the result."
    notes = f"{prefix}{work}\n{check}\nREADABILITY: Clear."
    assert demo.clean_observe_notes(notes) == notes
    assert demo.clean_observe_notes(notes, include_check=False) == (
        f"{prefix}{work}\nREADABILITY: Clear."
    )
    assert check in Snapshot(1, [], notes).current_assessment  # Explicit legacy compatibility.
    modern = Snapshot(1, [], notes, review_kind="work")
    assert modern.current_assessment == ("TASK: x² + 2x − 4 = 3\n" + work + "\nREADABILITY: Clear.")
    assert modern.notes == notes


@pytest.mark.parametrize("notes", ["WAIT", "First problem.\nx = 2", "A custom observation."])
def test_observe_cleanup_preserves_unstructured_legacy_notes(notes):
    assert demo.clean_observe_notes(notes) == notes
    assert Snapshot(1, [], notes).current_assessment == notes


def test_request_metadata_alone_does_not_turn_legacy_facts_into_structured_notes():
    request = 'REQUEST:\n{"text": "Check this", "ready": true}'
    notes = f"A custom observation before the metadata.\n{request}\nx + 1 = 2 is legible."
    assert demo.clean_observe_notes(notes) == notes
    assert demo.parse_page_request(notes) == (True, demo.PageRequest("Check this"))
    assert Snapshot(1, [], notes).current_assessment == (
        "A custom observation before the metadata.\nx + 1 = 2 is legible."
    )


def test_prefaced_wait_is_normalized_without_discarding_fields_in_a_mixed_response():
    prefaced = "No new writing is visible.\nThe previous notes still apply.\n\nWAIT"
    assert demo.clean_observe_notes(prefaced) == "WAIT"
    assert demo.observe_fields(prefaced) == {}
    assert demo.clean_observe_notes("WAIT\nWORK: x = 2\nWAIT") == "WORK: x = 2"


@pytest.mark.parametrize(
    "field, expected",
    [
        ("REVIEW: wait", "wait"),
        ("REVIEW: problem", "problem"),
        ("**REVIEW:** work", "work"),
        ("REVIEW: work extra", None),
        ('REVIEW: {"kind": "work"}', None),
        ("REVIEW: work\nREVIEW: wait", None),
        ("COMMENTS: REVIEW: work", None),
    ],
)
def test_review_field_is_a_plain_enum_and_never_part_of_the_factual_assessment(field, expected):
    facts = "TASK: Solve x + 1 = 2.\nWORK: x = 1."
    notes = field + "\n" + facts
    assert demo.parse_review_kind(notes) == expected
    if expected is not None:
        assert field in demo.clean_observe_notes(notes)
        assert Snapshot(1, [], notes).current_assessment == facts


def test_fresh_observation_rechecks_a_previous_no_for_unchanged_work():
    class ReconsideringBackend(ProbeBackend):
        async def output_probability(self, snapshot, history, student_context, **kwargs):
            await super().output_probability(snapshot, history, student_context, **kwargs)
            return 0.0 if snapshot.revision == 1 else 1.0

    async def run():
        backend = ReconsideringBackend({1: "WORK: x = 2", 2: "WORK: x = 2"})
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("first reading")
        session.start()
        try:
            first, first_gate = await asyncio.wait_for(backend.probes.get(), 2)
            first_gate.set()
            await eventually(lambda: session.probed_snapshot is first)
            assert not backend.replies
            session.submit_frame("fresh reading of the same work")
            second, second_gate = await asyncio.wait_for(backend.probes.get(), 2)
            assert second.revision == 2
            assert second.assessment_revision == first.assessment_revision
            assert backend.probe_assessments == [("WORK: x = 2", "None.")] * 2
            second_gate.set()
            await eventually(lambda: any(event["type"] == "done" for event in events))
            assert backend.replies == [2]
            assert session.last_reply_assessment == "WORK: x = 2"
        finally:
            backend.release_probes()
            await asyncio.wait_for(session.close(), 2)
        for task in session.tasks:
            task.result()
        assert sorted(backend.freed) == [1, 2]

    asyncio.run(run())


@pytest.mark.parametrize(
    "first_notes, changed_notes",
    [("First problem.", "A different problem."), ("WORK: x = 1.", "WORK: X = 1.")],
)
def test_speak_decision_is_rechecked_after_genuinely_changed_work(first_notes, changed_notes):
    async def run():
        backend = ProbeBackend({1: first_notes, 2: changed_notes})
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("first")
        session.start()
        try:
            first, first_gate = await asyncio.wait_for(backend.probes.get(), 2)
            assert first.revision == 1
            session.submit_frame("different work")
            await eventually(lambda: session.latest.revision == 2)
            first_gate.set()
            second, second_gate = await asyncio.wait_for(backend.probes.get(), 2)
            assert second.revision == 2
            assert not backend.replies
            assert backend.freed == [1]
            second_gate.set()
            await eventually(lambda: backend.replies == [2])
            await eventually(lambda: any(event["type"] == "done" for event in events))
            assert sum(event["type"] == "start" for event in events) == 1
        finally:
            backend.release_probes()
            await asyncio.wait_for(session.close(), 2)
        for task in session.tasks:
            task.result()
        assert sorted(backend.freed) == [1, 2]

    asyncio.run(run())


def test_case_only_symbol_correction_reopens_answered_work():
    async def run():
        backend = ProbeBackend({1: "WORK: x = 1.", 2: "WORK: X = 1.", 3: "WAIT"})
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("first")
        session.start()
        try:
            _, first_gate = await asyncio.wait_for(backend.probes.get(), 2)
            first_gate.set()
            await eventually(lambda: any(event["type"] == "done" for event in events))
            assert backend.replies == [1]
            session.submit_frame("case-only symbol correction")
            second, second_gate = await asyncio.wait_for(backend.probes.get(), 2)
            assert second.revision == 2
            second_gate.set()
            await eventually(lambda: sum(event["type"] == "done" for event in events) == 2)
            assert backend.replies == [1, 2]
            session.submit_frame("unchanged corrected work")
            await eventually(lambda: session.processed_revision == 3)
            backend.release_probes()
            await eventually(lambda: backend.active_probes == 0)
            for _ in range(10):
                await asyncio.sleep(0)
            assert backend.replies == [1, 2]
            assert sum(event["type"] == "start" for event in events) == 2
        finally:
            backend.release_probes()
            await asyncio.wait_for(session.close(), 2)
        for task in session.tasks:
            task.result()
        assert sorted(backend.freed) == [1, 2, 3]

    asyncio.run(run())


def test_speak_compares_with_last_reply_assessment_not_latest_unanswered_observation():
    class BaselineBackend(ProbeBackend):
        async def output_probability(self, snapshot, history, student_context, **kwargs):
            probability = await super().output_probability(
                snapshot, history, student_context, **kwargs
            )
            return 0.0 if snapshot.revision == 2 else probability

    async def run():
        notes = {
            1: "WORK: original equation.",
            2: "WORK: first new step.",
            3: "WORK: second new step.",
        }
        backend = BaselineBackend(notes)
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("original equation")
        session.start()
        try:
            _, first_gate = await asyncio.wait_for(backend.probes.get(), 2)
            assert backend.probe_assessments[-1] == (notes[1], "None.")
            first_gate.set()
            await eventually(lambda: any(event["type"] == "done" for event in events))
            assert session.last_reply_assessment == notes[1]
            session.submit_frame("first new step")
            _, second_gate = await asyncio.wait_for(backend.probes.get(), 2)
            assert backend.probe_assessments[-1] == (notes[2], notes[1])
            second_gate.set()
            await eventually(
                lambda: session.probed_snapshot is not None
                and session.probed_snapshot.revision == 2
            )
            assert backend.replies == [1]
            assert session.last_reply_assessment == notes[1]
            session.submit_frame("second new step")
            _, third_gate = await asyncio.wait_for(backend.probes.get(), 2)
            assert backend.probe_assessments[-1] == (notes[3], notes[1])
            third_gate.set()
            await eventually(lambda: sum(event["type"] == "done" for event in events) == 2)
            assert backend.replies == [1, 3]
            assert session.last_reply_assessment == notes[3]
        finally:
            backend.release_probes()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


def page_observation(text=None, *, clarify=False, ready=True, work="WORK: x + 1 = 2"):
    request = None if text is None else {"text": text, "clarify": clarify, "ready": ready}
    return f"REQUEST: {json.dumps(request)}\n{work}"


class PageBackend(Backend):
    def __init__(self, notes, *, hold_replies=False):
        super().__init__()
        self.notes = notes
        self.hold_replies = hold_replies
        self.replies = []
        self.reply_requests = asyncio.Queue()
        self.reply_gates = []
        self.speak_calls = []

    async def inspect(self, image, revision, previous_notes, stopped, **kwargs):
        self.observations.append((revision, image))
        notes = self.notes[revision]
        known, request = demo.parse_page_request(notes)
        facts = demo.clean_observe_notes(notes, include_check=False).strip()
        return Snapshot(
            revision,
            [len(self.observations)],
            notes,
            page_request=request,
            request_known=known,
            observation_valid=bool(facts) and facts.upper().rstrip(".") != "WAIT",
        )

    async def output_probability(self, snapshot, history, student_context, **kwargs):
        self.speak_calls.append(snapshot.revision)
        return 0.0

    async def reply(self, snapshot, history, question, stopped):
        gate = asyncio.Event()
        self.reply_gates.append(gate)
        if not self.hold_replies:
            gate.set()
        call = {
            "revision": snapshot.revision,
            "question": question,
            "history": history,
            "assessment": snapshot.current_assessment,
        }
        self.replies.append(call)
        self.reply_requests.put_nowait((call, gate))
        yield "Reply started."
        await gate.wait()
        assert all(block not in self.freed for block in snapshot.blocks)
        if not stopped():
            yield "Reply finished."

    def release_replies(self):
        self.hold_replies = False
        for gate in self.reply_gates:
            gate.set()


class ReadingBackend(PageBackend):
    def __init__(self, notes):
        super().__init__(notes)
        self.reader_inputs = []

    async def inspect(self, image, revision, previous_notes, stopped, **kwargs):
        self.observations.append((revision, image))
        self.reader_inputs.append(
            {"revision": revision, "previous_notes": previous_notes, **kwargs}
        )
        notes = self.notes[revision]
        if isinstance(notes, list):
            index = sum(item[0] == revision for item in self.observations) - 1
            notes = notes[min(index, len(notes) - 1)]
        known, request = demo.parse_page_request(notes)
        facts = demo.clean_observe_notes(notes, include_check=False).strip()
        return Snapshot(
            revision,
            [len(self.observations)],
            notes,
            page_request=request,
            request_known=known,
            observation_valid=bool(facts) and facts.upper().rstrip(".") != "WAIT",
        )


async def observe_frame(session, image="paper", **kwargs):
    session.submit_frame(image, **kwargs)
    revision = session.revision
    await eventually(lambda: session.latest is not None and session.latest.revision == revision)


def test_empty_video_observation_preserves_facts_and_retries_only_on_a_fresh_frame():
    async def run():
        backend = ReadingBackend(
            {1: "WORK: x = 1", 2: "Nothing changed.\n\nWAIT", 3: "WORK: x = 2"}
        )
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.auto = False
        session.start()
        try:
            await observe_frame(session)
            facts = session.latest.current_assessment
            await observe_frame(session)
            assert session.latest.current_assessment == facts
            assert session.needs_fresh_observation is True
            for _ in range(5):
                await asyncio.sleep(0)
            assert len(backend.reader_inputs) == 2 and not session.frames_ready.is_set()
            await observe_frame(session)
            assert backend.reader_inputs[-1]["force"] is True
            assert backend.reader_inputs[-1]["previous_notes"] == "None."
            assert session.latest.current_assessment == "WORK: x = 2"
            assert not session.needs_fresh_observation
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


def test_empty_photo_read_retries_once_and_new_photo_request_is_immediate():
    async def run():
        backend = ReadingBackend(
            {
                1: ["No changes.\nWAIT", "No changes.\nWAIT"],
                2: page_observation("Check this", work="WORK: x = 2"),
            }
        )
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.start()
        try:
            await observe_frame(session, trusted_still=True)
            await eventually(lambda: len(backend.reader_inputs) == 2)
            for _ in range(5):
                await asyncio.sleep(0)
            assert [item["revision"] for item in backend.reader_inputs] == [1, 1]
            assert backend.reader_inputs[1]["previous_notes"] == "None."
            assert backend.reader_inputs[1]["force"] is True
            assert not session.frames_ready.is_set() and not backend.replies
            await observe_frame(session, trusted_still=True)
            await eventually(lambda: any(event["type"] == "done" for event in events))
            assert len(backend.reader_inputs) == 3
            assert len(backend.replies) == 1 and backend.replies[0]["revision"] == 2
            assert backend.replies[0]["question"] == "Check this"
            assert not backend.speak_calls
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


def test_new_page_request_bypasses_speak_old_praise_and_reply_cooldown():
    async def run():
        backend = PageBackend({1: page_observation("correct or not?")})
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=3600)
        session.last_auto = demo.time.monotonic()
        session.history.extend(
            [
                {"role": "user", "content": "Check my previous work."},
                {"role": "assistant", "content": "Great, that was correct."},
            ]
        )
        session.submit_frame("new question")
        session.start()
        try:
            await eventually(lambda: any(event["type"] == "done" for event in events))
            assert [call["question"] for call in backend.replies] == ["correct or not?"]
            assert not backend.speak_calls
            start = next(event for event in events if event["type"] == "start")
            assert start["source"] == "page" and start["automatic"] is False
            assert list(session.history)[-2] == {"role": "user", "content": "correct or not?"}
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


@pytest.mark.parametrize("pretty", [False, True])
def test_multiline_page_greeting_is_routed_directly_without_metadata_in_reply_context(pretty):
    async def run():
        greeting = {"text": "Hello!", "ready": True, "clarify": False}
        facts = "COMMENTS: Hello!\nWORK: None."
        notes = f"REQUEST:\n{json.dumps(greeting, indent=2 if pretty else None)}\n{facts}"
        backend = PageBackend({1: notes})
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=3600)
        session.last_auto = demo.time.monotonic()
        session.submit_frame("handwritten greeting")
        session.start()
        try:
            await eventually(lambda: any(event["type"] == "done" for event in events))
            assert [call["question"] for call in backend.replies] == ["Hello!"]
            assert backend.replies[0]["assessment"] == facts
            assert not backend.speak_calls  # This backend's Speak score would be zero.
            assert next(event for event in events if event["type"] == "start")["source"] == "page"
            assert list(session.history)[0] == {"role": "user", "content": "Hello!"}
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


def test_answered_page_request_is_not_repeated_for_wait_or_rephrased_notes():
    async def run():
        backend = PageBackend(
            {
                1: page_observation("correct or not?"),
                2: "WAIT",
                3: "COMMENTS: The same question remains visible; work unchanged.",
                4: page_observation("correct or not?", work="WORK: Same correct calculation."),
                5: page_observation("correct   or not?"),
            }
        )
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("question")
        session.start()
        try:
            await eventually(lambda: any(event["type"] == "done" for event in events))
            for revision in range(2, 6):
                session.submit_frame("unchanged question")
                await eventually(lambda: session.processed_revision == revision)
            for _ in range(10):
                await asyncio.sleep(0)
            assert [call["question"] for call in backend.replies] == ["correct or not?"]
            assert session.visible_page_request.text == "correct   or not?"
            assert not session._has_pending_page_request()
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


def test_terminal_question_marks_do_not_repeat_a_page_request_or_rewrite_its_history():
    async def run():
        backend = PageBackend(
            {
                1: page_observation("am I right"),
                2: page_observation("am I right?"),
                3: page_observation("am I right？"),
            }
        )
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("question without final punctuation")
        session.start()
        try:
            await eventually(lambda: any(event["type"] == "done" for event in events))
            for revision in range(2, 4):
                session.submit_frame("question mark completed")
                await eventually(lambda: session.processed_revision == revision)
            for _ in range(10):
                await asyncio.sleep(0)
            assert [call["question"] for call in backend.replies] == ["am I right"]
            assert list(session.history)[0] == {"role": "user", "content": "am I right"}
            assert session.visible_page_request.text == "am I right？"
            assert session.page_request_id == session.handled_page_request_id == 1
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


@pytest.mark.parametrize(
    "first_question, second_question",
    [("Find x?", "Find X?"), ("Find x + 1?", "Find x - 1?")],
)
def test_page_request_identity_preserves_math_case_and_operators(first_question, second_question):
    async def run():
        backend = PageBackend(
            {1: page_observation(first_question), 2: page_observation(second_question)}
        )
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("first mathematical request")
        session.start()
        try:
            await eventually(lambda: any(event["type"] == "done" for event in events))
            session.submit_frame("one symbol in the request changed")
            await eventually(lambda: sum(event["type"] == "done" for event in events) == 2)
            assert [call["question"] for call in backend.replies] == [
                first_question,
                second_question,
            ]
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


@pytest.mark.parametrize("same_text", [False, True])
def test_page_request_draft_is_answered_once_when_completed(same_text):
    async def run():
        completed = "whats next step?"
        draft = completed if same_text else "whats ne"
        backend = PageBackend(
            {
                1: page_observation(draft, ready=False, clarify=not same_text),
                2: page_observation(completed),
                3: page_observation(completed),
            }
        )
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("request still being written")
        session.start()
        try:
            await eventually(lambda: session.processed_revision == 1)
            for _ in range(10):
                await asyncio.sleep(0)
            assert not backend.replies
            assert session.visible_page_request.ready is False
            assert not session._has_pending_page_request()
            draft_id = session.page_request_id
            session.submit_frame("completed request")
            await eventually(lambda: any(event["type"] == "done" for event in events))
            assert session.visible_page_request.ready is True
            assert session.page_request_id == draft_id + (not same_text)
            session.submit_frame("same completed request")
            await eventually(lambda: session.processed_revision == 3)
            for _ in range(10):
                await asyncio.sleep(0)
            assert [call["question"] for call in backend.replies] == [completed]
            assert next(event for event in events if event["type"] == "start")["source"] == "page"
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


def test_mathematical_feedback_continues_while_a_page_request_is_a_draft():
    class CheckingBackend(PageBackend):
        async def output_probability(self, snapshot, history, student_context, **kwargs):
            self.speak_calls.append(snapshot.revision)
            return 1.0

    async def run():
        backend = CheckingBackend(
            {
                1: page_observation("Find x", ready=False, work="WORK: x = 1."),
                2: page_observation("Find x", ready=False, work="WORK: X = 1."),
                3: page_observation("Find x", work="WORK: X = 1."),
            }
        )
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("math beside an unfinished request")
        session.start()
        try:
            await eventually(lambda: any(event["type"] == "done" for event in events))
            session.submit_frame("a mathematical symbol changed")
            await eventually(lambda: sum(event["type"] == "done" for event in events) == 2)
            assert backend.speak_calls == [1, 2]
            assert [event["source"] for event in events if event["type"] == "start"] == [
                "automatic",
                "automatic",
            ]
            assert [call["question"] for call in backend.replies] == [demo.AUTO_REPLY_REQUEST] * 2
            session.submit_frame("request now ready")
            await eventually(lambda: sum(event["type"] == "done" for event in events) == 3)
            assert backend.replies[-1]["question"] == "Find x"
            assert [event["source"] for event in events if event["type"] == "start"] == [
                "automatic",
                "automatic",
                "page",
            ]
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


def test_erased_reappearing_and_changed_page_requests_are_new_occurrences():
    async def run():
        backend = PageBackend(
            {
                1: page_observation("Find x."),
                2: page_observation(),
                3: page_observation("Find x."),
                4: page_observation("Explain the sign."),
            }
        )
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("first question")
        session.start()
        try:
            await eventually(lambda: any(event["type"] == "done" for event in events))
            session.submit_frame("question erased")
            await eventually(lambda: session.processed_revision == 2)
            assert session.visible_page_request is None
            assert not session._has_pending_page_request()
            session.submit_frame("same question rewritten")
            await eventually(lambda: sum(event["type"] == "done" for event in events) == 2)
            session.submit_frame("different question")
            await eventually(lambda: sum(event["type"] == "done" for event in events) == 3)
            assert [call["question"] for call in backend.replies] == [
                "Find x.",
                "Find x.",
                "Explain the sign.",
            ]
            assert session.handled_page_request_id == session.page_request_id
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


def test_page_request_arriving_during_reply_is_answered_after_it_without_interrupting():
    async def run():
        backend = PageBackend(
            {1: page_observation("Explain this step."), 2: page_observation("Find x.")},
            hold_replies=True,
        )
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("first question")
        session.start()
        try:
            first, first_gate = await asyncio.wait_for(backend.reply_requests.get(), 2)
            assert first["question"] == "Explain this step."
            session.submit_frame("second question")
            await eventually(lambda: session.processed_revision == 2)
            assert len(backend.replies) == 1
            assert not backend.freed
            first_gate.set()
            second, second_gate = await asyncio.wait_for(backend.reply_requests.get(), 2)
            assert second["question"] == "Find x."
            assert second["revision"] == 2
            assert backend.freed == [1]
            second_gate.set()
            await eventually(lambda: sum(event["type"] == "done" for event in events) == 2)
            assert not any(event.get("interrupted") for event in events)
            assert [
                message["content"] for message in session.history if message["role"] == "user"
            ] == ["Explain this step.", "Find x."]
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)
        assert sorted(backend.freed) == [1, 2]

    asyncio.run(run())


@pytest.mark.parametrize("chat_question", ["Explain my notation first.", "Find x."])
def test_chat_request_precedes_pending_page_request_and_wakes_its_followup(chat_question):
    async def run():
        backend = PageBackend({1: page_observation("Find x.")})
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_text(chat_question)
        session.submit_frame("page question")
        session.start()
        try:
            await eventually(lambda: sum(event["type"] == "done" for event in events) == 2)
            assert [call["question"] for call in backend.replies] == [chat_question, "Find x."]
            assert [event["source"] for event in events if event["type"] == "start"] == [
                "chat",
                "page",
            ]
            assert not backend.speak_calls
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


def test_auto_disabled_holds_page_request_until_enabled():
    async def run():
        backend = PageBackend({1: page_observation("Find x.")})
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.auto = False
        session.submit_frame("page question")
        session.start()
        try:
            await eventually(lambda: session.latest is not None)
            for _ in range(10):
                await asyncio.sleep(0)
            assert not backend.replies
            session.auto = True
            session.probes_ready.set()
            session.responses_ready.set()
            await eventually(lambda: any(event["type"] == "done" for event in events))
            assert [call["question"] for call in backend.replies] == ["Find x."]
            assert not backend.speak_calls
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


def test_stop_consumes_visible_page_request_without_blocking_a_new_one():
    async def run():
        backend = PageBackend(
            {1: page_observation("Find x."), 2: "WAIT", 3: page_observation("Explain this sign.")}
        )
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.auto = False
        session.submit_frame("pending page question")
        session.start()
        try:
            await eventually(lambda: session.latest is not None)
            session.interrupt()
            session.auto = True
            session.probes_ready.set()
            session.responses_ready.set()
            session.submit_frame("same question")
            await eventually(lambda: session.processed_revision == 2)
            for _ in range(10):
                await asyncio.sleep(0)
            assert not backend.replies
            session.submit_frame("new question")
            await eventually(lambda: any(event["type"] == "done" for event in events))
            assert [call["question"] for call in backend.replies] == ["Explain this sign."]
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


def test_ambiguous_page_request_keeps_clarification_intent_and_verbatim_history():
    async def run():
        backend = PageBackend({1: page_observation("Find x?", clarify=True)})
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("ambiguous request")
        session.start()
        try:
            await eventually(lambda: any(event["type"] == "done" for event in events))
            question = backend.replies[0]["question"]
            assert question != "Find x?"
            assert "Find x?" in question and "before solving" in question
            assert list(session.history)[0] == {"role": "user", "content": "Find x?"}
            assert next(event for event in events if event["type"] == "start")["source"] == "page"
            assert not backend.speak_calls
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


@pytest.mark.parametrize(
    "notes",
    [
        "COMMENTS: The student asks, 'correct or not?'",
        "REQUEST: {broken JSON}\nCOMMENTS: correct or not?",
        'REQUEST: {"text":"Find x.","clarify":"false"}',
        'REQUEST: {"text":"Find x.","ready":"false"}',
        'REQUEST: null\nTASK: Exercise text says "Find x."',
    ],
)
def test_unstructured_malformed_and_exercise_questions_do_not_force_page_replies(notes):
    async def run():
        backend = PageBackend({1: notes})
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit, auto_interval=0)
        session.submit_frame("not a recognized user request")
        session.start()
        try:
            await eventually(lambda: len(backend.speak_calls) == 1)
            for _ in range(10):
                await asyncio.sleep(0)
            assert session.visible_page_request is None
            assert not backend.replies
        finally:
            backend.release_replies()
            await asyncio.wait_for(session.close(), 2)

    asyncio.run(run())


class TemplateTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        if text in {"yes", "no"}:
            return [1 if text == "yes" else 2]
        return [ord(char) + 10 for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token - 10) for token in ids)

    def apply_chat_template(self, messages, *, add_generation_prompt=False, **kwargs):
        # Match the relevant Qwen edge case: a system-only conversation is invalid.
        if messages and all(message["role"] == "system" for message in messages):
            raise ValueError("A non-system message is required")
        text = ""
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    "<image>" if part["type"] == "image" else part["text"] for part in content
                )
            text += f"<{message['role']}>{content}</{message['role']}>\n"
        return text + ("<assistant>" if add_generation_prompt else "")


class FrameProcessor:
    def __init__(self, tokenizer):
        self.tokenizer, self.inputs = tokenizer, []

    def apply_chat_template(self, messages, **kwargs):
        text = self.tokenizer.apply_chat_template(messages, **kwargs)
        ids = torch.tensor([self.tokenizer.encode(text)])
        types = torch.zeros_like(ids)
        start = text.index("<image>")
        types[:, start : start + len("<image>")] = 1
        inputs = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": types,
            "pixel_values": torch.tensor([[0.1, 0.2, 0.3]]),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }
        self.inputs.append(inputs)
        return inputs.copy()


class CacheLLM:
    def __init__(self):
        self.tokenizer = TemplateTokenizer()
        self.processor = FrameProcessor(self.tokenizer)
        self.config = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=0))
        self.blocks, self.calls, self.freed = [], [], []
        self.system_started, self.system_resume = asyncio.Event(), asyncio.Event()
        self.system_resume.set()
        self.fail_next_forward = False

    async def create_block(self):
        block = SimpleNamespace(index=len(self.blocks), num_tokens=0, tokens=[])
        self.blocks.append(block)
        return block

    async def free_block(self, block):
        assert block not in self.freed, "double free"
        self.freed.append(block)

    async def forward(self, input_ids, *, cache_view=None, write_to=None, **kwargs):
        view = list(cache_view or [])
        block = write_to if write_to is not None else view[-1]
        assert all(item not in self.freed for item in [*view, block])
        ids = torch.as_tensor(input_ids).flatten().tolist()
        self.calls.append(SimpleNamespace(ids=ids, view=view, block=block, inputs=kwargs))
        if not view:
            self.system_started.set()
            await self.system_resume.wait()
        if self.fail_next_forward:
            self.fail_next_forward = False
            raise RuntimeError("injected prefill failure")
        block.tokens.extend(ids)
        block.num_tokens += len(ids)
        return SimpleNamespace(logits=torch.tensor([0.0, 5.0, 1.0]))


async def fixed_generation(llm, view, output, limit, stopped):
    await llm.forward([ord("x") + 10], cache_view=view)
    yield "x"


@pytest.mark.parametrize(
    "stage, finish_reason",
    [
        ("Observe", "eos"),
        ("Observe", "token_limit"),
        ("Observe", "stopped"),
        ("Reply", "eos"),
        ("Reply", "token_limit"),
        ("Reply", "stopped"),
        ("Reply", "consumer_closed"),
    ],
)
def test_generation_finish_reason_and_token_counts_reach_the_inference_trace(stage, finish_reason):
    async def run():
        llm = CacheLLM()
        limit = 4 if finish_reason == "token_limit" else 8
        backend = AsyncLLMBackend(
            llm,
            prompts=demo.Prompts(system="Tutor.", observe="Inspect.", speak="Decide."),
            observation_tokens=limit,
            reply_tokens=limit,
        )
        queue = asyncio.Queue()
        backend.trace = Trace(queue)
        tokens = iter(backend.encode("Note") + [0])
        sample_calls = 0

        async def sample(output, *, temperature):
            nonlocal sample_calls
            sample_calls += 1
            return torch.tensor(next(tokens))

        llm.sample = sample
        stopped = lambda: finish_reason == "stopped" and sample_calls >= 4
        snapshot = None
        try:
            if stage == "Observe":
                snapshot = await backend.observe(Image.new("RGB", (16, 16)), 1, "None", stopped)
                assert snapshot.notes == "Note"
            else:
                frame = await backend.prepare_frame(Image.new("RGB", (16, 16)), "None", [])
                snapshot = Snapshot(1, [frame], "work")
                replies = backend.reply(snapshot, [], "Explain", stopped)
                if finish_reason == "consumer_closed":
                    assert await anext(replies) == "N"
                    await replies.aclose()
                else:
                    answers = [text async for text in replies]
                    assert answers[-1] == "Note"
            records = [queue.get_nowait() for _ in range(queue.qsize())]
            end = next(
                record
                for record in records
                if record.get("stage") == stage and record["event"] == "end"
            )
            done = next(record for record in records if record["event"] == "generation_done")
            expected_tokens = 1 if finish_reason == "consumer_closed" else 4
            for record in (done, end):
                assert record["finish_reason"] == finish_reason
                assert record["generated_tokens"] == expected_tokens
                assert record["token_limit"] == limit
            assert end["status"] == (
                "interrupted" if finish_reason in {"stopped", "consumer_closed"} else "ok"
            )
        finally:
            if snapshot is not None:
                await backend.free(snapshot.blocks)
            await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


@pytest.mark.parametrize(
    "error, expected_status",
    [(RuntimeError("inference failed"), "error"), (asyncio.CancelledError(), "cancelled")],
)
def test_generation_stop_metadata_does_not_hide_an_actual_inference_error(error, expected_status):
    queue = asyncio.Queue()
    trace = Trace(queue)
    with pytest.raises(type(error)):
        with trace.span("Observe"):
            trace.generation_done(2, 384, "stopped")
            raise error
    records = [queue.get_nowait() for _ in range(queue.qsize())]
    end = next(record for record in records if record["event"] == "end")
    assert end["status"] == expected_status
    assert end["finish_reason"] == "stopped"


def test_closing_session_during_observe_traces_partial_but_does_not_publish_it():
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(
            llm, prompts=demo.Prompts(system="Tutor.", observe="Inspect.", speak="Decide.")
        )
        queue = asyncio.Queue()
        backend.trace = Trace(queue)
        sample_started, sample_resume = asyncio.Event(), asyncio.Event()
        tokens = iter(backend.encode("REQUEST: null\nWORK: unfinished") + [0])
        sample_calls = 0

        async def sample(output, *, temperature):
            nonlocal sample_calls
            sample_calls += 1
            if sample_calls == 2:
                sample_started.set()
                await sample_resume.wait()
            return torch.tensor(next(tokens))

        llm.sample = sample
        events = []

        async def emit(event):
            events.append(event)

        session = TutorSession(backend, emit)
        session.auto = False
        session.submit_frame(Image.new("RGB", (16, 16)))
        session.start()
        closing = None
        try:
            await asyncio.wait_for(sample_started.wait(), 2)
            closing = asyncio.create_task(session.close())
            await eventually(lambda: session.closed)
            assert not closing.done()
            sample_resume.set()
            await asyncio.wait_for(closing, 2)
            assert session.latest is None
            assert not any(event["type"] in {"observed", "start", "text"} for event in events)
            records = [queue.get_nowait() for _ in range(queue.qsize())]
            end = next(
                record
                for record in records
                if record.get("stage") == "Observe" and record["event"] == "end"
            )
            assert end["status"] == "interrupted"
            assert end["finish_reason"] == "stopped"
            assert end["generated_tokens"] == 2
            assert end["text"] == "RE"
        finally:
            sample_resume.set()
            await asyncio.wait_for(closing or session.close(), 2)
            await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


@pytest.mark.parametrize(
    "page_request_value, layout",
    [
        (None, "inline"),
        ({"text": "correct or not?", "clarify": False}, "inline"),
        ({"text": "Find x?", "clarify": True}, "inline"),
        ({"text": "whats ne", "clarify": True, "ready": False}, "inline"),
        ({"text": "Hello!", "ready": True, "clarify": False}, "next-line"),
        ({"text": "Hello!", "ready": True, "clarify": False}, "pretty"),
        (None, "next-line"),
        ({"text": "Hello!", "ready": True, "clarify": False}, "markdown"),
    ],
)
def test_observe_extracts_request_metadata_and_preserves_the_factual_assessment(
    page_request_value, layout
):
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(
            llm, prompts=demo.Prompts(system="Tutor.", observe="Inspect.", speak="Decide.")
        )
        facts = "TASK: Check the discriminant calculation.\nWORK: D = 4."
        check = "CHECK: D = 4 is wrong; the product -4(-6) is +24."
        payload = json.dumps(
            page_request_value, indent=2 if layout in {"pretty", "markdown"} else None
        )
        if layout == "inline":
            field = f"REQUEST: {payload}"
        elif layout == "markdown":
            field = f"**REQUEST:**\n{payload}**"
        else:
            field = f"REQUEST:\n{payload}"
        notes = f"{field}\nREVIEW: work\n{facts}\n{check}"

        async def generate(view, output, limit, stopped):
            await llm.forward(backend.encode(notes), cache_view=view)
            yield notes

        backend.generate = generate
        snapshot = await backend.observe(Image.new("RGB", (16, 16)), 1, "None", lambda: False)
        try:
            assert snapshot.request_known is True
            assert snapshot.page_request == (
                demo.PageRequest(**page_request_value) if page_request_value else None
            )
            assert snapshot.notes == notes
            assert snapshot.review_kind == "work" and snapshot.observation_valid is True
            assert snapshot.current_assessment == facts
        finally:
            await backend.free(snapshot.blocks)
            await backend.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "notes, valid, review_kind",
    [
        ("REQUEST: null\nREVIEW: wait\nTASK: none\nWORK: none", True, "wait"),
        (
            'REQUEST:\n{"text":"Hello!","ready":true}\nREVIEW: wait\nTASK: none\nWORK: none',
            True,
            "wait",
        ),
        ("REQUEST: null\nREVIEW: problem\nTASK: x + 1 = 2\nWORK: none", True, "problem"),
        ("WAIT", False, None),
        ("No new writing is visible.\n\nWAIT", False, None),
        ("REQUEST: null\nTASK: x + 1 = 2\nWORK: x = 1", True, None),
        ("REQUEST: null\nREVIEW: unknown\nTASK: x + 1 = 2\nWORK: x = 1", True, None),
        ("REQUEST: null\nREVIEW: work\nWORK: x = 1", True, "work"),
        ("REQUEST: null\nREVIEW: work\nTASK: x + 1 = 2", True, "work"),
        ("REQUEST: {broken JSON}\nREVIEW: work\nTASK: x + 1 = 2\nWORK: x = 1", True, "work"),
        ("REVIEW: work\nTASK: x + 1 = 2\nWORK: x = 1", True, "work"),
    ],
)
def test_real_observe_accepts_facts_without_a_complete_schema(notes, valid, review_kind):
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(
            llm, prompts=demo.Prompts(system="Tutor.", observe="Inspect.", speak="Decide.")
        )
        queue = asyncio.Queue()
        backend.trace = Trace(queue)

        async def generate(view, output, limit, stopped):
            await llm.forward(backend.encode(notes), cache_view=view)
            yield notes

        backend.generate = generate
        snapshot = await backend.observe(Image.new("RGB", (16, 16)), 1, "None.", lambda: False)
        try:
            assert snapshot.notes == notes
            assert snapshot.observation_valid is valid
            assert snapshot.review_kind == review_kind
            if valid:
                assert "REVIEW:" not in snapshot.current_assessment
            records = [queue.get_nowait() for _ in range(queue.qsize())]
            observed = next(
                record
                for record in records
                if record.get("stage") == "Observe" and record["event"] == "end"
            )
            assert observed["text"] == notes
        finally:
            await backend.free(snapshot.blocks)
            await backend.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "metadata",
    [
        'REQUEST:\n{"text": "Hello!", "ready": true} trailing junk',
        'REQUEST:\n{\n  "text": "Hello!"\n} {"text": "another value"}',
        'REQUEST:\n{\n  "text": "Hello!",\n  "ready": true,\n}',
        'REQUEST:\n{\n  "text": "Hello!",\n  "ready": true',
        'REQUEST:\n{"text": "  "}',
        'REQUEST:\n{"text": "Hello!", "ready": 1}',
        'REQUEST:\n{"text": "Hello!", "clarify": "false"}',
        "REQUEST:\nnull trailing junk",
    ],
)
def test_invalid_multiline_request_is_not_routed_and_its_metadata_is_not_copied(metadata):
    facts = "WORK: x + 1 = 2.\nCHECK: The equation is legible."
    notes = f"{metadata}\n{facts}"
    assert demo.parse_page_request(notes) == (False, None)
    assert Snapshot(1, [], notes).current_assessment == facts


def test_request_json_under_other_observation_fields_is_not_a_page_request():
    notes = 'COMMENTS:\n{"text": "Hello!", "ready": true}\nWORK: None.'
    assert demo.parse_page_request(notes) == (False, None)
    assert Snapshot(1, [], notes).current_assessment == notes


def test_system_prefix_is_initialized_once_and_preserves_multimodal_inputs():
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(llm)
        llm.system_resume.clear()
        first = asyncio.create_task(backend.start())
        await asyncio.wait_for(llm.system_started.wait(), 2)
        second = asyncio.create_task(backend.start())
        await asyncio.sleep(0)
        assert len(llm.blocks) == 1
        llm.system_resume.set()
        system, same_system = await asyncio.gather(first, second)
        assert system is same_system is backend.system_block
        frames = await asyncio.gather(
            *(backend.prepare_frame(Image.new("RGB", (16, 16)), str(i), []) for i in range(2))
        )
        assert sum(not call.view for call in llm.calls) == 1
        for frame in frames:
            call = next(call for call in llm.calls if call.block is frame)
            assert call.view == [system, frame]
            full = next(
                inputs
                for inputs in llm.processor.inputs
                if inputs["pixel_values"] is call.inputs["pixel_values"]
            )
            assert full["input_ids"].flatten().tolist() == system.tokens + frame.tokens
            size = len(backend.system_ids)
            for name in ("attention_mask", "mm_token_type_ids"):
                assert torch.equal(call.inputs[name], full[name][:, size:])
            assert call.inputs["pixel_values"] is full["pixel_values"]
            assert call.inputs["image_grid_thw"] is full["image_grid_thw"]
        await backend.free(frames)
        assert system not in llm.freed
        await backend.close()
        await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


def test_custom_prompts_and_system_blocks_are_private_and_initialized_lazily():
    async def run():
        llm = CacheLLM()
        prompts = demo.Prompts(system="Custom tutor rules.", observe="Check brackets {x}.")
        with pytest.raises(FrozenInstanceError):
            prompts.system = "Changed during inference"
        first = AsyncLLMBackend(llm, prompts=prompts)
        second = AsyncLLMBackend(
            llm, prompts=demo.Prompts(system=prompts.system, observe="Check the signs.")
        )
        assert first.system_block is second.system_block is None
        assert not llm.calls and not llm.blocks
        image = Image.new("RGB", (16, 16))
        first_frame = await first.prepare_frame(image, "previous {notes}", ["student {question}"])
        assert second.system_block is None
        second_frame = await second.prepare_frame(image, "None.", [])
        assert first.system_block is not second.system_block
        assert first.system_block.tokens == second.system_block.tokens
        assert sum(not call.view for call in llm.calls) == 2
        for backend, frame in ((first, first_frame), (second, second_frame)):
            system_text = llm.tokenizer.decode(backend.system_block.tokens)
            frame_text = llm.tokenizer.decode(frame.tokens)
            assert prompts.system in system_text
            assert demo.SYSTEM not in system_text
            assert demo.IMAGE_CAPTION in frame_text
            assert backend.prompts.observe not in frame_text
            call = next(call for call in llm.calls if call.block is frame)
            assert call.view == [backend.system_block, frame]
        first_text = llm.tokenizer.decode(first_frame.tokens)
        assert "previous {notes}" not in first_text and "student {question}" not in first_text
        first_observation, _ = first.observation_instruction("previous {notes}")
        assert prompts.observe in first_observation and "previous {notes}" in first_observation
        assert "student {question}" not in first_observation
        second_text = llm.tokenizer.decode(second_frame.tokens)
        assert prompts.observe not in second_text
        assert first_frame.tokens == second_frame.tokens
        await first.free([first_frame])
        await first.close()
        assert second.system_block not in llm.freed
        assert second_frame not in llm.freed
        await second.free([second_frame])
        await second.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


def test_observe_instructions_and_previous_facts_live_only_in_its_private_notes_block():
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(
            llm,
            prompts=demo.Prompts(system="Tutor.", observe="OBSERVE_ONLY_POLICY", speak="Decide."),
        )
        queue = asyncio.Queue()
        backend.trace = Trace(queue)
        previous = "TASK: Previous task.\nWORK: PRIVATE_PREVIOUS_WORK"
        raw = (
            "REQUEST: null\nREVIEW: work\nTASK: x² + 2x − 4 = 3\n"
            "WORK: x² + 2x − 4 = 3\nD = 4 + 16 = 20\nCHECK: D = 32."
        )

        async def generated_notes(view, output, limit, stopped):
            await llm.forward(backend.encode(raw), cache_view=view)
            yield raw

        backend.generate = generated_notes
        snapshot = await backend.observe(Image.new("RGB", (16, 16)), 1, previous, lambda: False)
        image, notes = snapshot.blocks
        try:
            instruction, instruction_ids = backend.observation_instruction(previous)
            image_text = llm.tokenizer.decode(image.tokens)
            assert demo.IMAGE_CAPTION in image_text
            assert (
                "OBSERVE_ONLY_POLICY" not in image_text
                and "PRIVATE_PREVIOUS_WORK" not in image_text
            )
            assert "Return REQUEST:" not in image_text
            private_calls = [call for call in llm.calls if call.block is notes]
            assert private_calls[0].ids == [*instruction_ids, *backend.observation_prefix_ids]
            assert notes.tokens[: len(instruction_ids)] == instruction_ids
            assert raw in llm.tokenizer.decode(notes.tokens)
            image_tokens = list(image.tokens)

            async def generated_reply(view, output, limit, stopped):
                async for text in fixed_generation(llm, view, output, limit, stopped):
                    yield text

            backend.generate = generated_reply
            before = len(llm.calls)
            assert [
                text
                async for text in backend.reply(snapshot, [], "Check this line.", lambda: False)
            ] == ["x"]
            for call in llm.calls[before:]:
                assert call.view[:-1] == [backend.system_block, image]
                cached_text = "".join(llm.tokenizer.decode(block.tokens) for block in call.view)
                assert "OBSERVE_ONLY_POLICY" not in cached_text
                assert "PRIVATE_PREVIOUS_WORK" not in cached_text
                assert "Return REQUEST:" not in cached_text and "D = 32" not in cached_text
                assert "D = 4 + 16 = 20" in cached_text
            assert image.tokens == image_tokens
            records = [queue.get_nowait() for _ in range(queue.qsize())]
            image_prompt = next(
                record
                for record in records
                if record.get("context") == "System | Image" and record["event"] == "prompt"
            )
            observe_prompt = next(
                record
                for record in records
                if record.get("context") == "System | Image 1 | Observe"
                and record["event"] == "prompt"
            )
            assert image_prompt["text"] == demo.IMAGE_CAPTION
            assert observe_prompt["text"] == instruction
            assert observe_prompt["input_tokens"] == len(instruction_ids)
            assert observe_prompt["response_prefix"] == demo.OBSERVATION_PREFIX
        finally:
            await backend.free(snapshot.blocks)
            await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


def test_shared_system_and_image_are_reused_without_private_observe_context():
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(
            llm,
            prompts=demo.Prompts(system="Custom tutor rules.", observe="Check signs carefully."),
        )

        async def generate(view, output, limit, stopped):
            async for text in fixed_generation(llm, view, output, limit, stopped):
                yield text

        backend.generate = generate
        image = Image.new("RGB", (16, 16))
        first = await backend.observe(image, 1, "None", lambda: False)
        events = []

        async def emit(event):
            events.append(event)

        second = await backend.inspect(
            image,
            2,
            first.notes,
            lambda: False,
            previous=first,
            student_context=[],
            event_gate=EventGate(),
            force=False,
            emit=emit,
        )
        system = backend.system_block
        assert len(first.blocks) == len(second.blocks) == 2
        assert system not in [*first.blocks, *second.blocks]
        event_call = next(
            call for call in llm.calls if "Compare the two paper" in llm.tokenizer.decode(call.ids)
        )
        assert event_call.view[:-1] == [system, first.blocks[0], second.blocks[0]]
        assert event_call.block in llm.freed
        await backend.output_probability(second, [], [])
        speak_call = llm.calls[-1]
        assert speak_call.view[:-1] == [system, second.blocks[0]]
        assert speak_call.block in llm.freed
        before = len(llm.calls)
        legacy = Snapshot(second.revision, second.blocks, second.notes)
        assert [text async for text in backend.reply(legacy, [], "Help", lambda: False)] == ["x"]
        for call in llm.calls[before:]:
            assert call.view[:-1] == [system, second.blocks[0]]
            assert call.block in llm.freed
        for call in llm.calls:
            if call.view:
                assert sum(block is system for block in call.view) == 1
                assert call.ids[: len(backend.system_ids)] != backend.system_ids
        session = TutorSession(backend, emit)
        await session.release(first)
        await session.release(second)
        assert system not in llm.freed
        await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


@pytest.mark.parametrize("with_review_metadata", [True, False])
def test_reply_excludes_internal_check_from_published_facts_and_private_kv(with_review_metadata):
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(
            llm, prompts=demo.Prompts(system="Tutor.", observe="Inspect.", speak="Decide.")
        )
        queue = asyncio.Queue()
        backend.trace = Trace(queue)
        facts = "TASK: x² + 2x − 4 = 3\nWORK: x² + 2x − 4 = 3\nD = 4 + 16 = 20"
        check = "CHECK: c = -7.\nD = 32."
        review = "REVIEW: work\n" if with_review_metadata else ""
        notes = f"REQUEST: null\n{review}{facts}\n{check}\nREADABILITY: Clear."

        async def generated_observation(view, output, limit, stopped):
            await llm.forward(backend.encode(notes), cache_view=view)
            yield notes

        backend.generate = generated_observation
        snapshot = await backend.observe(Image.new("RGB", (16, 16)), 1, "None.", lambda: False)
        image_block, observed_block = snapshot.blocks
        try:
            assert snapshot.observation_valid is True
            # Session publication removes internal checks before either response branch.
            snapshot.assessment = demo.clean_observe_notes(notes, include_check=False)
            assert snapshot.notes == notes
            assert check in llm.tokenizer.decode(observed_block.tokens)
            assert check not in snapshot.current_assessment
            assert "D = 4 + 16 = 20" in snapshot.current_assessment
            question = "Check the newly written calculation."
            messages = [
                {
                    "role": "user",
                    "content": demo.REPLY_TEMPLATE.format(
                        assessment=snapshot.current_assessment, question=question
                    ),
                }
            ]
            # Exact admission for image-only Reply; adding the Observe block would exceed it.
            cached_tokens = backend.system_block.num_tokens + image_block.num_tokens
            backend.context_limit = (
                cached_tokens + len(backend.encode_chat(messages)) + backend.reply_tokens + 1
            )

            async def generated_reply(view, output, limit, stopped):
                async for answer in fixed_generation(llm, view, output, limit, stopped):
                    yield answer

            backend.generate = generated_reply
            before = len(llm.calls)
            assert [
                answer async for answer in backend.reply(snapshot, [], question, lambda: False)
            ] == ["x"]
            reply_calls = llm.calls[before:]
            assert reply_calls[0].ids == backend.encode_chat(messages)
            for call in reply_calls:
                assert call.view[:-1] == [backend.system_block, image_block]
                assert observed_block not in call.view
                context = "".join(llm.tokenizer.decode(block.tokens) for block in call.view)
                assert "D = 32" not in context and "c = -7" not in context
                assert "D = 4 + 16 = 20" in context
            assert observed_block not in llm.freed and image_block not in llm.freed
            records = [queue.get_nowait() for _ in range(queue.qsize())]
            observe_end = next(
                record
                for record in records
                if record.get("stage") == "Observe" and record["event"] == "end"
            )
            assert observe_end["text"] == notes
            prompt = next(
                record
                for record in records
                if record.get("stage") == "Reply" and record["event"] == "prompt"
            )
            assert prompt["context"] == "System | Image 1 | Reply"
            assert prompt["cached_tokens"] == cached_tokens
            assert "D = 32" not in json.dumps(prompt, ensure_ascii=False)
        finally:
            await backend.free(snapshot.blocks)
            await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


def test_shared_system_failures_release_partial_blocks_and_allow_retry():
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(llm)
        llm.fail_next_forward = True
        with pytest.raises(RuntimeError, match="injected prefill failure"):
            await backend.start()
        assert backend.system_block is None
        assert llm.freed == llm.blocks
        system = await backend.start()
        llm.fail_next_forward = True
        with pytest.raises(RuntimeError, match="injected prefill failure"):
            await backend.prepare_frame(Image.new("RGB", (16, 16)), "None", [])
        assert system not in llm.freed
        assert llm.blocks[-1] in llm.freed
        await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


def test_context_budgets_include_shared_system_once():
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(llm)
        image = Image.new("RGB", (16, 16))
        frame = await backend.prepare_frame(image, "None", [])
        system_size = backend.system_block.num_tokens
        backend.context_limit = (
            system_size
            + frame.num_tokens
            + len(backend.observation_instruction("None")[1])
            + backend.observation_framing_tokens
            + backend.observation_tokens
            + backend.context_reserve
            - 1
        )
        with pytest.raises(ValueError, match="Image exceeds context budget"):
            await backend.prepare_frame(image, "None", [])
        probe_size = len(
            backend.encode_chat([{"role": "user", "content": "Changed? Answer only yes or no."}])
        )
        backend.context_limit = system_size + frame.num_tokens + probe_size
        with pytest.raises(ValueError, match="Probe exceeds context budget"):
            await backend.binary_probe([frame], "Changed?")
        backend.context_limit += 1
        assert await backend.binary_probe([frame], "Changed?") > 0.5
        snapshot = Snapshot(1, [frame], "work")
        question = [
            {
                "role": "user",
                "content": demo.REPLY_TEMPLATE.format(
                    assessment=snapshot.current_assessment, question="Help"
                ),
            }
        ]
        backend.context_limit = (
            system_size
            + frame.num_tokens
            + len(backend.encode_chat(question))
            + backend.reply_tokens
        )
        with pytest.raises(ValueError, match="Question exceeds the available context budget"):
            async for _ in backend.reply(snapshot, [], "Help", lambda: False):
                pass
        await backend.free([frame])
        await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


@pytest.mark.parametrize(
    "question", ["Please explain the solution in detail.", "Explain: " + "x" * 1950]
)
def test_long_reply_fits_an_admitted_frame_and_trims_old_dialogue(question):
    async def run():
        llm = CacheLLM()
        # This tokenizer counts characters as tokens; keep policy text out of the budget test.
        backend = AsyncLLMBackend(
            llm, prompts=demo.Prompts(system="Tutor rules.", observe="Inspect.", speak="Check.")
        )
        image = Image.new("RGB", (16, 16))
        student_context = [question]
        frame = await backend.prepare_frame(image, "", student_context)
        assert question not in llm.tokenizer.decode(frame.tokens)
        reserve = max(
            backend.context_reserve,
            backend.reply_tokens
            + len(
                backend.encode_chat(
                    [
                        {
                            "role": "user",
                            "content": demo.REPLY_TEMPLATE.format(
                                assessment="None.", question=question
                            ),
                        }
                    ]
                )
            )
            + backend.observation_tokens
            + 1,
        )
        padding = (
            backend.context_limit
            - backend.system_block.num_tokens
            - frame.num_tokens
            - len(backend.observation_instruction("")[1])
            - backend.observation_framing_tokens
            - backend.observation_tokens
            - reserve
        )
        assert padding > 0
        await backend.free([frame])
        explanation = "A detailed explanation of the solution.\n" * 20
        notes = "N" * backend.observation_tokens
        tokens = iter(backend.encode(notes + explanation) + [0])

        async def sample(output, *, temperature):
            return torch.tensor(next(tokens))

        llm.sample = sample
        # Exercise the real Observe header, generation, and closing at admission's boundary.
        snapshot = await backend.observe(
            image, 1, "N" * padding, lambda: False, student_context=student_context
        )
        assert snapshot.notes == notes
        history = [
            {"role": "user", "content": "Old question. " * 1000},
            {"role": "assistant", "content": "Old answer. " * 1000},
        ]
        calls_before = len(llm.calls)
        try:
            answers = [
                text async for text in backend.reply(snapshot, history, question, lambda: False)
            ]
            assert len(backend.encode(answers[-1])) > 256
            assert answers[-1] == explanation
            # Retain the current request even when old dialogue must be dropped.
            calls = llm.calls[calls_before:]
            assert calls[0].ids == backend.encode_chat(
                [
                    {
                        "role": "user",
                        "content": demo.REPLY_TEMPLATE.format(assessment=notes, question=question),
                    }
                ]
            )
            assert sum(block.num_tokens for block in calls[-1].view) < backend.context_limit
        finally:
            await backend.free(snapshot.blocks)
            await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


def test_speak_retains_the_request_and_start_of_a_long_reply():
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(llm)
        frame = await backend.prepare_frame(Image.new("RGB", (16, 16)), "None", [])
        question = "Please explain why the discriminant is negative."
        answer = "ANSWER_START " + "x" * (backend.reply_tokens - len("ANSWER_START "))
        history = [
            {"role": "user", "content": "Old question. " * 100},
            {"role": "assistant", "content": "Old answer. " * 100},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        try:
            await backend.output_probability(Snapshot(1, [frame], "work"), history, [])
            prompt = llm.tokenizer.decode(llm.calls[-1].ids)
            assert question in prompt
            assert answer in prompt
        finally:
            await backend.free([frame])
            await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


@pytest.mark.parametrize(
    "raw_notes",
    [
        'REQUEST: {"text":"Check this.","clarify":false}\nCHECK: Current work has a sign error.',
        "WAIT",
    ],
)
def test_latest_assessment_follows_old_dialogue_in_speak_and_reply(raw_notes):
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(llm)
        assessment = "CHECK: Current work has a sign error."
        old_reply = "I see the equation. What would you like to do with it?"
        frame = await backend.prepare_frame(Image.new("RGB", (16, 16)), "Earlier notes.", [])
        notes = await llm.create_block()
        await llm.forward(
            [
                *backend.observation_prefix_ids,
                *backend.encode(raw_notes),
                *backend.observation_suffix_ids,
            ],
            cache_view=[backend.system_block, frame, notes],
        )
        snapshot = Snapshot(
            2,
            [frame, notes],
            raw_notes,
            assessment=assessment if raw_notes == "WAIT" else None,
        )
        history = [
            {"role": "user", "content": "Please check the equation."},
            {"role": "assistant", "content": old_reply},
        ]
        saved_history = [dict(message) for message in history]

        async def generate(view, output, limit, stopped):
            async for text in fixed_generation(llm, view, output, limit, stopped):
                yield text

        backend.generate = generate
        try:
            replied_assessment = "EARLIER: only the original equation had been checked."
            await backend.output_probability(
                snapshot, history, [], replied_assessment=replied_assessment
            )
            speak_prompt = llm.tokenizer.decode(llm.calls[-1].ids)
            assert speak_prompt.rindex(assessment) > speak_prompt.rindex(old_reply)
            assert speak_prompt.rindex(assessment) > speak_prompt.rindex(replied_assessment)
            assert 'REQUEST: {"text":"Check this."' not in speak_prompt
            question = "Explain the current mistake."
            calls_before = len(llm.calls)
            assert [
                text async for text in backend.reply(snapshot, history, question, lambda: False)
            ] == ["x"]
            reply_prompt = llm.tokenizer.decode(llm.calls[calls_before].ids)
            assert reply_prompt.rindex(question) > reply_prompt.rindex(assessment)
            assert reply_prompt.rindex(assessment) > reply_prompt.rindex(old_reply)
            assert 'REQUEST: {"text":"Check this."' not in reply_prompt
            assert history == saved_history
        finally:
            await backend.free(snapshot.blocks)
            await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


def test_forced_frame_bypasses_event_probe_but_keeps_shared_system():
    async def run():
        llm = CacheLLM()
        backend = AsyncLLMBackend(llm)

        async def generate(view, output, limit, stopped):
            async for text in fixed_generation(llm, view, output, limit, stopped):
                yield text

        backend.generate = generate
        image = Image.new("RGB", (16, 16))
        snapshots, events = [], []

        async def emit(event):
            events.append(event)

        try:
            first = await backend.observe(image, 1, "None", lambda: False)
            snapshots.append(first)
            calls_before = len(llm.calls)
            paused = await backend.inspect(
                image,
                2,
                first.notes,
                lambda: False,
                previous=first,
                student_context=[],
                event_gate=EventGate(),
                force=True,
                emit=emit,
            )
            snapshots.append(paused)
            assert paused.revision == 2
            assert not events
            calls = llm.calls[calls_before:]
            assert all(
                "Compare the two paper" not in llm.tokenizer.decode(call.ids) for call in calls
            )
            assert all(call.view[0] is backend.system_block for call in calls)
        finally:
            for snapshot in snapshots:
                await backend.free(snapshot.blocks)
            await backend.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


class MemoryWebSocket:
    """Exercise the actual endpoint and session tasks without a network server."""

    def __init__(self):
        self.headers = {}
        self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()
        self.close_code = None

    async def accept(self):
        await asyncio.sleep(0)  # Allow simultaneous connection attempts.

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, event):
        await self.outgoing.put(event)

    async def close(self, code=1000, reason=None):
        self.close_code = code

    async def event(self):
        return await asyncio.wait_for(self.outgoing.get(), 2)

    async def disconnect(self):
        await self.incoming.put({"type": "websocket.disconnect", "code": 1000})


@pytest.fixture
def app_without_model(monkeypatch):
    closed = []

    async def close_llm():
        closed.append("llm")

    llm = SimpleNamespace(processor=object(), close=close_llm, backends=[])

    class SessionBackend:
        def __init__(self, model, *, prompts=None):
            assert model is llm
            self.llm = model
            self.prompts = prompts if prompts is not None else demo.Prompts()
            self.system_block = None
            llm.backends.append(self)

        async def close(self):
            closed.append(self)

    monkeypatch.setattr(demo, "create_llm", lambda *args: llm)
    monkeypatch.setattr(demo, "AsyncLLMBackend", SessionBackend)
    monkeypatch.setattr(demo, "FrameGate", lambda: SimpleNamespace(close=lambda: None))
    app = demo.make_app()
    routes = {route.path: route.endpoint for route in app.routes}
    return app, routes, llm, closed


def ready_event(prompts=None):
    return {"type": "ready", "prompts": asdict(prompts or demo.Prompts())}


def test_trace_forwards_speak_start_before_a_blocked_probe_finishes(monkeypatch, app_without_model):
    app, routes, llm, _ = app_without_model

    class SpeakingBackend(demo.AsyncLLMBackend):
        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)
            self.probe_started = asyncio.Event()
            self.probe_release = asyncio.Event()

        async def inspect(self, image, revision, previous_notes, stopped, **kwargs):
            return Snapshot(revision, [revision], "A question for the tutor.")

        @demo.traced("Speak", frame="snapshot", result=True)
        async def output_probability(self, snapshot, history, student_context, **kwargs):
            self.probe_started.set()
            await self.probe_release.wait()
            return 0.0

        async def free(self, blocks):
            pass

    monkeypatch.setattr(demo, "AsyncLLMBackend", SpeakingBackend)
    monkeypatch.setattr(demo, "FrameGate", lambda: None)
    jpeg = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(jpeg, format="JPEG")

    async def run():
        socket = MemoryWebSocket()
        async with app.router.lifespan_context(app):
            task = asyncio.create_task(routes["/ws"](socket))
            try:
                assert await socket.event() == ready_event()
                await socket.incoming.put(
                    {
                        "type": "websocket.receive",
                        "text": json.dumps({"type": "trace", "enabled": True}),
                    }
                )
                await socket.incoming.put({"type": "websocket.receive", "bytes": jpeg.getvalue()})
                records = []
                while not any(
                    record.get("stage") == "Speak" and record["event"] == "begin"
                    for record in records
                ):
                    event = await socket.event()
                    if event["type"] == "trace":
                        records.extend(event["events"])
                backend = llm.backends[0]
                assert backend.probe_started.is_set()
                assert not backend.probe_release.is_set()
                assert not any(record["event"] == "decision" for record in records)
                backend.probe_release.set()
                while not any(record["event"] == "decision" for record in records):
                    event = await socket.event()
                    if event["type"] == "trace":
                        records.extend(event["events"])
                decision = next(record for record in records if record["event"] == "decision")
                assert decision["speak"] is False
            finally:
                for backend in llm.backends:
                    backend.probe_release.set()
                await socket.disconnect()
                await asyncio.wait_for(task, 2)

    asyncio.run(run())


@pytest.mark.parametrize("subscribe_before_burst", [False, True])
def test_full_trace_preserves_hidden_records_and_bursts_before_subscription_or_during_backpressure(
    monkeypatch, app_without_model, subscribe_before_burst
):
    app, routes, llm, _ = app_without_model

    class PassiveSession(TutorSession):
        def start(self):
            pass  # Exercise trace transport independently of model generation.

    class BackpressuredSocket(MemoryWebSocket):
        def __init__(self):
            super().__init__()
            self.trace_waiting, self.trace_release = asyncio.Event(), asyncio.Event()

        async def send_json(self, event):
            if event["type"] == "trace":
                self.trace_waiting.set()
                await self.trace_release.wait()
            await super().send_json(event)

    monkeypatch.setattr(demo, "TutorSession", PassiveSession)

    async def run():
        socket = BackpressuredSocket()

        async def subscribe():
            await socket.incoming.put(
                {
                    "type": "websocket.receive",
                    "text": json.dumps({"type": "trace", "enabled": True}),
                }
            )

        async with app.router.lifespan_context(app):
            task = asyncio.create_task(routes["/ws"](socket))
            try:
                assert await socket.event() == ready_event()
                if subscribe_before_burst:
                    await subscribe()
                    await asyncio.wait_for(socket.trace_waiting.wait(), 2)
                trace = llm.backends[0].trace
                with trace.span("System", input_tokens=42) as span:
                    trace.add("prompt", role="system", text="Complete system prompt")
                    span.data["cached_tokens"] = 42
                with trace.span("Frame filter", frame=7) as span:
                    span.data.update(accepted=True, reason="refresh", score=0.25)
                trace.add(
                    "context",
                    stage="Reply",
                    frame=7,
                    context="System | Image 7 | Observe 7 | Reply",
                    source="page",
                )
                trace.add(
                    "prompt",
                    stage="Reply",
                    frame=7,
                    messages=[{"role": "user", "content": "Explain the calculation."}],
                    input_tokens=27,
                    cached_tokens=101,
                )
                trace.generation_done(18, 2048, "eos")
                for number in range(300):
                    trace.add("diagnostic", number=number)
                trace.add("reply_completed", text="Burst complete", source="page")
                if not subscribe_before_burst:
                    assert socket.outgoing.empty()
                    await subscribe()
                socket.trace_release.set()

                records, batches = [], []
                while not any(record.get("text") == "Burst complete" for record in records):
                    event = await socket.event()
                    assert event["type"] == "trace"
                    batches.append(event["events"])
                    records.extend(event["events"])
                assert records[0]["event"] == "session_started"
                assert records[0]["prompts"] == asdict(demo.Prompts())
                assert [record["number"] for record in records if "number" in record] == list(
                    range(300)
                )
                assert not any("dropped" in record for record in records)
                assert all(0 < len(batch) <= 16 for batch in batches)
                assert {record["session"] for record in records} == {1}
                assert [record["t_ms"] for record in records] == sorted(
                    record["t_ms"] for record in records
                )
                system = [record for record in records if record.get("stage") == "System"]
                assert [record["event"] for record in system] == ["begin", "prompt", "end"]
                assert system[1]["text"] == "Complete system prompt"
                assert system[2]["input_tokens"] == system[2]["cached_tokens"] == 42
                assert system[2]["duration_ms"] >= 0 and system[2]["status"] == "ok"
                filtered = next(
                    record
                    for record in records
                    if record.get("stage") == "Frame filter" and record["event"] == "end"
                )
                assert filtered["frame"] == 7 and filtered["reason"] == "refresh"
                assert filtered["accepted"] is True and filtered["score"] == 0.25
                context = next(record for record in records if record["event"] == "context")
                assert context["source"] == "page" and "Image 7" in context["context"]
                prompt = next(
                    record
                    for record in records
                    if record.get("stage") == "Reply" and record["event"] == "prompt"
                )
                assert prompt["messages"] == [
                    {"role": "user", "content": "Explain the calculation."}
                ]
                assert prompt["input_tokens"] == 27 and prompt["cached_tokens"] == 101
                generation = next(
                    record for record in records if record["event"] == "generation_done"
                )
                assert generation["generated_tokens"] == 18
                assert generation["token_limit"] == 2048 and generation["finish_reason"] == "eos"
            finally:
                socket.trace_release.set()
                await socket.disconnect()
                await asyncio.wait_for(task, 2)

    asyncio.run(run())


def test_full_trace_archive_keeps_its_own_restarts_and_excludes_other_connections(
    monkeypatch, app_without_model
):
    app, routes, llm, _ = app_without_model

    class PassiveSession(TutorSession):
        def start(self):
            pass

    monkeypatch.setattr(demo, "TutorSession", PassiveSession)

    async def run():
        sockets = [MemoryWebSocket(), MemoryWebSocket()]
        tasks = []

        async def command(socket, data):
            await socket.incoming.put({"type": "websocket.receive", "text": json.dumps(data)})

        async def collect(socket, marker):
            records = []
            while not any(record.get("text") == marker for record in records):
                event = await socket.event()
                assert event["type"] == "trace"
                records.extend(event["events"])
            return records

        async with app.router.lifespan_context(app):
            try:
                for socket in sockets:
                    tasks.append(asyncio.create_task(routes["/ws"](socket)))
                    assert await socket.event() == ready_event()
                original, peer = llm.backends
                original.trace.add("prompt", text="First connection before restart")
                peer.trace.add("prompt", text="Second connection only")
                replacement = demo.Prompts(
                    system="Restarted system", observe="Inspect current work"
                )
                await command(sockets[0], {"type": "restart", "prompts": asdict(replacement)})
                assert await sockets[0].event() == ready_event(replacement)
                fresh = llm.backends[-1]
                fresh.trace.add("prompt", text="First connection after restart")
                fresh.trace.add("reply_completed", text="First archive complete")
                peer.trace.add("reply_completed", text="Second archive complete")
                for socket in sockets:
                    await command(socket, {"type": "trace", "enabled": True})
                first, second = await asyncio.gather(
                    collect(sockets[0], "First archive complete"),
                    collect(sockets[1], "Second archive complete"),
                )
                assert [
                    (record["session"], record["text"])
                    for record in first
                    if record["event"] == "prompt"
                ] == [(1, "First connection before restart"), (2, "First connection after restart")]
                assert [
                    (record["session"], record["text"])
                    for record in second
                    if record["event"] == "prompt"
                ] == [(1, "Second connection only")]
                starts = [record for record in first if record["event"] == "session_started"]
                assert [(record["session"], record["prompts"]) for record in starts] == [
                    (1, asdict(demo.Prompts())),
                    (2, asdict(replacement)),
                ]
                assert any(record["event"] == "session_restart_requested" for record in first)
                assert not any(record["event"] == "session_restart_requested" for record in second)
                assert [record["t_ms"] for record in first] == sorted(
                    record["t_ms"] for record in first
                )
            finally:
                for socket in sockets:
                    await socket.disconnect()
                await asyncio.wait_for(asyncio.gather(*tasks), 2)

    asyncio.run(run())


def test_photo_and_video_page_requests_are_answered_on_first_reading(
    monkeypatch, app_without_model
):
    app, routes, llm, _ = app_without_model

    class StillBackend(demo.AsyncLLMBackend):
        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)
            self.reads, self.questions = [], []

        async def inspect(self, image, revision, previous_notes, stopped, **kwargs):
            self.reads.append(revision)
            question = "Hello!" if revision == 1 else "Thanks!"
            notes = page_observation(question, work="WORK: null")
            return Snapshot(
                revision,
                [],
                notes,
                page_request=demo.PageRequest(question),
                request_known=True,
                observation_valid=True,
            )

        async def output_probability(self, *args, **kwargs):
            raise AssertionError("Modern page addresses must bypass Speak")

        async def reply(self, snapshot, history, question, stopped):
            self.questions.append(question)
            yield "Acknowledged."

        async def free(self, blocks):
            pass

    monkeypatch.setattr(demo, "AsyncLLMBackend", StillBackend)
    monkeypatch.setattr(demo, "FrameGate", lambda: None)
    jpeg = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(jpeg, format="JPEG")

    async def run():
        socket = MemoryWebSocket()
        events = []

        async def until(kind):
            while True:
                event = await socket.event()
                events.append(event)
                if event["type"] == kind:
                    return event

        async with app.router.lifespan_context(app):
            task = asyncio.create_task(routes["/ws"](socket))
            try:
                assert await socket.event() == ready_event()
                await socket.incoming.put(
                    {
                        "type": "websocket.receive",
                        "text": json.dumps({"type": "frame", "force": False, "still": True}),
                    }
                )
                await socket.incoming.put({"type": "websocket.receive", "bytes": jpeg.getvalue()})
                await until("done")
                backend = llm.backends[0]
                assert backend.reads == [1] and backend.questions == ["Hello!"]
                assert (
                    next(event for event in events if event["type"] == "start")["source"] == "page"
                )
                await socket.incoming.put({"type": "websocket.receive", "bytes": jpeg.getvalue()})
                await until("done")
                assert backend.reads == [1, 2]
                assert backend.questions == ["Hello!", "Thanks!"]
            finally:
                await socket.disconnect()
                await asyncio.wait_for(task, 2)

    asyncio.run(run())


def test_forced_frame_header_applies_only_to_next_binary_payload(monkeypatch, app_without_model):
    app, routes, _, _ = app_without_model
    received = []
    still_flags = []

    class TransportSession(TutorSession):
        def start(self):
            pass  # Exercise image transport without starting model tasks.

        def submit_frame(self, image, *, force=False, trusted_still=False):
            super().submit_frame(image, force=force, trusted_still=trusted_still)
            received.append((self.revision, image.size, force))
            still_flags.append(trusted_still)

    monkeypatch.setattr(demo, "TutorSession", TransportSession)
    jpeg = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(jpeg, format="JPEG")
    data = jpeg.getvalue()

    async def run():
        socket = MemoryWebSocket()

        async def header(force, still=None):
            data = {"type": "frame", "force": force}
            if still is not None:
                data["still"] = still
            await socket.incoming.put({"type": "websocket.receive", "text": json.dumps(data)})

        async def frame(payload=data):
            await socket.incoming.put({"type": "websocket.receive", "bytes": payload})
            return await socket.event()

        async with app.router.lifespan_context(app):
            task = asyncio.create_task(routes["/ws"](socket))
            try:
                assert await socket.event() == ready_event()
                await header(True)
                assert await frame() == {"type": "frame_received", "revision": 1}
                assert received[-1] == (1, (16, 16), True)
                assert await frame() == {"type": "frame_received", "revision": 2}
                assert received[-1] == (2, (16, 16), False)

                # A bad JPEG consumes the marker; it must not force the next image.
                await header(True)
                assert (await frame(b"not an image"))["type"] == "error"
                assert len(received) == 2
                assert await frame() == {"type": "frame_received", "revision": 3}
                assert received[-1] == (3, (16, 16), False)

                await header(False)
                assert await frame() == {"type": "frame_received", "revision": 4}
                assert received[-1] == (4, (16, 16), False)
                await header(True)
                for invalid in (1, "true", None):
                    await header(invalid)
                    assert (await socket.event())["type"] == "error"
                assert await frame() == {"type": "frame_received", "revision": 5}
                assert received[-1] == (5, (16, 16), False)
                assert still_flags == [False] * 5

                await header(True, True)
                assert await frame() == {"type": "frame_received", "revision": 6}
                assert still_flags[-1] is True
                assert await frame() == {"type": "frame_received", "revision": 7}
                assert still_flags[-1] is False
                await header(True, True)
                assert (await frame(b"not an image"))["type"] == "error"
                assert await frame() == {"type": "frame_received", "revision": 8}
                assert still_flags[-1] is False
                await header(True, True)
                await header(False, "true")
                assert (await socket.event())["type"] == "error"
                assert await frame() == {"type": "frame_received", "revision": 9}
                assert still_flags[-1] is False

                await header(True, True)
                await socket.incoming.put(
                    {
                        "type": "websocket.receive",
                        "text": json.dumps({"type": "restart", "prompts": asdict(demo.Prompts())}),
                    }
                )
                assert await socket.event() == ready_event()
                assert await frame() == {"type": "frame_received", "revision": 1}
                assert received[-1] == (1, (16, 16), False) and still_flags[-1] is False
            finally:
                await socket.disconnect()
                await asyncio.wait_for(task, 2)

    asyncio.run(run())


@pytest.mark.parametrize("cancel_during_close", [False, True])
def test_multiuser_admission_isolation_and_slot_draining(
    monkeypatch, app_without_model, cancel_during_close
):
    app, routes, llm, closed = app_without_model
    sessions = []

    class TrackedSession(TutorSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closing, self.allow_close = asyncio.Event(), asyncio.Event()
            self.allow_close.set()
            sessions.append(self)

        async def close(self):
            self.closing.set()
            await self.allow_close.wait()
            await super().close()

    monkeypatch.setattr(demo, "TutorSession", TrackedSession)

    async def run():
        tasks, sockets = [], []

        async def connect():
            socket = MemoryWebSocket()
            sockets.append(socket)
            tasks.append(asyncio.create_task(routes["/ws"](socket)))
            return socket, await socket.event()

        async with app.router.lifespan_context(app):
            try:
                peers = await asyncio.gather(*(connect() for _ in range(demo.MAX_USERS)))
                assert all(event == ready_event() for _, event in peers)
                assert len(sessions) == demo.MAX_USERS == 4
                assert all(session.backend.llm is llm for session in sessions)
                assert len({id(session.backend) for session in sessions}) == demo.MAX_USERS
                assert len({id(session.history) for session in sessions}) == demo.MAX_USERS
                health = await routes["/health"]()
                assert health == {
                    "ready": True,
                    "busy": True,
                    "active_sessions": demo.MAX_USERS,
                    "max_sessions": demo.MAX_USERS,
                }

                # Text and emitted messages belong only to their own connection.
                await peers[0][0].incoming.put(
                    {"type": "websocket.receive", "text": json.dumps({"type": "text", "text": "A"})}
                )
                await eventually(lambda: bool(sessions[0].student_inputs))
                assert list(sessions[0].student_inputs) == ["A"]
                assert all(not session.student_inputs for session in sessions[1:])
                await sessions[0].send("text", text="Only for A")
                assert await peers[0][0].event() == {"type": "text", "text": "Only for A"}
                assert all(socket.outgoing.empty() for socket, _ in peers[1:])

                fifth, error = await connect()
                assert error["type"] == "error"
                assert str(demo.MAX_USERS) in error["text"]
                assert "ограничен" in error["text"].lower()
                await asyncio.wait_for(tasks[-1], 2)
                assert fifth.close_code == 1013
                assert len(app.state.connections) == demo.MAX_USERS

                sessions[0].allow_close.clear()
                await peers[0][0].disconnect()
                await asyncio.wait_for(sessions[0].closing.wait(), 2)
                assert len(app.state.connections) == demo.MAX_USERS
                rejected, error = await connect()
                assert error["type"] == "error"
                await asyncio.wait_for(tasks[-1], 2)
                assert rejected.close_code == 1013

                if cancel_during_close:
                    tasks[0].cancel()
                    await asyncio.sleep(0)
                    assert not tasks[0].done()
                    assert len(app.state.connections) == demo.MAX_USERS
                sessions[0].allow_close.set()
                await eventually(lambda: len(app.state.connections) == demo.MAX_USERS - 1)
                _, event = await connect()
                assert event == ready_event()
                assert len(app.state.connections) == demo.MAX_USERS
                assert closed == [sessions[0].backend]
                assert "llm" not in closed  # A disconnect must not close the shared model.
            finally:
                for session in sessions:
                    session.allow_close.set()
                for socket in sockets:
                    await socket.disconnect()
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
                assert all(
                    result is None
                    or (
                        cancel_during_close
                        and i == 0
                        and isinstance(result, asyncio.CancelledError)
                    )
                    for i, result in enumerate(results)
                )
            assert not app.state.connections
            assert (await routes["/health"]())["busy"] is False
        assert closed[-1] == "llm"
        assert len(closed) == len(llm.backends) + 1
        assert all(closed.count(backend) == 1 for backend in llm.backends)

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["gate", "start"])
def test_connection_setup_failure_releases_slot(monkeypatch, app_without_model, failure):
    app, routes, llm, closed = app_without_model

    def fail(*args, **kwargs):
        raise RuntimeError("injected session setup failure")

    if failure == "gate":
        monkeypatch.setattr(demo, "FrameGate", fail)
    else:
        monkeypatch.setattr(demo.TutorSession, "start", fail)

    async def run():
        async with app.router.lifespan_context(app):
            socket = MemoryWebSocket()
            await asyncio.wait_for(routes["/ws"](socket), 2)
            assert (await socket.event())["type"] == "error"
            assert socket.close_code is not None
            assert not app.state.connections
            assert (await routes["/health"]())["active_sessions"] == 0
            assert closed == llm.backends
        assert closed == [*llm.backends, "llm"]

    asyncio.run(run())


def test_prompt_restart_validates_before_draining_and_frees_only_its_own_caches(monkeypatch):
    sessions = []

    class RestartSession(TutorSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closing = asyncio.Event()
            sessions.append(self)

        def start(self):
            pass  # Seed actual cache blocks below without starting model generation.

        async def close(self):
            self.closing.set()
            await super().close()

    monkeypatch.setattr(demo, "TutorSession", RestartSession)
    monkeypatch.setattr(demo, "FrameGate", lambda: SimpleNamespace(close=lambda: None))

    async def run():
        llm = CacheLLM()
        llm_closed = []

        async def close_llm():
            llm_closed.append(True)

        llm.close = close_llm
        monkeypatch.setattr(demo, "create_llm", lambda *args: llm)
        app = demo.make_app()
        routes = {route.path: route.endpoint for route in app.routes}
        sockets = [MemoryWebSocket(), MemoryWebSocket()]
        release_reader = asyncio.Event()
        tasks = []

        async def restart(prompts):
            await sockets[0].incoming.put(
                {
                    "type": "websocket.receive",
                    "text": json.dumps({"type": "restart", "prompts": prompts}),
                }
            )

        async with app.router.lifespan_context(app):
            assert not llm.calls  # Startup loads no session's system prefix.
            try:
                for socket in sockets:
                    tasks.append(asyncio.create_task(routes["/ws"](socket)))
                    assert await socket.event() == ready_event()
                assert not llm.calls
                old, peer = sessions
                image = Image.new("RGB", (16, 16))
                for session in sessions:
                    frame = await session.backend.prepare_frame(image, "None.", [])
                    session.latest = Snapshot(1, [frame], "Earlier assessment")
                    session.history.append({"role": "assistant", "content": "Earlier hint"})
                old_system, peer_system = old.backend.system_block, peer.backend.system_block
                old_frame, peer_frame = old.latest.blocks[0], peer.latest.blocks[0]
                old.student_inputs.append("Old question")
                old.questions.append("Queued question")
                old.frame = (1, image, False)
                old.pending_hint = True
                old.event_gate.update(0.9)

                defaults = asdict(demo.Prompts())
                invalid = [
                    None,
                    {"system": "Missing observe"},
                    {**defaults, "extra": "not editable"},
                    {**defaults, "observe": 7},
                    {**defaults, "observe": "   "},
                    {**defaults, "system": "x" * (demo.MAX_PROMPT_CHARACTERS + 1)},
                    {**defaults, "system": "x" * demo.MAX_PROMPT_CHARACTERS},
                ]
                for prompts in invalid:
                    await restart(prompts)
                    error = await sockets[0].event()
                    assert error["type"] == "error" and error["operation"] == "restart"
                    assert not old.closed and len(sessions) == 2
                    assert old.latest.blocks == [old_frame]
                    assert not llm.freed

                async def reader():
                    await release_reader.wait()
                    assert old_system not in llm.freed and old_frame not in llm.freed
                    await old.send("text", text="Late output from old session")

                old.tasks = [asyncio.create_task(reader())]
                prompts = demo.Prompts(system="New rules.", observe="Check algebra signs.")
                await restart(asdict(prompts))
                await asyncio.wait_for(old.closing.wait(), 2)
                assert len(app.state.connections) == 2
                assert not llm.freed and len(sessions) == 2
                assert sockets[0].outgoing.empty()
                # Another user's session continues to emit while this one drains.
                await peer.send("text", text="Peer still working")
                assert await sockets[1].event() == {"type": "text", "text": "Peer still working"}
                release_reader.set()
                assert await sockets[0].event() == ready_event(prompts)
                fresh = sessions[-1]
                assert fresh is not old and fresh.backend is not old.backend
                assert fresh.backend.llm is peer.backend.llm is llm
                assert fresh.backend.prompts == prompts
                assert fresh.backend.system_block is None
                assert old_system in llm.freed and old_frame in llm.freed
                assert peer_system not in llm.freed and peer_frame not in llm.freed
                assert not fresh.history and not fresh.student_inputs and not fresh.questions
                assert fresh.frame is fresh.latest is None
                assert fresh.event_gate.probability is None and not fresh.pending_hint
                assert len(app.state.connections) == 2
                assert sockets[0].outgoing.empty() and not llm_closed
                new_frame = await fresh.backend.prepare_frame(image, "None.", [])
                fresh.latest = Snapshot(1, [new_frame], "Fresh assessment")
                assert prompts.system in llm.tokenizer.decode(fresh.backend.system_block.tokens)
            finally:
                release_reader.set()
                for socket in sockets:
                    await socket.disconnect()
                await asyncio.wait_for(asyncio.gather(*tasks), 2)
            assert len(llm.freed) == len(llm.blocks)
            assert not app.state.connections and not llm_closed
        assert llm_closed == [True]

    asyncio.run(run())


@pytest.mark.parametrize("capacity,graph_sizes", [(8, [1, 2, 4, 8]), (6, [1, 2, 4, 6])])
def test_multiuser_engine_capacity(monkeypatch, capacity, graph_sizes):
    assert demo.MAX_RUNNING_REQUESTS == demo.MAX_USERS * demo.BRANCHES_PER_USER
    monkeypatch.setattr(demo, "MAX_RUNNING_REQUESTS", capacity)
    calls = []

    def create(model, **kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setitem(sys.modules, "minisgl.llm", SimpleNamespace(AsyncLLM=create))
    config = demo.create_llm()
    assert len(calls) == 1
    assert config["max_running_req"] == config["cuda_graph_max_bs"] == capacity
    assert config["cuda_graph_bs"] == graph_sizes
    assert all(any(size >= batch for size in graph_sizes) for batch in range(1, capacity + 1))


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
