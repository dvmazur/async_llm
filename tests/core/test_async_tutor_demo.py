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
                if inputs["input_ids"].flatten().tolist() == system.tokens + frame.tokens
            )
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
            assert backend.prompts.observe in frame_text
            assert "output only WAIT" in frame_text
            assert "TASK:" in frame_text and "READABILITY:" in frame_text
            call = next(call for call in llm.calls if call.block is frame)
            assert call.view == [backend.system_block, frame]
        first_text = llm.tokenizer.decode(first_frame.tokens)
        assert "previous {notes}" in first_text and "student {question}" in first_text
        second_text = llm.tokenizer.decode(second_frame.tokens)
        assert prompts.observe not in second_text
        await first.free([first_frame])
        await first.close()
        assert second.system_block not in llm.freed
        assert second_frame not in llm.freed
        await second.free([second_frame])
        await second.close()
        assert len(llm.freed) == len(llm.blocks)

    asyncio.run(run())


def test_shared_system_is_read_once_by_each_branch_and_outlives_snapshots():
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
        assert speak_call.view[:-1] == [system, *second.blocks]
        assert speak_call.block in llm.freed
        before = len(llm.calls)
        assert [text async for text in backend.reply(second, [], "Help", lambda: False)] == ["x"]
        for call in llm.calls[before:]:
            assert call.view[:-1] == [system, *second.blocks]
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
            system_size + frame.num_tokens + backend.observation_tokens + 2048 - 1
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
        question = [{"role": "user", "content": "Help"}]
        snapshot = Snapshot(1, [frame], "work")
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


def test_forced_frame_header_applies_only_to_next_binary_payload(monkeypatch, app_without_model):
    app, routes, _, _ = app_without_model
    received = []

    class TransportSession(TutorSession):
        def start(self):
            pass  # Exercise image transport without starting model tasks.

        def submit_frame(self, image, *, force=False):
            super().submit_frame(image, force=force)
            received.append((self.revision, image.size, force))

    monkeypatch.setattr(demo, "TutorSession", TransportSession)
    jpeg = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(jpeg, format="JPEG")
    data = jpeg.getvalue()

    async def run():
        socket = MemoryWebSocket()

        async def header(force):
            await socket.incoming.put(
                {"type": "websocket.receive", "text": json.dumps({"type": "frame", "force": force})}
            )

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
