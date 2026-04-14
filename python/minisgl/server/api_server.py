from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Set, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from minisgl.async_reasoning.prompting import AsyncReasoningPrompting
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def _async_reasoning_prompt_variants(
    prompt: str | List[Dict[str, str]],
) -> Tuple[str | List[Dict[str, str]], str | List[Dict[str, str]]]:
    if not isinstance(prompt, str):
        assert len(prompt) == 1, "AsyncReasoning is not implemented for multiple messages, use /reset in shell mode"
        assert prompt[0].get("role") == "user"
        prompt = prompt[0].get("content")

    prompting = AsyncReasoningPrompting(prompt)
    prompt_reasoning = prompting.input_prompt + prompting.thinker_output_prefix
    prompt_summarization = prompting.input_prompt + prompting.thinker_output_prefix + prompting.writer_output_prefix
    return prompt_reasoning, prompt_summarization


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False
    # When True, AsyncReasoning: two prompt branches run in one tokenizer batch.
    async_reasoning: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False

    async_reasoning: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    # AsyncReasoning sub-requests deliver UserReply with child uids; route to parent.
    ack_route: Dict[int, int] = field(default_factory=dict)
    async_reasoning_children: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    def new_async_reasoning_user(self) -> Tuple[int, int, int]:
        """One logical AsyncReasoning session: two scheduler uids, both streamed under parent_uid."""
        parent_uid = self.uid_counter
        self.uid_counter += 1
        uid_reasoning = self.uid_counter
        self.uid_counter += 1
        uid_summarization = self.uid_counter
        self.uid_counter += 1
        self.ack_map[parent_uid] = []
        self.event_map[parent_uid] = asyncio.Event()
        self.ack_route[uid_reasoning] = parent_uid
        self.ack_route[uid_summarization] = parent_uid
        self.async_reasoning_children[parent_uid] = (uid_reasoning, uid_summarization)
        return parent_uid, uid_reasoning, uid_summarization

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                target_uid = self.ack_route.get(msg.uid, msg.uid)
                if target_uid not in self.ack_map:
                    continue
                self.ack_map[target_uid].append(msg)
                self.event_map[target_uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]
        children = self.async_reasoning_children.get(uid)
        required_finish: Set[int] = set(children) if children else {uid}
        finished_children: Set[int] = set()

        try:
            while True:
                await event.wait()
                event.clear()

                if uid not in self.ack_map:
                    break
                pending = self.ack_map[uid]
                self.ack_map[uid] = []
                for ack in pending:
                    yield ack
                    if ack.finished:
                        finished_children.add(ack.uid)
                if required_finish <= finished_children:
                    break
        finally:
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)
            self.async_reasoning_children.pop(uid, None)
            if children:
                for c in children:
                    self.ack_route.pop(c, None)

    async def stream_generate(self, uid: int):
        children = self.async_reasoning_children.get(uid)
        uid_reasoning: int | None = None
        uid_summarization: int | None = None
        if children:
            uid_reasoning, uid_summarization = children
        async for ack in self.wait_for_ack(uid):
            if uid_reasoning is not None and uid_summarization is not None:
                idx = 0 if ack.uid == uid_reasoning else 1
                payload = json.dumps({"index": idx, "text": ack.incremental_output})
                yield f"data: {payload}\n".encode()
            else:
                yield f"data: {ack.incremental_output}\n".encode()
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        children = self.async_reasoning_children.get(uid)
        if children:
            uid_reasoning, uid_summarization = children
            uid_to_index = {uid_reasoning: 0, uid_summarization: 1}
        else:
            uid_to_index = {}
        saw_role: Set[int] = set()

        async for ack in self.wait_for_ack(uid):
            idx = uid_to_index.get(ack.uid, 0)
            delta = {}
            if idx not in saw_role:
                delta["role"] = "assistant"
                saw_role.add(idx)
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": idx, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

        if children:
            end_chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [
                    {"delta": {}, "index": 0, "finish_reason": "stop"},
                    {"delta": {}, "index": 1, "finish_reason": "stop"},
                ],
            }
        else:
            end_chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
            }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        children = self.async_reasoning_children.pop(uid, None)
        if children:
            for c in children:
                self.ack_route.pop(c, None)
        wake = self.event_map.pop(uid, None)
        self.ack_map.pop(uid, None)
        if wake is not None:
            wake.set()
        logger.warning("Aborting request for user %s", uid)
        if children:
            await self.send_one(
                BatchTokenizerMsg(data=[AbortMsg(uid=c) for c in children])
            )
        else:
            await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    sp = SamplingParams(
        ignore_eos=req.ignore_eos,
        max_tokens=req.max_tokens,
    )
    if req.async_reasoning:
        text_reasoning, text_summarization = _async_reasoning_prompt_variants(req.prompt)
        assert isinstance(text_reasoning, str) and isinstance(text_summarization, str)
        parent_uid, uid_reasoning, uid_summarization = state.new_async_reasoning_user()
        await state.send_one(
            BatchTokenizerMsg(
                data=[
                    TokenizeMsg(uid=uid_reasoning, text=text_reasoning, sampling_params=sp),
                    TokenizeMsg(uid=uid_summarization, text=text_summarization, sampling_params=sp),
                ]
            )
        )
        stream_uid = parent_uid
    else:
        stream_uid = state.new_user()
        await state.send_one(
            TokenizeMsg(
                uid=stream_uid,
                text=req.prompt,
                sampling_params=sp,
            )
        )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(stream_uid), request, stream_uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    sp = SamplingParams(
        ignore_eos=req.ignore_eos,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )
    if req.async_reasoning:
        prompt_reasoning, prompt_summarization = _async_reasoning_prompt_variants(prompt)
        parent_uid, uid_reasoning, uid_summarization = state.new_async_reasoning_user()
        await state.send_one(
            BatchTokenizerMsg(
                data=[
                    TokenizeMsg(uid=uid_reasoning, text=prompt_reasoning, sampling_params=sp),
                    TokenizeMsg(
                        uid=uid_summarization, text=prompt_summarization, sampling_params=sp
                    ),
                ]
            )
        )
        stream_uid = parent_uid
    else:
        stream_uid = state.new_user()
        await state.send_one(
            TokenizeMsg(
                uid=stream_uid,
                text=prompt,
                sampling_params=sp,
            )
        )

    return StreamingResponse(
        state.stream_with_cancellation(
            state.stream_chat_completions(stream_uid), request, stream_uid
        ),
        media_type="text/event-stream",
    )


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    sp = SamplingParams(
        ignore_eos=req.ignore_eos,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )
    if req.async_reasoning:
        prompt_reasoning, prompt_summarization = _async_reasoning_prompt_variants(prompt)
        stream_uid, uid_reasoning, uid_summarization = state.new_async_reasoning_user()
        await state.send_one(
            BatchTokenizerMsg(
                data=[
                    TokenizeMsg(uid=uid_reasoning, text=prompt_reasoning, sampling_params=sp),
                    TokenizeMsg(
                        uid=uid_summarization, text=prompt_summarization, sampling_params=sp
                    ),
                ]
            )
        )
    else:
        stream_uid = state.new_user()
        await state.send_one(
            TokenizeMsg(
                uid=stream_uid,
                text=prompt,
                sampling_params=sp,
            )
        )

    async def _abort():
        await state.abort_user(stream_uid)

    return StreamingResponse(
        state.stream_generate(stream_uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )


async def read_stdin():
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        line = line.decode().rstrip("\n")


async def async_input(prompt=""):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            need_stop = False
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
                async_reasoning=bool(ENV.SHELL_ASYNC_REASONING),
            )
            cur_msg = ""
            cur_ar_streams: Dict[int, str] = {0: "", 1: ""}
            branch_header_shown: Set[int] = set()
            async for chunk in (await shell_completion(req)).body_iterator:
                if need_stop:
                    break
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                if req.async_reasoning:
                    try:
                        obj = json.loads(msg)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict) and "index" in obj and "text" in obj:
                        idx = int(obj["index"])
                        piece = str(obj["text"])
                        if idx not in branch_header_shown:
                            branch_label = "Reasoning" if idx == 0 else "Summarization"
                            print(f"\n--- {branch_label} ---\n", end="", flush=True)
                            branch_header_shown.add(idx)
                        print(piece, end="", flush=True)
                        cur_ar_streams[idx] = cur_ar_streams.get(idx, "") + piece
                        continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            if req.async_reasoning:
                history.append(
                    (
                        cmd,
                        "[reasoning]\n"
                        f"{cur_ar_streams.get(0, '')}\n\n"
                        "[summarization]\n"
                        f"{cur_ar_streams.get(1, '')}",
                    )
                )
            else:
                history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
