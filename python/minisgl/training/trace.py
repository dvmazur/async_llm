from dataclasses import dataclass, field, replace
from itertools import groupby


@dataclass(frozen=True)
class Token:
    token_id: int
    step: int
    block: int
    position: int
    view: tuple[tuple[int, int], ...]
    kind: str


@dataclass(frozen=True)
class TensorData:
    data: tuple
    shape: tuple[int, ...]
    dtype: str

    @classmethod
    def capture(cls, tensor):
        if tensor is None:
            return None
        return cls(
            tuple(tensor.detach().cpu().flatten().tolist()),
            tuple(tensor.shape),
            str(tensor.dtype).removeprefix("torch."),
        )

    def tensor(self):
        import torch

        dtype = getattr(torch, self.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unknown tensor dtype: {self.dtype}")
        return torch.tensor(self.data, dtype=dtype, device="cpu").reshape(self.shape)


@dataclass(frozen=True)
class ForwardRequest:
    """One request with ordered cache lengths captured before its batch writes."""

    input_ids: tuple[int, ...]
    write_to: int
    write_start: int
    cache_view: tuple[tuple[int, int], ...]
    pixel_values: TensorData | None = None
    image_grid_thw: TensorData | None = None
    mm_token_type_ids: TensorData | None = None
    mrope_positions: TensorData | None = None
    write_span: int | None = None
    cache_spans: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ForwardBatch:
    step: int
    kind: str
    requests: tuple[ForwardRequest, ...]


def _causal_view(view, block, advance):
    own = next((i for i in range(len(view) - 1, -1, -1) if view[i][0] == block), -1)
    return tuple((b, n + advance if i == own else n) for i, (b, n) in enumerate(view))


def _pack_tokens(tokens):
    batches = []
    for (step, kind), group in groupby(tokens, key=lambda t: (t.step, t.kind)):
        rows = list(group)
        written = {t.block for t in rows}
        requests = []
        for token in rows:
            view = (
                _causal_view(token.view, token.block, -1)
                if kind == "prefill"
                else tuple((b, n - int(b in written)) for b, n in token.view)
            )
            if kind == "prefill" and requests:
                previous, input_ids = requests[-1]
                size = len(input_ids)
                if (
                    token.block == previous.write_to
                    and token.position == previous.write_start + size
                    and token.view == _causal_view(previous.cache_view, token.block, size + 1)
                ):
                    input_ids.append(token.token_id)
                    continue
            requests.append(
                (ForwardRequest((), token.block, token.position, view), [token.token_id])
            )
        batches.append(
            ForwardBatch(
                step,
                kind,
                tuple(replace(request, input_ids=tuple(ids)) for request, ids in requests),
            )
        )
    return batches


@dataclass(init=False)
class Trace:
    """Executed request batches; internal prefill chunks share their request's step."""

    format_version: int = field(default=2, init=False)
    batches: list[ForwardBatch] = field(default_factory=list)
    steps: int = 0
    updates: list["BlockUpdate"] = field(default_factory=list)

    def __init__(self, tokens=(), steps=0, updates=(), *, batches=None):
        if batches is not None and tokens:
            raise ValueError("Pass either legacy tokens or request batches")
        self.batches = _pack_tokens(tokens) if batches is None else list(batches)
        self.steps = steps
        self.updates = list(updates)
        self._num_tokens = sum(len(r.input_ids) for b in self.batches for r in b.requests)

    def append(self, batch):
        """Append a recorded batch; keeps the token count without expanding requests."""
        self.batches.append(batch)
        self._num_tokens += sum(len(request.input_ids) for request in batch.requests)

    @property
    def num_tokens(self):
        return self._num_tokens

    @property
    def tokens(self):
        records = []
        for batch in self.batches:
            written = {request.write_to for request in batch.requests}
            for request in batch.requests:
                for offset, token_id in enumerate(request.input_ids):
                    view = (
                        _causal_view(request.cache_view, request.write_to, offset + 1)
                        if batch.kind == "prefill"
                        else tuple((b, n + int(b in written)) for b, n in request.cache_view)
                    )
                    records.append(
                        Token(
                            token_id,
                            batch.step,
                            request.write_to,
                            request.write_start + offset,
                            view,
                            batch.kind,
                        )
                    )
        return records

    @property
    def multimodal(self):
        return any(
            request.pixel_values is not None
            or request.image_grid_thw is not None
            or request.mm_token_type_ids is not None
            or request.mrope_positions is not None
            or request.write_span is not None
            or request.cache_spans is not None
            for batch in self.batches
            for request in batch.requests
        )

    @classmethod
    def from_dict(cls, data):
        version = data.get("format_version", 1)
        if type(version) is not int or version not in (1, 2):
            raise ValueError(f"Unsupported trace format_version: {version}")
        updates = [
            BlockUpdate(
                item["step"],
                item["block"],
                tuple(item.get("sources", ())),
                None if item.get("refs") is None else tuple(tuple(ref) for ref in item["refs"]),
            )
            for item in data.get("updates", ())
        ]
        if version == 1:
            if "batches" in data:
                raise ValueError("Request batches require trace format_version 2")
            tokens = [
                Token(**{**item, "view": tuple(tuple(ref) for ref in item["view"])})
                for item in data["tokens"]
            ]
            return cls(tokens, data.get("steps", 0), updates)
        if "tokens" in data:
            raise ValueError("Trace format_version 2 stores requests, not tokens")
        tensor_fields = (
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
            "mrope_positions",
        )
        batches = []
        for batch in data["batches"]:
            step, kind = batch["step"], batch["kind"]
            if type(step) is not int or step < 0 or (batches and step <= batches[-1].step):
                raise ValueError("Request batch steps must be nonnegative and strictly increasing")
            if kind not in {"prefill", "decode"} or not batch["requests"]:
                raise ValueError("Expected a nonempty prefill or decode batch")
            requests = []
            written = set()
            for item in batch["requests"]:
                request = {
                    **item,
                    "input_ids": tuple(item["input_ids"]),
                    "cache_view": tuple(tuple(ref) for ref in item["cache_view"]),
                }
                if request.pop("image_embeds", None) is not None:
                    raise ValueError("Trace image embeddings are unsupported; record pixel_values")
                if (
                    not request["input_ids"]
                    or (kind == "decode" and len(request["input_ids"]) != 1)
                    or request["write_to"] in written
                ):
                    raise ValueError(
                        "Each request needs input tokens and a distinct write block; decode takes one token"
                    )
                written.add(request["write_to"])
                if (
                    request["write_to"] < 0
                    or request["write_start"] < 0
                    or any(block < 0 or length < 0 for block, length in request["cache_view"])
                ):
                    raise ValueError(
                        "Block IDs, write positions and cache lengths must be nonnegative"
                    )
                if item.get("cache_spans") is not None:
                    request["cache_spans"] = tuple(item["cache_spans"])
                    if len(request["cache_spans"]) != len(request["cache_view"]) or any(
                        span < 0 for span in request["cache_spans"]
                    ):
                        raise ValueError(
                            "Cache spans must match the ordered view and be nonnegative"
                        )
                if item.get("write_span") is not None and item["write_span"] < 0:
                    raise ValueError("Write span must be nonnegative")
                for name in tensor_fields:
                    tensor = item.get(name)
                    if tensor is not None:
                        request[name] = TensorData(
                            tuple(tensor["data"]), tuple(tensor["shape"]), tensor["dtype"]
                        )
                requests.append(ForwardRequest(**request))
            batches.append(ForwardBatch(step, kind, tuple(requests)))
        return cls(steps=data.get("steps", 0), updates=updates, batches=batches)


@dataclass(frozen=True)
class BlockUpdate:
    """Replace from pre-update live (block, length) refs, or legacy source rows."""

    step: int
    block: int
    sources: tuple[int, ...] = ()
    refs: tuple[tuple[int, int], ...] | None = None


class RecordingSession:
    """Session proxy: install before submitting any work to AsyncLLM."""

    def __init__(self, session):
        self.session = session
        self.trace = Trace()
        self._contents = {}
        self._failed = False

    def __getattr__(self, name):
        return getattr(self.session, name)

    def _check(self, blocks):
        if self._failed:
            raise RuntimeError("Discard recorder after a failed inference call")
        for block in blocks:
            if block.num_tokens != self._contents.get(block.block_id, 0):
                raise ValueError("Unrecorded or modified block; attach recorder before prefill")

    def _call(self, call):
        try:
            return call()
        except Exception:
            self._failed = True
            raise

    def _run(self, batch, call):
        result = self._call(call)
        for request in batch.requests:
            self._contents[request.write_to] = request.write_start + len(request.input_ids)
        self.trace.append(batch)
        self.trace.steps += 1
        return result

    def _replace(self, block, refs):
        self._contents[block.block_id] = sum(length for _, length in refs)
        self.trace.updates.append(BlockUpdate(self.trace.steps, block.block_id, refs=refs))

    def merge_blocks(self, left, right):
        self._check([left, right])
        refs = ((left.block_id, left.num_tokens), (right.block_id, right.num_tokens))
        merged = self._call(lambda: self.session.merge_blocks(left, right))
        self._replace(merged, refs)
        return merged

    def append_block(self, left, right):
        self._check([left, right])
        refs = ((left.block_id, left.num_tokens), (right.block_id, right.num_tokens))
        result = self._call(lambda: self.session.append_block(left, right))
        self._replace(left, refs)
        return result

    def free_block(self, block):
        self._check([block])
        self._call(lambda: self.session.free_block(block))
        self._replace(block, ())

    def prefill_batch(self, jobs):
        jobs = list(jobs)
        requests = []
        for job in jobs:
            self._check([*job.context, job.block])
            if getattr(job, "image_embeds", None) is not None:
                raise ValueError(
                    "Record image requests with pixels before internal prefill chunking"
                )
            view = [*job.context, job.block]
            tensors = {
                name: TensorData.capture(getattr(job, name, None))
                for name in ("pixel_values", "image_grid_thw", "mm_token_type_ids")
            }
            positions = getattr(job, "mrope_rel", None)
            if positions is None and tensors["pixel_values"] is not None:
                derive = getattr(self.session, "_mrope_rel", None)
                if derive is not None:
                    positions = derive(
                        job.input_ids.flatten(), job.mm_token_type_ids, job.image_grid_thw
                    )
            span = getattr(job.block, "mrope_span", job.block.num_tokens)
            spans = tuple(getattr(block, "mrope_span", block.num_tokens) for block in view)
            multimodal = positions is not None or any(
                value is not None for value in tensors.values()
            )
            if positions is None and span != job.block.num_tokens:
                import torch

                positions = torch.arange(job.input_ids.numel()).expand(3, -1)
            requests.append(
                ForwardRequest(
                    tuple(int(token) for token in job.input_ids.flatten().tolist()),
                    job.block.block_id,
                    job.block.num_tokens,
                    tuple((block.block_id, block.num_tokens) for block in view),
                    **tensors,
                    mrope_positions=TensorData.capture(
                        None if positions is None else positions + span
                    ),
                    write_span=span if multimodal or span != job.block.num_tokens else None,
                    cache_spans=(
                        spans
                        if multimodal or any(n != block.num_tokens for n, block in zip(spans, view))
                        else None
                    ),
                )
            )
        batch = ForwardBatch(self.trace.steps, "prefill", tuple(requests))
        return self._run(batch, lambda: self.session.prefill_batch(jobs))

    def prefill_block(
        self,
        block,
        input_ids,
        context=None,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
    ):
        from minisgl.shared_cache.session import PrefillJob

        job = PrefillJob(
            block, input_ids, list(context or ()), pixel_values, image_grid_thw, mm_token_type_ids
        )
        return self.prefill_batch([job])[0]

    def decode_step(self, group, input_ids):
        contexts = list(group)
        requests = []
        for ctx, token_id in zip(contexts, input_ids.flatten().tolist(), strict=True):
            self._check([*ctx.cache_view, ctx.output_block])
            block = ctx.output_block
            span = getattr(block, "mrope_span", block.num_tokens)
            spans = tuple(getattr(b, "mrope_span", b.num_tokens) for b in ctx.cache_view)
            requests.append(
                ForwardRequest(
                    (int(token_id),),
                    block.block_id,
                    block.num_tokens,
                    tuple((b.block_id, b.num_tokens) for b in ctx.cache_view),
                    write_span=span if span != block.num_tokens else None,
                    cache_spans=(
                        spans
                        if any(n != b.num_tokens for n, b in zip(spans, ctx.cache_view))
                        else None
                    ),
                )
            )
        batch = ForwardBatch(self.trace.steps, "decode", tuple(requests))
        return self._run(batch, lambda: self.session.decode_step(group, input_ids))


def record(llm):
    """Wrap an idle, fresh AsyncLLM's session without changing its inference code."""
    if isinstance(llm.async_engine.session, RecordingSession):
        raise ValueError("This AsyncLLM is already recording")
    recorder = RecordingSession(llm.async_engine.session)
    llm.async_engine.session = recorder
    return recorder.trace
