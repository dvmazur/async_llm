from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass, field
from itertools import groupby

import torch

_IMAGE_INPUT_FIELDS = ("pixel_values", "image_grid_thw", "mm_token_type_ids")


def _validate_replay_options(
    query_chunk_size: int, gdn_segment_size: int, gdn_read_mode: str, gdn_read_backend: str
) -> None:
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if gdn_segment_size < 1:
        raise ValueError("gdn_segment_size must be positive")
    if gdn_read_mode not in {"expanded", "factorized"}:
        raise ValueError("gdn_read_mode must be expanded or factorized")
    if gdn_read_backend not in {"torch", "triton"}:
        raise ValueError("gdn_read_backend must be torch or triton")


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

    def to_tensor(self):
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


@dataclass(init=False)
class Trace:
    """Executed request batches; internal prefill chunks share their request's step."""

    format_version: int = field(default=2, init=False)
    batches: list[ForwardBatch] = field(default_factory=list)
    steps: int = 0
    updates: list["BlockUpdate"] = field(default_factory=list)

    # Runtime preparation is deliberately excluded from dataclass fields, so
    # asdict(trace) remains a portable record without tensors or cached plans.
    query_chunk_size = 128
    gdn_segment_size = 8
    gdn_read_mode = "expanded"
    gdn_cache_reused = True
    gdn_read_backend = "torch"
    attention_checkpointing = True
    cache_plans = True
    _option_names = (
        "query_chunk_size",
        "gdn_segment_size",
        "gdn_read_mode",
        "gdn_cache_reused",
        "gdn_read_backend",
        "attention_checkpointing",
        "cache_plans",
    )

    def __init__(
        self,
        *,
        batches: Iterable[ForwardBatch] = (),
        steps: int = 0,
        updates: Iterable["BlockUpdate"] = (),
    ) -> None:
        self.batches = list(batches)
        self.steps = steps
        self.updates = list(updates)
        self._num_tokens = sum(len(r.input_ids) for b in self.batches for r in b.requests)
        self._prepared_history = None
        self._device = torch.device("cpu")
        self._gdn, self._attention = {}, {}

    @torch.inference_mode(False)
    def to(self, device: torch.device | str = "cpu", **options) -> "Trace":
        """Prepare trace metadata on a device, optionally changing replay settings.

        Supported settings are query_chunk_size, attention_checkpointing,
        cache_plans, and the gdn_* settings documented in the training README.
        Recording data stays on the CPU and retains its serializable format.
        Returns self when the device and settings already match; otherwise returns
        a trace with an independent recording snapshot. Device-only copies share
        cached metadata keyed by device, never activations or model weights.
        """
        unknown = options.keys() - self._option_names
        if unknown:
            raise TypeError(f"Unknown replay options: {', '.join(sorted(unknown))}")
        device = torch.device(device)
        changed = any(getattr(self, key) != value for key, value in options.items())
        if not changed:
            self._prepare()
        result = self
        if device != self._device or changed:
            result = copy(self)
            result.batches, result.updates = list(self.batches), list(self.updates)
            for key, value in options.items():
                setattr(result, key, value)
            result._device = device
            if changed:
                result._gdn, result._attention = {}, {}
        result._prepare()
        if result.input_ids.device != device:
            result.input_ids = result.input_ids.to(device)
            result.order = result.order.to(device)
            if result.rope_positions is not None:
                result.rope_positions = result.rope_positions.to(device)
        return result

    @torch.inference_mode(False)
    def _prepare(self) -> "Trace":
        """Compile visibility lazily; recording changes invalidate derived metadata."""
        if getattr(self, "_replay_snapshot", False):
            return self
        _validate_replay_options(
            self.query_chunk_size, self.gdn_segment_size, self.gdn_read_mode, self.gdn_read_backend
        )
        history = (tuple(self.batches), tuple(self.updates))
        if history == self._prepared_history:
            return self
        from .mrope import MropeLayout

        layout = MropeLayout(self) if self.multimodal else None
        tokens = self.tokens
        if not tokens:
            raise ValueError("Cannot replay an empty trace")
        if any(a.step > b.step for a, b in zip(tokens, tokens[1:])) or any(
            a.step > b.step for a, b in zip(self.updates, self.updates[1:])
        ):
            raise ValueError("Trace must be in execution order")
        order = sorted(
            range(len(tokens)), key=lambda i: (tokens[i].block, tokens[i].step, tokens[i].position)
        )
        inverse = {row: i for i, row in enumerate(order)}
        spans, copies, lengths, current, rows = [], [], [], {}, []

        def append(version, row):
            source = inverse[row]
            parts = spans[version]
            if parts and parts[-1][1] == source:
                start, _, position = parts[-1]
                parts[-1] = (start, source + 1, position)
            else:
                parts.append((source, source + 1, lengths[version]))
            lengths[version] += 1

        def replace(block, refs=()):
            parents, position = [], 0
            for source, length in refs:
                parent = current.get(source)
                if source < 0 or length < 0 or length != (0 if parent is None else lengths[parent]):
                    raise ValueError("Mutation must reference a live block with its exact length")
                if length:
                    parents.append((parent, length, position))
                    position += length
            if layout is not None:
                layout.replace(block, refs)
            current[block] = len(spans)
            spans.append([])
            copies.append(parents)
            lengths.append(position)

        update_index = 0
        for step, group in groupby(enumerate(tokens), key=lambda item: item[1].step):
            group = list(group)
            while update_index < len(self.updates) and self.updates[update_index].step <= step:
                update = self.updates[update_index]
                replace(update.block, update.refs)
                update_index += 1
            for row, token in group:
                if token.block not in current:
                    replace(token.block)
            if layout is not None:
                layout.begin_step(step)
            for row, token in group:
                version = current[token.block]
                if token.position != lengths[version]:
                    raise ValueError("Writes must append to a contiguous block span")
                append(version, row)
            for row, token in group:
                if token.kind not in {"prefill", "decode"}:
                    raise ValueError("Unknown forward kind")
                own_lengths = [length for block, length in token.view if block == token.block]
                if any(length <= token.position for length in own_lengths):
                    raise ValueError("A token must attend to itself")
                if token.kind == "prefill" and (
                    not own_lengths or any(length != token.position + 1 for length in own_lengths)
                ):
                    raise ValueError("Prefill must include its own causal prefix")
                entries, prefix = [], 0
                total = sum(length for _, length in token.view)
                for block, length in token.view:
                    if length < 0:
                        raise ValueError("View length must be nonnegative")
                    if length == 0:
                        continue
                    if block not in current or length > lengths[current[block]]:
                        raise ValueError("View exposes future or unrecorded tokens")
                    entries.append((current[block], 0, length, total - 1 - prefix))
                    prefix += length
                if not own_lengths:
                    entries.append(
                        (current[token.block], token.position, token.position + 1, token.position)
                    )
                if layout is not None:
                    entries = [
                        (*entry[:3], position)
                        for entry, position in zip(entries, layout.queries[row], strict=True)
                    ]
                rows.append(entries)

        for update in self.updates[update_index:]:
            replace(update.block, update.refs)
        self.input_ids = torch.tensor([tokens[i].token_id for i in order], device=self._device)
        self.order = torch.tensor(order, device=self._device)
        self.spans, self.copies = spans, copies
        self.reads = [rows[i] for i in order]
        self.rope_positions = (
            None if layout is None else torch.tensor(layout.keys, device=self._device)[order]
        )
        self.rope_offsets = None if layout is None else layout.copy_offsets
        self._gdn, self._attention = {}, {}
        self._prepared_history = history
        self._num_tokens = len(tokens)
        return self

    def compute_source_spans(self, version, start, end, with_rope=False):
        self._prepare()
        stack = [(version, start, end, 0, 0)]
        while stack:
            version, start, end, offset, rope_offset = stack.pop()
            if version < 0:
                yield (start, end, offset, rope_offset) if with_rope else (start, end, offset)
                continue
            for left, right, position in reversed(self.spans[version]):
                lo, hi = max(start, position), min(end, position + right - left)
                if lo < hi:
                    stack.append(
                        (-1, left + lo - position, left + hi - position, offset + lo, rope_offset)
                    )
            for index in range(len(self.copies[version]) - 1, -1, -1):
                parent, length, position = self.copies[version][index]
                lo, hi = max(start, position), min(end, position + length)
                if lo < hi:
                    shift = self.rope_offsets[version][index] if with_rope else 0
                    stack.append(
                        (
                            parent,
                            lo - position,
                            hi - position,
                            offset + position,
                            rope_offset + shift,
                        )
                    )

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
        version = data.get("format_version")
        if type(version) is not int or version != 2:
            raise ValueError(f"Unsupported trace format_version: {version}")
        if "tokens" in data:
            raise ValueError("Trace format_version 2 stores requests, not tokens")
        if any("sources" in item for item in data.get("updates", ())):
            raise ValueError("BlockUpdate.sources is unsupported; use block refs")
        updates = [
            BlockUpdate(
                item["step"],
                item["block"],
                refs=tuple(tuple(ref) for ref in item["refs"]),
            )
            for item in data.get("updates", ())
        ]
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
                if "image_embeds" in request:
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
                for name in (*_IMAGE_INPUT_FIELDS, "mrope_positions"):
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
    """Replace from pre-update live (block, length) refs; empty refs clear a block."""

    step: int
    block: int
    refs: tuple[tuple[int, int], ...] = ()
