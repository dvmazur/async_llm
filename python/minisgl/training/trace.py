from dataclasses import dataclass, field


@dataclass(frozen=True)
class Token:
    token_id: int
    step: int
    block: int
    position: int
    view: tuple[tuple[int, int], ...]
    kind: str


@dataclass
class Trace:
    """KV input tokens; steps count session calls (prefill chunks share a step)."""

    tokens: list[Token] = field(default_factory=list)
    steps: int = 0
    updates: list["BlockUpdate"] = field(default_factory=list)


@dataclass(frozen=True)
class BlockUpdate:
    """Destination contents before `step`, referencing original trace token rows."""

    step: int
    block: int
    sources: tuple[int, ...]


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
            if block.num_tokens != len(self._contents.get(block.block_id, ())):
                raise ValueError("Unrecorded or modified block; attach recorder before prefill")

    def _call(self, call):
        try:
            return call()
        except Exception:
            self._failed = True
            raise

    def _run(self, records, call):
        result = self._call(call)
        for token in records:
            self._contents.setdefault(token.block, []).append(len(self.trace.tokens))
            self.trace.tokens.append(token)
        self.trace.steps += 1
        return result

    def _replace(self, block, sources):
        self._contents[block.block_id] = list(sources)
        self.trace.updates.append(BlockUpdate(self.trace.steps, block.block_id, tuple(sources)))

    def merge_blocks(self, left, right):
        self._check([left, right])
        sources = self._contents.get(left.block_id, []) + self._contents.get(right.block_id, [])
        merged = self._call(lambda: self.session.merge_blocks(left, right))
        self._replace(merged, sources)
        return merged

    def append_block(self, left, right):
        self._check([left, right])
        sources = self._contents.get(left.block_id, []) + self._contents.get(right.block_id, [])
        result = self._call(lambda: self.session.append_block(left, right))
        self._replace(left, sources)
        return result

    def free_block(self, block):
        self._check([block])
        self._call(lambda: self.session.free_block(block))
        self._replace(block, ())

    def prefill_batch(self, jobs):
        jobs = list(jobs)
        records = []
        for job in jobs:
            if job.pixel_values is not None:
                raise NotImplementedError("Training traces currently support text Qwen3 only")
            self._check([*job.context, job.block])
            context = tuple((b.block_id, b.num_tokens) for b in job.context)
            for offset, token_id in enumerate(job.input_ids.flatten().tolist()):
                position = job.block.num_tokens + offset
                records.append(
                    Token(
                        int(token_id),
                        self.trace.steps,
                        job.block.block_id,
                        position,
                        context + ((job.block.block_id, position + 1),),
                        "prefill",
                    )
                )
        return self._run(records, lambda: self.session.prefill_batch(jobs))

    def decode_step(self, group, input_ids):
        contexts = list(group)
        written = {c.output_block.block_id for c in contexts}
        records = []
        for ctx, token_id in zip(contexts, input_ids.flatten().tolist(), strict=True):
            self._check([*ctx.cache_view, ctx.output_block])
            view = tuple(
                (b.block_id, b.num_tokens + int(b.block_id in written)) for b in ctx.cache_view
            )
            records.append(
                Token(
                    int(token_id),
                    self.trace.steps,
                    ctx.output_block.block_id,
                    ctx.output_block.num_tokens,
                    view,
                    "decode",
                )
            )
        return self._run(records, lambda: self.session.decode_step(group, input_ids))


def record(llm):
    """Wrap an idle, fresh AsyncLLM's session without changing its inference code."""
    if isinstance(llm.async_engine.session, RecordingSession):
        raise ValueError("This AsyncLLM is already recording")
    recorder = RecordingSession(llm.async_engine.session)
    llm.async_engine.session = recorder
    return recorder.trace
