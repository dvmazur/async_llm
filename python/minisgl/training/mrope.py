import torch


class MropeLayout:
    """Block coordinate frames, independent of token-count visibility."""

    def __init__(self, trace):
        self.batches = {batch.step: batch for batch in trace.batches}
        self.spans = {}
        self.keys = []
        self.queries = []
        self.copy_offsets = []

    def replace(self, block, sources=(), refs=None):
        if sources:
            raise ValueError(
                "Multimodal cache mutations require block refs, not legacy source rows"
            )
        offsets, span = [], 0
        for source, length in refs or ():
            if length:
                offsets.append(span)
            span += self.spans.get(source, 0)
        self.copy_offsets.append(offsets)
        self.spans[block] = span

    def begin_step(self, step):
        batch = self.batches[step]
        writers = {request.write_to for request in batch.requests}
        new_spans = {}
        for request in batch.requests:
            size = len(request.input_ids)
            span = self.spans.get(request.write_to, 0)
            if request.write_span is not None and request.write_span != span:
                raise ValueError("Recorded write span disagrees with the cache history")
            spans = [self.spans.get(block, 0) for block, _ in request.cache_view]
            if request.cache_spans is not None and tuple(spans) != request.cache_spans:
                raise ValueError("Recorded cache spans disagree with the cache history")
            if request.mrope_positions is None:
                if request.pixel_values is not None:
                    raise ValueError("Image requests require recorded mRoPE positions")
                positions = torch.arange(span, span + size).expand(3, -1)
            else:
                positions = request.mrope_positions.tensor()
                if (
                    positions.shape != (3, size)
                    or positions.dtype not in (torch.int32, torch.int64)
                    or (positions < 0).any()
                ):
                    raise ValueError(
                        "mRoPE positions must be nonnegative integer coordinates [3, N]"
                    )
            if batch.kind == "prefill":
                if not request.cache_view or request.cache_view[-1][0] != request.write_to:
                    raise ValueError("Prefill must end its cache view with the write block")
                total = sum(spans[:-1])
            else:
                if size != 1 or not (positions == span).all():
                    raise ValueError("Decode requires one token at the current mRoPE span")
                spans = [
                    value + int(block in writers)
                    for (block, _), value in zip(request.cache_view, spans, strict=True)
                ]
                total = sum(spans)
            for offset, position in enumerate(positions.T.tolist()):
                queries, prefix = [], 0
                for index, ((block, length), value) in enumerate(
                    zip(request.cache_view, spans, strict=True)
                ):
                    if batch.kind == "decode":
                        length += int(block in writers)
                        query = (total - prefix - 1,) * 3
                    elif index == len(spans) - 1:
                        length += offset + 1
                        query = tuple(position)
                    else:
                        query = tuple(p + total - prefix for p in position)
                    if length:
                        queries.append(query)
                    prefix += value
                if request.write_to not in {block for block, _ in request.cache_view}:
                    queries.append(tuple(position))
                self.keys.append(position)
                self.queries.append(queries)
            new_spans[request.write_to] = max(span, int(positions.max()) + 1)
        self.spans.update(new_spans)
