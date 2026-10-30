"""Export the demo's executed trace and sampled next-token targets."""

import json
from dataclasses import asdict
from pathlib import Path

from minisgl.training.trace import record


class DemoRecorder:
    """Wrap a fresh AsyncLLM before any demo or probe prefills are submitted."""

    def __init__(self, llm):
        self.llm = llm
        self.trace = record(llm)
        self._samples = {}

    def __getattr__(self, name):
        return getattr(self.llm, name)

    async def async_generate(self, context, *, return_logits=False, **kwargs):
        stream = self.llm.async_generate(context, return_logits=True, **kwargs)
        try:
            while True:
                block = context.output_block
                block_id, position = block.block_id, block.num_tokens
                batch_cursor, row_cursor = len(self.trace.batches), self.trace.num_tokens
                try:
                    result = await anext(stream)
                except StopAsyncIteration:
                    return
                token, logits = result
                matches, row = [], row_cursor
                for batch in self.trace.batches[batch_cursor:]:
                    for request in batch.requests:
                        if (
                            batch.kind == "decode"
                            and request.write_to == block_id
                            and request.write_start == position
                        ):
                            matches.append(row)
                        row += len(request.input_ids)
                if len(matches) != 1 or matches[0] in self._samples:
                    raise RuntimeError(
                        "Could not uniquely align a sampled token to its decode input"
                    )
                # Label the input row now, including EOS or a final unflushed sample.
                self._samples[matches[0]] = (
                    int(token),
                    logits.float().log_softmax(-1)[token].item(),
                )
                yield result if return_logits else token
        finally:
            await stream.aclose()

    def save(self, path, *, model, problem, thinker_tokens, writer_tokens):
        """Write one SFT example without overwriting an existing destination."""
        targets, logprobs = [-100] * self.trace.num_tokens, [None] * self.trace.num_tokens
        for row, (token, logprob) in self._samples.items():
            targets[row], logprobs[row] = token, logprob
        data = {
            "model": str(model),
            "examples": [
                {
                    "prompt": problem,
                    "trace": asdict(self.trace),
                    "targets": targets,
                    "logprobs": logprobs,
                    "thinker_tokens": list(thinker_tokens),
                    "writer_tokens": list(writer_tokens),
                }
            ],
        }
        text = json.dumps(data, ensure_ascii=False, allow_nan=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            output.write(text + "\n")
        return path
