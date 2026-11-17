"""Minimal sampled-target adapter for the unchanged H+Delta role schedule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Forward:
    logits: object
    block: object
    row_cursor: int
    batch_cursor: int
    write_end: int


@dataclass
class Prediction:
    token: int
    row: int
    logprob: float
    role: str
    step: int
    emitted: bool = False

    def item(self):
        return self.token

    def view(self, *shape):
        import torch

        return torch.tensor(self.token, dtype=torch.long).view(*shape)


class TargetRecorder:
    """Delay labels until a prediction is emitted; unused lookahead stays masked."""

    def __init__(self, recorded):
        self.recorded = recorded
        self.step = 0
        self.samples = {}
        self.emission_index = 0

    def __getattr__(self, name):
        return getattr(self.recorded, name)

    async def __call__(self, *args, **kwargs):
        trace = self.recorded.get_trace()
        row_cursor, batch_cursor = trace.num_tokens, len(trace.batches)
        result = await self.recorded.forward(*args, **kwargs)
        return Forward(
            None if result.logits is None else result.logits.detach().clone(),
            result.block,
            row_cursor,
            batch_cursor,
            result.block.num_tokens,
        )

    def prediction(self, output, token, role):
        from minisgl_examples.training.recording import record_sample

        pending = {}
        record_sample(
            self.recorded,
            pending,
            block_id=output.block.block_id,
            write_end=output.write_end,
            row_cursor=output.row_cursor,
            batch_cursor=output.batch_cursor,
            token=token,
            logits=output.logits.flatten(),
        )
        if len(pending) != 1:
            raise RuntimeError("Could not uniquely align a prediction")
        row, (token, logprob) = next(iter(pending.items()))
        return Prediction(token, row, logprob, role, self.step)

    async def sample(self, output, *, role):
        token = int(output.logits.flatten().argmax().item())
        return self.prediction(output, token, role)

    def emit(self, prediction):
        if prediction.emitted or prediction.row in self.samples:
            raise RuntimeError("A sampled token was labeled twice")
        prediction.emitted = True
        self.samples[prediction.row] = {
            "row": prediction.row,
            "token": prediction.token,
            "logprob": prediction.logprob,
            "role": prediction.role,
            "step": prediction.step,
            "emission": self.emission_index,
        }
        self.emission_index += 1

    def record_decision(self, output, token):
        self.emit(self.prediction(output, int(token), "actor"))

    def export(self):
        from dataclasses import asdict

        trace = self.recorded.get_trace()
        targets, logprobs = [-100] * trace.num_tokens, [None] * trace.num_tokens
        for row, sample in self.samples.items():
            targets[row], logprobs[row] = sample["token"], sample["logprob"]
        samples = [self.samples[row] for row in sorted(self.samples)]
        return {
            "trace": asdict(trace),
            "targets": targets,
            "logprobs": logprobs,
            "samples": samples,
        }
