"""Actual recording context + actual demo + CPU scheduler/model stub."""
import asyncio
from contextlib import nullcontext
import json

import pytest
import torch

from eval.async_thoughts import solve, trace_summary
from eval.math500 import read_config
from minisgl.core import SamplingParams
from minisgl.training import Trace
from minisgl.shared_cache.session import PrefillJob
from tests.core.test_async_llm import _make_llm
from tests.eval.test_async_thoughts import Tokenizer, config


def test_gemma_payload_does_not_shift_positional_qwen_job_fields():
    block, ids, context, pixels, grid, types, embeds, rel, span = [object() for _ in range(9)]
    job = PrefillJob(block, ids, context, pixels, grid, types, embeds, rel, span)
    assert job.mm_token_type_ids is types
    assert job.image_embeds is embeds and job.mrope_rel is rel and job.mrope_span is span
    assert job.image_position_ids is None
    positions = object()
    gemma = PrefillJob(block, ids, context, image_position_ids=positions)
    assert gemma.image_position_ids is positions and gemma.mm_token_type_ids is None


def test_math_demo_exports_real_training_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *a, **k: nullcontext())

    class Formatting(Tokenizer):
        def decode(self, ids, **kwargs):
            reverse = {value: key for key, value in self.vocab.items()}
            return "".join(reverse.get(int(i), "x") for i in ids)

        def encode(self, text, **kwargs):
            ids = [self.vocab[text]] if text in self.vocab else [20, 21]
            # Keep all input IDs inside the stub's vocabulary.
            return torch.tensor([ids]) if kwargs.get("return_tensors") == "pt" else ids

    async def scenario():
        llm, session = _make_llm()
        llm.tokenizer = Formatting()
        settings = dict(config(think=6, answer=5), model_path="stub-model")
        try:
            return await solve(llm, "full math problem", settings,
                SamplingParams(temperature=0), {0}, trace_path=tmp_path / "trace.json"), session
        finally:
            await llm.close()

    row, session = asyncio.run(scenario())
    payload = json.loads((tmp_path / "trace.json").read_text())
    example, = payload["examples"]
    trace = Trace.from_dict(example["trace"])
    assert payload["model"] == "stub-model"
    assert example["thinker_tokens"] == row["thinker_token_ids"]
    assert example["writer_tokens"] == row["token_ids"]
    assert len(example["targets"]) == len(example["logprobs"]) == trace.num_tokens
    assert sum(t != -100 for t in example["targets"]) == row["generated_tokens"]
    assert all((t == -100) == (p is None) for t, p in zip(example["targets"], example["logprobs"]))
    assert trace_summary(trace)["decode"]["batches"] == len(session.decode_calls)
    assert row["training_trace"]["input_rows"] == trace.num_tokens
    assert row["training_trace"]["export_s"] >= 0
    assert any(t == -100 for t in example["targets"])  # Prompts/probes aren't labels.


def test_trace_export_refuses_overwrite(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text("previous result")
    llm = type("LLM", (), {"tokenizer": Tokenizer()})()
    with pytest.raises(FileExistsError):
        asyncio.run(solve(llm, "problem", config(), None, {0}, trace_path=path))
    assert path.read_text() == "previous result"


@pytest.mark.parametrize("mode,expected", [("async_thoughts", True), ("plain", False)])
def test_trace_default_and_explicit_control(tmp_path, mode, expected):
    path = tmp_path / "config.json"
    data = dict(model_path="stub", mode=mode)
    path.write_text(json.dumps(data))
    assert read_config(path)["record_training_trace"] is expected
    data["record_training_trace"] = False
    path.write_text(json.dumps(data))
    assert read_config(path)["record_training_trace"] is False
