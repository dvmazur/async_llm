"""Parallel-thinking scheduling, trace supervision, notebooks, and process launching."""

import ast
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from minisgl.training import Trace
from minisgl_examples.training.parallel_thinking import demo
from minisgl_examples.training.parallel_thinking.async_helpers import isolated

from tests.scripts.training_helpers import collect_example


@pytest.fixture(autouse=True)
def cpu_profiling(monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())


@pytest.mark.parametrize("synchronized", [True, False])
def test_recorded_schedules_preserve_samples_and_publications(synchronized):
    result, example = collect_example(synchronized)
    trace, targets = example["trace"], example["targets"]
    assert result["generated_tokens"] == {"Alice": 4, "Bob": 4}
    assert result["completed_steps"] == 4
    if synchronized:
        assert result["max_token_gap"] == 1
    else:
        assert result["max_token_gap"] > 1
    assert len(targets) == trace.num_tokens == len(example["logprobs"])
    assert sum(target != -100 for target in targets) == 10
    assert any(update.refs for update in trace.updates)
    assert any(not update.refs for update in trace.updates)
    assert any(
        len(request.cache_view) == 6 for batch in trace.batches for request in batch.requests
    )
    assert Trace.from_dict(asdict(trace)) == trace
    assert all(
        target == (token.token_id + 2) % 32
        for token, target in zip(trace.tokens, targets)
        if target != -100
    )
    if synchronized:
        assert any(len(batch.requests) == 2 for batch in trace.batches if batch.kind == "decode")


@pytest.mark.parametrize("path", sorted(Path(demo.__file__).parent.glob("*.ipynb")))
def test_notebook_cells_compile_without_stale_outputs(path):
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            compile(
                "".join(cell["source"]), str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )
            assert cell["execution_count"] is None and not cell["outputs"]


def test_isolated_process_respects_accelerator_selection(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "preserved")
    assert isolated(os.getenv, "CUDA_VISIBLE_DEVICES") == "preserved"
