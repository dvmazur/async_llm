"""Correction rollout structure and CLI checks without a GPU or Hub access."""

import argparse
import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from minisgl.training import Trace
from minisgl_examples.training.corrections import collect as collector
from minisgl_examples.training.corrections.grading import boxed_answers, grade_text
from minisgl_examples.training.corrections.rollout import (
    INJECTION_PREFIX,
    collect_correction_example,
    problem_shards,
    validate_correction_example,
)

from tests.training.helpers import make_llm


class Tokenizer:
    eos_token_id = 31

    def encode(self, text, **kwargs):
        if text == "\n":
            return [4]
        if text.startswith(INJECTION_PREFIX):
            return [5, 6, 7]
        if text == "correction":
            return [8]
        if text.endswith("questioncorrection"):
            return [1, 2, 3, 8]
        return [1, 2, 3]

    def apply_chat_template(self, messages, **kwargs):
        return "CHAT: " + messages[0]["content"]

    def decode(self, tokens, **kwargs):
        return ""


@pytest.mark.parametrize("k_steps", [0, 2])
def test_correction_rollout_updates_only_the_prompt_and_aligns_targets(k_steps):
    async def run():
        llm, _ = make_llm()
        llm.tokenizer = Tokenizer()
        try:
            return await collect_correction_example(
                llm,
                {"problem_shards": ["question", "correction"], "answer": "answer"},
                dataset="dataset",
                dataset_revision="revision",
                dataset_split="train",
                dataset_index=3,
                k_steps=k_steps,
                max_tokens=4,
                precision="bfloat16",
            )
        finally:
            await llm.close()

    example = asyncio.run(run())
    report = validate_correction_example(example)
    assert isinstance(example["trace"], Trace)
    assert example["dataset_index"] == 3
    assert example["output_tokens"] == [0, 0, 0, 0]
    assert report["supervised_tokens"] == 4
    assert len(example["targets"]) == report["trace_tokens"]
    assert example["targets"].count(-100) == report["trace_tokens"] - 4
    assert example["answer_evaluation"]["label"] == "incorrect"
    trace = example["trace"]
    assert validate_correction_example({**example, "trace": asdict(trace)}) == report
    requests = [(batch.step, req) for batch in trace.batches for req in batch.requests]
    first_step, first_request = requests[0]
    prompt = first_request.write_to
    prompt_writes = [(step, req) for step, req in requests if req.write_to == prompt]
    output_writes = [(step, req) for step, req in requests if req.write_to != prompt]
    assert [req.input_ids for _, req in output_writes] == [(4,), (0,), (0,), (0,)]
    if k_steps == 0:
        assert "questioncorrection" in example["prompt"]
        assert [req.input_ids for _, req in prompt_writes] == [(1, 2, 3, 8)]
        assert example["injection_trace_step"] == first_step
    else:
        assert example["prompt"].endswith("question")
        assert [req.input_ids for _, req in prompt_writes] == [(1, 2, 3), (5, 6, 7)]
        correction_step = prompt_writes[-1][0]
        assert example["injection_trace_step"] == correction_step
        assert output_writes[k_steps - 1][0] < correction_step < output_writes[k_steps][0]
    assert dict(output_writes[k_steps][1].cache_view)[prompt] == (4 if k_steps == 0 else 6)


def test_eos_before_correction_is_recorded_but_does_not_stop_injection():
    async def run():
        llm, _ = make_llm()
        tokenizer = Tokenizer()
        tokenizer.eos_token_id = 0
        llm.tokenizer = tokenizer
        try:
            return await collect_correction_example(
                llm,
                {"problem_shards": ["question", "correction"], "answer": "answer"},
                dataset="dataset",
                dataset_revision="revision",
                dataset_split="train",
                dataset_index=0,
                k_steps=2,
                max_tokens=4,
                precision="bfloat16",
            )
        finally:
            await llm.close()

    example = asyncio.run(run())
    assert example["output_tokens"] == [0, 0, 0]
    assert example["eos_token_positions"] == [0, 1, 2]
    assert example["eos_before_correction"]
    assert example["stopped_reason"] == "eos"


def test_grading_retains_all_correct_incorrect_and_mixed_answers():
    equivalent = lambda expected, actual: actual.strip() == expected
    text = r"first \boxed{41}, nested \boxed{\frac{84}{2}}, final \boxed{42}"
    assert boxed_answers(text) == ["41", r"\frac{84}{2}", "42"]
    result = grade_text(text, "42", equivalent=equivalent)
    assert result["label"] == "mixed"
    assert [item["correct"] for item in result["answers"]] == [False, False, True]
    assert grade_text(r"\boxed{42}\boxed{42}", "42", equivalent=equivalent)["label"] == "correct"
    assert grade_text("no answer", "42", equivalent=equivalent)["label"] == "incorrect"


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"problem_shards": ["only one"]},
        {"problem_shards": ["", "second"]},
        {"problem_shards": "two strings"},
    ],
)
def test_problem_shards_rejects_invalid_dataset_rows(row):
    with pytest.raises(ValueError, match="two nonempty"):
        problem_shards(row)


def test_collector_refuses_file_output_before_loading_dataset(tmp_path, monkeypatch):
    output = tmp_path / "traces"
    output.write_text("keep")
    monkeypatch.setattr(collector.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        collector,
        "load_table",
        lambda *args: pytest.fail("loaded the dataset before checking the destination"),
    )
    args = argparse.Namespace(
        output=output,
        start_index=0,
        count=1,
        k_steps=[0, 2],
        max_tokens=4,
        model="model",
        dataset="dataset",
        dataset_revision="revision",
        split="train",
        precision="bfloat16",
        concurrency=1,
    )
    with pytest.raises(FileExistsError):
        asyncio.run(collector.collect_dataset_traces(args))


def test_collector_checkpoints_each_trace_and_resumes(tmp_path, monkeypatch):
    class Table:
        num_rows = 1

        def slice(self, index, count):
            return self

        def to_pylist(self):
            return [{"problem_shards": ["first", "second"], "answer": "42"}]

    class LLM:
        async def close(self):
            pass

    active = 0
    peak_active = 0

    async def collect(llm, item, **options):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0)
        active -= 1
        return {
            "trace": Trace(),
            "targets": [],
            "output_tokens": [],
            "answer_evaluation": {"label": "incorrect"},
            "answer_before_correction": False,
            "eos_before_correction": False,
            "k_steps": options["k_steps"],
        }

    monkeypatch.setattr(collector.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(collector, "load_table", lambda *args: Table())
    monkeypatch.setattr(collector, "AsyncLLM", lambda *args, **kwargs: LLM())
    monkeypatch.setattr(collector, "collect_correction_example", collect)
    args = SimpleNamespace(
        output=tmp_path / "run",
        start_index=0,
        count=1,
        k_steps=[0, 16],
        max_tokens=32,
        model="model",
        dataset="dataset",
        dataset_revision="revision",
        split="train",
        precision="bfloat16",
        concurrency=2,
        cuda_graphs=False,
        memory_ratio=0.7,
        max_seq_len=128,
        page_size=1,
        distributed_port=2361,
    )
    first = asyncio.run(collector.collect_dataset_traces(args))
    assert first["collected"] == 2 and first["skipped"] == 0
    assert peak_active == 2
    assert sorted(path.name for path in (args.output / "traces").iterdir()) == [
        "0000-k0000.json",
        "0000-k0016.json",
    ]
    second = asyncio.run(collector.collect_dataset_traces(args))
    assert second == {"output": str(args.output), "collected": 0, "skipped": 2}
    config_path = args.output / "config.json"
    config = json.loads(config_path.read_text())
    assert config.pop("correction_protocol") == "prompt_only"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Existing run configuration differs"):
        asyncio.run(collector.collect_dataset_traces(args))


def test_cli_defaults_to_pinned_dataset_and_bfloat16(tmp_path):
    args = collector.parser().parse_args(["--output", str(tmp_path / "traces.json")])
    assert args.dataset == collector.DATASET
    assert args.dataset_revision == collector.DATASET_REVISION
    assert args.model == "Qwen/Qwen3.5-4B"
    assert args.precision == "bfloat16"
    assert args.k_steps == [0, 16, 32, 64, 128, 256, 512]
    assert args.max_tokens == 1024
    assert args.concurrency == 1


@pytest.mark.parametrize(
    ("concurrency", "sizes"),
    [(1, [1]), (8, [1, 2, 4, 8]), (16, [1, 2, 4, 8, 16]), (24, [1, 2, 4, 8, 16, 24])],
)
def test_cuda_graph_batch_sizes_cover_concurrency(concurrency, sizes):
    assert collector.cuda_graph_batch_sizes(concurrency) == sizes
