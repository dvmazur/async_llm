"""CPU checks for saved-trace readouts, answer limits, and resume integrity."""

import asyncio
import gzip
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from minisgl_examples.training.parallel_thinking import budget_curve_eval as budget_eval
from minisgl_examples.training.parallel_thinking.report import COMMON

CUE = "\n\nGiven the current thoughts, the answer is \\boxed{"


class Tokenizer:
    vocab = {"<think>": 90, "</think>": 91}
    eos_token_id = 99
    pieces = {1: r"Problem with an old \boxed{123}", 3: "reasoning", 4: " text",
              10: "7", 11: "}", 12: r"\sqrt{", 15: "x", 70: CUE,
              90: "<think>", 91: "</think>", 99: "<|im_end|>"}

    def __init__(self):
        self.encodings = []

    def encode(self, text, **kwargs):
        self.encodings.append(text)
        assert text in (CUE, "</think>" + CUE), "Saved token IDs were decoded and re-tokenized"
        return [70] if text == CUE else [91, 70]

    def decode(self, ids, **kwargs):
        return "".join(self.pieces.get(int(token), f"<{token}>") for token in ids)


class FakeLLM:
    def __init__(self, samples, *, fail_on=None):
        self.tokenizer = Tokenizer()
        self.samples = iter(samples)
        self.fail_on = fail_on
        self.forwards = []
        self.blocks = []
        self.freed = []
        self.sample_count = 0

    async def create_block(self):
        block = SimpleNamespace(block_id=len(self.blocks), num_tokens=0, token_ids=[])
        self.blocks.append(block)
        return block

    async def free_block(self, block):
        self.freed.append(block)

    async def forward(self, input_ids=None, *, cache_view=None, write_to=None, return_logits=True):
        view = cache_view.cache_view if hasattr(cache_view, "cache_view") else cache_view
        block = write_to or view[-1]
        ids = list(input_ids) if input_ids is not None else [cache_view.next_input_id]
        self.forwards.append(ids)
        if self.fail_on == len(self.forwards):
            raise RuntimeError("Simulated readout failure")
        block.token_ids.extend(ids)
        block.num_tokens += len(ids)
        if not return_logits:
            return SimpleNamespace(logits=None)
        token = next(self.samples)
        self.sample_count += 1
        logits = torch.full((128,), -1000.0)
        logits[token] = 1000.0
        return SimpleNamespace(logits=logits)


def rollout():
    return {
        "layout": "native_workers", "publication_policy": "encoded_separator_v1",
        "workers": ["Alice", "Bob"], "prompt_ids": [1, 90],
        "worker_token_ids": {"Alice": [3, 4, 99], "Bob": [4, 3, 15]},
        "decode_rounds": 3, "eos_token_ids": [99], "eos_token_id": 99,
        "other_header_ids": [], "own_header_ids": [],
        "initial_worker_prefix_ids": {"Alice": [], "Bob": []}, "publications": [],
        "finisher": {"suffix_ids": [111], "sampled_token_ids": [112, 113]},
    }


def source_case(tmp_path, name="first", *, index=0):
    directory = tmp_path / "shards" / name
    directory.mkdir(parents=True)
    config = {**deepcopy(COMMON), "mode": "hogwild", "workers": [2], "indices": [index],
              "publication_policy": "encoded_separator_v1", "multi_worker_layout": "native"}
    row = {"mode": "hogwild", "workers": 2, "index": index, "unique_id": f"test/sample-{index}.json",
           "gold": "7", "prediction": "123", "correct": False, "grading_status": "incorrect",
           "trace_path": f"traces/hogwild-w2-i{index:03d}.json.gz"}
    payload = {
        "format": "hogwild-inference-v1",
        "case": {k: row[k] for k in ("mode", "workers", "index", "unique_id")},
        **{k: config[k] for k in ("model", "model_revision", "dataset", "dataset_revision", "seed")},
        "rollout": rollout(), "metrics": {}, "engine": {},
    }
    path = directory / row["trace_path"]
    path.parent.mkdir()
    blob = gzip.compress(json.dumps(payload).encode(), mtime=0)
    path.write_bytes(blob)
    row.update(trace_sha256=hashlib.sha256(blob).hexdigest(), trace_bytes=len(blob))
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "results.jsonl").write_text(json.dumps(row) + "\n")
    return {"directory": directory, "row": row, "config": config}, payload


def test_readout_input_preserves_saved_ids_and_excludes_original_finisher():
    trace = rollout()
    original = deepcopy(trace)
    tok = Tokenizer()
    result = budget_eval.build_readout_input(trace, 1000, tok)
    assert result["view_token_ids"] == [1, 90, 4, 3, 15, 3, 4]
    assert result["input_ids"] == [1, 90, 4, 3, 15, 3, 4, 91, 70]
    assert result["effective_rounds"] == 3
    assert result["reasoning_tokens"] == 5
    assert trace == original
    assert not ({99, 111, 112, 113} & set(result["input_ids"]))


def test_last_thinking_tag_controls_cue_after_worker_reordering():
    trace = rollout()
    trace["worker_token_ids"] = {"Alice": [3, 91, 99], "Bob": [4, 90, 15]}
    result = budget_eval.build_readout_input(trace, 1000, Tokenizer())
    # Alice is last in the logical view, so her closing tag owns the final state.
    assert result["input_ids"][-1:] == [70]
    assert result["input_ids"][-2] == 91
    assert result["input_ids"].count(91) == 1


def test_readout_uses_eight_token_cap_without_old_box_fallback_or_repair():
    llm = FakeLLM([15] * 9)
    result = asyncio.run(budget_eval.greedy_readout(llm, [1, 90, 3, 91, 70], 8, [99], [91, 70]))
    assert result["prediction"] is None
    assert result["answer_token_ids"] == [15] * 8
    assert llm.sample_count == 8
    assert llm.freed == llm.blocks


@pytest.mark.parametrize("cap,prediction,expected_ids", [
    (3, None, [12, 10, 11]),
    (8, r"\sqrt{7}", [12, 10, 11, 11]),
])
def test_nested_box_requires_complete_outer_brace(cap, prediction, expected_ids):
    llm = FakeLLM([12, 10, 11, 11, 15])
    result = asyncio.run(budget_eval.greedy_readout(llm, [1, 70], cap, [99], [70]))
    assert result["prediction"] == prediction
    assert result["answer_token_ids"] == expected_ids
    assert llm.sample_count == len(expected_ids)


def test_eos_counts_one_sample_and_keeps_old_answer_unavailable():
    llm = FakeLLM([99, 10, 11])
    result = asyncio.run(budget_eval.greedy_readout(llm, [1, 70], 8, [99], [70]))
    assert result["prediction"] is None
    assert result["answer_token_ids"] == [99]
    assert result["stop_reason"] == "eos"
    assert llm.sample_count == 1


def test_readout_releases_its_block_after_failure():
    llm = FakeLLM([15] * 8, fail_on=2)
    with pytest.raises(RuntimeError, match="Simulated"):
        asyncio.run(budget_eval.greedy_readout(llm, [1, 70], 8, [99], [70]))
    assert llm.freed == llm.blocks


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "positive_inf", "negative_inf"])
@pytest.mark.parametrize("fault_step", [1, 2], ids=["prefill", "decode"])
def test_nonfinite_logits_reject_the_readout_and_release_cache(nonfinite, fault_step):
    class NonfiniteLLM(FakeLLM):
        async def forward(self, *args, **kwargs):
            result = await super().forward(*args, **kwargs)
            if len(self.forwards) == fault_step:
                result.logits[0] = nonfinite
            return result

    llm = NonfiniteLLM([15] * 8)
    with pytest.raises(RuntimeError, match="[Ff]inite"):
        asyncio.run(budget_eval.greedy_readout(llm, [1, 70], 8, [99], [70]))
    assert len(llm.forwards) == fault_step
    assert llm.freed == llm.blocks


def test_source_catalog_ignores_uncommitted_tail_and_rejects_duplicate_cases(tmp_path):
    case, _ = source_case(tmp_path)
    results = case["directory"] / "results.jsonl"
    with results.open("ab") as stream:
        stream.write(b'{"mode":')
    catalog = budget_eval.read_catalog(tmp_path)
    assert set(catalog) == {("hogwild", 2, 0)}
    source_case(tmp_path, "duplicate")
    with pytest.raises(ValueError):
        budget_eval.read_catalog(tmp_path)


def test_source_trace_checks_hash_and_identity_before_use(tmp_path):
    case, payload = source_case(tmp_path)
    assert budget_eval.load_source_trace(case)["rollout"] == payload["rollout"]
    path = case["directory"] / case["row"]["trace_path"]
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 1
    path.write_bytes(blob)
    with pytest.raises(ValueError):
        budget_eval.load_source_trace(case)
    payload["case"]["index"] = 1
    blob = gzip.compress(json.dumps(payload).encode(), mtime=0)
    path.write_bytes(blob)
    case["row"].update(trace_sha256=hashlib.sha256(blob).hexdigest(), trace_bytes=len(blob))
    with pytest.raises(ValueError):
        budget_eval.load_source_trace(case)


def test_catalog_rejects_sources_from_a_different_model_revision(tmp_path):
    case, _ = source_case(tmp_path)
    case["config"]["model_revision"] = "different-model-weights"
    (case["directory"] / "config.json").write_text(json.dumps(case["config"]))
    with pytest.raises(ValueError, match="Unpinned"):
        budget_eval.read_catalog(tmp_path)


def test_resume_requires_same_protocol_and_unique_committed_keys(tmp_path):
    config = {"protocol": "budget-readout-v1", "budgets": [1000, 2000], "answer_tokens": 8}
    (tmp_path / "config.json").write_text(json.dumps(config))
    row = {"mode": "native", "workers": 1, "index": 4, "budget": 1000,
           "correct": True, "answer_tokens": 2, "generated_tokens": 50}
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert set(budget_eval.read_existing(tmp_path, config)) == {("native", 1, 4, 1000)}
    with pytest.raises(ValueError):
        budget_eval.read_existing(tmp_path, {**config, "answer_tokens": 16})
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError):
        budget_eval.read_existing(tmp_path, config)


def test_resume_archives_only_uncommitted_tail_and_preserves_committed_bytes(tmp_path):
    config = {"protocol": "budget-readout-v1", "budgets": [1000], "answer_tokens": 8}
    (tmp_path / "config.json").write_text(json.dumps(config))
    row = {"mode": "native", "workers": 1, "index": 4, "budget": 1000,
           "correct": True, "answer_tokens": 2, "generated_tokens": 50}
    committed = (json.dumps(row) + "\n").encode()
    tail = b'{"mode":"native"'
    path = tmp_path / "results.jsonl"
    path.write_bytes(committed + tail)
    existing = budget_eval.read_existing(tmp_path, config)
    assert existing == {("native", 1, 4, 1000): row}
    assert path.read_bytes() == committed
    archive = tmp_path / "recovery" / f"incomplete-results-{hashlib.sha256(tail).hexdigest()}.bin"
    assert archive.read_bytes() == tail
    assert budget_eval.read_existing(tmp_path, config) == existing
    assert list((tmp_path / "recovery").iterdir()) == [archive]


@pytest.mark.parametrize("bad_committed", ["duplicate", "malformed"])
def test_invalid_committed_rows_are_rejected_before_partial_tail_is_modified(tmp_path, bad_committed):
    config = {"protocol": "budget-readout-v1", "budgets": [1000], "answer_tokens": 8}
    (tmp_path / "config.json").write_text(json.dumps(config))
    row = {"mode": "native", "workers": 1, "index": 4, "budget": 1000}
    committed = (json.dumps(row) + "\n").encode()
    committed += committed if bad_committed == "duplicate" else b'{"mode":\n'
    original = committed + b'{"mode":"native"'
    path = tmp_path / "results.jsonl"
    path.write_bytes(original)
    with pytest.raises(ValueError):
        budget_eval.read_existing(tmp_path, config)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("recovery/incomplete-results-*.bin"))


@pytest.mark.parametrize("marker", ["quality-hold.json", "multiworker-stopped.json", "superseded.json"])
def test_readouts_require_an_active_source_run(tmp_path, marker):
    budget_eval.assert_source_active(tmp_path)
    (tmp_path / marker).write_text("{}\n")
    with pytest.raises(RuntimeError):
        budget_eval.assert_source_active(tmp_path)


def test_early_stop_context_reuses_exact_answer_across_budgets(tmp_path):
    case, trace = source_case(tmp_path)
    llm = FakeLLM([10, 11, 10, 11])
    committed = []
    rows = asyncio.run(budget_eval.evaluate_case(
        llm, case, trace, [1, 3, 1000, 2000], 8, [99], {}, on_result=committed.append,
    ))
    assert [row["budget"] for row in rows] == [1, 3, 1000, 2000]
    assert [row["reused_readout"] for row in rows] == [False, False, True, True]
    assert [row["generated_tokens"] for row in rows] == [4, 7, 7, 7]
    assert all(row["correct"] and row["answer_tokens"] == 2 for row in rows)
    assert rows[1]["context_sha256"] == rows[2]["context_sha256"] == rows[3]["context_sha256"]
    assert len(llm.blocks) == 2 and llm.sample_count == 4
    assert committed == rows


def test_committed_context_can_supply_later_budget_after_resume(tmp_path):
    case, trace = source_case(tmp_path)
    first = asyncio.run(budget_eval.evaluate_case(FakeLLM([10, 11]), case, trace, [1000], 8, [99], {}))[0]
    existing = {("hogwild", 2, 0, 1000): first}
    llm = FakeLLM([])
    rows = asyncio.run(budget_eval.evaluate_case(llm, case, trace, [1000, 2000], 8, [99], existing))
    assert len(rows) == 1 and rows[0]["budget"] == 2000
    assert rows[0]["reused_readout"] is True
    assert rows[0]["answer_token_ids"] == first["answer_token_ids"]
    assert rows[0]["prediction"] == first["prediction"] == "7"
    assert not llm.blocks and not llm.forwards


def test_resume_rejects_changed_source_even_when_its_budget_is_already_complete(tmp_path):
    case, trace = source_case(tmp_path)
    first = asyncio.run(budget_eval.evaluate_case(FakeLLM([10, 11]), case, trace, [1000], 8, [99], {}))[0]
    case["row"]["trace_sha256"] = "changed-source-trace"
    llm = FakeLLM([])
    with pytest.raises(ValueError, match="Source trace changed"):
        asyncio.run(budget_eval.evaluate_case(
            llm, case, trace, [1000], 8, [99], {("hogwild", 2, 0, 1000): first},
        ))
    assert not llm.blocks


def test_identical_context_in_another_case_does_not_cross_case_cache(tmp_path):
    first_case, first_trace = source_case(tmp_path, "first", index=0)
    first = asyncio.run(budget_eval.evaluate_case(
        FakeLLM([10, 11]), first_case, first_trace, [1000], 8, [99], {},
    ))[0]
    next_case, next_trace = source_case(tmp_path, "next", index=1)
    llm = FakeLLM([10, 11])
    rows = asyncio.run(budget_eval.evaluate_case(
        llm, next_case, next_trace, [1000], 8, [99], {("hogwild", 2, 0, 1000): first},
    ))
    assert len(rows) == 1 and rows[0]["index"] == 1
    assert rows[0]["reused_readout"] is False
    assert len(llm.blocks) == 1 and llm.sample_count == 2


def test_each_result_is_committed_before_a_later_readout_can_fail(tmp_path):
    case, trace = source_case(tmp_path)
    llm = FakeLLM([10, 11, 10], fail_on=3)
    committed = []
    with pytest.raises(RuntimeError, match="Simulated"):
        asyncio.run(budget_eval.evaluate_case(
            llm, case, trace, [1, 3], 8, [99], {}, on_result=committed.append,
        ))
    assert len(committed) == 1 and committed[0]["budget"] == 1
    assert committed[0]["prediction"] == "7"
    assert llm.freed == llm.blocks


def test_source_hold_after_first_commit_blocks_even_a_reused_readout(tmp_path):
    case, trace = source_case(tmp_path)
    llm = FakeLLM([10, 11])
    committed = []
    checks = []

    def check_source():
        checks.append(len(committed))
        if committed:
            raise RuntimeError("Source run is on hold")

    with pytest.raises(RuntimeError, match="on hold"):
        asyncio.run(budget_eval.evaluate_case(
            llm, case, trace, [1000, 2000], 8, [99], {},
            on_result=committed.append, check_source=check_source,
        ))
    assert checks == [0, 1]
    assert len(committed) == 1 and committed[0]["budget"] == 1000
    assert committed[0]["prediction"] == "7"
    assert llm.sample_count == 2 and llm.freed == llm.blocks
