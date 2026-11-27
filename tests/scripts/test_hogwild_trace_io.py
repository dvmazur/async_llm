"""Trace the seven MATH-500 conditions through the evaluator on the CPU scheduler."""

import gzip
import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.training.trace import Trace
from minisgl_examples.training.parallel_thinking import eval as evaluation
from minisgl_examples.training.parallel_thinking import trace_io
from minisgl_examples.training.parallel_thinking.rollout import (
    MathFormatting,
    select_terminal_worker_answer,
)

from tests.core.test_async_llm import _make_llm
from tests.training.helpers import copy_operations


class TinyTokenizer:
    vocab = {"<think>": 24, "</think>": 25, "ĊĊ": 6}
    eos_token_id = 10

    def apply_chat_template(self, messages, **options):
        assert messages[0]["role"] == "user" and messages[0]["content"].endswith("What is seven?")
        assert options == {"tokenize": False, "add_generation_prompt": True,
                           "enable_thinking": True, "reasoning_effort": "xhigh"}
        return "prompt"

    def encode(self, text, **options):
        assert options == {"add_special_tokens": False}
        cue = "\n\nThe final answer is \\boxed{"
        return {
            "prompt": [1, 24, 2], "\n\n": [6], MathFormatting.s1_finisher_suffix: [11, 12],
            cue: [11, 12], "</think>" + cue: [25, 11, 12],
        }[text]

    def decode(self, ids, *, skip_special_tokens=False):
        pieces = {1: "Question", 2: " ", 4: "Seven", 6: ".\n\n", 8: "We conclude", 10: "<|im_end|>",
                  11: "\\boxed", 12: "{",
                  14: "7}", 24: "<think>", 25: "</think>"}
        return "".join(pieces.get(i, str(i)) for i in ids if not (skip_special_tokens and i == 10))


def test_trace_audit_reselects_answer_and_finisher_from_token_ids(monkeypatch):
    tokenizer = TinyTokenizer()
    monkeypatch.setattr(trace_io, "selection_tokenizer", lambda *args: tokenizer)
    streams = {"Alice": [11, 12, 14, 10], "Bob": [4, 6, 8, 4]}
    answer, selection = select_terminal_worker_answer(tokenizer, streams, eos_worker="Alice", eos_token_ids=[10])
    rollout = {"workers": ["Alice", "Bob"], "worker_token_ids": streams, "eos_worker": "Alice",
               "eos_token_ids": [10], "answer_selection": selection, "finisher": {"used": False}}
    data = {"model": "fixture", "model_revision": "fixture", "rollout": rollout}
    row = {"eos_worker": "Alice", "prediction": answer}
    manifest = {"answer_selection_policy": "eos_worker_last_box_v1", "eos_token_ids": [10]}
    trace_io.verify_answer_selection(data, row, manifest)
    # Producers may sort this unordered set; generation metadata need not.
    manifest["eos_token_ids"] = [99, 10]
    rollout["eos_token_ids"] = [10, 99]
    trace_io.verify_answer_selection(data, row, manifest)
    manifest["eos_token_ids"] = rollout["eos_token_ids"] = [10]
    row["prediction"] = "999"
    with pytest.raises(ValueError, match="Prediction differs"):
        trace_io.verify_answer_selection(data, row, manifest)
    row["prediction"] = answer
    rollout["answer_selection"] = {**selection, "selected_answer": "999"}
    with pytest.raises(ValueError, match="selection differs"):
        trace_io.verify_answer_selection(data, row, manifest)
    rollout["answer_selection"] = selection
    rollout["finisher"]["used"] = True
    with pytest.raises(ValueError, match="Finisher decision"):
        trace_io.verify_answer_selection(data, row, manifest)
    rollout["finisher"]["used"] = False
    rollout["eos_token_ids"] = [999]
    with pytest.raises(ValueError, match="EOS tokens differ"):
        trace_io.verify_answer_selection(data, row, manifest)
    rollout["eos_token_ids"] = [10]
    rollout["eos_worker"] = row["eos_worker"] = "Bob"
    with pytest.raises(ValueError, match="terminating worker"):
        trace_io.verify_answer_selection(data, row, manifest)


def session_signature(session):
    """Compare inputs and ordered read/write blocks, ignoring process-global block IDs."""
    names = {}

    def name(block):
        return names.setdefault(block.block_id, len(names))

    prefills = [(call["ids"], name(call["block"]), [name(b) for b in call["context"] or []])
                for call in session.prefill_calls]
    decodes = [(call["input_ids"], [name(b) for b in call["write_to"]],
                [[name(b) for b in view] for view in call["structure"]]) for call in session.decode_calls]
    return prefills, decodes


@pytest.mark.parametrize("mode,workers", [("native", 1), *(("hogwild", n) for n in (1, 2, 4, 6, 8, 16))])
def test_evaluator_saves_complete_trace_with_identical_generation(tmp_path, monkeypatch, mode, workers):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 512}}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": False, "eos_token_id": [10, 16]}))
    data = {"unique_id": "test/algebra/7.json", "problem": "What is seven?", "answer": "7"}
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(json.dumps(data) + "\n")
    sessions = []

    def cpu_model(path, **options):
        llm, session = _make_llm()
        copy_operations(session)
        llm.tokenizer = TinyTokenizer()
        llm.engine = SimpleNamespace(
            max_seq_len=512, sampler=llm.async_engine.sampler,
            config=SimpleNamespace(get_default_sampling_params=SamplingParams),
        )
        sessions.append(session)
        return llm

    monkeypatch.setattr(evaluation, "AsyncLLM", cpu_model)
    monkeypatch.setattr(evaluation, "snapshot_download", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(evaluation.math500, "load_math500", lambda: (dataset, [data]))
    monkeypatch.setattr(evaluation, "gpu_preflight", lambda: {"index": "7", "uuid": "CPU"})
    monkeypatch.setattr(trace_io, "selection_tokenizer", lambda *args: TinyTokenizer())
    monkeypatch.setattr(evaluation.subprocess, "check_output", lambda *args, **kwargs: "fixture\n")
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())
    args = ["--mode", mode, "--workers", str(workers), "--limit", "1", "--budget", "6", "--no-cuda-graphs"]
    if mode == "hogwild":
        args += ["--finisher-tokens", "3"]

    plain, traced = tmp_path / "plain", tmp_path / "traced"
    with torch.device("cpu"):
        evaluation.main([*args, "--output", str(plain)])
        evaluation.main([*args, "--output", str(traced), "--save-traces"])

    baseline = json.loads((plain / "results.jsonl").read_text())
    row = json.loads((traced / "results.jsonl").read_text())
    manifest = json.loads((traced / "config.json").read_text())
    assert {k: row[k] for k in baseline if k != "wall_seconds"} == {
        k: v for k, v in baseline.items() if k != "wall_seconds"
    }
    assert session_signature(sessions[0]) == session_signature(sessions[1])
    assert not (plain / "traces").exists() and "trace_path" not in baseline
    assert json.loads((traced / "completed.json").read_text())["cases"] == 1
    trace_io.verify_trace(traced, row, manifest)
    path = traced / row["trace_path"]
    blob = path.read_bytes()
    assert row["trace_bytes"] == len(blob) and row["trace_sha256"] == hashlib.sha256(blob).hexdigest()
    content = json.loads(gzip.decompress(blob))
    assert content["format"] == trace_io.TRACE_FORMAT and content["case"]["unique_id"] == data["unique_id"]
    rollout = content["rollout"]
    assert rollout["prompt_ids"] == [1, 24, 2]
    assert set(rollout["worker_text"]) == set(rollout["workers"])
    assert len(rollout["workers"]) == workers and list(rollout["worker_token_ids"].values()) == [[4, 6, 8, 10]] * workers
    assert row["eos_token_id"] == rollout["eos_token_id"] == 10
    assert all(text.endswith("<|im_end|>") for text in rollout["worker_text"].values())
    expected_pubs = workers if mode == "hogwild" else 0
    assert len(rollout["publications"]) == expected_pubs
    assert [event["round"] for event in rollout["publications"]] == [3 if workers > 1 else 2] * expected_pubs
    if workers > 1:
        assert rollout["publication_policy"] == manifest["publication_policy"] == "encoded_separator_v1"
    assert rollout["finisher"]["sampled_token_ids"] == ([14] if mode == "hogwild" else [])
    assert row["prediction"] == ("7" if mode == "hogwild" else None)

    executed = Trace.from_dict(content["engine"])
    assert executed.steps == len(sessions[1].prefill_batches) + len(sessions[1].decode_calls)
    assert sum(bool(update.refs) for update in executed.updates) == expected_pubs
    assert {10, 16}.isdisjoint(t.token_id for t in executed.tokens)
    if mode == "hogwild" and workers == 2:
        def save_altered_trace():
            rewritten = gzip.compress(json.dumps(content).encode())
            path.write_bytes(rewritten)
            row.update(trace_sha256=hashlib.sha256(rewritten).hexdigest(), trace_bytes=len(rewritten))

        path.write_bytes(bytes([blob[0] ^ 1]) + blob[1:])
        with pytest.raises(ValueError, match="SHA-256"):
            trace_io.verify_trace(traced, row, manifest)
        transfer = next(update for update in content["engine"]["updates"] if update["refs"])
        transfer["refs"][1][1] += 1
        save_altered_trace()
        with pytest.raises(ValueError, match="stale"):
            trace_io.verify_trace(traced, row, manifest)
        transfer["refs"][1][1] -= 1
        for event in rollout["publications"]:
            event["round"] += 1
        save_altered_trace()
        with pytest.raises(ValueError, match="encoded separator"):
            trace_io.verify_trace(traced, row, manifest)
        for event in rollout["publications"]:
            event["round"] -= 1

        roles = rollout["block_ids"]
        requests = [request for batch in content["engine"]["batches"] for request in batch["requests"]]
        for block, message in ((roles["history"], "Common cache inputs"),
                               (roles["workers"][rollout["workers"][0]], "exactly once"),
                               (roles["answer"], "Answer cache inputs")):
            request = next(request for request in requests if request["write_to"] == block)
            original = request["input_ids"][0]
            request["input_ids"][0] = 99
            save_altered_trace()
            # The stored lengths and transfers still agree, and the checksum
            # is valid. The input tokens must also match the sampled streams.
            with pytest.raises(ValueError, match=message):
                trace_io.verify_trace(traced, row, manifest)
            request["input_ids"][0] = original
        rollout["worker_token_ids"][rollout["workers"][0]].pop()
        save_altered_trace()
        with pytest.raises(ValueError, match="count"):
            trace_io.verify_trace(traced, row, manifest)
