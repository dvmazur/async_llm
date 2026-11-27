"""Chronological publication through the real CPU AsyncLLM scheduler."""

import asyncio
import hashlib
from contextlib import nullcontext

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.training import RecordedAsyncLLM
from minisgl_examples.training.parallel_thinking.native import run_native
from minisgl_examples.training.parallel_thinking.single_worker import run_single_worker

from tests.core.test_async_llm import _make_llm
from tests.training.helpers import copy_operations


class Format:
    workers = ("Alice",)
    current_step_header = "others"
    current_worker_header = "own"
    incomplete_step = "<...>"
    sep = "\n\n"
    s1_finisher_suffix = r"finish \boxed{"
    s1_collab_message = "Quick check: am I doing redundant work? (yes/no): "

    def apply_chat_template(self, problem):
        return "scaffold"

    def get_step_prefix(self, worker, step):
        return f"{worker} {step}: "

    def is_end_of_step(self, ids):
        return ids[-1] in (6, 14)


class Tokenizer:
    vocab = {"<|im_start|>": 22, "<|endoftext|>": 31}

    def __init__(self):
        self.encodings = []

    def apply_chat_template(self, messages, **options):
        assert messages == [{"role": "user", "content": "problem"}]
        assert options == {"tokenize": False, "add_generation_prompt": True,
                           "enable_thinking": True, "reasoning_effort": "xhigh"}
        return "native"

    def encode(self, text, **options):
        assert options == {"add_special_tokens": False}
        self.encodings.append(text)
        return {
            "native": [1, 2], "scaffold": [1, 2], "others": [3, 4], "own": [5, 6],
            "Alice 1: Hi, I'm Alice.": [7, 8], "Alice 2: ": [15, 16],
            "Alice 2: " + Format.s1_collab_message: [17, 18],
            Format.incomplete_step + Format.sep: [9, 10], Format.s1_finisher_suffix: [11, 12],
        }[text]

    def decode(self, ids, **options):
        if list(ids) == [4, 6, 8, 10]:
            return r"answer \boxed{42}"
        if list(ids) == [14]:
            return r"\frac{1}{"
        if list(ids) == [14, 16]:
            return r"\frac{1}{2}}"
        return "ids(" + ",".join(map(str, ids)) + ")"


@pytest.fixture(autouse=True)
def cpu_profiling(monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())


def exercise(*, scaffold=False, publish=True, native=False, fail_transfer=False, cap=128, budget=8, eos=None,
             capture_trace=False):
    async def run():
        llm, session = _make_llm()
        tok = llm.tokenizer = Tokenizer()
        copy_operations(session)
        decoded, transfers, created = [], [], []
        decode, append, create = session.decode_step, session.append_block, session.create_block

        def audited_decode(group, input_ids):
            for blocks, token in zip(group.cache_structure, input_ids.tolist()):
                prefix = [value for block in blocks for value in block.token_ids]
                decoded.append((prefix + [token], len(blocks)))
            return decode(group, input_ids)

        def audited_append(left, right):
            transfers.append((left.token_ids.copy(), right.token_ids.copy()))
            if fail_transfer:
                raise RuntimeError("simulated append failure")
            return append(left, right)

        def audited_create():
            block = create()
            created.append(block)
            return block

        session.decode_step, session.append_block, session.create_block = audited_decode, audited_append, audited_create
        try:
            args = {"problem": "problem", "max_steps": budget, "sampling": SamplingParams(temperature=0),
                    "eos_token_ids": ([20, 31] if scaffold else [10, 31]) if eos is None else eos,
                    "max_context_tokens": cap}
            with (RecordedAsyncLLM(llm) if capture_trace else nullcontext(llm)) as model:
                if native:
                    result = await run_native(model, **args, capture_trace=capture_trace)
                else:
                    result = await run_single_worker(
                        model, Format(), **args, answer_steps=4, publish_paragraphs=publish,
                        scaffold=scaffold, reminder_interval=1, capture_trace=capture_trace,
                    )
            if capture_trace:
                session.captured_trace = model.get_trace()
            return result, decoded, transfers, session, tok
        finally:
            assert all(b.num_tokens == 0 and not b.token_ids for b in created)
            await llm.close()

    with torch.device("cpu"):
        return asyncio.run(run())


def test_native_prompt_transfer_preserves_every_next_token_boundary():
    native, native_reads, _, _, _ = exercise(native=True)
    on, on_reads, transfers, _, _ = exercise()
    off, off_reads, off_transfers, _, _ = exercise(publish=False)
    expected = [[1, 2], [1, 2, 4], [1, 2, 4, 6], [1, 2, 4, 6, 8]]
    for reads in (native_reads, on_reads, off_reads):
        assert [ids for ids, _ in reads] == expected
    assert [depth for _, depth in on_reads] == [1, 1, 2, 2]
    assert [depth for _, depth in off_reads] == [1, 1, 1, 1]
    # Separator 6 is still pending during transfer. It appears once, on
    # the next decode, and that forward supplies the next sampled token.
    assert transfers == [([], [1, 2, 4])]
    assert not off_transfers
    assert on["prompt_token_ids_sha256"] == off["prompt_token_ids_sha256"]
    assert on["first_boundary_prefix_sha256"] == off["first_boundary_prefix_sha256"] == hashlib.sha256(b"[4,6]").hexdigest()
    assert on["first_boundary_round"] == off["first_boundary_round"] == on["first_publication_round"] == 2
    assert on["completed_steps"] == 1 and off["completed_steps"] == 0 and off["first_publication_round"] is None
    for result in (native, on, off):
        assert result["boxed_answer"] == "42"
        assert result["decode_rounds"] == result["total_generated_tokens"] == 4
        assert result["stop_reason"] == "eos" and result["eos_token_id"] == 10
        assert result["answer_generated_tokens"] == 0 and not result["used_finisher"]
        assert "trace" not in result and "current_ids" not in result


def test_scaffold_labels_and_cached_finisher_consume_only_new_tokens():
    result, reads, transfers, session, tok = exercise(scaffold=True)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    assert [ids for ids, _ in reads[:4]] == [
        prompt, prompt + [10], prompt + [10, 12], prompt + [10, 12, 14, 17, 18],
    ]
    assert transfers == [([], prompt + [10, 12])]
    assert any(call["ids"] == [14, 17] for call in session.prefill_calls)
    assert result["inserted_worker_tokens"] == 4
    assert result["first_boundary_prefix_sha256"] == hashlib.sha256(b"[7,8,10,12,14]").hexdigest()
    assert result["first_boundary_round"] == result["first_publication_round"] == 3
    assert all(not text.startswith("ids(") for text in tok.encodings)
    assert [call["ids"] for call in session.prefill_calls] == [prompt[:-1], [14, 17], [11, 12]]
    assert reads[-1] == (prompt + [10, 12, 14, 17, 18, 11, 12, 14], 2)
    assert result["finisher_mode"] == "cached"
    assert result["boxed_answer"] == r"\frac{1}{2}" and result["finisher_stop_reason"] == "answer"
    assert result["reasoning_generated_tokens"] == 4 and result["answer_generated_tokens"] == 2
    assert result["total_generated_tokens"] == 6 and result["eos_token_id"] == 20


def test_finisher_replaces_pending_eos_before_any_publication():
    result, reads, transfers, session, _ = exercise(eos=[4])
    assert transfers == []
    assert [call["ids"] for call in session.prefill_calls] == [[1], [11, 12]]
    assert reads == [([1, 2], 1), ([1, 2, 11, 12, 14], 1)]
    assert result["reasoning_stop_reason"] == "eos" and result["eos_token_id"] == 4
    assert result["reasoning_generated_tokens"] == 1 and result["answer_generated_tokens"] == 2
    assert result["boxed_answer"] == r"\frac{1}{2}" and result["peak_context_tokens"] == 6


def test_captured_scaffold_records_samples_and_actual_published_cache_rows():
    plain, original_reads, original_transfers, _, _ = exercise(scaffold=True)
    result, reads, transfers, session, _ = exercise(scaffold=True, capture_trace=True)
    trace = result.pop("trace")
    assert result == plain and reads == original_reads and transfers == original_transfers
    assert trace["prompt_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert trace["initial_worker_prefix_ids"] == {"Alice": [7, 8]}
    assert trace["worker_token_ids"] == {"Alice": [10, 12, 14, 20]}
    assert trace["publications"] == [{
        "round": 3, "worker": "Alice", "step": 1,
        "committed_token_count": len(transfers[0][1]), "next_prefix_ids": [17, 18],
    }]
    active = trace["block_ids"]["workers"]["Alice"]
    history = trace["block_ids"]["history"]
    assert trace["block_ids"]["answer"] == active
    assert any(update.block == history and (active, 10) in update.refs
               for update in session.captured_trace.updates if update.refs)
    assert trace["finisher"] == {
        "used": True, "mode": "cached", "worker_marker_ids": [], "suffix_ids": [11, 12],
        "sampled_token_ids": [14, 16], "stop_reason": "answer", "eos_token_id": None,
    }


def test_trace_keeps_both_sampled_eos_without_forwarding_them():
    result, _, _, session, _ = exercise(eos=[4, 14], capture_trace=True)
    trace = result["trace"]
    assert trace["worker_token_ids"] == {"Alice": [4]} and trace["publications"] == []
    assert trace["eos_token_id"] == 4 and trace["finisher"]["sampled_token_ids"] == [14]
    assert trace["finisher"]["eos_token_id"] == 14 and trace["finisher"]["stop_reason"] == "eos"
    assert [t.token_id for t in session.captured_trace.tokens] == [1, 2, 11, 12]


def test_second_publication_appends_only_the_encoded_active_paragraph():
    result, reads, transfers, _, _ = exercise(budget=12, eos=[20, 31])
    assert transfers == [([], [1, 2, 4]), ([1, 2, 4], [6, 8, 10, 12])]
    sampled = list(range(4, 21, 2))
    assert [ids for ids, _ in reads[:9]] == [[1, 2, *sampled[:n]] for n in range(9)]
    assert result["completed_steps"] == 2 and result["decode_rounds"] == 9
    assert result["eos_token_id"] == 20


def test_finisher_consumes_pending_label_once_if_budget_ends_at_publication():
    result, reads, _, session, _ = exercise(scaffold=True, budget=3)
    assert result["reasoning_stop_reason"] == "max_steps" and result["first_publication_round"] == 3
    assert [call["ids"] for call in session.prefill_calls][-2:] == [[14, 17], [18, 11, 12]]
    assert reads[-1] == ([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 18, 11, 12, 14], 2)
    assert result["boxed_answer"] == r"\frac{1}{2}" and result["finisher_mode"] == "cached"


def test_context_reserves_the_final_answer_and_reports_early_stop():
    result, _, _, _, _ = exercise(cap=10)
    assert result["decode_rounds"] == 2
    assert result["reasoning_stop_reason"] == "context_limit"
    assert result["used_finisher"] and result["peak_context_tokens"] <= 10


def test_partially_completed_rollout_releases_both_blocks():
    with pytest.raises(RuntimeError, match="simulated append failure"):
        exercise(fail_transfer=True)
