"""Multi-worker inference budgets and cleanup through the real CPU scheduler."""

import asyncio
import hashlib
from contextlib import nullcontext

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM
from minisgl_examples.training.parallel_thinking.rollout import (
    TERMINAL_ANSWER_POLICY,
    MathFormatting,
    extract_boxed_answer,
    run_agents,
    select_terminal_worker_answer,
)

from tests.core.test_async_cache_engine import StubSession
from tests.core.test_async_llm import _make_llm
from tests.scripts.training_helpers import StubFormatting
from tests.training.helpers import copy_operations


@pytest.fixture(autouse=True)
def cpu_profiling(monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())


class WorkerFormatting(StubFormatting):
    s1_collab_message = "Wait, am I doing redundant work? (yes/no):"

    def __init__(self, workers):
        self.workers = tuple(f"Worker{i}" for i in range(workers))


def run_cases(workers, *options, formatting=None):
    async def run():
        llm, session = _make_llm()
        llm.tokenizer = formatting or WorkerFormatting(workers)
        copy_operations(session)
        blocks, results = [], []
        create = session.create_block

        def tracked_create():
            block = create()
            blocks.append(block)
            return block

        session.create_block = tracked_create
        try:
            for case in options:
                recording = case.pop("record_engine", False)
                with RecordedAsyncLLM(llm) if recording else nullcontext(llm) as runner:
                    result = await run_agents(
                        runner, llm.tokenizer, problem="test", synchronized=case.pop("synchronized", True),
                        max_steps=case.pop("max_steps", 4), answer_steps=case.pop("answer_steps", 2), **case,
                    )
                if recording:
                    session.recorded_trace = runner.get_trace()
                assert all(block.num_tokens == 0 and block.token_ids == [] for block in blocks)
                results.append(result)
            return results, session
        finally:
            assert all(block.num_tokens == 0 and block.token_ids == [] for block in blocks)
            await llm.close()

    with torch.device("cpu"):
        return asyncio.run(run())


@pytest.mark.parametrize("workers", [1, 4, 16])
def test_round_budget_and_repeated_examples_release_all_blocks(workers):
    results, session = run_cases(workers, {}, {})
    assert results[0]["generated_tokens"] == results[1]["generated_tokens"]
    for result in results:
        assert set(result["generated_tokens"].values()) == {4}
        assert result["reasoning_generated_tokens"] == workers * 4
        assert result["answer_generated_tokens"] == 2
        assert result["total_generated_tokens"] == workers * 4 + 2
        assert result["decode_rounds"] == result["rounds"] == 4
        assert result["completed_steps"] == workers * 2
        assert result["first_publication_round"] == 2
        assert result["stop_reason"] == "max_steps"
        assert result["eos_token_id"] is None and result["eos_worker"] is None
        assert any(len(call["input_ids"]) == workers for call in session.decode_calls)
        assert max(len(view) for call in session.decode_calls for view in call["structure"]) == workers + 4


@pytest.mark.parametrize("workers", [1, 4, 16])
def test_aggregate_budget_has_no_partial_batch_overshoot(workers):
    budget = workers * 2 + 1
    (result,), _ = run_cases(workers, {"max_steps": 10, "total_token_budget": budget})
    assert result["reasoning_generated_tokens"] == budget
    assert result["generated_tokens"]["Worker0"] == 3
    assert all(value == 2 for name, value in result["generated_tokens"].items() if name != "Worker0")
    assert result["stop_reason"] == "token_budget"
    assert result["decode_rounds"] == 3


@pytest.mark.parametrize("workers", [1, 4, 16])
def test_context_limit_reserves_final_answer_and_reports_stop(workers):
    # Two-token prompt, two headers, two-token prefixes, one round, and
    # four tokens reserved for the finisher prefix and its two output samples.
    cap = 3 * workers + 10
    (result,), _ = run_cases(workers, {"max_context_tokens": cap})
    assert result["decode_rounds"] == 1
    assert result["stop_reason"] == "context_limit"
    assert result["answer_generated_tokens"] == 2
    assert result["peak_context_tokens"] <= cap


def test_context_limit_rejects_oversize_prompt_and_cleans_up():
    with pytest.raises(ValueError, match="final-answer reserve"):
        run_cases(4, {"max_context_tokens": 2})


@pytest.mark.parametrize("workers, cap", [(1, 14), (4, 26)])
def test_context_limit_covers_new_paragraph_prefixes(workers, cap):
    (result,), _ = run_cases(workers, {"max_context_tokens": cap})
    assert result["decode_rounds"] == 2
    assert result["completed_steps"] == 0
    assert result["first_publication_round"] is None
    assert result["first_boundary_round"] == (2 if workers == 1 else None)
    assert result["stop_reason"] == "context_limit"
    assert result["peak_context_tokens"] == cap


def test_metrics_only_path_preserves_counts_without_token_details():
    (detailed, compact), _ = run_cases(4, {}, {"retain_details": False})
    for field in ("generated_tokens", "reasoning_generated_tokens", "answer_generated_tokens",
                  "total_generated_tokens", "decode_rounds", "completed_steps", "max_token_gap",
                  "peak_context_tokens", "stop_reason", "answer", "boxed_answer"):
        assert compact[field] == detailed[field]
    for field in ("transcript", "generated_text", "progress", "merge_events", "history_ids"):
        assert field in detailed and field not in compact


def test_all_supplied_eos_ids_stop_reasoning():
    (result,), _ = run_cases(4, {"eos_token_ids": [4, 31]})
    assert result["decode_rounds"] == 1
    assert result["stop_reason"] == "eos"
    assert result["eos_token_id"] == 4 and result["eos_worker"] == "Worker0"
    assert result["answer_generated_tokens"] == 1
    assert result["finisher_eos_token_id"] == 4


def test_collaboration_reminders_are_injected_at_publication():
    (result,), _ = run_cases(4, {"reminder_interval": 3})
    assert result["reminder_events"]
    assert result["reminder_events"][0]["worker"] == "Worker0"
    assert result["reminder_events"][0]["generated_tokens"] == 8


@pytest.mark.parametrize("synchronized", [True, False])
def test_publication_ablation_skips_cache_appends_and_prefix_injection(synchronized):
    options = {"synchronized": synchronized, "retain_details": False}
    (on, off), _ = run_cases(1, options.copy(), {**options, "publish_paragraphs": False})
    assert on["completed_steps"] == 2 and off["completed_steps"] == 0
    assert on["first_publication_round"] == (2 if synchronized else None)
    assert off["first_publication_round"] is None
    assert off["inserted_worker_tokens"] == 2 < on["inserted_worker_tokens"]
    for result in (on, off):
        assert result["stop_reason"] == "max_steps" and result["reasoning_generated_tokens"] == 4
        assert result["used_finisher"] and result["answer_generated_tokens"] == 2
        assert "trace" not in result and "current_ids" not in result
    if synchronized:
        prompt_digest = hashlib.sha256(b"[1,2,1,2,1,2,1,2]").hexdigest()
        boundary_digest = hashlib.sha256(b"[1,2,4,6]").hexdigest()
        for result in (on, off):
            assert result["prompt_token_ids_sha256"] == prompt_digest
            assert result["first_boundary_round"] == 2
            assert result["first_boundary_prefix_sha256"] == boundary_digest
    else:
        for result in (on, off):
            assert result["prompt_token_ids_sha256"] is None
            assert result["first_boundary_round"] is None and result["first_boundary_prefix_sha256"] is None


def test_disabled_publication_scans_only_until_the_first_boundary():
    class CountChecks(WorkerFormatting):
        checks = 0

        def is_end_of_step(self, ids):
            self.checks += 1
            return super().is_end_of_step(ids)

    fmt = CountChecks(1)
    (result,), _ = run_cases(1, {"publish_paragraphs": False}, formatting=fmt)
    assert fmt.checks == result["first_boundary_round"] == 2
    assert result["history_ids"] == [] and result["merge_events"] == []


def test_disabled_publication_can_finish_without_any_boundary():
    (result,), _ = run_cases(1, {"max_steps": 1, "publish_paragraphs": False})
    assert result["first_publication_round"] is None and result["first_boundary_round"] is None
    assert result["first_boundary_prefix_sha256"] is None and result["prompt_token_ids_sha256"]


def test_existing_numeric_answer_skips_finisher_but_not_reasoning():
    class BoxFormatting(WorkerFormatting):
        def decode(self, ids):
            return r"\boxed{17}" if 6 in ids else super().decode(ids)

        def get_final_answer(self, text):
            answer = extract_boxed_answer(text, numeric_only=True)
            return float(answer) if answer is not None else None

    (result,), _ = run_cases(4, {"finish_if_missing": True}, formatting=BoxFormatting(4))
    assert result["decode_rounds"] == 4
    assert result["stop_reason"] == "max_steps"
    assert result["boxed_answer"] == "17" and result["answer"] == 17
    assert not result["used_finisher"] and result["answer_generated_tokens"] == 0


def test_numeric_extraction_ignores_invalid_or_nonfinite_later_boxes():
    assert extract_boxed_answer(r"\boxed{17} \boxed{bad} \boxed{nan}", numeric_only=True) == "17"


class FlatFormatting(WorkerFormatting):
    """Distinct segment IDs let us assert the exact fresh finisher input."""

    s1_finisher_suffix = "finish \\boxed{"

    def __init__(self, workers):
        super().__init__(workers)
        self.flat_text = []

    def encode(self, text, **kwargs):
        if text.startswith("ids(") and text.endswith(self.s1_finisher_suffix):
            self.flat_text.append(text)
            return [13, 14, 15, 16, 17]
        return {
            "test": [1, 2], "others": [3, 4], "own": [5, 6],
            self.incomplete_step + self.sep: [9, 10], self.s1_finisher_suffix: [11, 12],
        }.get(text, [7, 8])

    def decode(self, ids):
        if list(ids) == [19]:
            return r"\frac{1}{"
        if list(ids) == [19, 21]:
            return r"\frac{1}{2}}"
        return "ids(" + ",".join(map(str, ids)) + ")"


def test_flat_finisher_uses_fresh_serialized_response_and_stops_on_nested_box():
    fmt = FlatFormatting(2)
    (result,), session = run_cases(2, {
        "max_steps": 1, "answer_steps": 4, "flat_finisher": True, "symbolic_answers": True,
    }, formatting=fmt)
    # The problem, others-header, each still-open worker step and each
    # incomplete marker appear in order. The own-header does not appear.
    assert fmt.flat_text == ["ids(1,2,3,4,7,8,10,9,10,7,8,10,9,10)" + fmt.s1_finisher_suffix]
    prefill = next(call for call in session.prefill_calls if call["ids"] == [13, 14, 15, 16, 17])
    assert not prefill["context"]
    assert session.decode_calls[-1]["structure"] == [[prefill["block"]]]
    assert result["finisher_mode"] == "flat" and result["used_finisher"]
    assert result["answer_generated_tokens"] == 2 and result["finisher_stop_reason"] == "answer"
    assert result["answer"] == result["boxed_answer"] == r"\frac{1}{2}"


def test_cached_finisher_reuses_published_history_and_unfinished_worker_blocks(monkeypatch):
    fmt = FlatFormatting(2)
    forward = AsyncLLM.forward
    suffix_views = []

    async def observe_forward(self, input_ids=None, cache_view=None, **kwargs):
        if input_ids == [11] and isinstance(cache_view, AsyncContext):
            suffix_views.append([(block, list(block.token_ids)) for block in cache_view.cache_view])
        return await forward(self, input_ids, cache_view, **kwargs)

    monkeypatch.setattr(AsyncLLM, "forward", observe_forward)
    (result,), session = run_cases(2, {
        "max_steps": 3, "answer_steps": 2, "cached_finisher": True,
        "max_context_tokens": 26, "retain_details": False,
    }, formatting=fmt)
    assert result["completed_steps"] == 2 and result["used_finisher"]
    assert result["finisher_mode"] == "cached" and result["peak_context_tokens"] == 26
    assert result["answer_generated_tokens"] == 2 and not fmt.flat_text
    assert "trace" not in result and "transcript" not in result

    # The only multi-token forwards after prompt/header setup add two new
    # delimiters. The stub worker prefix and suffix each prefill a single ID
    # (decode path), so an extra prompt/history prefill cannot go unnoticed.
    assert [call["ids"] for call in session.prefill_calls] == [
        [1, 2], [3, 4], [5, 6], [9, 10], [9, 10],
    ]
    prompt, others, own, left_marker, right_marker = session.prefill_calls
    common, header = prompt["block"], others["block"]
    left, right = left_marker["block"], right_marker["block"]
    assert left_marker["context"] == [common, header]
    assert right_marker["context"] == [common, header, left]
    (snapshot,) = suffix_views
    assert [b for b, _ in snapshot[:-1]] == [common, header, left, right]
    assert own["block"] not in [b for b, _ in snapshot]
    assert [ids for _, ids in snapshot] == [
        [1, 2, 7, 8, 10, 12, 7, 8, 10, 12], [3, 4],
        [7, 8, 10, 9, 10], [7, 8, 10, 9, 10], [],
    ]
    # The live view persists for the suffix and both generation steps.
    tail = session.decode_calls[-3:]
    assert [call["input_ids"] for call in tail] == [[11], [12], [14]]
    assert all(call["structure"] == [[b for b, _ in snapshot]] for call in tail)


def test_cached_finisher_rejects_trace_collection_and_conflicting_modes():
    with pytest.raises(ValueError, match="inference-only"):
        run_cases(2, {"cached_finisher": True, "samples": {}})
    with pytest.raises(ValueError, match="inference-only"):
        run_cases(2, {"cached_finisher": True, "record_engine": True})
    with pytest.raises(ValueError, match="synchronized workers"):
        run_cases(2, {"capture_trace": True, "synchronized": False})
    with pytest.raises(ValueError, match="either"):
        run_cases(2, {"cached_finisher": True, "flat_finisher": True})


@pytest.mark.parametrize("both_stop", [False, True])
def test_cached_finisher_discards_only_terminal_eos_samples(monkeypatch, both_stop):
    class DistinctWorkers(FlatFormatting):
        def encode(self, text, **kwargs):
            if text.startswith("Worker0 1"):
                return [7, 17]  # Stub decodes 19 (EOS).
            if text.startswith("Worker1 1"):
                return [7, 21]  # Stub decodes 23.
            return super().encode(text, **kwargs)

    suffix_views = []
    forward = AsyncLLM.forward

    async def inspect(self, input_ids=None, cache_view=None, **kwargs):
        if input_ids == [11] and isinstance(cache_view, AsyncContext):
            suffix_views.append([list(block.token_ids) for block in cache_view.cache_view])
        return await forward(self, input_ids, cache_view, **kwargs)

    monkeypatch.setattr(AsyncLLM, "forward", inspect)
    (result,), session = run_cases(2, {
        "cached_finisher": True, "finish_if_missing": True, "eos_token_ids": [19, 23] if both_stop else [19],
        "capture_trace": True, "record_engine": True,
    }, formatting=DistinctWorkers(2))
    assert result["decode_rounds"] == 1 and result["reasoning_generated_tokens"] == 2
    assert result["eos_worker"] == "Worker0" and result["eos_token_id"] == 19
    assert result["stop_reason"] == "eos" and result["used_finisher"]
    peer = [7, 21] if both_stop else [7, 21, 23]
    assert result["current_ids"] == [[7, 17], peer]
    assert suffix_views == [[[1, 2], [3, 4], [7, 17, 9, 10], [*peer, 9, 10], []]]
    fed = [token for call in session.decode_calls for token in call["input_ids"]]
    assert 19 not in fed and fed.count(23) == (0 if both_stop else 1)
    assert result["trace"]["worker_token_ids"] == {"Worker0": [19], "Worker1": [23]}


class NativeFormatting(FlatFormatting):
    vocab = {"<think>": 24, "</think>": 25}
    cue = "\n\nThe final answer is \\boxed{"

    def apply_chat_template(self, messages, **kwargs):
        if isinstance(messages, str):
            return messages
        assert messages == [{"role": "user", "content": "test"}]
        assert kwargs == {"tokenize": False, "add_generation_prompt": True,
                          "enable_thinking": True, "reasoning_effort": "xhigh"}
        return "native"

    def encode(self, text, **kwargs):
        if text in ("test", "native"):
            return [1, 24, 2]
        if text == self.cue:
            return [11, 12]
        if text == "</think>" + self.cue:
            return [25, 11, 12]
        if text == self.sep:
            return [22]
        return super().encode(text, **kwargs)

    def is_end_of_step(self, ids):
        return ids[-1] in (8, 9)


class TerminalAnswerFormatting(NativeFormatting):
    """The first worker finishes; a later unfinished peer has a draft box."""

    def decode(self, ids):
        tokens = {
            24: "<think>", 25: "</think>", 27: r"\boxed{6r^2-4r-24}",
            29: "\n\n", 31: "<|im_end|>", 4: "I should check ",
            6: "the coefficients. ", 8: "\n\n", 10: r"\boxed{A=6,B=-4,C=-24}",
            12: "final ", 14: "42}",
        }
        return "".join(tokens.get(i, f"[{i}]") for i in ids)

    def is_end_of_step(self, ids):
        return ids[-1] in (8, 29)


@pytest.mark.parametrize("capture_trace", [False, True])
def test_terminal_worker_answer_survives_publication_and_ignores_peer_draft(monkeypatch, capture_trace):
    draws = iter([25, 4])

    async def draw(self, logits, **params):
        return torch.tensor(next(draws))

    monkeypatch.setattr(AsyncLLM, "sample", draw)
    (result,), session = run_cases(2, {
        "native_workers": True, "symbolic_answers": True, "max_steps": 4,
        "retain_details": capture_trace, "capture_trace": capture_trace,
    }, formatting=TerminalAnswerFormatting(2))
    assert result["eos_worker"] == "Worker0" and result["decode_rounds"] == 4
    assert result["completed_steps"] == 2
    assert result["boxed_answer"] == r"6r^2-4r-24"
    assert not result["used_finisher"] and result["answer_generated_tokens"] == 0
    selection = result["answer_selection"]
    assert selection == {
        "policy": TERMINAL_ANSWER_POLICY, "worker": "Worker0", "reason": "terminal_final_box",
        "answer_token_start": 0, "answer_token_end": 3, "selected_answer": r"6r^2-4r-24",
    }
    assert [call["ids"] for call in session.prefill_calls] == [[1, 24, 2]]
    if capture_trace:
        assert result["history_ids"] == [25, 27, 29, 4, 6, 8]
        assert result["current_ids"] == [[], [10]]
        assert result["trace"]["answer_selection"] == selection
        assert result["trace"]["worker_token_ids"] == {"Worker0": [25, 27, 29, 31], "Worker1": [4, 6, 8, 10]}


def test_budget_stop_with_existing_worker_box_still_uses_cached_finisher(monkeypatch):
    draws = iter([25, 4])

    async def draw(self, logits, **params):
        return torch.tensor(next(draws))

    monkeypatch.setattr(AsyncLLM, "sample", draw)
    fmt = TerminalAnswerFormatting(2)
    (result,), session = run_cases(2, {
        "native_workers": True, "symbolic_answers": True, "max_steps": 3,
    }, formatting=fmt)
    assert result["stop_reason"] == "max_steps" and result["eos_worker"] is None
    assert result["used_finisher"] and result["finisher_mode"] == "cached"
    assert result["boxed_answer"] == "42" and result["answer_generated_tokens"] == 1
    assert result["answer_selection"]["reason"] == "no_eos" and not fmt.flat_text
    assert result["completed_steps"] == 0 and result["history_ids"] == []
    assert result["current_ids"] == [[25, 27, 29], [4, 6, 8]]
    assert all(27 not in call["ids"] for call in session.prefill_calls)


@pytest.mark.parametrize("stream, eos_worker, reason", [
    ([25, 4, 31], "Worker0", "missing_final_box"),
    ([25, 27], "Worker0", "invalid_terminal_stream"),
    ([25, 27, 31], None, "no_eos"),
])
def test_terminal_selector_requires_own_complete_box_and_eos(stream, eos_worker, reason):
    answer, selection = select_terminal_worker_answer(
        TerminalAnswerFormatting(2), {"Worker0": stream, "Worker1": [25, 27, 31]},
        eos_worker=eos_worker, eos_token_ids=[31],
    )
    assert answer is None and selection["reason"] == reason
    assert selection["selected_answer"] is None


@pytest.mark.parametrize("stream", [
    [27, 31], [25, 27, 24, 31], [27, 25, 4, 31],
])
def test_terminal_selector_does_not_infer_shared_thinking_phase_from_own_stream(stream):
    answer, selection = select_terminal_worker_answer(
        TerminalAnswerFormatting(2), {"Worker0": stream, "Worker1": [25, 10]},
        eos_worker="Worker0", eos_token_ids=[31],
    )
    assert answer == r"6r^2-4r-24"
    assert selection["policy"] == "eos_worker_last_box_v1"
    assert selection["answer_token_start"] == 0 and selection["answer_token_end"] == len(stream) - 1


def test_terminal_selector_obeys_recorded_eos_tie_break_without_voting():
    answer, selection = select_terminal_worker_answer(
        TerminalAnswerFormatting(2), {"Worker0": [25, 10, 31], "Worker1": [25, 27, 31]},
        eos_worker="Worker0", eos_token_ids=[31],
    )
    assert answer == "A=6,B=-4,C=-24" and selection["worker"] == "Worker0"


def test_native_workers_share_live_peer_kv_and_publish_without_replaying(monkeypatch):
    fmt = NativeFormatting(2)
    draws = iter([4, 5])
    seed_logits, snapshots = [], []
    decode = StubSession.decode_step

    async def draw(self, logits, **params):
        assert int(logits.argmax()) == 4 and params["top_k"] == 1
        seed_logits.append(logits)
        return torch.tensor(next(draws))

    def record_decode(self, group, inputs):
        snapshots.append((inputs.flatten().tolist(), [
            [list(block.token_ids) for block in view] for view in group.cache_structure
        ]))
        return decode(self, group, inputs)

    monkeypatch.setattr(AsyncLLM, "sample", draw)
    monkeypatch.setattr(StubSession, "decode_step", record_decode)
    (result,), session = run_cases(2, {
        "native_workers": True, "sampling": SamplingParams(top_k=1), "reminder_interval": 1,
    }, formatting=fmt)
    assert len(seed_logits) == 2 and seed_logits[0] is seed_logits[1]
    assert result["generated_tokens"] == {"Worker0": 4, "Worker1": 4}
    assert result["inserted_worker_tokens"] == 0 and result["reminder_events"] == []
    assert result["first_publication_round"] == 4 and result["completed_steps"] == 2
    assert result["publication_policy"] == "encoded_separator_v1"
    prompt = [1, 24, 2]
    assert [call["ids"] for call in session.prefill_calls] == [prompt, [25, 11]]
    assert snapshots[:4] == [
        ([4, 5], [[prompt, [], []], [prompt, [], []]]),
        ([6, 7], [[prompt, [5], [4]], [prompt, [4], [5]]]),
        ([8, 9], [[prompt, [5, 7], [4, 6]], [prompt, [4, 6], [5, 7]]]),
        ([10, 11], [[[1, 24, 2, 4, 6, 8, 5, 7, 9], [], []], [[1, 24, 2, 4, 6, 8, 5, 7, 9], [], []]]),
    ]
    assert result["history_ids"] == [4, 6, 8, 5, 7, 9]
    assert result["current_ids"] == [[10], [11]]
    assert result["finisher_mode"] == "cached" and not fmt.flat_text


def test_native_seeded_eos_is_counted_and_omitted_from_cached_answer(monkeypatch):
    async def draw(self, logits, **params):
        return torch.tensor(next(first))

    first = iter([31, 25])  # Terminal worker; peer enters the answer phase.
    monkeypatch.setattr(AsyncLLM, "sample", draw)
    (result,), session = run_cases(2, {"native_workers": True}, formatting=NativeFormatting(2))
    assert result["stop_reason"] == "eos" and result["eos_worker"] == "Worker0"
    assert result["decode_rounds"] == 1 and result["reasoning_generated_tokens"] == 2
    assert not result["completed_steps"] and result["current_ids"] == [[], [25]]
    assert [call["ids"] for call in session.prefill_calls] == [[1, 24, 2]]
    # No new closing tag when the peer already supplied it. Only its pending
    # token, the two separators, new cue and then answer tokens are encoded.
    assert [call["input_ids"] for call in session.decode_calls] == [[25], [22], [22], [11], [12], [14]]


@pytest.mark.parametrize("budget,eos,rounds,history,current,publications,inputs", [
    (4, [31], 4, [4, 6, 5, 7, 9], [[8, 10], [11]], [(3, "Worker0", 2), (4, "Worker1", 3)],
     [[4, 5], [6, 7], [8, 9], [10, 11]]),
    (4, [8, 31], 3, [4, 6], [[], [5, 7, 9]], [(3, "Worker0", 2)], [[4, 5], [6, 7], [9]]),
    (2, [31], 2, [], [[4, 6], [5, 7]], [], [[4, 5], [6, 7]]),
])
def test_native_delayed_publication_preserves_staggered_eos_and_budget_tokens(
    monkeypatch, budget, eos, rounds, history, current, publications, inputs,
):
    class StaggeredFormatting(NativeFormatting):
        def is_end_of_step(self, ids):
            return ids[-1] in (6, 9)

    draws = iter([4, 5])

    async def draw(self, logits, **params):
        return torch.tensor(next(draws))

    monkeypatch.setattr(AsyncLLM, "sample", draw)
    (result,), session = run_cases(2, {
        "native_workers": True, "max_steps": budget, "eos_token_ids": eos,
        "capture_trace": True, "record_engine": True,
    }, formatting=StaggeredFormatting(2))
    assert result["decode_rounds"] == rounds and result["reasoning_generated_tokens"] == 2 * rounds
    assert result["history_ids"] == history and result["current_ids"] == current
    trace = result["trace"]
    assert trace["publication_policy"] == "encoded_separator_v1"
    assert [(p["round"], p["worker"], p["committed_token_count"]) for p in trace["publications"]] == publications
    assert [call["input_ids"] for call in session.decode_calls[:len(inputs)]] == inputs
    # Only the original prompt and new answer cue are prefills. Separator,
    # next-paragraph token and an unfinished peer each enter decode once.
    assert [call["ids"] for call in session.prefill_calls] == [[1, 24, 2], [25, 11]]
    assert [call["input_ids"] for call in session.decode_calls[len(inputs):]] == [[22], [22], [12], [14]]
    assert all(len(req.input_ids) == 1 for batch in session.recorded_trace.batches if batch.kind == "decode"
               for req in batch.requests)
    if result["eos_token_id"] is not None:
        assert trace["worker_token_ids"]["Worker0"][-1] == 8
        assert all(8 not in req.input_ids for batch in session.recorded_trace.batches for req in batch.requests)


def test_legacy_answer_cue_uses_last_tag_in_finisher_view():
    class LastTagInPeer(NativeFormatting):
        def encode(self, text, **kwargs):
            if text.startswith("Worker1 1"):
                return [7, 23]  # Peer samples </think>.
            return super().encode(text, **kwargs)

    (result,), session = run_cases(2, {
        "max_steps": 1, "cached_finisher": True, "native_answer_cue": True,
    }, formatting=LastTagInPeer(2))
    assert result["used_finisher"] and result["current_ids"][1][-1] == 25
    assert [call["ids"] for call in session.prefill_calls] == [
        [1, 24, 2], [3, 4], [5, 6], [9, 10], [9, 10],
    ]
    assert [call["input_ids"] for call in session.decode_calls[-3:]] == [[11], [12], [14]]


@pytest.mark.parametrize("native_workers", [False, True])
def test_inference_trace_reconstructs_worker_history_without_extra_forwards(monkeypatch, native_workers):
    class DistinctReminder(NativeFormatting):
        def encode(self, text, **kwargs):
            if self.s1_collab_message in text:
                return [15, 16]
            return super().encode(text, **kwargs)

        def is_end_of_step(self, ids):
            return ids[-1] in (8, 9) if native_workers else len(ids) == 4

    draws = iter([4, 5, 4, 5])

    async def draw(self, logits, **params):
        return torch.tensor(next(draws))

    monkeypatch.setattr(AsyncLLM, "sample", draw)
    rounds = 4 if native_workers else 3
    options = {"native_workers": native_workers, "cached_finisher": True, "max_steps": rounds,
               "eos_token_ids": [16, 31], "answer_steps": 4, "reminder_interval": 2}
    (baseline, captured), session = run_cases(
        2, options.copy(), {**options, "capture_trace": True, "retain_details": False, "record_engine": True},
        formatting=DistinctReminder(2),
    )
    assert "trace" not in baseline and "transcript" not in captured
    trace = captured["trace"]
    assert trace["workers"] == ["Worker0", "Worker1"]
    assert trace["prompt_ids"] == baseline["prompt_ids"]
    assert trace["decode_rounds"] == baseline["decode_rounds"] == rounds
    assert trace["boxed_answer"] == baseline["boxed_answer"]
    assert [len(ids) for ids in trace["worker_token_ids"].values()] == [rounds, rounds]
    assert len(trace["publications"]) == baseline["completed_steps"] == 2
    assert [event["worker"] for event in trace["publications"]] == trace["workers"]
    assert all(event["round"] == (4 if native_workers else 2) for event in trace["publications"])
    assert [event["next_prefix_ids"] for event in trace["publications"]] == (
        [[], []] if native_workers else [[7, 8], [15, 16]]
    )

    # Reconstruct the published and unfinished token streams from the trace.
    history = []
    active = {w: ids.copy() for w, ids in trace["initial_worker_prefix_ids"].items()}
    for round_id in range(1, trace["decode_rounds"] + 1):
        for worker in trace["workers"]:
            token = trace["worker_token_ids"][worker][round_id - 1]
            if token not in trace["eos_token_ids"]:
                active[worker].append(token)
        for event in trace["publications"]:
            if event["round"] == round_id:
                worker = event["worker"]
                length = event["committed_token_count"]
                assert length == len(active[worker]) - int(native_workers)
                history.extend(active[worker][:length])
                active[worker] = active[worker][length:] + event["next_prefix_ids"]
    assert history == baseline["history_ids"]
    assert list(active.values()) == baseline["current_ids"]
    finisher = trace["finisher"]
    assert finisher["sampled_token_ids"] == [14, 16]
    assert finisher["used"] and finisher["mode"] == "cached" and finisher["stop_reason"] == "eos"
    assert finisher["eos_token_id"] == 16
    assert finisher["worker_marker_ids"] == ([22] if native_workers else [9, 10])
    assert finisher["suffix_ids"] == ([25, 11, 12] if native_workers else [11, 12])
    assert captured["total_generated_tokens"] == baseline["total_generated_tokens"]

    # The wrapper records the same real batches, including cache appends,
    # without any replay or additional engine requests.
    for kind, calls, key in (("prefill", session.prefill_calls, "ids"), ("decode", session.decode_calls, "input_ids")):
        actual = [call[key] for call in calls]
        middle = len(actual) // 2
        assert actual[:middle] == actual[middle:]
        batches = [batch for batch in session.recorded_trace.batches if batch.kind == kind]
        executed = ([list(req.input_ids) for batch in batches for req in batch.requests] if kind == "prefill"
                    else [[req.input_ids[0] for req in batch.requests] for batch in batches])
        assert executed == actual[middle:]
    assert any(update.refs for update in session.recorded_trace.updates)
    roles = trace["block_ids"]
    assert roles["history"] == session.recorded_trace.batches[0].requests[0].write_to
    assert roles["answer"] == session.recorded_trace.batches[-1].requests[0].write_to
    assert (roles["own_header"] is None) == native_workers
    assert (roles["others_header"] is None) == native_workers
    assert set(roles["workers"].values()) == {
        update.refs[1][0] for update in session.recorded_trace.updates if update.refs
    }


def test_complete_symbolic_worker_answer_skips_finisher():
    class SymbolicFormatting(FlatFormatting):
        def decode(self, ids):
            return r"candidate \boxed{\sqrt{2}}" if 10 in ids else super().decode(ids)

    fmt = SymbolicFormatting(2)
    (result,), _ = run_cases(2, {
        "max_steps": 2, "flat_finisher": True, "symbolic_answers": True,
    }, formatting=fmt)
    assert result["decode_rounds"] == 2 and result["stop_reason"] == "max_steps"
    assert not result["used_finisher"] and result["finisher_stop_reason"] == "existing_answer"
    assert result["boxed_answer"] == r"\sqrt{2}"
    assert not fmt.flat_text


def test_flat_retokenization_checks_exact_context_length_before_prefill():
    class ExpandedFormatting(FlatFormatting):
        def encode(self, text, **kwargs):
            if text.startswith("ids(") and text.endswith(self.s1_finisher_suffix):
                return [17] * 60
            return super().encode(text, **kwargs)

    (result,), session = run_cases(2, {
        "max_steps": 1, "max_context_tokens": 32, "flat_finisher": True,
    }, formatting=ExpandedFormatting(2))
    assert result["reasoning_stop_reason"] == "max_steps"
    assert result["stop_reason"] == result["finisher_stop_reason"] == "context_limit"
    assert not result["used_finisher"] and result["answer_generated_tokens"] == 0
    assert result["peak_context_tokens"] <= 32
    assert all(len(call["ids"]) < 60 for call in session.prefill_calls)


def test_flat_finisher_reserves_markers_and_records_its_own_eos():
    # At this cap the four incomplete markers leave no room for another
    # worker round. The fake tokenizer's final flattened prefix fits and
    # its greedy next token is EOS 19, distinct from a reasoning EOS.
    (result,), _ = run_cases(4, {
        "max_context_tokens": 24, "flat_finisher": True, "eos_token_ids": [19, 31],
    }, formatting=FlatFormatting(4))
    assert result["decode_rounds"] == 0 and result["reasoning_stop_reason"] == "context_limit"
    assert result["eos_token_id"] is None and result["eos_worker"] is None
    assert result["answer_generated_tokens"] == 1 and result["finisher_eos_token_id"] == 19
    assert result["finisher_stop_reason"] == "eos"


@pytest.mark.parametrize("causal_headers", [False, True])
def test_header_control_prefills_with_expected_parents(causal_headers):
    _, session = run_cases(1, {"max_steps": 1, "causal_headers": causal_headers}, formatting=FlatFormatting(1))
    prompt, others, own = [next(call for call in session.prefill_calls if call["ids"] == ids)
                           for ids in ([1, 2], [3, 4], [5, 6])]
    if causal_headers:
        assert others["context"] == [prompt["block"]]
        assert own["context"] == [prompt["block"], others["block"]]
    else:
        assert not others["context"] and not own["context"]


@pytest.mark.parametrize("text, expected", [
    (r"first \boxed{12}, then \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"\boxed{\left\{x\right.}", r"\left\{x\right."),
    (r"first \boxed{12}, then \boxed{incomplete", "12"),
    (r"\boxed{}", None),
    ("no answer", None),
])
def test_nested_boxed_answer_extraction(text, expected):
    assert extract_boxed_answer(text) == expected


def test_single_worker_prompt_has_no_phantom_collaborator():
    class Tokenizer:
        vocab = {"SEP": 1}

        def encode(self, text, **kwargs):
            return [1]

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    prompt = MathFormatting(Tokenizer(), workers=["Alice"]).apply_chat_template("Problem")
    assert "no other assistants" in prompt
    assert "Bob" not in prompt and " and Alice" not in prompt
