"""
Mixed prefill/decode batches, in four tiers.

A mixed batch is an *extend* batch whose trailing reqs happen to have ``extend_len == 1``.
Nothing downstream of the scheduler branches on "mixed" -- the attention backends, the LM
head gather and the position/mapping builders all just see a varlen extend batch -- so the
only thing that has to hold is the layout:

    reqs = [ extend reqs ... | decode reqs ... ]
             ^-- num_prefill ^^-- num_decode --^

``Scheduler._process_last_data`` relies on exactly that to decide which reqs get their
prefix inserted into the radix cache (``i < batch.num_prefill``); if the decode segment
were not a contiguous suffix, every decode step would re-insert its prefix.

The tiers, cheapest first:

1. layout + the hybrid guard -- pure CPU, no engine.
2. the budget arithmetic in ``Scheduler._schedule_next_batch``, against stub managers.
   Decode rows are charged against the same ``max_extend_tokens`` budget as extend rows,
   because that budget is what sizes the pynccl forward buffer.
3. the same arithmetic against the *real* ``PrefillManager``/``DecodeManager`` plus the
   real page allocator and radix cache -- CUDA, but no model weights.
4. end-to-end equivalence: the same greedy prompts must produce the same token ids with
   mixed batching on and off.  Needs CUDA and a model; runs each policy in its own
   subprocess because ``Engine.__init__`` allows only one engine per process.  See
   ``E2E_PROMPTS`` for why the prompts are what they are -- the two policies use
   different attention kernels, so this is argmax-stable, not bit-exact.

Run::

    pytest tests/core/test_mixed_batch.py -v
    MINISGL_E2E_MODEL=Qwen/Qwen3-0.6B pytest tests/core/test_mixed_batch.py -v  # + tier 4
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import List

import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.scheduler.utils import mix_batches, resolve_mixed_batch

E2E_MODEL_PATH = os.environ.get("MINISGL_E2E_MODEL", "")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA (pinned host staging buffers)"
)
requires_e2e = pytest.mark.skipif(
    not E2E_MODEL_PATH or not torch.cuda.is_available(),
    reason="Set MINISGL_E2E_MODEL to an HF model path and ensure CUDA is available",
)


def _make_req(uid: int, *, input_len: int, cached_len: int, cls=Req) -> Req:
    return cls(
        input_ids=torch.zeros(input_len, dtype=torch.int32),
        table_idx=uid,
        cached_len=cached_len,
        output_len=8,
        uid=uid,
        sampling_params=None,  # type: ignore[arg-type]
        cache_handle=None,  # type: ignore[arg-type]
    )


def _extend_reqs(n: int, *, offset: int = 0) -> List[Req]:
    # cached_len > 0 keeps these on the same code path as a partial prefix-cache hit
    return [_make_req(offset + i, input_len=64, cached_len=16) for i in range(n)]


def _decode_reqs(n: int, *, offset: int = 0) -> List[Req]:
    # a decode req is an extend req with exactly one new token
    return [_make_req(offset + i, input_len=32, cached_len=31) for i in range(n)]


def _extend_req_of_rows(rows: int, *, uid: int = 0) -> Req:
    """An extend req contributing exactly ``rows`` query rows."""
    return _make_req(uid, input_len=rows + 1, cached_len=1)


def _prefill_rows(batch: Batch) -> int:
    return sum(req.extend_len for req in batch.reqs[: batch.num_prefill])


def _cached_indices(batch: Batch) -> List[int]:
    """The reqs `_process_last_data` would insert into the prefix cache."""
    return [i for i in range(batch.size) if i < batch.num_prefill]


class TestBatchSegments:
    def test_decode_batch_is_all_decode(self):
        batch = Batch(reqs=_decode_reqs(4), phase="decode")
        assert batch.num_decode == 4
        assert batch.num_prefill == 0
        assert not batch.is_mixed

    def test_prefill_batch_is_all_prefill(self):
        batch = Batch(reqs=_extend_reqs(3), phase="prefill")
        assert batch.num_decode == 0
        assert batch.num_prefill == 3
        assert not batch.is_mixed

    def test_decode_phase_overrides_stale_num_decode(self):
        # __post_init__ is what keeps every existing construction site correct
        batch = Batch(reqs=_decode_reqs(2), phase="decode", num_decode=0)
        assert batch.num_decode == 2


class TestMixBatches:
    def test_decode_segment_is_a_contiguous_suffix(self):
        prefill = Batch(reqs=_extend_reqs(2), phase="prefill")
        decode = Batch(reqs=_decode_reqs(3, offset=2), phase="decode")
        mixed = mix_batches(prefill, decode)
        assert mixed is not None

        assert mixed.size == 5
        assert mixed.num_prefill == 2 and mixed.num_decode == 3
        assert mixed.is_mixed
        assert mixed.reqs[: mixed.num_prefill] == prefill.reqs
        assert mixed.reqs[mixed.num_prefill :] == decode.reqs
        # every req in the decode segment contributes exactly one query row
        assert all(req.extend_len == 1 for req in mixed.reqs[mixed.num_prefill :])

    def test_mixed_batch_runs_as_an_extend_batch(self):
        mixed = mix_batches(
            Batch(reqs=_extend_reqs(1), phase="prefill"),
            Batch(reqs=_decode_reqs(2, offset=1), phase="decode"),
        )
        assert mixed is not None
        # is_prefill routes it to the extend kernels and the LM-head last-token gather
        assert mixed.is_prefill and not mixed.is_decode
        assert max(req.extend_len for req in mixed.reqs) > 1

    def test_passthrough_when_one_side_is_empty(self):
        prefill = Batch(reqs=_extend_reqs(2), phase="prefill")
        decode = Batch(reqs=_decode_reqs(2), phase="decode")
        assert mix_batches(prefill, None) is prefill
        assert mix_batches(None, decode) is decode
        assert mix_batches(None, None) is None


class TestPrefixCachingRule:
    """`i < batch.num_prefill` must select exactly the extend reqs, in all three shapes."""

    def test_pure_prefill_caches_every_req(self):
        batch = Batch(reqs=_extend_reqs(3), phase="prefill")
        assert _cached_indices(batch) == [0, 1, 2]

    def test_pure_decode_caches_nothing(self):
        batch = Batch(reqs=_decode_reqs(3), phase="decode")
        assert _cached_indices(batch) == []

    def test_mixed_caches_only_the_extend_prefix(self):
        mixed = mix_batches(
            Batch(reqs=_extend_reqs(2), phase="prefill"),
            Batch(reqs=_decode_reqs(3, offset=2), phase="decode"),
        )
        assert mixed is not None
        assert _cached_indices(mixed) == [0, 1]

    def test_chunked_reqs_stay_in_the_prefill_segment(self):
        # a ChunkedReq is never sampled and never enters the decode manager, so it can
        # only ever appear in the extend segment
        chunked = _make_req(0, input_len=64, cached_len=16, cls=ChunkedReq)
        assert not chunked.can_decode
        mixed = mix_batches(
            Batch(reqs=[chunked], phase="prefill"),
            Batch(reqs=_decode_reqs(2, offset=1), phase="decode"),
        )
        assert mixed is not None
        assert mixed.reqs[0] is chunked
        assert _cached_indices(mixed) == [0]


# =============================================================================
# The hybrid guard (CPU only -- a pure predicate)
# =============================================================================


class TestHybridGuard:
    """A hybrid model must never see a mixed batch: it would be wrong *and* silent."""

    def test_hybrid_disables_mixed_batch(self):
        assert resolve_mixed_batch(enabled=True, is_hybrid=True) is False

    def test_non_hybrid_keeps_it_enabled(self):
        assert resolve_mixed_batch(enabled=True, is_hybrid=False) is True

    def test_the_knob_still_wins_when_off(self):
        assert resolve_mixed_batch(enabled=False, is_hybrid=False) is False
        assert resolve_mixed_batch(enabled=False, is_hybrid=True) is False


# =============================================================================
# Budget arithmetic (CPU only -- stub managers, no engine)
# =============================================================================


class _BudgetFillingPrefill:
    """Stands in for ``PrefillManager``: records every budget it is handed, and returns
    an extend batch that uses the whole of it -- the worst case a correct manager can
    produce, so the row-count invariants below are tight."""

    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty
        self.budgets: List[int] = []

    def schedule_next_batch(self, budget: int) -> Batch | None:
        self.budgets.append(budget)
        if self.empty or budget <= 0:
            return None
        return Batch(reqs=[_extend_req_of_rows(budget)], phase="prefill")


class _FixedDecode:
    """Stands in for ``DecodeManager``, which hands out *all* running reqs every step."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.calls = 0

    def schedule_next_batch(self) -> Batch | None:
        self.calls += 1
        if self.size == 0:
            return None
        return Batch(reqs=_decode_reqs(self.size, offset=1000), phase="decode")


def _schedule(*, budget: int, decode: int, mixed: bool = True, no_prefill: bool = False):
    """Run the real ``Scheduler._schedule_next_batch`` against stub managers.

    ``_prepare_batch`` is the identity so the batch comes back unprepared: everything it
    would do (page allocation, attn metadata) needs a GPU and none of it is under test.
    """
    from minisgl.scheduler.scheduler import Scheduler

    stub = SimpleNamespace(
        enable_mixed_batch=mixed,
        prefill_budget=budget,
        prefill_manager=_BudgetFillingPrefill(empty=no_prefill),
        decode_manager=_FixedDecode(decode),
        _prepare_batch=lambda batch: batch,
    )
    return Scheduler._schedule_next_batch(stub), stub


class TestMixedBudget:
    def test_decode_rows_are_charged_against_the_prefill_budget(self):
        batch, stub = _schedule(budget=100, decode=30)
        assert stub.prefill_manager.budgets == [70]
        assert batch is not None
        assert (batch.num_prefill, batch.num_decode) == (1, 30)
        assert _prefill_rows(batch) + batch.num_decode == 100

    def test_full_budget_goes_to_prefill_when_nothing_is_decoding(self):
        batch, stub = _schedule(budget=100, decode=0)
        assert stub.prefill_manager.budgets == [100]
        assert batch is not None and batch.num_decode == 0
        assert _prefill_rows(batch) == 100

    @pytest.mark.parametrize("decode", [100, 101, 250])
    def test_prefill_is_skipped_once_decode_fills_the_budget(self, decode):
        """No zero/negative budget may reach the prefill manager -- ``try_add_one`` would
        treat it as "no room" anyway, but the guard is what keeps that an invariant."""
        batch, stub = _schedule(budget=100, decode=decode)
        assert stub.prefill_manager.budgets == []
        assert batch is not None
        assert batch.num_prefill == 0 and batch.num_decode == decode

    @pytest.mark.parametrize("decode", [0, 1, 50, 99, 100, 150])
    def test_a_mixed_forward_never_overruns_the_row_budget(self, decode):
        """The invariant that sizes the pynccl forward buffer.

        Decode rows themselves are bounded by ``max_running_req``, not by this budget (a
        pure decode batch has always been able to exceed it), so what mixed batching must
        guarantee is that it never *adds* extend rows on top of them.
        """
        budget = 100
        batch, _ = _schedule(budget=budget, decode=decode)
        assert batch is not None
        if batch.num_decode < budget:
            assert _prefill_rows(batch) + batch.num_decode <= budget
        else:
            assert _prefill_rows(batch) == 0

    def test_idle_scheduler_produces_nothing(self):
        for mixed in (True, False):
            batch, _ = _schedule(budget=100, decode=0, mixed=mixed, no_prefill=True)
            assert batch is None


class TestAlternatingBudget:
    """With mixed batching off, the two phases take turns and never share a budget."""

    def test_prefill_wins_and_gets_the_whole_budget(self):
        batch, stub = _schedule(budget=100, decode=30, mixed=False)
        assert stub.prefill_manager.budgets == [100]
        assert batch is not None
        assert batch.num_decode == 0 and _prefill_rows(batch) == 100

    def test_decode_runs_only_when_no_prefill_batch_forms(self):
        batch, stub = _schedule(budget=100, decode=30, mixed=False, no_prefill=True)
        assert batch is not None
        assert (batch.num_prefill, batch.num_decode) == (0, 30)
        assert stub.decode_manager.calls == 1


# =============================================================================
# Real managers, no model (CUDA -- the page allocator and radix cache are real)
# =============================================================================


PAGE_SIZE = 1
NUM_PAGES = 8192
NUM_SLOTS = 16
MAX_SEQ = 512


@pytest.fixture(scope="module")
def global_ctx():
    """``RadixPrefixCache`` reads ``page_size`` off the global context, which normally only
    ``Engine`` installs.  ``set_global_ctx`` is single-shot, hence module scope."""
    from minisgl import core

    if core._GLOBAL_CTX is None:
        core.set_global_ctx(core.Context(page_size=PAGE_SIZE))
    return core.get_global_ctx()


@pytest.fixture
def managers(global_ctx):
    """The scheduler's manager trio over a real page pool, without any model weights."""
    from minisgl.kvcache import PageAllocator
    from minisgl.scheduler.cache import CacheManager
    from minisgl.scheduler.decode import DecodeManager
    from minisgl.scheduler.prefill import PrefillManager
    from minisgl.scheduler.table import TableManager

    device = torch.device("cuda")
    page_table = torch.zeros((NUM_SLOTS + 1, MAX_SEQ), dtype=torch.int32, device=device)
    table_manager = TableManager(NUM_SLOTS, page_table)
    cache_manager = CacheManager(PageAllocator(NUM_PAGES, PAGE_SIZE, device), page_table, "radix")
    decode_manager = DecodeManager(PAGE_SIZE)
    return SimpleNamespace(
        prefill_manager=PrefillManager(cache_manager, table_manager, decode_manager),
        decode_manager=decode_manager,
        cache_manager=cache_manager,
        table_manager=table_manager,
    )


def _submit(prefill_manager, *, count: int, input_len: int, max_tokens: int = 8) -> None:
    from minisgl.message import UserMsg

    for uid in range(count):
        prefill_manager.add_one_req(
            UserMsg(
                uid=uid,
                input_ids=torch.arange(
                    uid * input_len, (uid + 1) * input_len, dtype=torch.int32
                ),
                sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
            )
        )


@requires_cuda
class TestRealPrefillBudget:
    def test_rows_stop_at_the_budget_and_the_tail_is_chunked(self, managers):
        _submit(managers.prefill_manager, count=4, input_len=100)
        batch = managers.prefill_manager.schedule_next_batch(250)

        assert batch is not None
        assert sum(req.extend_len for req in batch.reqs) == 250
        # taken in arrival order, and only the last one is split
        assert [req.uid for req in batch.reqs] == [0, 1, 2]
        assert [isinstance(req, ChunkedReq) for req in batch.reqs] == [False, False, True]
        # the partial req goes back to the *head*, so the order never changes
        assert [p.uid for p in managers.prefill_manager.pending_list] == [2, 3]

    @pytest.mark.parametrize("budget", [0, -1, -300])
    def test_a_non_positive_budget_admits_nothing(self, managers, budget):
        _submit(managers.prefill_manager, count=2, input_len=100)
        assert managers.prefill_manager.schedule_next_batch(budget) is None
        assert len(managers.prefill_manager.pending_list) == 2


@requires_cuda
class TestRealMixedBudget:
    """Tier 2's invariant again, with nothing stubbed but ``_prepare_batch``."""

    @pytest.mark.parametrize("num_decode", [0, 1, 8, 16])
    def test_total_rows_never_exceed_max_extend_tokens(self, managers, num_decode):
        from minisgl.scheduler.scheduler import Scheduler

        budget = 64
        # far more pending prefill work than the budget can hold, so the extend segment
        # is only ever limited by the budget the scheduler grants it
        _submit(managers.prefill_manager, count=8, input_len=100)
        managers.decode_manager.running_reqs = set(_decode_reqs(num_decode, offset=1000))

        stub = SimpleNamespace(
            enable_mixed_batch=True,
            prefill_budget=budget,
            prefill_manager=managers.prefill_manager,
            decode_manager=managers.decode_manager,
            _prepare_batch=lambda batch: batch,
        )
        batch = Scheduler._schedule_next_batch(stub)

        assert batch is not None
        assert batch.num_decode == num_decode
        assert sum(req.extend_len for req in batch.reqs) == budget
        assert _prefill_rows(batch) == budget - num_decode


# =============================================================================
# End-to-end equivalence (CUDA + model weights)
# =============================================================================


E2E_MAX_TOKENS = 12
# Small enough that the offline feeder still has prompts to prefill while earlier ones are
# already decoding -- which is the only way a mixed batch ever forms.
E2E_EXTEND_BUDGET = 64
_RESULT_SENTINEL = "__MIXED_BATCH_RESULT__ "

# The two policies route decode rows through different attention kernels -- a mixed batch
# is an extend batch (fa), a pure decode batch replays a CUDA graph (fi) -- so the logits
# are close but not bit-identical, and only a *confident* argmax survives the difference.
# Hence prompts with one obvious continuation each: on near-tied logits a single flip
# cascades and the sequences diverge completely without anything being wrong.
E2E_PROMPTS = [
    "Count up: one, two, three, four, five, six, seven, eight, nine,",
    "The capital of France is Paris. The capital of Italy is Rome. The capital of Japan is",
    "2 + 2 = 4. 3 + 3 = 6. 4 + 4 = 8. 5 + 5 =",
    "a b c d e f g h i j k l m n o p q r s",
    "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday,",
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy",
    "red red red red red red red red red red red red red red red",
    "10, 20, 30, 40, 50, 60, 70, 80,",
]


def _run_policy(mixed: bool) -> dict:
    """Generate in a subprocess: ``Engine.__init__`` refuses a second engine per process,
    and a fresh process also means a cold prefix cache, so neither policy gets to read KV
    the other one wrote."""
    proc = subprocess.run(
        [sys.executable, __file__, "mixed" if mixed else "alternating"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    # the engine logs to stdout too, so pick the payload out by its sentinel
    payloads = [ln for ln in proc.stdout.splitlines() if ln.startswith(_RESULT_SENTINEL)]
    assert len(payloads) == 1, f"no result line in:\n{proc.stdout[-4000:]}"
    return json.loads(payloads[0][len(_RESULT_SENTINEL) :])


@requires_e2e
class TestMixedEquivalence:
    """Mixing a decode row into an extend batch must not change what that row samples."""

    def test_same_tokens_with_and_without_mixed_batching(self):
        mixed = _run_policy(mixed=True)
        alternating = _run_policy(mixed=False)

        # otherwise the comparison is vacuous: no mixed batch was ever built
        assert mixed["mixed_batches"] > 0, "no mixed batch formed; tune E2E_EXTEND_BUDGET"
        assert alternating["mixed_batches"] == 0

        assert len(mixed["tokens"]) == len(E2E_PROMPTS)
        for i, (got, want) in enumerate(zip(mixed["tokens"], alternating["tokens"])):
            assert len(want) == E2E_MAX_TOKENS
            assert got == want, f"prompt {i!r} diverged: {got} vs {want}"


def _subprocess_main(policy: str) -> None:
    """Entry point for ``_run_policy``; not collected by pytest."""
    from minisgl.llm import LLM

    class CountingLLM(LLM):
        mixed_batches = 0
        total_batches = 0

        def _schedule_next_batch(self):
            forward_input = super()._schedule_next_batch()
            if forward_input is not None:
                self.total_batches += 1
                self.mixed_batches += forward_input.batch.is_mixed
            return forward_input

    llm = CountingLLM(
        E2E_MODEL_PATH,
        enable_mixed_batch=(policy == "mixed"),
        max_extend_tokens=E2E_EXTEND_BUDGET,
        max_running_req=8,
        cuda_graph_bs=[2, 4, 8],
        cuda_graph_max_bs=8,
        memory_ratio=0.3,
        max_seq_len_override=1024,
    )
    results = llm.generate(
        E2E_PROMPTS,
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=E2E_MAX_TOKENS),
    )
    payload = {
        "mixed_batches": llm.mixed_batches,
        "total_batches": llm.total_batches,
        "tokens": [r["token_ids"] for r in results],
    }
    print(_RESULT_SENTINEL + json.dumps(payload), flush=True)


if __name__ == "__main__":
    _subprocess_main(sys.argv[1])
