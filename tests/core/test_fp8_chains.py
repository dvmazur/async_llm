"""CPU topology/runner-contract tests. These do not claim learned-model parity."""
from copy import deepcopy
from dataclasses import dataclass, field

import pytest
import torch

from tests.e2e.fp8.chains import LAYOUTS, fixtures, plan, snapshot, check_coverage, materialize
from tests.e2e.fp8.chains import count_moe_calls, check_moe_execution


def cases():
    return [dict(name=f"chain_{i}", prompt_ids=list(range(40)) + [90+i]*(12+i*5),
                 teacher_tokens=list(range(100+i*8, 108+i*8)), cold_steps=[0, 3, 7]) for i in range(3)]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("step", [0, 3, 7])
def test_plan_matches_flat_causal_history_at_each_checkpoint(layout, step):
    source = cases()
    nodes, chains = plan(source, layout, step)
    for case, chain in zip(source, chains):
        assert [t for name in chain for t in nodes[name]["ids"]] == case["prompt_ids"] + case["teacher_tokens"][:step]
        for depth, name in enumerate(chain):
            assert nodes[name]["parents"] == chain[:depth]
    if layout == "chain-shared":
        assert [len(c) for c in chains] == [2, 3, 4]
        assert all(c[0] == "common" for c in chains)
        assert nodes[chains[0][-1]]["ids"] and not nodes[chains[1][-1]]["ids"]


@dataclass
class Block:
    token_ids: list = field(default_factory=list)

    @property
    def num_tokens(self):
        return len(self.token_ids)


@dataclass
class Job:
    block: Block
    input_ids: torch.Tensor
    context: list


class FakeSession:
    def __init__(self):
        self.calls = []
        self.buffer = torch.empty(3, 1)

    def create_block(self):
        return Block()

    def prefill_batch(self, jobs):
        self.calls.append(jobs)
        writes = {id(j.block) for j in jobs}
        for i, job in enumerate(jobs):
            assert not writes & {id(b) for b in job.context}, "dependent jobs in same prefill batch"
            assert all(b.num_tokens for b in job.context), "uninitialized parent"
            job.block.token_ids.extend(job.input_ids.tolist())
            ids = [t for b in job.context + [job.block] for t in b.token_ids]
            self.buffer[i, 0] = sum((pos+1)*token for pos, token in enumerate(ids))
        return [self.buffer[i:i+1] for i in range(len(jobs))]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("step", [0, 3, 7])
def test_materialization_obeys_dependencies_shares_prefix_once_and_owns_logits(layout, step):
    source, blocks, session = cases(), {}, FakeSession()
    chains, logits = materialize(session, source, layout, step, blocks, Job)
    expected = [sum((pos+1)*t for pos, t in enumerate(c["prompt_ids"] + c["teacher_tokens"][:step])) for c in source]
    torch.testing.assert_close(logits[:, 0], torch.tensor(expected, dtype=torch.float32))
    session.buffer.zero_()
    torch.testing.assert_close(logits[:, 0], torch.tensor(expected, dtype=torch.float32))
    snapshot(source, blocks, chains, step, [2, 0, 1])
    if layout == "chain-shared":
        assert sum(j.block is blocks["common"] for call in session.calls for j in call) == 1


def make_coverage(layout):
    source, blocks = cases(), {}
    chains, _ = materialize(FakeSession(), source, layout, 0, blocks, Job)
    coverage = dict(layout=layout, decode_forward_calls=7, forwards=[])
    for step in range(8):
        order = [(i + step) % 3 for i in range(3)]
        if step:
            for i in order:
                blocks[chains[i][-1]].token_ids.append(source[i]["teacher_tokens"][step-1])
        coverage["forwards"].append(snapshot(source, blocks, chains, step, order))
    return source, coverage


@pytest.mark.parametrize("layout", LAYOUTS)
def test_coverage_accepts_real_chain_growth_and_worker_permutation(layout):
    source, coverage = make_coverage(layout)
    check_coverage(coverage, source, layout)


@pytest.mark.parametrize("broken", ["flat", "copied_prefix", "reordered_history", "changed_prefix", "no_decode", "missing_step", "wrong_writer", "same_order"])
def test_coverage_rejects_false_multiblock_coverage(broken):
    source, coverage = make_coverage("chain-shared")
    if broken == "flat":
        for forward in coverage["forwards"]:
            for row in forward:
                row["block_ids"] = [row["write_id"]]
                row["lengths"] = [sum(row["lengths"])]
    elif broken == "copied_prefix":
        for forward in coverage["forwards"]:
            for row in forward:
                row["block_ids"][0] += int(row["case"][-1]) + 10
    elif broken == "reordered_history":
        coverage["forwards"][1][0]["history_sha256"] = "wrong history"
    elif broken == "changed_prefix":
        row = coverage["forwards"][1][0]
        row["lengths"][0] += 1
        row["lengths"][-1] -= 1
    elif broken == "no_decode":
        coverage["decode_forward_calls"] = 0
    elif broken == "missing_step":
        coverage["forwards"].pop()
    elif broken == "wrong_writer":
        coverage["forwards"][0][0]["write_id"] = -1
    else:
        for forward in coverage["forwards"]:
            forward.sort(key=lambda r: r["case"])
    with pytest.raises(AssertionError):
        check_coverage(coverage, source, "chain-shared")


def test_fixtures_are_tiny_text_only_and_reference_histories_are_not_modified():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return messages[0]["content"]
        def encode(self, text, **kwargs):
            return list(text.encode())
    source = fixtures(Tokenizer(), 8)
    original = deepcopy(source)
    assert len(source) == 3 and all(len(c["teacher_tokens"]) == 8 and "image_tensors" not in c for c in source)
    for layout in LAYOUTS:
        for step in (0, 3, 7):
            plan(source, layout, step)
    assert source == original


@pytest.mark.parametrize("field", ["num_experts", "fp8_expert_tensors", "moe_fp8_prefill_calls", "moe_fp8_decode_calls"])
def test_moe_claim_requires_both_real_phases_and_quantized_experts(field):
    storage = dict(num_experts=4, fp8_expert_tensors=2, moe_fp8_prefill_calls=3, moe_fp8_decode_calls=7)
    check_moe_execution(storage)
    storage[field] = 0
    with pytest.raises(AssertionError):check_moe_execution(storage)


def test_moe_observer_preserves_outputs_routes_and_restores_on_failure(monkeypatch):
    from types import SimpleNamespace
    import minisgl.core as core
    import minisgl.moe.fused as moe
    state = SimpleNamespace(is_prefill=True)
    monkeypatch.setattr(core, "get_global_ctx", lambda: SimpleNamespace(batch=state))
    x, routes, scales, output = object(), object(), object(), object()
    w = torch.empty(1, dtype=torch.float8_e4m3fn)
    calls = []
    def native(*args, **kwargs):
        assert args == (x, w, w, routes)
        assert kwargs == dict(w1_scale=scales, w2_scale=scales)
        calls.append(args)
        return output
    monkeypatch.setattr(moe, "fused_experts_impl", native)
    with pytest.raises(RuntimeError, match="deliberate"):
        with count_moe_calls(True) as counts:
            assert moe.fused_experts_impl(x, w, w, routes, w1_scale=scales, w2_scale=scales) is output
            state.is_prefill = False
            assert moe.fused_experts_impl(x, w, w, routes, w1_scale=scales, w2_scale=scales) is output
            assert counts == dict(prefill=1, decode=1, mixed=0)
            state.is_mixed = True
            assert moe.fused_experts_impl(x, w, w, routes, w1_scale=scales, w2_scale=scales) is output
            assert counts == dict(prefill=1, decode=1, mixed=1)
            raise RuntimeError("deliberate")
    assert moe.fused_experts_impl is native and len(calls) == 3
    with count_moe_calls(False) as counts:
        assert counts == dict(prefill=0, decode=0, mixed=0)
        assert moe.fused_experts_impl is native


def test_mixed_moe_coverage_cannot_be_replaced_by_separate_prefill_and_decode():
    storage = dict(num_experts=4, fp8_expert_tensors=2, moe_fp8_prefill_calls=3,
                   moe_fp8_decode_calls=7, moe_fp8_mixed_calls=0)
    check_moe_execution(storage)
    with pytest.raises(AssertionError, match="mixed batches"):
        check_moe_execution(storage, require_mixed=True)
    storage["moe_fp8_mixed_calls"] = 2
    check_moe_execution(storage, require_mixed=True)
