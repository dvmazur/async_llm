"""Cross-report checks and wording for the paper renderer use only supplied figures."""

from copy import deepcopy

import pytest
from minisgl_examples.demos.async_thoughts.math500_tulu_paper import (
    ARMS,
    CANDIDATE,
    ORDINARY,
    cost_lines,
    render,
)


def fixture():
    n = 2
    truth = {name: [i != 1, i <= 1] for i, name in enumerate(ARMS)}
    cases = [
        {"rank": i, "unique_id": f"id{i}", "outcomes": {arm: val[i] for arm, val in truth.items()}}
        for i in range(n)
    ]
    summaries = {}
    full = {}
    equal = {}
    paired = {}

    def constant(amt):
        return {"n": n, "sum": amt * n, "mean": amt, "median": amt}

    for i, (arm, answers) in enumerate(truth.items()):
        correct = sum(answers)
        writer = 1 if arm == "direct" else 2
        thinker = 0 if arm in (ORDINARY, "direct") else 1
        tot = (writer + thinker) * n
        s = {
            "coverage": n,
            "correct": correct,
            "writer_tokens": constant(writer),
            "thinker_tokens": constant(thinker),
        }
        # Legacy comparison reports have token distributions but no recorded scheduler counts.
        if not arm.startswith("archived-"):
            s.update(
                content_tokens=constant(writer + thinker),
                decode_batches=constant(4),
                forward_batches=constant(5),
            )
        summaries[arm] = s
        full[arm] = {
            "content": tot,
            "correct": correct,
            "writer_content": writer * n,
            "thinker_content": thinker * n,
            "recorded_decode_batches": 4 * n,
            "recorded_forward_batches": 5 * n,
        }
        spent = min(4, tot)
        gains = sum(a and not b for a, b in zip(answers, truth[ORDINARY]))
        losses = sum(b and not a for a, b in zip(answers, truth[ORDINARY]))
        equal[arm] = {
            "content": spent,
            "writer_content": min(writer * n, spent),
            "thinker_content": max(0, spent - writer * n),
            "correct": correct,
            "uniform_cap": 2,
            "residual_from_target": spent - 4,
            "status": "exact" if spent == 4 else "below_target_at_natural_completion",
            "only_this_arm_vs_full_ordinary": gains,
            "only_full_ordinary": losses,
        }
        if arm != CANDIDATE:
            gains = [j for j in range(n) if truth[CANDIDATE][j] and not answers[j]]
            losses = [j for j in range(n) if answers[j] and not truth[CANDIDATE][j]]
            paired[arm] = {
                "n": n,
                "baseline_correct": correct,
                "candidate_correct": sum(truth[CANDIDATE]),
                "gained": gains,
                "lost": losses,
                "gained_count": len(gains),
                "lost_count": len(losses),
                "delta_percentage_points": 100 * (len(gains) - len(losses)) / n,
                "exact_two_sided_sign_p": 1.0,
            }
    summaries[CANDIDATE].update(
        admission_votes={"applied": 3, "yes": 2},
        continuation_votes={"applied": 5, "yes": 2},
        positive_probe_count=constant(2),
        probe_count=constant(5),
        joint_decode_batches=constant(1),
        writer_tokens_after_last_thinker_decode=constant(1),
    )
    c = {
        "coverage": n,
        "target": 64,
        "summary": summaries,
        "cases": cases,
        "paired_against_evidence_turns": paired,
    }
    b = {
        "coverage": n,
        "expected": 64,
        "partial": True,
        "rank_ids": ["id0", "id1"],
        "full_observed": full,
        "ordinary_content_target": 4,
        "full_ordinary_correct": sum(truth[ORDINARY]),
        "equal_aggregate_content": equal,
    }
    return c, b


def test_render_measured_partial_and_costs():
    c, b = fixture()
    r = render(c, b)
    assert "Partial frozen cohort, 2 of 64" in r
    assert "| Ordinary (archived) |" in r
    assert "| Admission | 3 | 2 | 1 |" in r
    assert "Natural output below target" in r and "## Rental receipts" not in r
    receipt = {
        "pod": "p1",
        "estimated_usd": 0.31,
        "upper_estimate_usd": 0.42,
        "posted_usd": 0,
        "billing": {"metadata": {"recordCount": 0, "query": {"podId": "p1"}}},
    }
    with_cost = render(c, b, [receipt])
    assert (
        "no charge had posted" in with_cost
        and "receipt estimate $0.3100" in with_cost
        and "posted $0" not in with_cost
    )
    with pytest.raises(ValueError, match="duplicate"):
        cost_lines([receipt, deepcopy(receipt)])


@pytest.mark.parametrize("mismatch", ["rank", "score", "tokens", "physical", "aggregate", "paired"])
def test_renderer_rejects_inconsistent_inputs(mismatch):
    c, b = fixture()
    if mismatch == "rank":
        b["rank_ids"].reverse()
    if mismatch == "score":
        b["full_observed"][ORDINARY]["correct"] -= 1
    if mismatch == "tokens":
        b["full_observed"]["serial-thinking"]["thinker_content"] += 1
    if mismatch == "physical":
        b["full_observed"]["direct"]["recorded_decode_batches"] -= 1
    if mismatch == "aggregate":
        b["equal_aggregate_content"]["direct"]["residual_from_target"] = 0
    if mismatch == "paired":
        c["paired_against_evidence_turns"]["direct"]["gained_count"] += 1
    with pytest.raises(ValueError):
        render(c, b)
