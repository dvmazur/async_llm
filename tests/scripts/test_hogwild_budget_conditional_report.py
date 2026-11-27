import pytest

from minisgl_examples.training.parallel_thinking import budget_conditional_report as report


def pair(rounds, budget, *, original_correct=True, forced_correct=False):
    original = dict(decoding_steps=rounds, unique_id="problem", gold="17", trace_sha256="a" * 64,
                    prediction="original", correct=original_correct, used_finisher="True",
                    grading_status="correct" if original_correct else "incorrect")
    forced = dict(unique_id="problem", gold="17", source_trace_sha256="a" * 64,
                  effective_rounds=min(rounds, budget), prediction="forced", correct=forced_correct,
                  grading_status="correct" if forced_correct else "incorrect", answer_tokens=8)
    return original, forced


@pytest.mark.parametrize("rounds,budget,selected", [(256, 255, "forced"), (256, 256, "original"), (256, 257, "original")])
def test_answer_source_uses_reasoning_completion_including_exact_boundary(rounds, budget, selected):
    original, forced = pair(rounds, budget)
    result = report.conditional_row(("native", 1, 0, budget), original, forced)
    assert result["answer_source"] == selected
    assert result["prediction"] == selected
    assert result["correct"] == (selected == "original")
    assert result["original_used_finisher"] == "True"


def test_completed_incorrect_answer_stays_incorrect_even_if_forced_answer_is_correct():
    original, forced = pair(100, 256, original_correct=False, forced_correct=True)
    result = report.conditional_row(("hogwild", 16, 0, 256), original, forced)
    assert result["answer_source"] == "original" and result["correct"] is False


@pytest.mark.parametrize("field,value", [("source_trace_sha256", "b" * 64), ("unique_id", "other"),
                                         ("gold", "18"), ("effective_rounds", 255)])
def test_mismatched_trace_identity_or_effective_cutoff_is_rejected(field, value):
    original, forced = pair(256, 256)
    forced[field] = value
    with pytest.raises(ValueError, match="Trace identity or cutoff mismatch"):
        report.conditional_row(("native", 1, 0, 256), original, forced)


def test_requested_three_table_ranges():
    assert [budgets for _, budgets in report.BUDGET_RANGES] == [
        (256, 512, 1024), tuple(range(1000, 10001, 1000)), tuple(range(10000, 16001, 1000))]
