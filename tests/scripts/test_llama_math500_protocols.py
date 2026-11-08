import pytest
from minisgl_examples.demos.async_thoughts.math500 import ARMS, last_box, schedule, verify_smoke


def test_strict_grading_uses_last_literal_balanced_box():
    assert last_box(r"\boxed{2} then \boxed{ \frac{3}{2} }") == r"\frac{3}{2}"
    assert last_box(r"\boxed{2} then \boxed{") is None
    assert last_box(r"\boxed {5}") is None


def test_paired_schedule_finishes_each_problem_before_advancing():
    rows = [{"unique_id": str(i)} for i in range(5)]
    jobs = list(schedule(rows, paired=3, limit=4))
    assert [r for r, _, _ in jobs] == [0] * 3 + [1] * 3 + [2] * 3 + [3]
    assert [a for _, _, a in jobs] == [*ARMS, *ARMS[1:], ARMS[0], ARMS[2], *ARMS[:2], "native-chat"]
    assert all(row["unique_id"] == str(rank) for rank, row, _ in jobs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("same_prompt", False),
        ("matching_native_thinker_prefix", 0),
        ("native_box", "6"),
        ("writer_saw_thinker_grow", False),
    ],
)
def test_failed_smoke_cannot_launch_the_benchmark(field, value):
    row = {
        "same_prompt": True,
        "matching_native_thinker_prefix": 32,
        "native_box": "5",
        "ordinary_box": "5",
        "writer_saw_thinker_grow": True,
        "joint_decode_batches": 3,
    }
    verify_smoke([row])
    with pytest.raises(ValueError):
        verify_smoke([{**row, field: value}])
