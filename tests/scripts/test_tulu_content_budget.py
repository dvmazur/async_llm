"""Generated-content cap replay preserves joint steps and strict prefix grading."""

import pytest
from minisgl_examples.demos.async_thoughts.math500_token_budget import Case
from minisgl_examples.demos.async_thoughts.math500_tulu_content_budget import (
    ORDINARY,
    content_schedule,
    ordinary_case,
    summarize,
)


class Tokenizer:
    @staticmethod
    def decode(ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join({0: "work ", 1: r"\boxed{1}", 2: r"\boxed{2}"}[i] for i in ids)


def events(*items):
    return [{"batch_step": step, "role": role, "token": token} for step, role, token in items]


def test_content_batches_count_accepted_roles_indivisibly_and_skip_terminal_samples():
    # Decoder metadata may arrive in either order within a physical joint step.
    sample = events(
        (5, "thinker", 10),
        (8, "writer", 1),
        (8, "thinker", 11),
        (10, "thinker", 77),
        (10, "writer", 2),
        (12, "writer", 99),
    )
    ends, written = content_schedule(sample, [10, 11], [1, 2])
    assert ends == [0, 1, 3, 4] and written == [3, 4]
    # 77 represents the in-flight final thought that the generator did not return.
    with pytest.raises(ValueError, match="Sample differs"):
        content_schedule(sample, [10, 15], [1, 2])
    with pytest.raises(ValueError, match="role in a sample batch"):
        content_schedule(events((3, "writer", 1), (3, "writer", 2)), [], [1, 2])


def test_serial_thoughts_consume_budget_before_any_visible_answer_and_eos_is_free():
    schedule = events(
        (1, "thinker", 10),
        (2, "thinker", 11),
        (3, "thinker", 99),
        (5, "writer", 1),
        (6, "writer", 2),
        (7, "writer", 99),
    )
    assert content_schedule(schedule, [10, 11], [1, 2]) == ([0, 1, 2, 3, 4], [3, 4])


def test_common_caps_full_steps_and_closest_aggregate_target():
    # Archived ordinary costs two content tokens and answers 2. The candidates' joint
    # prefix can spend 1 or 3, so the tie with target 2 selects the smaller cap.
    ref = Case("2", [0, 2], [1, 2], [0, 1, 3, 4], [3, 4], 3, 4, 3, 7, {})
    short = Case("2", [0, 2], [2], [0, 1], [1], 3, 4, 2, 3, {})
    serial = Case("2", [0, 2], [2, 1], [0, 1, 2, 3], [2, 3], 3, 4, 5, 7, {})
    report = summarize(
        {ORDINARY: [ordinary_case(ref)], "joint": [ref], "short": [short], "serial": [serial]},
        Tokenizer(),
        caps=[1, 2, 3, 4],
    )
    assert report["ordinary_content_target"] == 2
    j = report["common_caps"][2]["joint"]
    assert (j["content"], j["writer_content"], j["correct"]) == (1, 0, 0)
    j = report["common_caps"][3]["joint"]
    assert (j["content"], j["thinker_content"], j["correct"]) == (3, 2, 0)
    assert report["full_observed"]["joint"]["correct"] == 1
    assert report["full_observed"]["joint"]["recorded_forward_batches"] == 7
    j = report["equal_aggregate_content"]["joint"]
    assert (j["status"], j["content"], j["residual_from_target"]) == ("closest_uniform_cap", 1, -1)
    s = report["equal_aggregate_content"]["short"]
    assert (s["status"], s["content"], s["correct"]) == ("below_target_at_natural_completion", 1, 1)
    # At cap 2 the serial writer has already boxed 2; its later box overwrites this grade.
    assert report["common_caps"][2]["serial"]["correct"] == 1
    assert report["full_observed"]["serial"]["correct"] == 0
