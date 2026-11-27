"""Cutoff views preserve publication order, pending samples and prompt ownership."""

from copy import deepcopy

import pytest
from minisgl_examples.training.parallel_thinking.budget_views import reconstruct_cutoff


def trajectory(layout="native_workers"):
    streams = {"Alice": [10, 11, 12, 13, 14], "Bob": [20, 21, 22, 23, 99]}
    if layout in {"native", "causal"}:
        streams = {"Qwen" if layout == "native" else "Alice": [10, 11, 12, 13, 99]}
    return {
        "layout": layout, "publication_policy": "encoded_separator_v1",
        "workers": list(streams), "prompt_ids": [1, 2], "worker_token_ids": streams,
        "decode_rounds": 5, "eos_token_ids": [99], "eos_token_id": 99,
        "other_header_ids": [], "own_header_ids": [],
        "initial_worker_prefix_ids": {name: [] for name in streams}, "publications": [],
        "finisher": {"suffix_ids": [700], "sampled_token_ids": [701, 702]},
    }


def publication(round_number, worker, count):
    return {"round": round_number, "worker": worker, "step": 1,
            "committed_token_count": count, "next_prefix_ids": []}


def test_delayed_publication_keeps_boundary_pending_until_next_round():
    trace = trajectory()
    trace["publications"] = [publication(3, "Alice", 2)]
    before = reconstruct_cutoff(trace, 2)
    assert before["common_history_ids"] == [1, 2]
    assert before["encoded_residual_ids"] == {"Alice": [10], "Bob": [20]}
    assert before["pending_token_ids"] == {"Alice": [11], "Bob": [21]}
    assert before["view_token_ids"] == [1, 2, 20, 10]

    after = reconstruct_cutoff(trace, 3)
    assert after["common_history_ids"] == [1, 2, 10, 11]
    assert after["encoded_residual_ids"] == {"Alice": [], "Bob": [20, 21]}
    assert after["pending_token_ids"] == {"Alice": [12], "Bob": [22]}
    assert after["view_token_ids"] == [1, 2, 10, 11, 20, 21]
    assert reconstruct_cutoff(trace, 3, include_pending=True)["view_token_ids"] == [
        1, 2, 10, 11, 20, 21, 22, 12,
    ]


def test_shared_history_preserves_event_order_without_duplicate_worker_tokens():
    trace = trajectory()
    trace["publications"] = [publication(3, "Bob", 2), publication(3, "Alice", 2),
                             publication(5, "Alice", 2)]
    result = reconstruct_cutoff(trace, 5, include_pending=True)
    assert result["common_history_ids"] == [1, 2, 20, 21, 10, 11, 12, 13]
    assert result["view_order"] == ["common_history", "Bob", "Alice"]
    assert result["view_token_ids"] == [1, 2, 20, 21, 10, 11, 12, 13, 22, 23, 14]
    assert result["pending_token_ids"] == {"Alice": [14], "Bob": []}
    assert result["publication_count"] == 3


def test_peer_order_follows_recorded_worker_order_and_places_first_worker_last():
    trace = trajectory()
    trace["workers"] = ["Alice", "Bob", "Carol"]
    trace["worker_token_ids"]["Carol"] = [30, 31, 32, 33, 34]
    result = reconstruct_cutoff(trace, 2, include_pending=True)
    assert result["view_order"] == ["common_history", "Bob", "Carol", "Alice"]
    assert result["view_token_ids"] == [1, 2, 20, 21, 30, 31, 10, 11]


@pytest.mark.parametrize("layout", ["native", "causal", "native_workers"])
def test_early_stop_clamps_and_excludes_eos_and_original_finisher(layout):
    trace = trajectory(layout)
    original = deepcopy(trace)
    result = reconstruct_cutoff(trace, 1024, include_pending=True)
    assert result["effective_rounds"] == 5 and result["clamped_to_saved_run"]
    assert not ({99, 700, 701, 702} & set(result["view_token_ids"]))
    assert trace == original
    if layout == "native_workers":
        assert result["view_token_ids"] == [1, 2, 20, 21, 22, 23, 10, 11, 12, 13, 14]
    else:
        assert result["view_token_ids"] == [1, 2, 10, 11, 12, 13]


def test_causal_h1_first_publication_moves_prompt_exactly_once():
    trace = trajectory("causal")
    trace["publications"] = [publication(2, "Alice", 3), publication(4, "Alice", 2)]
    before = reconstruct_cutoff(trace, 1)
    assert before["common_history_ids"] == []
    assert before["encoded_residual_ids"] == {"Alice": [1, 2]}
    at_first = reconstruct_cutoff(trace, 2, include_pending=True)
    assert at_first["common_history_ids"] == [1, 2, 10]
    assert at_first["encoded_residual_ids"] == {"Alice": []}
    assert at_first["view_token_ids"] == [1, 2, 10, 11]
    at_second = reconstruct_cutoff(trace, 4, include_pending=True)
    assert at_second["common_history_ids"] == [1, 2, 10, 11, 12]
    assert at_second["view_token_ids"] == [1, 2, 10, 11, 12, 13]


def test_one_round_has_no_encoded_worker_sample_and_retains_each_pending_sample():
    result = reconstruct_cutoff(trajectory(), 1, include_pending=True)
    assert result["encoded_residual_ids"] == {"Alice": [], "Bob": []}
    assert result["pending_token_ids"] == {"Alice": [10], "Bob": [20]}
    assert result["view_token_ids"] == [1, 2, 20, 10]


def test_first_worker_eos_does_not_drop_peers_latest_nonterminal_sample():
    trace = trajectory()
    trace["worker_token_ids"] = {"Alice": [10, 11, 12, 13, 99], "Bob": [20, 21, 22, 23, 24]}
    result = reconstruct_cutoff(trace, 5, include_pending=True)
    assert result["pending_token_ids"] == {"Alice": [], "Bob": [24]}
    assert result["view_token_ids"] == [1, 2, 20, 21, 22, 23, 24, 10, 11, 12, 13]
    assert reconstruct_cutoff(trace, 5)["view_token_ids"] == [1, 2, 20, 21, 22, 23, 10, 11, 12, 13]


@pytest.mark.parametrize("layout", ["legacy", "scaffold", "unknown"])
def test_unsupported_layout_is_rejected(layout):
    with pytest.raises(ValueError, match="Unsupported cutoff layout"):
        reconstruct_cutoff(trajectory(layout), 2)


def test_old_publication_policy_and_inconsistent_event_counts_are_rejected():
    trace = trajectory()
    trace["publication_policy"] = "sampled_boundary_v1"
    with pytest.raises(ValueError, match="encoded_separator_v1"):
        reconstruct_cutoff(trace, 2)
    trace["publication_policy"] = "encoded_separator_v1"
    trace["publications"] = [publication(3, "Alice", 3)]
    with pytest.raises(ValueError, match="encoded prefix"):
        reconstruct_cutoff(trace, 2)


@pytest.mark.parametrize("cutoff", [0, -1, True, 1.5])
def test_invalid_cutoff_is_rejected(cutoff):
    with pytest.raises(ValueError, match="positive integer"):
        reconstruct_cutoff(trajectory(), cutoff)
