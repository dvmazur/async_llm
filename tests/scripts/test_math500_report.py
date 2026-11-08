import copy
from types import SimpleNamespace

import pytest
from minisgl_examples.demos.async_thoughts.math500_report import (
    common_prefix,
    paired,
    validate_batches,
    verify_ar,
    verify_ordinary,
)


def trace():
    def req(block, ids, view):
        return {"write_to": block, "write_start": view[-1][1], "input_ids": ids, "cache_view": view}

    steps = [
        ("prefill", [req(10, [1, 2], [[10, 0]])]),
        ("decode", [req(11, [3], [[10, 2], [11, 0]])]),
        ("decode", [req(11, [4], [[10, 2], [11, 1]])]),
        ("prefill", [req(12, [98], [[10, 2], [11, 2], [12, 0]])]),
        ("decode", [req(11, [5], [[10, 2], [11, 2]]), req(12, [99], [[10, 2], [11, 2], [12, 1]])]),
        ("decode", [req(12, [6], [[10, 2], [11, 3], [12, 2]]), req(11, [7], [[10, 2], [11, 3]])]),
    ]
    return {
        "format_version": 2,
        "steps": len(steps),
        "updates": [],
        "batches": [
            {"kind": kind, "step": i, "requests": reqs} for i, (kind, reqs) in enumerate(steps)
        ],
    }


def audit(trace):
    prompting = SimpleNamespace(
        native_turns=SimpleNamespace(prompt_ids=[1, 2, 3], writer_ids=[98, 99])
    )
    return verify_ar(trace, "native-chat", prompting, None, [1, 2, 3], [6], [4, 5, 7], True)


def test_native_witness_checks_peer_visibility_and_seed_tokens():
    t = trace()
    summary = audit(t)
    assert summary["joint_decode_batches"] == 2 and summary["writer_saw_thinker_grow"]
    assert summary["writer_first_visible_thinker_block_tokens"] == 2
    assert summary["first_visible_generated_thought_tokens"] == 2
    t["batches"][1]["requests"][0]["input_ids"] = [0]
    with pytest.raises(ValueError, match="Thinker seed"):
        audit(t)


def test_view_length_and_writer_chain_corruption_fail():
    t = trace()
    t["batches"][4]["requests"][1]["cache_view"][1][1] = 3
    with pytest.raises(ValueError, match="pre-step view"):
        audit(t)
    t = trace()
    t["batches"][5]["requests"][0]["input_ids"] = [7]
    with pytest.raises(ValueError, match="Decode token chain"):
        audit(t)


def test_decode_inputs_check_capped_and_eos_outputs_separately():
    t = trace()
    t["batches"] = t["batches"][:1]
    for i, token in enumerate([3, 4], 1):
        pos = 1 + i
        t["batches"].append(
            {
                "kind": "decode",
                "step": i,
                "requests": [
                    {
                        "input_ids": [token],
                        "write_to": 10,
                        "write_start": pos,
                        "cache_view": [[10, pos]],
                    }
                ],
            }
        )
    t["steps"] = 3
    verify_ordinary(t, [1, 2, 3], [4], True)
    verify_ordinary(t, [1, 2, 3], [4, 5], False)
    with pytest.raises(ValueError, match="Decode token chain"):
        verify_ordinary(t, [1, 2, 3], [4, 5], True)
    tampered = copy.deepcopy(t)
    tampered["batches"][2]["requests"][0]["write_start"] = 1
    with pytest.raises(ValueError, match="block write"):
        validate_batches(tampered)


def test_sign_test_uses_only_discordant_pairs_and_never_changes_denominator():
    baseline = {str(i): False for i in range(5)}
    candidate = {str(i): True for i in range(5)}
    result = paired(baseline, candidate)
    assert result["exact_two_sided_sign_p"] == 2 / 32
    assert result["n"] == result["gained_count"] == 5 and result["lost_count"] == 0
    assert paired(baseline, baseline)["exact_two_sided_sign_p"] == 1.0
    assert common_prefix([1, 2, 3], [1, 2, 6, 7]) == 2


def test_terminal_receipt_cannot_upgrade_incomplete_schedule(monkeypatch, tmp_path):
    import json

    from minisgl_examples.demos.async_thoughts import math500_report as report

    rows = [{"unique_id": str(i), "index": i} for i in range(2)]
    settings = {
        "dataset_revision": report.DATASET_REVISION,
        "dataset_sha256": report.DATASET_SHA,
        "order": report.ORDER,
        "ordered_ids": ["0", "1"],
        "sampling": "greedy",
        "dtype": "bfloat16",
        "budget": 32,
        "context": 80,
        "paired": 1,
        "limit": 2,
        "probe_period": 20,
        "smoke_only": False,
        "source_sha256": dict.fromkeys(
            ["math500.py", "engine.py", "demo.py", "short_retry.py"], "0" * 64
        ),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(settings))
    (tmp_path / "rows").mkdir()
    (tmp_path / "completed.json").write_text(
        json.dumps(
            {"planned_rows": 4, "settings_sha256": report.digest(report.encode_json(settings))}
        )
    )
    monkeypatch.setattr(report, "ordered_rows", lambda path: rows)
    with pytest.raises(ValueError, match="Terminal receipt"):
        report.report(tmp_path, None, None)
    (tmp_path / "completed.json").unlink()
    (tmp_path / "rows/000.native-chat.json").write_text("{}")
    (tmp_path / "environment.json").write_text(
        json.dumps({"gpu": "test", "effective_context": 80, "eos_token_ids": [1]})
    )
    (tmp_path / "smoke.json").write_text("[]")
    monkeypatch.setattr(report, "verify_smoke", lambda _: None)
    with pytest.raises(ValueError, match="prefix of the immutable schedule"):
        report.report(tmp_path, None, None)
