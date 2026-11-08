import json
from pathlib import Path

import pytest
from minisgl_examples.demos.async_thoughts import math500_import as imp
from minisgl_examples.demos.async_thoughts import math500_report
from minisgl_examples.demos.async_thoughts.math500 import digest, encode_json


def fixture(tmp, monkeypatch):
    origin = tmp / "first"
    origin.mkdir()
    for name in ("rows", "traces"):
        (origin / name).mkdir()
    sources = tmp / "sources"
    sources.mkdir()
    names = ("math500.py", "engine.py", "demo.py", "short_retry.py")
    for n in names:
        (sources / n).write_bytes(n.encode())
    manifest = {
        "paired": 2,
        "limit": 2,
        "ordered_ids": ["a", "b"],
        "torch": "T",
        "visible_devices": "GPU-old",
        "model": "models--test",
        "model_revision": "abc",
        "source_sha256": {n: digest(n.encode()) for n in names},
    }
    hardware = {"gpu": "GPU-A", "effective_context": 100, "cuda": "13", "eos_token_ids": [9]}
    for name, value in [
        ("manifest.json", manifest),
        ("smoke.json", []),
        ("environment.json", hardware),
    ]:
        (origin / name).write_bytes(encode_json(value))
    packed = b"checked-test-trace"
    (origin / "traces/000.ordinary.json.gz").write_bytes(packed)
    case = {
        "rank": 0,
        "arm": "ordinary",
        "unique_id": "a",
        "trace_sha256": digest(packed),
        "pure_exact": True,
    }
    (origin / "rows/000.ordinary.json").write_bytes(encode_json(case))
    report = {
        "settings_sha256": digest(encode_json(manifest)),
        "schedule": {"verified_rows": 1},
        "cases": [case],
        "environment": hardware,
    }
    previous = tmp / "report.json"
    previous.write_bytes(encode_json(report))
    monkeypatch.setattr(math500_report, "report", lambda *args, **kw: report)
    monkeypatch.setattr(imp, "ordered_rows", lambda p: [{"unique_id": "a"}, {"unique_id": "b"}])
    return origin, sources, previous, case


def test_import_copies_only_verified_prefix_and_binds_an_independent_session(tmp_path, monkeypatch):
    origin, sources, previous, case = fixture(tmp_path, monkeypatch)
    out = tmp_path / "resume"
    before = (origin / "rows/000.ordinary.json").read_bytes()
    receipt = imp.prepare(
        origin, out, Path("dataset"), None, sources, "pod-old", previous, paired=1, expect_rows=1
    )
    assert receipt["imported_rows"] == 1 and receipt["target_planned_rows"] == 3
    assert not (out / "manifest.json").exists()
    assert (out / "rows/000.ordinary.json").read_bytes() == before
    assert not (out / "rows/000.ordinary.json").samefile(origin / "rows/000.ordinary.json")
    bound = imp.bind(out, sources, "pod-new", "GPU-new", "T2", "GPU-B", "13")
    assert imp.bind(out, sources, "pod-new", "GPU-new", "T2", "GPU-B", "13") == bound
    target = imp.read(out / "manifest.json")
    hardware = {"gpu": "GPU-B", "cuda": "13", "eos_token_ids": [9]}
    assert target["limit"] == target["paired"] == 1 and target["visible_devices"] == "GPU-new"
    cases = [case.copy(), {"rank": 0, "arm": "working-notes"}]
    provenance = imp.session_provenance(out, target, hardware, cases)
    assert (provenance["imported_rows"], provenance["new_rows"]) == (1, 1)
    assert [(c["session"], c["imported"]) for c in cases] == [("pod-old", True), ("pod-new", False)]
    assert provenance["origin_environment"]["gpu"] == "GPU-A"
    assert (origin / "rows/000.ordinary.json").read_bytes() == before
    with pytest.raises(ValueError, match="incompatible"):
        imp.bind(out, sources, "wrong-pod", "GPU-new", "T2", "GPU-B", "13")
    with pytest.raises(ValueError, match="Observed destination"):
        imp.session_provenance(out, target, dict(hardware, gpu="unbound"), cases)
    with pytest.raises(ValueError, match="source/model/schedule"):
        imp.session_provenance(out, dict(target, model_revision="other"), hardware, cases)
    (out / "rows/000.ordinary.json").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Imported row changed"):
        imp.validate_import(out)
    assert (origin / "rows/000.ordinary.json").read_bytes() == before


def test_prepare_rejects_stale_report_or_non_prefix_without_partial_output(tmp_path, monkeypatch):
    origin, sources, previous, case = fixture(tmp_path, monkeypatch)
    out = tmp_path / "resume"
    stale = json.loads(previous.read_bytes())
    stale["cases"] = []
    previous.write_bytes(encode_json(stale))
    with pytest.raises(ValueError, match="Prior report"):
        imp.prepare(
            origin,
            out,
            Path("dataset"),
            None,
            sources,
            "pod-old",
            previous,
            paired=1,
            expect_rows=1,
        )
    with pytest.raises(ValueError, match="expected prefix"):
        imp.prepare(
            origin, out, Path("dataset"), None, sources, "pod-old", None, paired=1, expect_rows=2
        )
    assert not out.exists()
    (sources / "engine.py").write_text("changed")
    with pytest.raises(ValueError, match="Measured source"):
        imp.prepare(
            origin, out, Path("dataset"), None, sources, "pod-old", None, paired=1, expect_rows=1
        )


def test_repeated_continuation_preserves_first_acquisition_and_hardware(tmp_path, monkeypatch):
    first, sources, prior, old_case = fixture(tmp_path, monkeypatch)
    second = tmp_path / "second"
    imp.prepare(first, second, Path("dataset"), None, sources, "A", prior, paired=1)
    imp.bind(second, sources, "B", "GPU-b", "T-B", "GPU-B", "12")
    env_b = {"gpu": "GPU-B", "cuda": "12", "eos_token_ids": [9], "effective_context": 100}
    (second / "environment.json").write_bytes(encode_json(env_b))
    (second / "smoke.json").write_bytes((first / "smoke.json").read_bytes())
    new_trace = b"new-b-trace"
    new_case = {
        "rank": 0,
        "arm": "working-notes",
        "unique_id": "a",
        "trace_sha256": digest(new_trace),
        "pure_exact": False,
    }
    (second / "traces/000.working-notes.json.gz").write_bytes(new_trace)
    (second / "rows/000.working-notes.json").write_bytes(encode_json(new_case))
    cases = [old_case.copy(), new_case.copy()]
    manifest = imp.read(second / "manifest.json")
    lineage = imp.session_provenance(second, manifest, env_b, cases)
    proof = {
        "settings_sha256": digest(encode_json(manifest)),
        "environment": env_b,
        "cases": cases,
        "schedule": {"verified_rows": 2},
        "session_provenance": lineage,
    }
    monkeypatch.setattr(math500_report, "report", lambda *a, **kw: proof)
    third = tmp_path / "third"
    with pytest.raises(ValueError, match="Origin ID"):
        imp.prepare(second, third, Path("dataset"), None, sources, "wrong", None, paired=1)
    receipt = imp.prepare(second, third, Path("dataset"), None, sources, "B", None, paired=1)
    assert [r["acquired_session"] for r in receipt["rows"]] == ["A", "B"]
    assert "prior_import.json" in receipt["origin_catalog"]
    with pytest.raises(ValueError, match="prior session ID"):
        imp.bind(third, sources, "A", "GPU-c", "T-C", "GPU-C", "13")
    imp.bind(third, sources, "C", "GPU-c", "T-C", "GPU-C", "13")
    env_c = {"gpu": "GPU-C", "cuda": "13", "eos_token_ids": [9]}
    result_cases = [old_case.copy(), new_case.copy(), {"rank": 0, "arm": "native-chat"}]
    result = imp.session_provenance(third, imp.read(third / "manifest.json"), env_c, result_cases)
    assert [(r["session"], r["imported"]) for r in result_cases] == [
        ("A", True),
        ("B", True),
        ("C", False),
    ]
    assert [result["sessions"][s]["environment"]["gpu"] for s in "ABC"] == [
        "GPU-A",
        "GPU-B",
        "GPU-C",
    ]
    assert result["new_rows"] == 1 and result["imported_rows"] == 2
    assert (third / "origin/prior_session.json").read_bytes() == (
        second / "session.json"
    ).read_bytes()
