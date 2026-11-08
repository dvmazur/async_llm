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
