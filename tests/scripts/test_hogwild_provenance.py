import hashlib
import json

import pytest

from minisgl_examples.training.parallel_thinking import sweep


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    files = (
        "scripts/minisgl_examples/training/parallel_thinking/sweep.py",
        "scripts/minisgl_examples/training/parallel_thinking/eval.py",
        "python/minisgl/engine.py", "python/minisgl/kernel.cu",
        "run_hogwild.sh", "run_hogwild_budget.sh", "pyproject.toml", "uv.lock",
        "scripts/pyproject.toml", "docs/hogwild/requirements.txt",
    )
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n")
    monkeypatch.setattr(sweep, "ROOT", root)
    monkeypatch.setattr(sweep, "__file__", str(root / files[0]))
    output = tmp_path / "run"
    output.mkdir()
    return root, output, files


def test_new_run_freezes_sources_and_resume_verifies_them(checkout):
    root, output, files = checkout
    sweep.snapshot_sources(output)
    expected = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in files}
    assert json.loads((output / "source-files.json").read_text()) == expected
    assert json.loads((output / "active-source-files.json").read_text()) == expected
    for relative in files:
        assert (output / "source-snapshot" / relative).read_bytes() == (root / relative).read_bytes()
    before = (output / "source-files.json").read_bytes()
    sweep.snapshot_sources(output)
    sweep.verify_source_snapshot(output)
    assert (output / "source-files.json").read_bytes() == before
    (root / files[1]).write_text("changed evaluator")
    with pytest.raises(sweep.IncompatibleRun, match="changed"):
        sweep.verify_source_snapshot(output)
    with pytest.raises(sweep.IncompatibleRun, match="changed"):
        sweep.snapshot_sources(output)
    assert (output / "source-files.json").read_bytes() == before


def test_resume_rejects_corrupted_snapshot(checkout):
    _, output, files = checkout
    sweep.snapshot_sources(output)
    (output / "source-snapshot" / files[1]).write_text("changed archived evaluator")
    with pytest.raises(sweep.IncompatibleRun, match="snapshot changed"):
        sweep.snapshot_sources(output)


def test_existing_results_require_their_original_manifest(checkout):
    _, output, _ = checkout
    config = output / "shards/first/config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    with pytest.raises(sweep.IncompatibleRun, match="original source manifests"):
        sweep.snapshot_sources(output)
    assert not (output / "source-files.json").exists()
