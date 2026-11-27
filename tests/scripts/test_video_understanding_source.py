"""HF source loading with a synthetic clip and decoder."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from minisgl_examples.training.video_understanding import prepare as loader
from PIL import Image


@pytest.fixture
def source(tmp_path, monkeypatch):
    payload = b"synthetic video bytes"
    annotation = tmp_path / "annotation.json"
    row = {"video": loader.VIDEO, "conversation": [{"content": "Synthetic question"}]}
    annotation.write_text(json.dumps([row]))
    cache = tmp_path / "hf-cache"
    downloads = []
    reads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(annotation)

    class Archive:
        def __init__(self, url):
            assert loader.REVISION in url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def infolist(self):
            return [SimpleNamespace(filename="videos/" + loader.VIDEO, file_size=len(payload))]

        def read(self, member):
            reads.append(member.filename)
            return payload

    def decode(args, check):
        assert check
        output = Path(args[-1])
        assert output.parent == cache
        Image.new("RGB", (4, 2), "red").save(output)

    monkeypatch.setattr(loader, "cache_dir", lambda: cache)
    monkeypatch.setattr(loader, "VIDEO_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr("remotezip.RemoteZip", Archive)
    monkeypatch.setattr(loader.subprocess, "run", decode)
    loader.prepare.cache_clear()
    yield cache, row, downloads, reads
    loader.prepare.cache_clear()


def test_download_and_cached_reload_have_identical_manifests(source):
    cache, row, downloads, reads = source
    first = loader.prepare()
    assert first["annotation"] == row
    assert downloads[0] == {
        "repo_id": loader.REPO,
        "repo_type": "dataset",
        "revision": loader.REVISION,
        "filename": "VAD/anno.json",
    }
    assert reads == ["videos/" + loader.VIDEO]
    for frame in first["frames"]:
        assert loader.frame_path(frame).parent == cache
        assert hashlib.sha256(loader.frame_path(frame).read_bytes()).hexdigest() == frame["sha256"]
    loader.prepare.cache_clear()
    assert loader.prepare() == first
    assert len(reads) == 1


def test_corrupt_cached_clip_is_rejected(source):
    cache, _, _, _ = source
    loader.prepare()
    (cache / loader.VIDEO).write_bytes(b"corrupt")
    loader.prepare.cache_clear()
    with pytest.raises(ValueError, match="Cached video SHA256 mismatch"):
        loader.prepare()


def test_corrupt_download_is_not_cached(source, monkeypatch):
    cache, _, _, _ = source
    monkeypatch.setattr(loader, "VIDEO_SHA256", "incorrect")
    with pytest.raises(ValueError, match="Source video SHA256 mismatch"):
        loader.prepare()
    assert not (cache / loader.VIDEO).exists()
