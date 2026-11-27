"""HF loading and integrity checks using a tiny synthetic Parquet, not VQA data."""

import hashlib
import json
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from minisgl_examples.training.image_steering import dataset
from PIL import Image


@pytest.fixture
def hf_sample(tmp_path, monkeypatch):
    images = {}
    for side, color in (("before", "red"), ("after", "blue")):
        stream = BytesIO()
        Image.new("RGB", (3, 2), color).save(stream, format="PNG")
        payload = stream.getvalue()
        images[f"image_{side}"] = {
            "bytes": payload,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    row = {
        "id": dataset.SAMPLE_ID,
        "width": 3,
        "height": 2,
        "inputs": {**images, "text_shard_1": "Synthetic question", "text_shard_2": "New image"},
        "labels": {"answer_before": "8", "answer_after": "9"},
        "provenance_json": json.dumps({"source": "synthetic test"}),
    }
    path = tmp_path / "synthetic.parquet"
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    def write(rows):
        pq.write_table(pa.Table.from_pylist(rows), path)
        monkeypatch.setattr(
            dataset, "DATASET_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
        )
        dataset.load_sample.cache_clear()

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    write([row])
    yield row, path, calls, write
    dataset.load_sample.cache_clear()


def test_pinned_hf_record_supplies_all_task_inputs(hf_sample):
    row, _, calls, _ = hf_sample
    task = dataset.task()
    assert task["question"] == row["inputs"]["text_shard_1"]
    assert task["answer_before"] == "8" and task["answer_after"] == "9"
    assert task["provenance"] == {"source": "synthetic test"}
    assert "New image" in dataset.notification()
    with dataset.load_image("before") as image:
        assert image.size == (3, 2) and image.getpixel((0, 0)) == (255, 0, 0)
    with dataset.load_image("after") as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)
    assert calls == [
        {
            "repo_id": dataset.DATASET_REPO,
            "repo_type": "dataset",
            "filename": dataset.DATASET_FILE,
            "revision": dataset.DATASET_REVISION,
        }
    ]


def test_corrupt_download_is_rejected(hf_sample):
    _, path, _, _ = hf_sample
    path.write_bytes(b"corrupt download")
    with pytest.raises(ValueError, match="dataset SHA256 mismatch"):
        dataset.load_sample()


def test_corrupt_image_is_rejected(hf_sample):
    row, _, _, write = hf_sample
    row["inputs"]["image_after"]["sha256"] = "incorrect"
    write([row])
    with pytest.raises(ValueError, match="after image SHA256 mismatch"):
        dataset.load_sample()


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_record_is_rejected(hf_sample, duplicate):
    row, _, _, write = hf_sample
    if duplicate:
        write([row, row])
    else:
        row["id"] = "another-record"
        write([row])
    with pytest.raises(ValueError, match="exactly one dataset record"):
        dataset.load_sample()


def test_download_failure_does_not_fall_back_to_bundled_assets(hf_sample, monkeypatch):
    def fail(**kwargs):
        raise OSError("HF unavailable")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)
    with pytest.raises(OSError, match="HF unavailable"):
        dataset.load_sample()
