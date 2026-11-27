"""Load the pinned image pair from Hugging Face; keep dataset bytes out of Git."""

import hashlib
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

DATASET_REPO = "MrDarkTesla/sharded_vqa"
DATASET_REVISION = "e9b46c377fcf6e228d26028336d7bb2d806df904"
DATASET_FILE = "sharded_vqa.parquet"
DATASET_SHA256 = "18125e129d92e04f459a9ee02cb6282c3372e3953a213c983026c7dadbe82a35"
SAMPLE_ID = "0f6c4193c4c4860c"


@lru_cache(maxsize=1)
def load_sample():
    """Download to the HF cache and read only the selected record into Python."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=DATASET_FILE,
            revision=DATASET_REVISION,
        )
    )
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != DATASET_SHA256:
        raise ValueError("Hugging Face dataset SHA256 mismatch")
    rows = pq.read_table(path, filters=[("id", "=", SAMPLE_ID)]).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one dataset record for {SAMPLE_ID}")
    sample = rows[0]
    for side in ("before", "after"):
        image = sample["inputs"][f"image_{side}"]
        if hashlib.sha256(image["bytes"]).hexdigest() != image["sha256"]:
            raise ValueError(f"Dataset {side} image SHA256 mismatch")
    return sample


def task():
    sample = load_sample()
    return {
        "id": sample["id"],
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "file": DATASET_FILE,
            "sha256": DATASET_SHA256,
        },
        "question": sample["inputs"]["text_shard_1"],
        "answer_before": sample["labels"]["answer_before"],
        "answer_after": sample["labels"]["answer_after"],
        "images": {
            side: sample["inputs"][f"image_{side}"]["sha256"] for side in ("before", "after")
        },
        "provenance": json.loads(sample["provenance_json"]),
    }


def load_image(side):
    """Decode directly from HF dataset bytes, without extracting image files."""
    from PIL import Image

    if side not in {"before", "after"}:
        raise ValueError("Image side must be before/after")
    sample = load_sample()
    with Image.open(BytesIO(sample["inputs"][f"image_{side}"]["bytes"])) as image:
        image.load()
        if image.size != (sample["width"], sample["height"]):
            raise ValueError("Dataset image dimensions mismatch")
        return image.convert("RGB")


def notification():
    text = load_sample()["inputs"]["text_shard_2"]
    return (
        "</think><|im_end|>\n<|im_start|>user\n"
        + text
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
