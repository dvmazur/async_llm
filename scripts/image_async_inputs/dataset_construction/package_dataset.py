"""Offline, self-contained Parquet export and model-input loader."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


IMAGE = pa.struct([("bytes", pa.binary()), ("sha256", pa.string())])
SCHEMA = pa.schema([
    ("id", pa.string()), ("dataset", pa.string()), ("source_id", pa.string()),
    ("source_split", pa.string()), ("category", pa.string()),
    ("difficulty_group", pa.string()), ("group_id", pa.string()),
    ("width", pa.int32()), ("height", pa.int32()),
    ("inputs", pa.struct([
        ("image_before", IMAGE), ("image_after", IMAGE),
        ("text_shard_1", pa.string()), ("text_shard_2", pa.string()),
    ])),
    ("labels", pa.struct([
        ("answer_before", pa.string()), ("answer_after", pa.string()),
        ("choices", pa.list_(pa.string())),
        ("evaluation_metadata_json", pa.string()),
    ])),
    ("provenance_json", pa.string()),
])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def read_records(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {r["id"]: r for r in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate IDs in {path.name}")
    return result


def image_info(data):
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        if im.format != "PNG" or im.mode != "RGB" or im.info:
            raise ValueError("Expected RGB PNG with empty embedded metadata")
        return im.size


def local_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes dataset directory")
    return path


def build_rows(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name in ("inputs", "annotations"):
        if digest((root / f"{name}.jsonl").read_bytes()) != manifest[f"{name}_sha256"]:
            raise ValueError(f"Manifest checksum mismatch: {name}")
    inputs = read_records(root / "inputs.jsonl")
    annotations = read_records(root / "annotations.jsonl")
    if inputs.keys() != annotations.keys() or len(inputs) != manifest["count"]:
        raise ValueError("Input/annotation IDs or manifest count do not match")
    rows = []
    for sid, inp in sorted(inputs.items()):
        ann = annotations[sid]
        src, review = ann["source"], ann["review"]
        if ann["status"] != "accepted" or review["decision"] != "accept":
            raise ValueError(f"Not accepted: {sid}")
        if inp["text_shard_1"] != src["question"] or review["answer_after"] != src["answer"]:
            raise ValueError(f"Question/answer mismatch: {sid}")
        packed = {k: inp[k] for k in ("text_shard_1", "text_shard_2")}
        sizes = []
        for state in ("before", "after"):
            norm = ann["normalization"]["images"][state]
            if inp[f"image_{state}"] != norm["path"]:
                raise ValueError(f"Normalized image path mismatch: {sid}")
            data = local_file(root, norm["path"]).read_bytes()
            if digest(data) != norm["sha256"]:
                raise ValueError(f"Image checksum mismatch: {sid}/{state}")
            sizes.append(image_info(data))
            packed[f"image_{state}"] = {"bytes": data, "sha256": digest(data)}
        if sizes[0] != sizes[1] or list(sizes[0]) != ann["normalization"]["target_size"]:
            raise ValueError(f"Image dimension mismatch: {sid}")
        # Explicit allowlist: never bundle credentials, API payloads or workspace paths.
        provenance = {k: src[k] for k in (
            "dataset", "source_id", "source_split", "source_url", "license",
            "source_revision", "original", "selection_seed", "selection_offset",
            "construction_notes") if k in src}
        provenance.update({"review": review, "proposal": ann["proposal"],
                           "raw_after_sha256": ann["after_sha256"],
                           "raw_before_sha256": ann["before_sha256"],
                           "processor_parity": ann["normalization"]["processor_parity"]})
        original = src["original"]
        eval_meta = {k: original[k] for k in (
            "unit", "precision", "answer_type", "question_type", "reasoning_a_type")
            if k in original}
        # Keep prior equivalence evidence, without inventing new answer aliases.
        checks_path = root / "evidence" / sid / "checks.json"
        if checks_path.exists():
            checks = json.loads(checks_path.read_text())
            eval_meta["validated_answers"] = {
                state: {k: checks[state][k] for k in (
                    "expected", "matches", "semantic_grade", "reviewed_equivalence")
                    if k in checks[state]} | {"observed_answer": checks[state]["result"]["answer"]}
                for state in ("before", "after")}
        rows.append({"id": sid, **{k: src[k] for k in (
            "dataset", "source_id", "source_split", "category", "difficulty_group")},
            "group_id": ann["group_id"], "width": sizes[0][0], "height": sizes[0][1],
            "inputs": packed,
            "labels": {"answer_before": review["answer_before"],
                       "answer_after": review["answer_after"],
                       "choices": original.get("choices") or original.get("options") or [],
                       "evaluation_metadata_json": dumps(eval_meta)},
            "provenance_json": dumps(provenance)})
    return rows, manifest


def validate_package(path):
    table = pq.read_table(path)
    if not table.schema.equals(SCHEMA, check_metadata=False):
        raise ValueError("Unsupported package schema")
    metadata = table.schema.metadata or {}
    if metadata.get(b"image_async.schema_version") != b"1":
        raise ValueError("Unsupported package version")
    manifest = json.loads(metadata[b"image_async.manifest"])
    rows = table.to_pylist()
    if len(rows) != manifest["count"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Invalid row count or duplicate IDs")
    for row in rows:
        for state in ("before", "after"):
            asset = row["inputs"][f"image_{state}"]
            if digest(asset["bytes"]) != asset["sha256"]:
                raise ValueError("Embedded image checksum mismatch")
            if image_info(asset["bytes"]) != (row["width"], row["height"]):
                raise ValueError("Embedded image dimension mismatch")
    return rows


def model_context(row, stage):
    """Return only current image and text, never labels or edit descriptions.

    Stage 2 describes the updated context, NOT a second image to append to history.
    The async evaluator must replace the old image and append the new text at k.
    """
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    inp = row["inputs"]
    state = "before" if stage == 1 else "after"
    with Image.open(io.BytesIO(inp[f"image_{state}"]["bytes"])) as im:
        picture = im.copy()
    text = inp["text_shard_1"]
    if stage == 2:
        text += "\n\n" + inp["text_shard_2"]
    return {"image": picture, "text": text}


def package(root, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rows, manifest = build_rows(root)
    metadata = {
        b"image_async.schema_version": b"1",
        b"image_async.manifest": dumps(manifest).encode(),
        b"image_async.readme": Path(__file__).with_name("PACKAGE.md").read_bytes(),
    }
    table = pa.Table.from_pylist(rows, schema=SCHEMA.with_metadata(metadata))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".package-", suffix=".parquet", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        pq.write_table(table, temporary, compression="zstd", row_group_size=1)
        validate_package(temporary)
        # Atomic publish without overwriting another process's output.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(output.resolve()), "count": len(rows),
            "bytes": output.stat().st_size, "sha256": digest(output.read_bytes())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/diverse_corrections_v2"))
    parser.add_argument("--output", type=Path, default=Path("datasets/diverse_corrections_v2.parquet"))
    parser.add_argument("--validate", type=Path, help="Validate a standalone package instead of building")
    args = parser.parse_args()
    if args.validate:
        print(dumps({"count": len(validate_package(args.validate)), "valid": True}))
    else:
        print(dumps(package(args.source, args.output)))
