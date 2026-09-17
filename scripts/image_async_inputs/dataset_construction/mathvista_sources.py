"""Download pinned MathVista testmini and select reproducible real-source candidates."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import urllib.request

from PIL import Image
import pyarrow.parquet as pq

from pipeline import digest, import_sources, require, save


REPO = "AI4Math/MathVista"


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def category(row):
    context = row["metadata"]["context"].lower()
    if "geometry" in context:
        return "geometry"
    if "table" in context:
        return "table"
    if "chart" in context or "plot" in context:
        return "chart"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many per category for next batch.")
    args = parser.parse_args()
    require(args.per_category > 0 and args.offset >= 0, "Invalid selection size/offset.")
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache / "source_lock.json"
    if lock.exists():
        spec = json.loads(lock.read_text())
    else:
        revision = fetch(f"https://huggingface.co/api/datasets/{REPO}")["sha"]
        tree = fetch(f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/data")
        file = next(r for r in tree if r["path"].startswith("data/testmini-")
                    and r["path"].endswith(".parquet"))
        spec = {"repo": REPO, "revision": revision, "file": file["path"],
                "sha256": file["lfs"]["oid"]}
        save(lock, spec)
    parquet = cache / Path(spec["file"]).name
    if not parquet.exists():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{spec['revision']}/{spec['file']}"
        temporary = parquet.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        require(digest(temporary) == spec["sha256"], "Downloaded source hash mismatch.")
        temporary.replace(parquet)
    require(digest(parquet) == spec["sha256"], "Cached source hash mismatch.")
    table = pq.read_table(parquet)
    metadata = table.drop(["decoded_image"]).to_pylist()
    print("Source contexts:", dict(Counter(r["metadata"]["context"] for r in metadata)), flush=True)
    selected = []
    for name in ("chart", "geometry", "table"):
        eligible = [(i, r) for i, r in enumerate(metadata) if category(r) == name
                    and r["answer"] and r["metadata"]["language"].lower() == "english"]
        # Prefer math-targeted problems, but retain all as candidates for model screening.
        eligible.sort(key=lambda pair: (
            pair[1]["metadata"]["category"] != "math-targeted-vqa",
            hashlib.sha256(f"{args.seed}:{pair[1]['pid']}".encode()).hexdigest()))
        selected.extend((name, i, r) for i, r in eligible[args.offset:args.offset + args.per_category])
    root = args.workspace.resolve()
    require(not root.exists(), "Use a new workspace for each candidate batch.")
    (root / "records").mkdir(parents=True)
    staging = root / "sources"
    staging.mkdir()
    normalized = []
    for name, index, row in selected:
        raw = table["decoded_image"][index].as_py()["bytes"]
        image = Image.open(io.BytesIO(raw))
        image.load()
        ext = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(image.format)
        require(ext, "Unexpected source image format; preserve source bytes explicitly.")
        image_path = staging / f"{row['pid']}{ext}"
        image_path.write_bytes(raw)
        question = row["question"]
        if row.get("choices"):
            question += "\nChoices:\n" + "\n".join(
                f"({chr(65 + i)}) {choice}" for i, choice in enumerate(row["choices"]))
        if row.get("unit"):
            question += f"\nAnswer unit: {row['unit']}."
        if row.get("precision") is not None:
            question += f"\nDecimal places: {int(row['precision'])}."
        normalized.append({"dataset": REPO, "source_id": str(row["pid"]),
                           "source_split": "testmini", "category": name,
                           "question": question, "answer": str(row["answer"]),
                           "image": image_path.name,
                           "source_url": f"https://huggingface.co/datasets/{REPO}/blob/{spec['revision']}/README.md",
                           "license": "MathVista CC-BY-SA-4.0 contributions; underlying source image/question rights retained; evaluation use only",
                           "source_revision": spec["revision"], "original": row,
                           "selection_seed": args.seed, "selection_offset": args.offset})
    manifest = staging / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in normalized))
    import_sources(root, manifest)
    save(root / "selection.json", {"source": spec, "seed": args.seed, "offset": args.offset,
                                   "selected": [{"pid": r["source_id"], "category": r["category"]}
                                                for r in normalized]})
    print(f"Prepared {len(normalized)} real MathVista candidates in {root}")


if __name__ == "__main__":
    main()
