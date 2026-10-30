"""CPU-only, standard-library construction pipeline. See PIPELINE.md."""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path


NOTICE = (
    "The earlier picture contained an error. It has now been corrected. "
    "Recheck your reasoning using the current picture."
)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record_path(root, sample_id):
    require(len(sample_id) == 16 and all(c in "0123456789abcdef" for c in sample_id),
            "Use the 16-character sample ID from the report.")
    return root / "records" / f"{sample_id}.json"


def records(root):
    return [read(p) for p in sorted((root / "records").glob("*.json"))]


def import_sources(root, manifest):
    """Validate the whole batch before copying any assets; imports are idempotent."""
    existing = {r["id"]: r for r in records(root)}
    planned = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        for key in ("dataset", "source_id", "source_split", "category", "question",
                    "answer", "image", "source_url", "license"):
            require(isinstance(source.get(key), str) and source[key].strip(),
                    f"Missing nonempty string: {key}")
        image = (manifest.parent / source["image"]).resolve()
        require(image.is_file(), f"Missing image: {image}")
        require(image.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"),
                "Use PNG, JPEG, or WebP source images.")
        identity = json.dumps([source[k] for k in ("dataset", "source_split", "source_id")])
        sample_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
        checksum = digest(image)
        source = {**source, "image": f"assets/{sample_id}/after{image.suffix.lower()}"}
        candidate = {"id": sample_id, "schema_version": 1, "status": "imported",
                     "source": source, "after_sha256": checksum,
                     "group_id": checksum, "history": ["imported"]}
        previous = existing.get(sample_id) or planned.get(sample_id, (None,))[0]
        if previous:
            require(previous["source"] == source and previous["after_sha256"] == checksum,
                    f"Source changed for {sample_id}; use a new workspace/version.")
            continue
        planned[sample_id] = (candidate, image)
    for sample_id, (candidate, image) in planned.items():
        destination = root / candidate["source"]["image"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, destination)
        save(record_path(root, sample_id), candidate)
    print(f"Imported {len(planned)} new sources; existing sources preserved.")


def transition(root, sample_id, command, payload):
    path = record_path(root, sample_id)
    record = read(path)
    if command == "propose":
        require(record["status"] == "imported", "Proposal requires imported status.")
        for key in ("edit_type", "changed_fact_before", "changed_fact_after",
                    "expected_answer_before", "solution_before", "solution_after",
                    "editing_method", "reasoning_change"):
            require(isinstance(payload.get(key), str) and payload[key].strip(),
                    f"Proposal needs {key}.")
        require(payload["expected_answer_before"] != record["source"]["answer"],
                "Answers must differ; reviewer must also check mathematical equivalence.")
        record["proposal"] = payload
        record["status"] = "edit_proposed"
    elif command == "attach":
        require(record["status"] == "edit_proposed", "Attach requires an edit proposal.")
        image = Path(payload["image"]).resolve()
        require(image.is_file(), "Edited image does not exist.")
        require(image.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"),
                "Use PNG, JPEG, or WebP edited images.")
        require(digest(image) != record["after_sha256"], "Before and after files are identical.")
        destination = root / "assets" / sample_id / f"before{image.suffix.lower()}"
        shutil.copyfile(image, destination)
        record["before_image"] = str(destination.relative_to(root))
        record["before_sha256"] = digest(destination)
        record["editing_approval"] = payload["approval_note"]
        record["status"] = "rendered"
    elif command == "review":
        require(record["status"] in ("rendered", "normalized"), "Review requires rendered or normalized status.")
        for key in ("reviewer", "notes", "answer_before", "answer_after", "decision"):
            require(isinstance(payload.get(key), str) and payload[key].strip(),
                    f"Review needs {key}.")
        require(payload["decision"] in ("accept", "reject"), "Decision: accept or reject.")
        if payload["decision"] == "accept":
            require(record["status"] == "normalized", "Acceptance requires normalization first.")
            for key in ("isolated_edit", "readable", "both_solvable", "visual_required",
                        "answers_distinct", "source_terms_checked", "same_dimensions"):
                require(payload.get(key) is True, f"Acceptance requires {key}=true.")
            require(payload["answer_before"] == record["proposal"]["expected_answer_before"],
                    "Review disagrees with proposed answer; reject and revise in a new version.")
            require(payload["answer_after"] == record["source"]["answer"],
                    "Review disagrees with source answer; reject and investigate.")
            record["normalization"]["semantic_review"] = "reviewer_accepted_normalized_pair"
        record["review"] = payload
        record["status"] = "accepted" if payload["decision"] == "accept" else "rejected"
    record["history"].append(record["status"])
    save(path, record)


def normalize_record(root, sample_id, max_aspect_change=0.01, alignment_review=None):
    from normalize_pair import normalize

    path = record_path(root, sample_id)
    record = read(path)
    require(record["status"] == "rendered", "Normalization requires rendered status.")
    require(digest(root / record["before_image"]) == record["before_sha256"] and
            digest(root / record["source"]["image"]) == record["after_sha256"],
            "Raw asset changed before normalization.")
    output = root / "assets" / sample_id / "normalized"
    require(max_aspect_change <= 0.01 or alignment_review,
            "Increasing aspect tolerance requires a recorded alignment review.")
    result = normalize(root / record["before_image"], root / record["source"]["image"], output,
                       max_aspect_change=max_aspect_change)
    if alignment_review:
        result["alignment_review"] = alignment_review
    for image in result["images"].values():
        image["path"] = str((output / image["path"]).relative_to(root))
    record["normalization"] = result
    record["status"] = "normalized"
    record["history"].append("normalized")
    save(path, record)


def report(root):
    rows = records(root)
    counts = {}
    cards = []
    for record in rows:
        status = record["status"]
        counts[status] = counts.get(status, 0) + 1
        source = record["source"]
        normalized = record.get("normalization", {}).get("images", {})
        pictures = "".join(
            f'<figure><figcaption>{stage}</figcaption><img src="{html.escape(image)}"></figure>'
            for stage, image in (("Before", normalized.get("before", {}).get("path", record.get("before_image"))),
                                  ("After", normalized.get("after", {}).get("path", source["image"]))) if image
        )
        cards.append(f'<article><h2>{record["id"]}: {html.escape(status)}</h2>'
                     f'<p>{html.escape(source["category"])} — {html.escape(source["question"])}</p>'
                     f'<div>{pictures}</div><pre>{html.escape(json.dumps(record, indent=2))}</pre>'
                     '</article>')
    (root / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Dataset construction review</title>'
        '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}'
        'article{border-top:1px solid #aaa;padding:20px 0}div{display:flex}'
        'figure{width:48%;margin:1%}img{width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Evaluator-only review packet</h1>' + "".join(cards))
    print(json.dumps({"total": len(rows), "status_counts": counts,
                      "review_packet": str(root / "review.html")}, indent=2))


def export(root, output):
    accepted = [r for r in records(root) if r["status"] == "accepted"]
    require(accepted, "No accepted pairs to export.")
    require(not output.exists(), "Export destination already exists; choose a new version.")
    # Check all source assets before starting an export.
    for record in accepted:
        require("normalization" in record, "Accepted legacy record must be normalized/reviewed before export.")
        for image, expected in ((record["source"]["image"], record["after_sha256"]),
                                (record["before_image"], record["before_sha256"]),
                                *((v["path"], v["sha256"]) for v in record["normalization"]["images"].values())):
            require(digest(root / image) == expected, f"Asset changed: {image}")
    output.mkdir(parents=True)
    inputs = []
    for record in accepted:
        source = record["source"]
        normalized = record["normalization"]["images"]
        for image in (source["image"], record["before_image"],
                      normalized["before"]["path"], normalized["after"]["path"]):
            target = output / image
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / image, target)
        inputs.append({"id": record["id"], "image_before": normalized["before"]["path"],
                       "image_after": normalized["after"]["path"], "text_shard_1": source["question"],
                       "text_shard_2": NOTICE})
    for filename, values in (("inputs.jsonl", inputs), ("annotations.jsonl", accepted)):
        (output / filename).write_text("".join(json.dumps(r) + "\n" for r in values))
    save(output / "manifest.json", {"schema_version": 1, "count": len(accepted),
                                    "correction_notice": NOTICE,
                                    "inputs_sha256": digest(output / "inputs.jsonl"),
                                    "annotations_sha256": digest(output / "annotations.jsonl")})
    print(f"Exported {len(accepted)} accepted pairs to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("report")
    commands.add_parser("import").add_argument("manifest", type=Path)
    normalizer = commands.add_parser("normalize")
    normalizer.add_argument("id")
    normalizer.add_argument("--max-aspect-change", type=float, default=0.01)
    normalizer.add_argument("--alignment-review")
    processor = commands.add_parser("processor-check")
    processor.add_argument("id")
    processor.add_argument("--model", required=True)
    processor.add_argument("--revision")
    for name in ("propose", "review"):
        sub = commands.add_parser(name)
        sub.add_argument("id")
        sub.add_argument("payload", type=Path)
    sub = commands.add_parser("attach")
    sub.add_argument("id")
    sub.add_argument("image", type=Path)
    sub.add_argument("--approval-note", required=True,
                     help="Record the actual user-agreed editing method; not a substitute for approval.")
    commands.add_parser("export").add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve()
    try:
        if args.command == "init":
            (root / "records").mkdir(parents=True, exist_ok=True)
        else:
            require((root / "records").is_dir(), "Initialize the workspace first.")
            if args.command == "import":
                import_sources(root, args.manifest.resolve())
            elif args.command == "report":
                report(root)
            elif args.command == "export":
                export(root, args.output.resolve())
            elif args.command == "normalize":
                normalize_record(root, args.id, args.max_aspect_change, args.alignment_review)
            elif args.command == "processor-check":
                from normalize_pair import processor_parity
                path = record_path(root, args.id)
                record = read(path)
                require(record["status"] == "normalized", "Processor check requires normalized status.")
                result = processor_parity(root / "assets" / args.id / "normalized", args.model, args.revision)
                record["normalization"]["processor_parity"] = result
                save(path, record)
            else:
                payload = ({"image": str(args.image), "approval_note": args.approval_note}
                           if args.command == "attach" else read(args.payload))
                transition(root, args.id, args.command, payload)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
