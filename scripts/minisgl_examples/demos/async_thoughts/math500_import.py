"""Import verified MATH-500 rows and bind their continuation to a new GPU session.

Run `prepare` against the last archived evaluation and its measured source files.
Supply `--expect-rows N` to assert a known prefix length. Use a new directory and
keep the prior evaluation intact. Copy the complete prepared directory to the new
host, including its origin receipts. A GPU UUID in CUDA_VISIBLE_DEVICES identifies
the physical device across sessions. With that variable set, run `bind`
with a unique session ID and the installed measured source directory. Then launch
the unchanged math500 evaluator with --paired N --limit N and the original budget,
context, model revision and memory ratio. Its normal resume path skips the imported
rows. math500_report checks both sessions and groups timings by acquisition session.

If a continuation stops early, repeat `prepare` with that continuation as --origin
and its bound session ID as --origin-session. Retain each evaluation as an audit
artifact. Previously imported rows retain the session that first generated them.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from pathlib import Path

from .math500 import digest, encode_json, ordered_rows, save, schedule


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def checked_copy(src, dst, expected):
    data = src.read_bytes()
    require(digest(data) == expected, f"Input changed while copying {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("xb") as stream:
        stream.write(data)


def check_sources(root, manifest):
    expected = manifest["source_sha256"]
    require(
        set(expected) == {"math500.py", "demo.py", "engine.py", "short_retry.py"},
        "Unexpected measured source catalog",
    )
    for name, sha in expected.items():
        require(digest((root / name).read_bytes()) == sha, f"Measured source changed: {name}")


def target_settings(origin, paired, visible_devices, torch_version):
    require(0 < paired <= len(origin["ordered_ids"]), "Invalid paired continuation size")
    return {
        **origin,
        "paired": paired,
        "limit": paired,
        "ordered_ids": origin["ordered_ids"][:paired],
        "visible_devices": visible_devices,
        "torch": torch_version,
    }


def stems(settings):
    return [
        f"{rank:03d}.{arm}"
        for rank, _, arm in schedule(settings["ordered_ids"], settings["paired"], settings["limit"])
    ]


def prepare(
    origin,
    destination,
    dataset,
    tokenizer,
    sources,
    origin_session,
    prior_report,
    paired=64,
    expect_rows=None,
):
    from .math500_report import report

    require(
        origin_session and not destination.resolve().is_relative_to(origin.resolve()),
        "Use a named origin and a separate destination",
    )
    if destination.exists():
        raise FileExistsError(f"Use a fresh import destination: {destination}")
    source = read(origin / "manifest.json")
    check_sources(sources, source)
    origin_rows = {p.stem: digest(p.read_bytes()) for p in (origin / "rows").glob("*.json")}
    verified = report(origin, dataset, tokenizer, sources)
    upstream = verified.get("session_provenance")
    if upstream:
        require(
            upstream["destination_session"] == origin_session,
            "Origin ID differs from its bound session",
        )
    if prior_report:
        prior = read(prior_report)
        require(
            prior["settings_sha256"] == verified["settings_sha256"]
            and prior["schedule"] == verified["schedule"]
            and prior["cases"] == verified["cases"],
            "Prior report differs from the independently verified snapshot",
        )

    new = target_settings(source, paired, source["visible_devices"], source["torch"])
    ranked = ordered_rows(dataset)
    planned = [f"{r:03d}.{arm}" for r, _, arm in schedule(ranked, paired, paired)]
    cases = {f"{c['rank']:03d}.{c['arm']}": c for c in verified["cases"] if c["rank"] < paired}
    require(
        (expect_rows is None or len(cases) == expect_rows)
        and set(cases) == set(planned[: len(cases)]),
        "Imported cases must form the expected prefix of the new schedule",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".math500-import-", dir=destination.parent
    ) as temporary:
        work = Path(temporary) / "evaluation"
        (work / "origin").mkdir(parents=True)
        (work / "rows").mkdir()
        (work / "traces").mkdir()
        catalog = {}
        for name in ("manifest.json", "environment.json", "smoke.json"):
            sha = digest((origin / name).read_bytes())
            checked_copy(origin / name, work / "origin" / name, sha)
            catalog[name] = sha
        if upstream:
            for name, src_name in (
                ("prior_import.json", "imported.json"),
                ("prior_session.json", "session.json"),
            ):
                sha = digest((origin / src_name).read_bytes())
                checked_copy(origin / src_name, work / "origin" / name, sha)
                catalog[name] = sha
        # Preserve the verified input snapshot even when the original report also contained
        # unrelated historical comparisons. The row records are checked against it above.
        save(work / "origin" / "report.json", encode_json(verified))
        catalog["report.json"] = digest((work / "origin" / "report.json").read_bytes())
        rows = []
        for stem in planned[: len(cases)]:
            row = origin / "rows" / (stem + ".json")
            row_sha = origin_rows[stem]
            checked_copy(row, work / "rows" / row.name, row_sha)
            case = cases[stem]
            copied = read(work / "rows" / row.name)
            require(
                all(copied[key] == case[key] for key in case if key in copied),
                f"Verified case changed: {stem}",
            )
            packed = origin / "traces" / (stem + ".json.gz")
            checked_copy(packed, work / "traces" / packed.name, case["trace_sha256"])
            rows.append(
                {
                    "stem": stem,
                    "rank": case["rank"],
                    "arm": case["arm"],
                    "unique_id": case["unique_id"],
                    "row_sha256": row_sha,
                    "trace_sha256": case["trace_sha256"],
                }
            )
            if upstream:
                rows[-1]["acquired_session"] = case["session"]
        receipt = {
            "format_version": 1,
            "origin_session": origin_session,
            "origin_eval": str(origin.resolve()),
            "origin_catalog": catalog,
            "origin_settings_sha256": verified["settings_sha256"],
            "origin_environment": read(origin / "environment.json"),
            "target_paired": paired,
            "target_planned_rows": len(stems(new)),
            "imported_rows": len(rows),
            "rows": rows,
        }
        save(work / "imported.json", encode_json(receipt))
        validate_import(work)
        work.rename(destination)
    return receipt


def validate_import(directory):
    receipt = read(directory / "imported.json")
    require(receipt["format_version"] == 1 and receipt["origin_session"], "Invalid origin session")
    catalog = receipt["origin_catalog"]
    originals = {"manifest.json", "environment.json", "smoke.json", "report.json"}
    prior_files = {"prior_import.json", "prior_session.json"}
    require(set(catalog) in (originals, originals | prior_files), "Invalid origin evidence catalog")
    for name, sha in catalog.items():
        require(
            digest((directory / "origin" / name).read_bytes()) == sha,
            f"Origin evidence hash mismatch: {name}",
        )
    origin = read(directory / "origin/manifest.json")
    proof = read(directory / "origin/report.json")
    upstream = proof.get("session_provenance")
    require(bool(upstream) == ("prior_import.json" in catalog), "Origin session lineage missing")
    if upstream:
        parent_binding = read(directory / "origin/prior_session.json")
        require(
            upstream["destination_session"]
            == receipt["origin_session"]
            == parent_binding["session"]
            and upstream["import_receipt_sha256"]
            == catalog["prior_import.json"]
            == parent_binding["import_receipt_sha256"]
            and parent_binding["target_settings_sha256"] == receipt["origin_settings_sha256"],
            "Origin acquisition lineage mismatch",
        )
    require(
        proof["settings_sha256"]
        == digest(encode_json(origin))
        == receipt["origin_settings_sha256"],
        "Origin manifest receipt mismatch",
    )
    require(
        receipt["origin_environment"]
        == read(directory / "origin/environment.json")
        == proof["environment"],
        "Origin hardware receipt mismatch",
    )
    records = receipt["rows"]
    target = target_settings(
        origin, receipt["target_paired"], origin["visible_devices"], origin["torch"]
    )
    planned = stems(target)
    source_cases = {
        f"{c['rank']:03d}.{c['arm']}": c
        for c in proof["cases"]
        if c["rank"] < receipt["target_paired"]
    }
    require(
        receipt["imported_rows"] == len(records) == len(source_cases)
        and receipt["target_planned_rows"] == len(planned)
        and [r["stem"] for r in records] == planned[: len(records)],
        "Wrong imported schedule",
    )
    for row in records:
        stem = row["stem"]
        c = source_cases[stem]
        require(
            all(row[key] == c[key] for key in ("unique_id", "rank", "arm", "trace_sha256")),
            "Wrong origin case",
        )
        acquired = row.get("acquired_session", receipt["origin_session"])
        require(
            acquired == c.get("session", receipt["origin_session"]),
            "Imported row was attributed to a different acquisition session",
        )
        require(
            digest((directory / "rows" / (stem + ".json")).read_bytes()) == row["row_sha256"]
            and digest((directory / "traces" / (stem + ".json.gz")).read_bytes())
            == row["trace_sha256"],
            f"Imported row changed: {stem}",
        )
    return receipt, origin


def origin_sessions(directory, receipt, origin):
    """Recover acquisition identities from the verified input report, including early v1 reports."""
    upstream = read(directory / "origin/report.json").get("session_provenance")
    latest = {
        "environment": receipt["origin_environment"],
        "torch": origin["torch"],
        "visible_devices": origin["visible_devices"],
        "manifest_sha256": receipt["origin_settings_sha256"],
    }
    if upstream is None:
        return {receipt["origin_session"]: latest}
    if "sessions" in upstream:
        sessions = dict(upstream["sessions"])
    else:
        sessions = {
            upstream[f"{role}_session"]: {
                "environment": upstream[f"{role}_environment"],
                "torch": upstream[f"{role}_torch"],
                "visible_devices": upstream[f"{role}_visible_devices"],
                "manifest_sha256": upstream[f"{role}_manifest_sha256"],
            }
            for role in ("origin", "destination")
        }
    require(
        sessions.get(receipt["origin_session"]) == latest,
        "Latest source session metadata disagrees with origin evidence",
    )
    return sessions


def bind(directory, sources, session, visible_devices, torch_version, gpu, cuda, model=None):
    require(
        session
        and gpu
        and cuda
        and torch_version
        and visible_devices
        and "," not in visible_devices,
        "A named destination session with exactly one visible GPU is required",
    )
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt, origin = validate_import(directory)
        require(
            session not in origin_sessions(directory, receipt, origin),
            "A prior session ID cannot be rebound",
        )
        check_sources(sources, origin)
        if model:
            snapshot = model.resolve()
            require(
                snapshot.name == origin["model_revision"]
                and snapshot.parent.name == "snapshots"
                and snapshot.parent.parent.name == origin["model"],
                "Destination model path differs from original Hub checkpoint",
            )
        manifest = target_settings(origin, receipt["target_paired"], visible_devices, torch_version)
        if not (directory / "session.json").exists():
            require(
                {p.stem for p in (directory / "rows").glob("*.json")}
                == {r["stem"] for r in receipt["rows"]}
                and not any(
                    (directory / name).exists()
                    for name in ("smoke.json", "environment.json", "completed.json")
                ),
                "Bind the new session before generating new rows",
            )
        info = {
            "session": session,
            "gpu": gpu,
            "cuda": cuda,
            "torch": torch_version,
            "visible_devices": visible_devices,
            "target_settings_sha256": digest(encode_json(manifest)),
            "import_receipt_sha256": digest((directory / "imported.json").read_bytes()),
        }
        files = [("session.json", info), ("manifest.json", manifest)]
        for name, value in files:
            path = directory / name
            if path.exists():
                require(read(path) == value, f"Refusing to replace an incompatible {path}")
        for name, value in files:
            path = directory / name
            if not path.exists():
                save(path, encode_json(value))
    return info


def session_provenance(directory, settings, environment, cases):
    receipt, origin = validate_import(directory)
    info = read(directory / "session.json")
    sessions = origin_sessions(directory, receipt, origin)
    require(
        info["session"] not in sessions
        and info["import_receipt_sha256"] == digest((directory / "imported.json").read_bytes()),
        "Destination session receipt mismatch",
    )
    require(
        settings
        == target_settings(origin, receipt["target_paired"], info["visible_devices"], info["torch"])
        and info["target_settings_sha256"] == digest(encode_json(settings)),
        "Destination source/model/schedule differs from binding",
    )
    require(
        environment["gpu"] == info["gpu"]
        and environment["cuda"] == info["cuda"]
        and environment["eos_token_ids"] == receipt["origin_environment"]["eos_token_ids"],
        "Observed destination GPU, CUDA or EOS do not match the bound sessions",
    )
    imported = {
        r["stem"]: r.get("acquired_session", receipt["origin_session"]) for r in receipt["rows"]
    }
    keys = {f"{case['rank']:03d}.{case['arm']}" for case in cases}
    require(
        len(keys) == len(cases)
        and set(imported).issubset(keys)
        and set(imported.values()).issubset(sessions),
        "Combined report omitted imported rows or their acquisition session",
    )
    for case in cases:
        stem = f"{case['rank']:03d}.{case['arm']}"
        case["imported"] = stem in imported
        case["session"] = imported.get(stem, info["session"])
    sessions[info["session"]] = {
        "environment": environment,
        "torch": info["torch"],
        "visible_devices": info["visible_devices"],
        "manifest_sha256": info["target_settings_sha256"],
    }
    return {
        "origin_session": receipt["origin_session"],
        "destination_session": info["session"],
        "origin_environment": receipt["origin_environment"],
        "origin_torch": origin["torch"],
        "origin_visible_devices": origin["visible_devices"],
        "destination_environment": environment,
        "destination_torch": info["torch"],
        "destination_visible_devices": info["visible_devices"],
        "origin_manifest_sha256": receipt["origin_settings_sha256"],
        "destination_manifest_sha256": info["target_settings_sha256"],
        "import_receipt_sha256": info["import_receipt_sha256"],
        "sessions": sessions,
        "imported_rows": len(imported),
        "new_rows": len(cases) - len(imported),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for arg in ("origin", "destination", "dataset", "tokenizer", "source-dir"):
        prep.add_argument("--" + arg, type=Path, required=True)
    prep.add_argument("--origin-session", required=True)
    prep.add_argument("--prior-report", type=Path)
    prep.add_argument("--paired", type=int, default=64)
    prep.add_argument(
        "--expect-rows", type=int, help="Fail if the verified prefix has a different length."
    )
    run = sub.add_parser("bind")
    for arg in ("eval-dir", "source-dir"):
        run.add_argument("--" + arg, type=Path, required=True)
    run.add_argument("--session", required=True)
    run.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        result = prepare(
            args.origin,
            args.destination,
            args.dataset,
            tok,
            args.source_dir,
            args.origin_session,
            args.prior_report,
            args.paired,
            args.expect_rows,
        )
        print(
            json.dumps(
                {k: result[k] for k in ("origin_session", "imported_rows", "target_planned_rows")}
            )
        )
    else:
        import torch

        require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Expose one GPU before binding",
        )
        print(
            json.dumps(
                bind(
                    args.eval_dir,
                    args.source_dir,
                    args.session,
                    os.environ.get("CUDA_VISIBLE_DEVICES"),
                    str(torch.__version__),
                    torch.cuda.get_device_name(),
                    torch.version.cuda,
                    args.model,
                )
            )
        )


if __name__ == "__main__":
    main()
