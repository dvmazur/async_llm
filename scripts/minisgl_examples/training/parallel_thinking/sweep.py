"""Run the MATH-500 comparison on selected physical GPUs, checking each completed trace."""

import argparse
import fcntl
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import NamedTuple
from uuid import uuid4

from . import report

ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = ROOT / "run_hogwild.sh"
DEFAULT_GPUS = (1, 2)
CONDITIONS = report.CONDITIONS
BUDGET, SEED, TOTAL = 16384, 42, 500
# These matched MATH-500 IDs come from the repaired H2 (12/12) and H4/H16
# (6/6) pilot. Ordinary Qwen also solved all six wider-pilot problems.
H2_PILOT = (0, 1, 2, 45, 109, 149, 226, 261, 278, 288, 297, 306)
WIDE_PILOT = (62, 85, 156, 227, 230, 314)
ORDER = (("native", 1), ("hogwild", 2), ("hogwild", 1), *(("hogwild", n) for n in (4, 6, 8, 16)))
DISK_RESERVE = 2 * 1024**3
HOLD = "quality-hold.json"
STOP_MARKERS = (HOLD, "multiworker-stopped.json", "superseded.json")
POLL_SECONDS = 5
REPEATED_TOKEN_LIMIT = 128


class MultiWorkerProtocol(NamedTuple):
    layout: str = "native"
    answer_cue: str = "qwen"


class IncompatibleRun(ValueError):
    """Existing attempts belong to a different evaluator build or protocol."""


class GPUsUnavailable(RuntimeError):
    """No permitted free device remains for pending work; completed cases are preserved."""


def check_run_open(root):
    for name in STOP_MARKERS:
        if (root / name).exists():
            raise ValueError(f"This run is closed to new attempts: {root / name}")


def selected_gpus(gpus):
    devices = tuple(gpus)
    if not devices or any(type(gpu) is not int or gpu < 0 for gpu in devices) or len(set(devices)) != len(devices):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    return devices


def log(message):
    print(message, flush=True)


def source_hashes(save_traces=False):
    scripts = [Path(__file__).with_name(name)
               for name in ("eval.py", "rollout.py", "native.py", "single_worker.py", "math500.py")]
    if save_traces:
        scripts.append(Path(__file__).with_name("trace_io.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (*scripts, LAUNCHER)}


def snapshot_sources(root, *, verified_import=False):
    """Freeze this checkout before any evaluator starts, and verify it on resume."""
    files = set(Path(__file__).parent.glob("*.py"))
    files.update(p for p in (ROOT / "python/minisgl").rglob("*")
                 if p.is_file() and p.suffix in {".py", ".cu", ".cuh", ".h", ".cpp"})
    files.update(ROOT / name for name in (
        "run_hogwild.sh", "run_hogwild_budget.sh", "pyproject.toml", "uv.lock",
        "scripts/pyproject.toml", "docs/hogwild/requirements.txt",
    ))
    hashes = {str(p.relative_to(ROOT)): report.digest(p.read_bytes()) for p in sorted(files)}
    manifest = root / "source-files.json"
    active = root / "active-source-files.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != hashes or not active.exists() or json.loads(active.read_text()) != hashes:
            raise IncompatibleRun("Source or environment changed since this run was created")
        for relative, expected in hashes.items():
            if report.digest((root / "source-snapshot" / relative).read_bytes()) != expected:
                raise IncompatibleRun(f"Saved source snapshot changed: {relative}")
        return
    if active.exists() or (not verified_import and any((root / "shards").glob("*/config.json"))):
        raise IncompatibleRun("Existing results require their original source manifests")
    for relative in hashes:
        target = root / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    encoded = json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    report.atomic_write(active, encoded)
    report.atomic_write(manifest, encoded)


def verify_source_snapshot(root):
    manifest = root / "active-source-files.json"
    if manifest.exists():
        for relative, expected in json.loads(manifest.read_text()).items():
            if report.digest((ROOT / relative).read_bytes()) != expected:
                raise IncompatibleRun(f"Source changed during this sweep: {relative}")


def current_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def check_native_reuse_build(hashes):
    import ast

    report.validate_target_sources(hashes)
    # Python imports these files by content. Shared-checkout executable bits
    # may differ; content changes, symlink changes and deletions still fail.
    subprocess.run(["git", "-c", "core.filemode=false", "diff", "--quiet", report.REFERENCE_COMMIT, "--", "python/minisgl",
                    "pyproject.toml", "uv.lock", "scripts/pyproject.toml"], cwd=ROOT, check=True)
    # native.py imports the answer extractor from the otherwise changed rollout.
    source = Path(__file__).with_name("rollout.py").read_text()
    extractor = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "extract_boxed_answer")
    if report.digest(ast.get_source_segment(source, extractor).encode()) != "d88eb77230ce7b271bb008eacfc16def6f2c0ee2faa17622bfccd539b30daaac":
        raise ValueError("Native boxed-answer extraction changed since the audited comparison")


def import_native(old_root: Path, new_root: Path, hashes: dict, multi_worker=MultiWorkerProtocol()):
    """Freeze the eligible native cases as one atomic, auditable snapshot."""
    old, new = old_root.resolve(), new_root.resolve()
    destination = new / report.NATIVE_SNAPSHOT
    if new.is_relative_to(old) or old.is_relative_to(new):
        raise ValueError("Native source and destination must be separate run directories")
    check_native_reuse_build(hashes)
    if destination.exists():
        keys, meta = report.verify_native_reuse(new, hashes)
        if meta["source_run"] != str(old):
            raise ValueError(f"Existing native snapshot is from {meta['source_run']}, not {old}")
        return keys
    new.mkdir(parents=True, exist_ok=True)
    catalog = report.load_native_catalog()
    raw_catalog = json.dumps([catalog[i] for i in range(len(catalog))], ensure_ascii=False, separators=(",", ":")).encode()
    with TemporaryDirectory(dir=new.parent, prefix=".hogwild-native-import-") as tmp:
        stage = Path(tmp)
        base = stage / report.NATIVE_SNAPSHOT
        base.mkdir()
        (base / "catalog.json").write_bytes(raw_catalog)
        entries = []
        for manifest in sorted((old / "shards").glob("*/config.json")):
            raw_config = manifest.read_bytes()
            if json.loads(raw_config)["mode"] != "native":
                continue
            results = manifest.with_name("results.jsonl")
            completed = manifest.with_name("completed.json")
            if not results.exists() and not completed.exists():
                continue
            data = results.read_bytes()
            if not data and not completed.exists():
                continue
            relative = manifest.relative_to(old)
            target = base / relative
            target.parent.mkdir(parents=True)
            target.write_bytes(raw_config)
            target.with_name("results.jsonl").write_bytes(data)
            entry = {
                "source_manifest": str(relative), "manifest_sha256": report.digest(raw_config),
                "results_sha256": report.digest(data), "cases": len(data.splitlines()), "completed_sha256": None,
            }
            if completed.exists():
                marker = completed.read_bytes()
                target.with_name("completed.json").write_bytes(marker)
                entry["completed_sha256"] = report.digest(marker)
            entries.append(entry)
        meta = {"snapshot_format": 1, "cases": sum(entry["cases"] for entry in entries),
                "source_commit": report.REFERENCE_COMMIT, "source_run": str(old),
                "target_source_sha256": hashes, "catalog_sha256": report.digest(raw_catalog), "shards": entries}
        (base / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
        keys, _ = report.verify_native_reuse(stage, hashes)
        if (new / "shards").exists() and keys & read_existing(new / "shards", hashes, multi_worker=multi_worker):
            raise ValueError("A native case in the requested snapshot already exists in the corrected run")
        base.rename(destination)
    log(f"Imported {len(keys)} verified native rows from {old} into {destination}")
    return keys


def done_with_native_reuse(root, hashes, conditions, multi_worker=MultiWorkerProtocol()):
    done = read_existing(root / "shards", hashes, conditions, multi_worker)
    if (root / "reused-native").exists():
        if ("native", 1) not in conditions:
            raise ValueError("An isolated one-worker run cannot contain a native baseline import")
        check_native_reuse_build(hashes)
        imported, _ = report.verify_native_reuse(root, hashes)
        if done & imported:
            raise ValueError(f"Duplicate executed/imported case {min(done & imported)}")
        done.update(imported)
    return done


def check_attempt(directory, manifest, hashes, conditions, multi_worker):
    version = 4 if any(Path(p).name == "trace_io.py" for p in hashes) else 3
    if report.manifest_version(manifest) != version:
        raise IncompatibleRun(f"Cannot resume a different benchmark protocol; expected v{version}: {directory}")
    report.check_config(manifest)
    workers = manifest.get("workers", [])
    condition = (manifest.get("mode"), workers[0] if len(workers) == 1 else None)
    if condition not in conditions:
        raise IncompatibleRun(f"Unexpected condition {condition} in {directory}")
    one_worker = condition == ("hogwild", 1)
    multiple = condition[1] > 1
    required = {
        "cuda_graphs": True,
        "single_worker_layout": "causal" if one_worker else None,
        "multi_worker_layout": multi_worker.layout if multiple else None,
        "multi_worker_answer_cue": multi_worker.answer_cue if multiple else None,
        "answer_selection_policy": "eos_worker_last_box_v1" if multiple and multi_worker.layout == "native" else None,
        "publication_policy": "encoded_separator_v1" if multiple and multi_worker.layout == "native" else None,
        "context_cap": min(manifest["native_context"], BUDGET * condition[1] + 8192),
    }
    if report.canonical_sources(manifest["source_sha256"]) != report.canonical_sources(hashes):
        raise IncompatibleRun(f"Unexpected source_sha256 in {directory}: {manifest['source_sha256']!r}")
    for key, value in required.items():
        if (key not in manifest and key not in ("answer_selection_policy", "publication_policy")) or manifest.get(key) != value:
            raise IncompatibleRun(f"Unexpected {key} in {directory}: {manifest.get(key)!r}")


def attempt_rows(directory, hashes, conditions=CONDITIONS, multi_worker=MultiWorkerProtocol(), *, monitor=None):
    """Verify every committed row against its actual on-disk trace, including a resumed attempt."""
    manifest = json.loads((directory / "config.json").read_text())
    check_attempt(directory, manifest, hashes, conditions, multi_worker)
    traced = report.manifest_version(manifest) == 4
    catalog = (monitor.catalog if monitor is not None else report.load_native_catalog()) if traced else None
    rows = set()
    path = directory / "results.jsonl"
    if path.exists():
        for line_number, row in report.read_results(path, partial=not (directory / "completed.json").exists()):
            repetitions = ()
            try:
                report.check_task(row, manifest)
                if traced:
                    if monitor is not None:
                        repetitions = verify_monitored_case(directory, row, manifest, catalog)
                    else:
                        report.verify_traced_case(directory, row, manifest, catalog)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid saved result at {path}:{line_number}: {exc}") from exc
            key = (row["mode"], row["workers"], row["index"])
            if key in rows:
                raise ValueError(f"Duplicate or unexpected result {key} in {path}")
            rows.add(key)
            if monitor is not None:
                monitor.observe(directory, row, repetitions)
    completed = directory / "completed.json"
    if completed.exists() and (json.loads(completed.read_text())["cases"] != len(rows) or len(rows) != len(manifest["indices"])):
        raise ValueError(f"Incomplete attempt marked completed: {directory}")
    return rows


def read_existing(shards, hashes, conditions=CONDITIONS, multi_worker=MultiWorkerProtocol(), *, monitor=None):
    done = set()
    for attempt in sorted(p for p in shards.iterdir() if p.is_dir()):
        if not (attempt / "config.json").exists() and not (attempt / "results.jsonl").exists():
            continue  # Interrupted before the evaluator wrote its manifest.
        rows = attempt_rows(attempt, hashes, conditions, multi_worker, monitor=monitor)
        conflicts = done & rows
        if conflicts:
            raise ValueError(f"Duplicate result {min(conflicts)} in {attempt}")
        done.update(rows)
    return done


def gpu_available(gpu):
    """Check before starting a subprocess; the evaluator checks again immediately before loading."""
    fields = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(", ")
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True).strip()
    index, uuid, memory, utilization = fields
    if int(index) != gpu or not uuid.startswith("GPU-"):
        raise RuntimeError(f"Expected physical GPU {gpu}, got {fields}")
    # Match the evaluator's allowance for idle driver memory.
    return not processes and 0 <= int(memory) <= 8 and int(utilization) == 0


def pilot_indices(condition):
    if condition == ("hogwild", 2):
        return H2_PILOT
    if condition[1] == 1:
        # Both baseline and H1 solved these six H2 cases and the six wider
        # cases, permitting direct paired checks across all seven conditions.
        return (*WIDE_PILOT, 0, 2, 45, 109, 149, 226)
    return WIDE_PILOT


def repeated_sampled_tokens(streams):
    """Describe each worker's longest consecutive run, never shared-history text."""
    evidence = []
    for worker, tokens in streams.items():
        longest = length = 0
        previous = None
        token_id = end = None
        for index, token in enumerate(tokens):
            length = length + 1 if token == previous else 1
            previous = token
            if length > longest:
                longest, token_id, end = length, token, index + 1
        if longest >= REPEATED_TOKEN_LIMIT:
            evidence.append({"kind": "identical_token_run", "worker": worker, "token_id": token_id, "run_length": longest,
                             "start_index": end - longest, "end_index_exclusive": end,
                             "threshold": REPEATED_TOKEN_LIMIT, "stream": "worker_token_ids"})
    return evidence


def repeated_sampled_phrases(streams, exclude_workers=()):
    """Flag low-diversity sampled tails heuristically; this does not establish a cause."""
    evidence = []
    for worker, tokens in streams.items():
        if worker in exclude_workers:
            continue
        tail = tokens[-2000:]
        windows = len(tail) - 7
        if windows < 1000:
            continue
        counts = Counter(tuple(tail[i:i + 8]) for i in range(windows))
        pattern, occurrences = counts.most_common(1)[0]
        if len(counts) * 4 <= windows and occurrences >= 16:
            evidence.append({
                "kind": "low_diversity_8gram_tail", "heuristic": True, "worker": worker,
                "stream": "worker_token_ids", "start_index": len(tokens) - len(tail),
                "end_index_exclusive": len(tokens), "window_count": windows,
                "unique_windows": len(counts), "unique_fraction": len(counts) / windows,
                "pattern_token_ids": list(pattern), "pattern_occurrences": occurrences,
                "thresholds": {"min_windows": 1000, "max_unique_fraction": 0.25, "min_occurrences": 16},
            })
    return evidence


def verify_monitored_case(directory, row, manifest, catalog):
    # Run inside the verifier process. Reuse its parsed trace and return only
    # compact evidence, not the large engine history, to the coordinator.
    data = report.verify_traced_case(directory, row, manifest, catalog)
    streams = data["rollout"]["worker_token_ids"]
    evidence = repeated_sampled_tokens(streams)
    return evidence + repeated_sampled_phrases(streams, {entry["worker"] for entry in evidence})


class QualityMonitor:
    def __init__(self, root, catalog, repetition_policy="hold"):
        if repetition_policy not in ("hold", "record"):
            raise ValueError(f"Unknown repetition policy: {repetition_policy}")
        self.root, self.catalog = root, catalog
        self.repetition_policy = repetition_policy
        self.rows = {}
        self.lock = threading.Lock()

    def hold(self, message, evidence=()):
        # Link a complete JSON file atomically. Concurrent observers preserve
        # the first cause and no automatic restart removes this marker.
        payload = {"time": datetime.now(timezone.utc).isoformat(), "reason": str(message), "cases": list(evidence)}
        with NamedTemporaryFile("w", dir=self.root, delete=False) as tmp:
            filename = Path(tmp.name)
            try:
                json.dump(payload, tmp, indent=2)
                tmp.flush()
                try:
                    os.link(filename, self.root / HOLD)
                except FileExistsError:
                    pass
            finally:
                filename.unlink(missing_ok=True)
        raise RuntimeError(f"Quality hold: {message}. See {self.root / HOLD}")

    def check_space(self):
        free = shutil.disk_usage(self.root).free
        if free < DISK_RESERVE:
            self.hold(f"Only {free} bytes free; the trace volume requires a {DISK_RESERVE}-byte reserve")

    def record_suspicious(self, evidence):
        directory = self.root / "suspicious-samples"
        directory.mkdir(exist_ok=True)
        target = directory / f"{evidence['mode']}-w{evidence['workers']}-i{evidence['index']:03d}.json"
        payload = {"format": 1, "repetition_policy": "record", "case": evidence,
                   "cause_status": "unresolved; recorded cache/token integrity validated"}
        with NamedTemporaryFile("w", dir=directory, delete=False) as tmp:
            temporary = Path(tmp.name)
            try:
                json.dump({"time": datetime.now(timezone.utc).isoformat(), **payload}, tmp, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    existing = json.loads(target.read_text())
                    existing.pop("time", None)
                    if existing != payload:
                        self.hold(f"Conflicting suspicious-sample record: {target}", (evidence,))
                else:
                    log(f"Recorded suspicious sampled repetition: {target}")
            finally:
                temporary.unlink(missing_ok=True)

    def observe(self, directory, row, repetitions=()):
        condition = (row["mode"], row["workers"])
        index = row["index"]
        evidence = {key: row[key] for key in ("mode", "workers", "index", "correct", "grading_status", "decoding_steps",
                                              "used_finisher", "stop_reason", "finisher_stop_reason", "finisher_tokens",
                                              "published_paragraphs", "trace_sha256")}
        evidence["trace"] = str(directory.relative_to(self.root) / row["trace_path"])
        evidence.update({key: row[key] for key in ("generated_tokens", "reasoning_tokens", "total_decoding_steps",
                                                 "prediction", "gold") if key in row})
        if repetitions:
            evidence["repeated_sampled_tokens"] = list(repetitions)
            if self.repetition_policy == "hold":
                self.hold(f"Suspicious sampled repetition in {condition} index {index}", (evidence,))
            self.record_suspicious(evidence)
        with self.lock:
            group = self.rows.setdefault(condition, {})
            if index in group:
                if group[index] != evidence:
                    self.hold(f"Conflicting traced case {condition} index {index}", (group[index], evidence))
                return
            group[index] = evidence
            expected_pilot = pilot_indices(condition)
            pilot = [group[i] for i in expected_pilot if i in group]
            # H6/H8 have no matched historical pilot, so use the corroborated
            # failure-pattern check below, not the known-correct accuracy gate.
            if condition[1] not in (6, 8):
                for ids in (expected_pilot[:6], expected_pilot):
                    if set(ids) <= group.keys() and sum(not group[i]["correct"] for i in ids) * 2 >= len(ids):
                        self.hold(f"At least half of {len(ids)} matched pilot cases failed in {condition}", [group[i] for i in ids])
            start = index // 16 * 16
            ids = range(start, min(start + 16, TOTAL))
            window = [group[i] for i in ids] if all(i in group for i in ids) else []
            if condition[1] == 1:
                native, single = self.rows.get(("native", 1), {}), self.rows.get(("hogwild", 1), {})
                for cohort in (WIDE_PILOT, (0, 2, 45, 109, 149, 226), ids):
                    if len(cohort) < 6 or not all(i in native and i in single for i in cohort):
                        continue
                    gap = sum(int(native[i]["correct"]) - int(single[i]["correct"]) for i in cohort)
                    if gap >= 3 and gap * 3 >= len(cohort):
                        self.hold(f"Paired native accuracy exceeds H1 by {gap}/{len(cohort)} cases",
                                  [r for i in cohort for r in (native[i], single[i])])
            # On ordinary bulk items, a marked matched baseline collapse is
            # suspicious even if all generated boxes parse and termination
            # looks normal. Check also when ordinary results arrive last.
            if len(ids) == 16:
                native = self.rows.get(("native", 1), {})
                for (mode, n), other in self.rows.items():
                    if mode != "hogwild" or not all(i in native and i in other for i in ids):
                        continue
                    if sum(native[i]["correct"] for i in ids) >= 12 and sum(other[i]["correct"] for i in ids) <= 4:
                        self.hold(f"Matched 16-case collapse for Hogwild {n} (native >=12 correct, Hogwild <=4)",
                                  [r for i in ids for r in (native[i], other[i])])
            for cohort in (pilot if len(pilot) == len(expected_pilot) else [], window):
                if len(cohort) < 6:
                    continue
                failures = [r for r in cohort if not r["correct"] and (
                    r["grading_status"] == "unparsed_prediction" or
                    (r["used_finisher"] and r["stop_reason"] == "eos" and
                     (r["decoding_steps"] <= 256 or r["finisher_stop_reason"] == "eos")))]
                if len(failures) >= 4 and len(failures) * 3 >= len(cohort) * 2:
                    self.hold(f"Repeated unparsed predictions or early EOS with failed finishers in {condition}", cohort)
            # Stable pairs make accidental omission of the publication path
            # visible while allowing short problems with zero natural boundaries.
            if condition[0] == "hogwild" and len(window) >= 8 and all(r["published_paragraphs"] == 0 for r in window):
                if sum(r["decoding_steps"] >= 512 and not r["correct"] for r in window) >= 6:
                    self.hold(f"Repeated long incorrect Hogwild runs without any publication in {condition}", window)

    def status(self, phase):
        with self.lock:
            counts = {f"{mode}-w{n}": {"verified": len(group), "correct": sum(r["correct"] for r in group.values())}
                      for (mode, n), group in self.rows.items()}
            native = self.rows.get(("native", 1), {})
            matched = {}
            for (mode, count), group in self.rows.items():
                if mode == "hogwild":
                    ids = sorted(set(pilot_indices((mode, count))) & set(pilot_indices(("native", 1))) & group.keys() & native.keys())
                    matched[str(count)] = {
                        "indices": ids, "native_correct": sum(native[i]["correct"] for i in ids),
                        "hogwild_correct": sum(group[i]["correct"] for i in ids),
                        "disagree": [i for i in ids if group[i]["correct"] != native[i]["correct"]],
                    }
            progress = {"phase": phase, "conditions": counts, "paired_pilots": matched,
                        "repetition_policy": self.repetition_policy,
                        "suspicious_cases": sum(bool(row.get("repeated_sampled_tokens"))
                                                for group in self.rows.values() for row in group.values())}
            report.atomic_write(self.root / "progress.json", json.dumps(progress, indent=2) + "\n")


def stop_child(proc):
    """Stop only the subprocess group created by this coordinator."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def watch_attempt(proc, directory, hashes, condition, indices, multi_worker, monitor, stop, phase, logfile, verifier):
    """Validate newline-committed rows incrementally; never inspect an in-flight sidecar."""
    manifest, offset, log_offset, seen = None, 0, 0, set()
    result = directory / "results.jsonl"
    requested = set(indices)
    try:
        while True:
            check_run_open(monitor.root)
            if stop.is_set():
                raise RuntimeError("Another evaluator stopped the sweep")
            monitor.check_space()
            code = proc.poll()
            with logfile.open("rb") as stream:
                stream.seek(max(0, log_offset - 96))
                tail = stream.read().lower()
                log_offset = stream.tell()
            for signal_text in (b"device-side assert triggered", b"illegal memory access", b"cuda out of memory",
                                b"probability tensor contains either", b"non-finite logits", b"nonfinite logits"):
                if signal_text in tail:
                    raise ValueError(f"{signal_text.decode()} reported by {logfile}")
            if result.exists():
                if manifest is None:
                    manifest = json.loads((directory / "config.json").read_text())
                    check_attempt(directory, manifest, hashes, (condition,), multi_worker)
                    if manifest["indices"] != list(indices):
                        raise ValueError(f"Evaluator started unexpected indices in {directory}")
                with result.open("rb") as stream:
                    stream.seek(offset)
                    for line in stream:
                        if not line.endswith(b"\n"):
                            if (directory / "completed.json").exists():
                                raise ValueError(f"Unterminated result row in {result}")
                            break
                        row = json.loads(line)
                        report.check_task(row, manifest)
                        # Math-Verify implements timeouts with SIGALRM, which
                        # requires the main thread of the verifier process.
                        repetitions = verifier.submit(verify_monitored_case, directory, row, manifest, monitor.catalog).result()
                        index = row["index"]
                        if index not in requested or index in seen:
                            raise ValueError(f"Duplicate or unexpected trace for index {index} in {result}")
                        monitor.observe(directory, row, repetitions)
                        seen.add(index)
                        offset += len(line)
                        monitor.status(phase)
            if code is not None:
                return code, seen
            stop.wait(POLL_SECONDS)
    except Exception as exc:
        if not stop.is_set() and not any((monitor.root / m).exists() for m in STOP_MARKERS):
            monitor.hold(str(exc))
        raise
    finally:
        stop_child(proc)


def run_sweep(root, shard_size, lock_fd, only_one_worker=False, reuse_native_from=None, multi_worker=MultiWorkerProtocol(),
              save_traces=False, gpus=DEFAULT_GPUS, repetition_policy="hold", no_wait_gpus=False):
    gpus = selected_gpus(gpus)
    check_run_open(root)
    if save_traces and (reuse_native_from is not None or (root / report.NATIVE_SNAPSHOT).exists()):
        raise ValueError("The traced benchmark requires a fresh execution for every native and Hogwild case")
    shards, logs = root / "shards", root / "logs"
    shards.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    conditions = (("hogwild", 1),) if only_one_worker else ORDER
    hashes = source_hashes(save_traces)
    monitor = QualityMonitor(root, report.load_native_catalog(), repetition_policy) if save_traces else None
    if monitor:
        monitor.check_space()
        try:
            done = read_existing(shards, hashes, conditions, multi_worker, monitor=monitor)
        except IncompatibleRun:
            raise
        except (OSError, KeyError, TypeError, ValueError) as exc:
            monitor.hold(f"Invalid saved attempt: {exc}")
    else:
        if reuse_native_from is not None:
            if only_one_worker:
                raise ValueError("The isolated one-worker sweep cannot import native results")
            import_native(reuse_native_from, root, hashes, multi_worker)
        done = done_with_native_reuse(root, hashes, conditions, multi_worker)
    goal = TOTAL * len(conditions)
    stop = threading.Event()
    dropped_gpus = set()
    availability_lock = threading.Lock()

    def drop_gpu(gpu):
        with availability_lock:
            dropped_gpus.add(gpu)
        log(f"GPU {gpu} occupied; dropping it for this coordinator (--no-wait-gpus)")

    def work(gpu, jobs, phase, verifier):
        waiting = False
        try:
            while not stop.is_set() and not jobs.empty():
                check_run_open(root)
                verify_source_snapshot(root)
                if monitor:
                    monitor.check_space()
                if not gpu_available(gpu):
                    if no_wait_gpus:
                        drop_gpu(gpu)
                        return
                    if not waiting:
                        log(f"GPU {gpu} occupied; waiting for it to become free")
                    waiting = True
                    stop.wait(45)
                    continue
                waiting = False
                try:
                    mode, count, indices, retry = jobs.get_nowait()
                except queue.Empty:
                    return
                name = f"{mode}-w{count}-i{indices[0]:03d}-{indices[-1]:03d}-{uuid4().hex[:12]}"
                out, logfile = shards / name, logs / f"{name}.log"
                command = [
                    os.environ.get("HOGWILD_EVALUATOR_LAUNCHER", str(LAUNCHER)),
                    "--dataset", "math500", "--mode", mode, "--workers", str(count),
                    "--indices", *map(str, indices), "--seed", str(SEED), "--budget", str(BUDGET), "--output", str(out),
                ]
                if monitor:
                    command.append("--save-traces")
                if (mode, count) == ("hogwild", 1):
                    command.extend(["--single-worker-layout", "causal"])
                elif count > 1:
                    command.extend(["--multi-worker-layout", multi_worker.layout,
                                    "--multi-worker-answer-cue", multi_worker.answer_cue])
                check_run_open(root)
                log(f"GPU {gpu} starting {phase} {name} ({len(indices)} problems), log={logfile}")
                with logfile.open("x") as stream:
                    opts = {"cwd": ROOT, "env": {**os.environ, "HOGWILD_GPU": str(gpu)}, "stdout": stream,
                            "stderr": subprocess.STDOUT, "pass_fds": (lock_fd,)}
                    if monitor:
                        child = subprocess.Popen(command, **opts, start_new_session=True)
                        code, rows = watch_attempt(child, out, hashes, (mode, count), indices, multi_worker, monitor, stop,
                                                  phase, logfile, verifier)
                    else:
                        code = subprocess.run(command, **opts).returncode
                        saved = attempt_rows(out, hashes, ((mode, count),), multi_worker) if (out / "config.json").exists() else set()
                        rows = {key[2] for key in saved}
                        if not rows <= set(indices):
                            raise ValueError(f"Unexpected indices in {out}")
                missing = [i for i in indices if i not in rows]
                if code == 0:
                    marker = out / "completed.json"
                    if missing or not marker.exists() or json.loads(marker.read_text())["cases"] != len(rows):
                        raise RuntimeError(f"Evaluator exited successfully with incomplete records: {out}")
                    log(f"GPU {gpu} completed {name}: {len(rows)} verified cases")
                    continue
                with logfile.open("rb") as stream:
                    stream.seek(max(0, logfile.stat().st_size - 8192))
                    tail = stream.read().decode(errors="replace").lower()
                occupied = f"gpu {gpu} is occupied" in tail
                if no_wait_gpus and occupied and missing:
                    # The evaluator's own preflight caught an allocation race.
                    # Another permitted free worker may take these cases, but
                    # this device is never polled or launched again in this run.
                    jobs.put((mode, count, missing, retry))
                    drop_gpu(gpu)
                    return
                transient = occupied or any(s in tail for s in (
                    "readtimeout", "connecttimeout",
                    "connection reset by peer", "temporary failure in name resolution",
                ))
                if transient and retry < 1 and missing:
                    jobs.put((mode, count, missing, retry + 1))
                    log(f"Retrying {len(missing)} missing cases after transient failure: {logfile}")
                    if no_wait_gpus:
                        if not gpu_available(gpu):
                            drop_gpu(gpu)
                            return
                    else:
                        stop.wait(15)
                else:
                    raise RuntimeError(f"{name} failed with exit {code}; inspect {logfile}")
        except BaseException as exc:
            try:
                if monitor and isinstance(exc, Exception) and not stop.is_set() and not any((root / m).exists() for m in STOP_MARKERS):
                    monitor.hold(str(exc))
            finally:
                stop.set()
            raise

    for phase in (("pilot", "benchmark") if monitor else ("benchmark",)):
        jobs = queue.Queue()
        for start in ([0] if phase == "pilot" else range(0, TOTAL, shard_size)):
            for mode, count in conditions:
                pilots = pilot_indices((mode, count)) if monitor else ()
                cohort = pilots if phase == "pilot" else [i for i in range(start, min(start + shard_size, TOTAL)) if i not in pilots]
                missing = [i for i in cohort if (mode, count, i) not in done]
                if missing:
                    jobs.put((mode, count, missing, 0))
        phase_gpus = tuple(gpu for gpu in gpus if gpu not in dropped_gpus)
        log(f"{phase}: {jobs.qsize()} pending shards on GPUs {phase_gpus}, target {goal} verified cases")
        if monitor:
            monitor.status(phase)
        validators = ProcessPoolExecutor(max_workers=len(gpus), mp_context=get_context("spawn")) if monitor else nullcontext()
        with validators as verifier, ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(work, gpu, jobs, phase, verifier) for gpu in phase_gpus]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                stop.set()
                raise
        if not jobs.empty():
            payload = {"time": datetime.now(timezone.utc).isoformat(), "status": "gpu_unavailable",
                       "phase": phase, "pending_shards": jobs.qsize(), "configured_gpus": list(gpus),
                       "dropped_gpus": sorted(dropped_gpus), "no_wait_gpus": no_wait_gpus}
            report.atomic_write(root / "gpu-unavailable.json", json.dumps(payload, indent=2) + "\n")
            if monitor:
                monitor.status("gpu_unavailable")
            raise GPUsUnavailable(f"No free configured GPU remains; {jobs.qsize()} {phase} shards are pending")
        if phase == "pilot":
            for condition in conditions:
                if not set(pilot_indices(condition)) <= monitor.rows.get(condition, {}).keys():
                    monitor.hold(f"Pilot gate incomplete for {condition}")
            log("All matched pilot cohorts verified. Bulk shards may start.")
    finished = (sum(len(group) for group in monitor.rows.values()) if monitor else
                len(done_with_native_reuse(root, hashes, conditions, multi_worker)))
    if finished != goal:
        if monitor:
            monitor.hold(f"Only {finished}/{goal} verified traces")
        raise RuntimeError(f"Only {finished}/{goal} verified cases")
    log(f"All {goal} rows validated; generating report")
    report_command = [sys.executable, "-m", "minisgl_examples.training.parallel_thinking.report", str(root)]
    if only_one_worker:
        report_command.append("--only-one-worker")
    try:
        subprocess.run(report_command, check=True, cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        if monitor:
            monitor.hold(f"Final trace verification failed: {exc}")
        raise
    if monitor:
        monitor.status("complete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Output root; reuse it to resume an interrupted sweep")
    parser.add_argument("--gpus", type=int, nargs="+", required=True, metavar="ID",
                        help="Physical GPU IDs to consider for this invocation")
    parser.add_argument("--no-wait-gpus", action="store_true",
                        help="Drop occupied devices permanently; exit if no free device remains for pending work")
    parser.add_argument("--shard-size", type=int, choices=range(50, 101), metavar="50..100", default=50)
    parser.add_argument("--only-one-worker", action="store_true", help="Run only corrected causal Hogwild on all 500 problems")
    parser.add_argument("--reuse-native-from", type=Path, metavar="OLD_RUN_DIR",
                        help="Import audited native baseline rows for an untraced comparison")
    parser.add_argument("--regrade-from", type=Path, metavar="HELD_RUN_DIR",
                        help="Import reusable traces under the corrected answer selector into a separate run")
    parser.add_argument("--reuse-baselines-from", type=Path, metavar="HELD_RUN_DIR",
                        help="Reuse unchanged traced native and causal W1 cases; regenerate every multiworker case")
    parser.add_argument("--save-traces", action="store_true", help="Require a fresh verified inference trace for every case and monitor matched pilots")
    parser.add_argument("--repetition-policy", choices=("hold", "record"), default="hold",
                        help="Hold on suspicious repetition, or record evidence and continue (requires --save-traces)")
    parser.add_argument("--multi-worker-layout", choices=("collaboration", "native"), default=MultiWorkerProtocol().layout)
    parser.add_argument("--multi-worker-answer-cue", choices=("paper", "qwen"), default=MultiWorkerProtocol().answer_cue)
    args = parser.parse_args(argv)
    if args.repetition_policy == "record" and not args.save_traces:
        parser.error("--repetition-policy record requires --save-traces")
    try:
        gpus = selected_gpus(args.gpus)
    except ValueError as exc:
        parser.error(str(exc))
    root = args.run_dir.resolve()
    check_run_open(root)
    if args.reuse_baselines_from is not None:
        if (not args.save_traces or args.only_one_worker or args.reuse_native_from is not None
                or args.regrade_from is not None):
            parser.error("--reuse-baselines-from requires a full traced sweep and cannot combine with other imports")
        from .baseline_reuse import import_baselines

        import_baselines(args.reuse_baselines_from, root)
    if args.regrade_from is not None:
        if not args.save_traces or args.only_one_worker or args.reuse_native_from is not None:
            parser.error("--regrade-from requires a full traced sweep and cannot combine with native reuse")
        from .regrade import import_regraded

        import_regraded(args.regrade_from, root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".sweep.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, f"Another coordinator or evaluator still holds {root}/.sweep.lock\n")
        try:
            snapshot_sources(root, verified_import=args.regrade_from is not None or args.reuse_baselines_from is not None)
            run_sweep(root, args.shard_size, lock.fileno(), args.only_one_worker, args.reuse_native_from,
                      MultiWorkerProtocol(args.multi_worker_layout, args.multi_worker_answer_cue), args.save_traces,
                      gpus=gpus, repetition_policy=args.repetition_policy, no_wait_gpus=args.no_wait_gpus)
        except GPUsUnavailable as exc:
            parser.exit(3, f"GPU unavailable: {exc}. Completed cases are preserved.\n")
        verify_source_snapshot(root)
        completed = {
            "cases": TOTAL if args.only_one_worker else TOTAL * len(CONDITIONS),
            "source_manifest_sha256": report.digest((root / "source-files.json").read_bytes()),
            "active_source_manifest_sha256": report.digest((root / "active-source-files.json").read_bytes()),
            "tasks_sha256": report.digest((root / "report/tasks.csv").read_bytes()),
        }
        report.atomic_write(root / "complete.json", json.dumps(completed, indent=2) + "\n")


if __name__ == "__main__":
    main()
