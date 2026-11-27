"""Persist and validate one compressed inference trace per scored evaluation case."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

TRACE_FORMAT = "hogwild-inference-v1"
IDENTITY = ("mode", "workers", "index", "unique_id")
METRICS = (
    "correct", "grading_status", "gold", "prediction", "generated_tokens", "reasoning_tokens", "finisher_tokens",
    "decoding_steps", "total_decoding_steps", "used_finisher", "finisher_stop_reason", "finisher_eos_token_id",
    "stop_reason", "eos_token_id", "published_paragraphs",
)


def _path(row: dict) -> Path:
    return Path("traces") / f"{row['mode']}-w{row['workers']}-i{row['index']:03d}.json.gz"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _engine_json(value):
    if is_dataclass(value):
        return {f.name: v for f in fields(value) if (v := getattr(value, f.name)) is not None}
    raise TypeError(f"Cannot serialize {type(value).__name__} as an inference trace")


def _digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


@lru_cache(maxsize=2)
def selection_tokenizer(model: str, revision: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)


def verify_answer_selection(data: dict, row: dict, manifest: dict) -> None:
    """Recompute the answer choice from sampled token IDs, independently of the saved prediction."""
    from .rollout import TERMINAL_ANSWER_POLICY, extract_boxed_answer, select_terminal_worker_answer

    _require(manifest["answer_selection_policy"] == TERMINAL_ANSWER_POLICY, "Unknown answer selector")
    t = data["rollout"]
    _require(set(t["eos_token_ids"]) == set(manifest["eos_token_ids"]), "Trace EOS tokens differ from the manifest")
    streams, eos = t["worker_token_ids"], set(manifest["eos_token_ids"])
    terminal = next((name for name in t["workers"] if streams[name] and streams[name][-1] in eos), None)
    _require(terminal == t["eos_worker"] == row["eos_worker"], "Incorrect terminating worker")
    tokenizer = selection_tokenizer(data["model"], data["model_revision"])
    answer, selection = select_terminal_worker_answer(
        tokenizer, streams, eos_worker=terminal, eos_token_ids=manifest["eos_token_ids"],
    )
    _require(t.get("answer_selection") == selection, "Saved answer selection differs from terminal tokens")
    finisher = t["finisher"]
    _require(finisher["used"] == (answer is None), "Finisher decision differs from the answer selector")
    if answer is None:
        ids = finisher["sampled_token_ids"]
        if ids and ids[-1] in eos:
            ids = ids[:-1]
        text = tokenizer.decode(finisher["suffix_ids"]) + tokenizer.decode(ids)
        answer = extract_boxed_answer(text)
    _require(answer == row["prediction"], "Prediction differs from independently selected answer")


def write_trace(attempt: Path, row: dict, manifest: dict, problem: str, tokenizer, rollout: dict, engine):
    """Publish immutable complete bytes before the caller appends its result row."""
    rel = _path(row)
    path = attempt / rel
    path.parent.mkdir(exist_ok=True)
    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=False)

    rollout["worker_text"] = {name: decode(ids) for name, ids in rollout["worker_token_ids"].items()}
    rollout["prompt_text"] = decode(rollout["prompt_ids"])
    rollout["finisher"]["text"] = decode(rollout["finisher"]["sampled_token_ids"])
    payload = {
        "format": TRACE_FORMAT,
        "case": {k: row[k] for k in IDENTITY},
        "model": manifest["model"], "model_revision": manifest["model_revision"],
        "dataset": manifest["dataset"], "dataset_revision": manifest["dataset_revision"], "seed": manifest["seed"],
        "problem": problem, "metrics": {k: row[k] for k in METRICS}, "rollout": rollout, "engine": engine,
    }
    with NamedTemporaryFile(dir=path.parent, prefix=".partial-", delete=False) as temp:
        tmp = Path(temp.name)
    try:
        with tmp.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as gz:
            encoder = json.JSONEncoder(default=_engine_json, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            # Group encoder fragments to avoid a gzip write for every integer.
            buf = []
            for fragment in encoder.iterencode(payload):
                buf.append(fragment)
                if len(buf) >= 8192:
                    gz.write("".join(buf).encode())
                    buf.clear()
            if buf:
                gz.write("".join(buf).encode())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    row.update(trace_path=rel.as_posix(), trace_sha256=_digest(path), trace_bytes=path.stat().st_size)


def _check_engine(engine, trajectory):
    """Follow the recorded block lengths and transfers without materializing any KV or running inference."""
    batches = engine["batches"]
    _require(engine["format_version"] == 2 and engine["steps"] == len(batches), "Incomplete MiniSGL trace")
    if not batches:
        _require(trajectory["layout"] == "native" and trajectory["decode_rounds"] == 0
                 and not engine["updates"], "Missing engine requests")
        return
    _require(batches[0]["kind"] == "prefill", "Missing initial prefill")
    roles = trajectory["block_ids"]
    workers = roles["workers"]
    _require(set(workers) == set(trajectory["workers"]) and len(set(workers.values())) == len(workers),
             "Invalid worker block IDs")
    blocks = {b for role, b in roles.items() if role != "workers" and b is not None} | set(workers.values())
    lengths = dict.fromkeys(blocks, 0)
    pubs = iter(trajectory["publications"])
    updates = iter(engine["updates"])
    update = next(updates, None)
    for step in range(len(batches) + 1):
        while update is not None and update["step"] == step:
            target, refs = update["block"], update["refs"]
            _require(target in blocks and all(lengths.get(block) == size for block, size in refs),
                     "Cache transfer references stale or unknown KV")
            if refs:
                event = next(pubs, None)
                _require(event is not None and target == roles["history"] and len(refs) == 2
                         and refs[0] == [target, lengths[target]]
                         and refs[1] == [workers[event["worker"]], event["committed_token_count"]],
                         "Publication differs from the recorded KV transfer")
            lengths[target] = sum(n for _, n in refs)
            update = next(updates, None)
        if step == len(batches):
            break
        batch = batches[step]
        _require(batch["step"] == step and batch["kind"] in ("prefill", "decode") and batch["requests"],
                 "Invalid engine batch")
        written = {}
        for req in batch["requests"]:
            target, ids, view = req["write_to"], req["input_ids"], req["cache_view"]
            _require(target in blocks and target not in written and req["write_start"] == lengths[target],
                     "Invalid cache write position")
            _require(ids and all(type(i) is int and i >= 0 for i in ids)
                     and (batch["kind"] != "decode" or len(ids) == 1), "Invalid engine input IDs")
            _require(view and view[-1][0] == target and all(lengths.get(block) == size for block, size in view),
                     "Invalid cache view")
            written[target] = lengths[target] + len(ids)
        lengths.update(written)
    _require(update is None and next(pubs, None) is None and not any(lengths.values()),
             "Incomplete engine cache updates or publications")


def _check_encoded_separators(engine, trajectory):
    """Check token identity and delayed boundaries for the native publication policy."""
    _require(trajectory["layout"] == "native_workers", "Encoded separators require native workers")
    committed = dict.fromkeys(trajectory["workers"], 0)
    for event in trajectory["publications"]:
        worker = event["worker"]
        committed[worker] += event["committed_token_count"]
        _require(committed[worker] == event["round"] - 1,
                 "Publication did not commit the preceding round's encoded separator")
    roles = trajectory["block_ids"]
    encoded = {roles["history"]: [], roles["answer"]: []}
    encoded.update({block: [] for block in roles["workers"].values()})
    for batch in engine["batches"]:
        for request in batch["requests"]:
            encoded[request["write_to"]].extend(request["input_ids"])
    _require(encoded[roles["history"]] == trajectory["prompt_ids"],
             "Common cache inputs differ from the original prompt")
    finisher = trajectory["finisher"]
    markers = finisher["worker_marker_ids"] if finisher["used"] else []
    eos = set(trajectory["eos_token_ids"])
    for worker, block in roles["workers"].items():
        tokens = trajectory["worker_token_ids"][worker]
        reasoning = tokens[:-1] if tokens and tokens[-1] in eos else tokens
        _require(not any(token in eos for token in reasoning), "Worker continued after a sampled EOS")
        _require(encoded[block] == reasoning + markers,
                 "Worker cache inputs do not encode each sampled token exactly once")
    answer = finisher["sampled_token_ids"]
    expected = finisher["suffix_ids"] + answer[:-1] if finisher["used"] else []
    _require(encoded[roles["answer"]] == expected, "Answer cache inputs differ from its suffix and sampled tokens")


def verify_trace(attempt: Path, row: dict, manifest: dict) -> dict:
    """Validate trace integrity, case identity, sampled streams and execution history on CPU."""
    rel = _path(row)
    _require(manifest.get("trace_format") == TRACE_FORMAT, "Invalid trace protocol")
    _require(row.get("trace_path") == rel.as_posix(), "Missing or unexpected trace path")
    path = attempt / rel
    try:
        _require(path.is_file() and row["trace_bytes"] == path.stat().st_size > 0, "Missing or truncated trace")
        _require(_digest(path) == row["trace_sha256"], "Trace SHA-256 mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        _require(data["format"] == TRACE_FORMAT and data["case"] == {k: row[k] for k in IDENTITY},
                 "Trace case identity mismatch")
        _require(data["metrics"] == {k: row[k] for k in METRICS}, "Trace metrics mismatch")
        for k in ("model", "model_revision", "dataset", "dataset_revision", "seed"):
            _require(data[k] == manifest[k], f"Trace {k} mismatch")
        t = data["rollout"]
        _require(t.get("publication_policy") == manifest.get("publication_policy"),
                 "Trace publication policy differs from the manifest")
        _require(t["boxed_answer"] == row["prediction"] and t["decode_rounds"] == row["decoding_steps"]
                 and t["eos_token_id"] == row["eos_token_id"], "Trace answer, step or EOS mismatch")
        names, streams = t["workers"], t["worker_token_ids"]
        _require(len(names) == len(set(names)) == row["workers"] and set(names) == set(streams),
                 "Trace worker identities mismatch")
        _require(bool(t["prompt_ids"]) and all(type(i) is int and i >= 0 for i in t["prompt_ids"]), "Invalid prompt IDs")
        _require(all(isinstance(streams[n], list) and all(type(i) is int and i >= 0 for i in streams[n])
                     for n in names), "Invalid sampled token IDs")
        counts = {n: len(streams[n]) for n in names}
        _require(counts == row["per_worker_tokens"] and sum(counts.values()) == row["reasoning_tokens"]
                 and all(n == row["decoding_steps"] for n in counts.values()), "Trace reasoning count mismatch")
        fin = t["finisher"]
        ft = fin["sampled_token_ids"]
        _require(len(ft) == row["finisher_tokens"] and fin["used"] == row["used_finisher"]
                 and fin["stop_reason"] == row["finisher_stop_reason"] and fin["eos_token_id"] == row["finisher_eos_token_id"]
                 and sum(counts.values()) + len(ft) == row["generated_tokens"]
                 and row["total_decoding_steps"] == row["decoding_steps"] + len(ft),
                 "Trace finisher count/stop mismatch")
        if row["eos_token_id"] is not None:
            _require(any(streams[n] and streams[n][-1] == row["eos_token_id"] for n in names),
                     "Reasoning EOS missing from sampled trace")
        if fin["eos_token_id"] is not None:
            _require(bool(ft) and ft[-1] == fin["eos_token_id"], "Finisher EOS missing from sampled trace")
        pubs = t["publications"]
        _require(len(pubs) == row["published_paragraphs"], "Publication count mismatch")
        rounds = [e["round"] for e in pubs]
        _require(rounds == sorted(rounds) and all(1 <= rnd <= row["decoding_steps"] for rnd in rounds)
                 and all(e["worker"] in streams and e["committed_token_count"] > 0 for e in pubs),
                 "Invalid publication events")
        _require(set(t["worker_text"]) == set(streams) and isinstance(fin["text"], str),
                 "Missing decoded worker or finisher text")
        _check_engine(data["engine"], t)
        if t.get("publication_policy") == "encoded_separator_v1":
            _check_encoded_separators(data["engine"], t)
        if row["workers"] > 1 and manifest.get("answer_selection_policy") is not None:
            verify_answer_selection(data, row, manifest)
        return data
    except (OSError, EOFError, IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid inference trace {path}: {error}") from error
