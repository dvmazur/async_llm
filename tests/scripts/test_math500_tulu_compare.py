"""CPU fixtures for the offline Tülu baseline/async comparison."""

import gzip
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from shutil import copytree

import pytest
from minisgl_examples.demos.async_thoughts import math500_tulu_compare as compare
from minisgl_examples.demos.async_thoughts.math500 import digest, encode_json
from minisgl_examples.demos.async_thoughts.math500_tulu_paired import make_prompt
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def tokenizer():
    alphabet = sorted(ByteLevel.alphabet())
    backend = Tokenizer(BPE({c: i for i, c in enumerate(alphabet)}, []))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteDecoder()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")
    tok.add_tokens(["note", r"\boxed{2}"])
    tok.chat_template = (
        "{% for m in messages %}{{'<|' + m.role + '|>\\n' + m.content}}"
        "{% if m.role == 'assistant' %}{{eos_token}}{% endif %}{{'\\n'}}{% endfor %}"
        "{% if add_generation_prompt %}{{'<|assistant|>\\n'}}{% endif %}"
    )
    return tok


def dump(path, obj):
    path.write_bytes(encode_json(obj))


def make_manifest(directory, task):
    source = Path(compare.__file__).parent
    m = {
        "schema_version": 1,
        "model": compare.CHECKPOINT,
        "model_revision": compare.REVISION,
        "dataset_revision": compare.DATASET_REVISION,
        "dataset_sha256": compare.DATASET_SHA,
        "order": compare.ORDER,
        "ordered_ids": [task["unique_id"]],
        "arms": list(compare.ARMS),
        "start_rank": 0,
        "limit": 1,
        "budget": 16384,
        "serial_draft_cap": 8192,
        "memory_ratio": 0.85,
        "context": 40960,
        "probe_period": 20,
        "llama_gate": "evidence-turns",
        "probe_include_pending": True,
        "dtype": "bfloat16",
        "sampling": "greedy",
        "smoke_only": False,
        "source_sha256": {s: digest((source / s).read_bytes()) for s in compare.SOURCES},
    }
    dump(directory / "manifest.json", m)
    return m, digest(encode_json(m))


class Batches:
    def __init__(self):
        self.sizes = defaultdict(int)
        self.batches = []
        self.samples = []

    def add(self, kind, *entries):
        step = len(self.batches)
        requests = []
        for blocks, ids, sample in entries:
            requests.append(
                {
                    "write_to": blocks[-1],
                    "cache_view": [[b, self.sizes[b]] for b in blocks],
                    "input_ids": ids,
                    "write_start": self.sizes[blocks[-1]],
                }
            )
            if sample:
                role, token = sample
                self.samples.append({"role": role, "batch_step": step, "token": token})
        for q in requests:
            self.sizes[q["write_to"]] += len(q["input_ids"])
        self.batches.append({"kind": kind, "step": step, "requests": requests})

    def trace(self):
        return {
            "format_version": 2,
            "steps": len(self.batches),
            "batches": self.batches,
            "updates": [],
        }


def make_record(arm, tok, task, manifest_hash):
    prompt = make_prompt(tok, task["problem"], arm)
    note, box = [tok.encode(s, add_special_tokens=False)[0] for s in ("note", r"\boxed{2}")]
    eos = tok.eos_token_id
    b = Batches()
    b.add("prefill", ([0], prompt.initial[:-1], None))
    if arm == "direct":
        b.add("decode", ([0], prompt.initial[-1:], ("writer", box)))
        b.add("decode", ([0], [box], ("writer", eos)))
        thought, tstop = [], None
    elif arm == "serial-thinking":
        b.add("decode", ([0], prompt.initial[-1:], ("thinker", note)))
        b.add("decode", ([0], [note], ("thinker", eos)))
        b.add("prefill", ([0], prompt.bridge[:-1], None))
        b.add("decode", ([0], prompt.bridge[-1:], ("writer", box)))
        b.add("decode", ([0], [box], ("writer", eos)))
        thought, tstop = [note], eos
    else:
        b.add("decode", ([0, 1], prompt.initial[-1:], ("thinker", note)))
        b.add("prefill", ([0, 1, 2], prompt.bridge[:-1], None))
        b.add(
            "decode",
            ([0, 1], [note], ("thinker", note)),
            ([0, 1, 2], prompt.bridge[-1:], ("writer", box)),
        )
        # Writer EOS arrives alongside a final thinker sample that the scheduler discards.
        b.add("decode", ([0, 1], [note], ("thinker", 42)), ([0, 1, 2], [box], ("writer", eos)))
        thought, tstop = [note, note], None
    trace = {"format_version": 1, "trace": b.trace(), "samples": b.samples, "probe_events": []}
    row = {
        "rank": 0,
        "index": task["index"],
        "unique_id": task["unique_id"],
        "arm": arm,
        "gold": "2",
        "settings_sha256": manifest_hash,
        "initial_prompt_ids": prompt.initial,
        "bridge_ids": prompt.bridge,
        "prompt_ids_sha256": digest(encode_json(prompt.initial)),
        "user_message": prompt.user,
        "user_message_sha256": digest(prompt.user.encode()),
        "system_message": prompt.system,
        "followup_message": prompt.followup,
        "thinker_ids": thought,
        "thinker_stop_id": tstop,
        "writer_ids": [box],
        "writer_stop_id": eos,
        "writer_text": r"\boxed{2}",
        "writer_box": "2",
        "pure_exact": True,
        "thinker_samples": sum(e["role"] == "thinker" for e in b.samples),
        "writer_samples": 2,
        "forward_batches": len(b.batches),
        "decode_batches": sum(v["kind"] == "decode" for v in b.batches),
        "joint_decode_batches": 2 if arm == "evidence-turns" else 0,
        "elapsed_seconds": 1.0,
        "shared_prefix_writer_tokens": 1,
        "shared_prefix_exact": True,
    }
    return row, trace


def save_record(directory, row, trace):
    data = gzip.compress(encode_json(trace), mtime=0)
    stem = f"{row['rank']:03}.{row['arm']}"
    (directory / "traces" / f"{stem}.json.gz").write_bytes(data)
    rowfile = directory / "rows" / f"{stem}.json"
    dump(rowfile, {**row, "trace_sha256": digest(data)})
    return rowfile


@pytest.fixture
def journal(tmp_path, tokenizer):
    task = {"unique_id": "test/one", "index": 12, "problem": "What is 1+1?", "answer": "2"}
    (tmp_path / "rows").mkdir()
    (tmp_path / "traces").mkdir()
    manifest, sha = make_manifest(tmp_path, task)
    for arm in compare.ARMS:
        row, trace = make_record(arm, tokenizer, task, sha)
        save_record(tmp_path, row, trace)
    ref = {
        task["unique_id"]: {
            "study_rank": 0,
            "gold": "2",
            "prompt_ids_sha256": row["prompt_ids_sha256"],
            "ordinary_exact": False,
            "native_exact": True,
            "writer_tokens": 1,
            "thinker_tokens": 2,
            "ordinary_tokens": 1,
            "ordinary_seconds": 1,
            "pure_seconds": 1,
        }
    }
    return task, ref, manifest, sha


def test_physical_chat_chains_and_duplicate_import(tmp_path, tokenizer, journal):
    task, ref, _, _ = journal
    resume = tmp_path / "resumed"
    for subdir in ("rows", "traces"):
        copytree(tmp_path / subdir, resume / subdir)
    (resume / "manifest.json").write_bytes((tmp_path / "manifest.json").read_bytes())
    report = compare.comparison([tmp_path, resume], [task], ref, tokenizer, 1)
    assert report["coverage"] == 1 and set(report["arm_counts"].values()) == {1}
    assert all(report["summary"][arm]["correct"] == 1 for arm in compare.ARMS)
    assert report["summary"]["evidence-turns"]["joint_decode_batches"]["sum"] == 2


@pytest.mark.parametrize("arm", compare.ARMS)
def test_rejects_rehashed_wrong_physical_seed(tmp_path, tokenizer, journal, arm):
    task, ref, manifest, sha = journal
    row, trace = make_record(arm, tokenizer, task, sha)
    # Tokens, grade, journal checksums and buffer lengths still agree, but the initial seed token is wrong.
    decode = next(b for b in trace["trace"]["batches"] if b["kind"] == "decode")
    decode["requests"][0]["input_ids"][0] += 1
    path = save_record(tmp_path, row, trace)
    with pytest.raises(ValueError, match="decode (inputs|chain)"):
        compare.verify_row(path, tokenizer, [task], ref, manifest, sha)


def test_rejects_stale_thinker_view_and_extra_unaccepted_token(tmp_path, tokenizer, journal):
    task, ref, m, sha = journal
    row, trace = make_record("evidence-turns", tokenizer, task, sha)
    original = deepcopy(trace)
    trace["trace"]["batches"][-1]["requests"][1]["cache_view"][1][1] -= 1
    with pytest.raises(ValueError, match="pre-step view"):
        compare.verify_row(save_record(tmp_path, row, trace), tokenizer, [task], ref, m, sha)
    row["thinker_ids"].pop()  # Only one concurrently sampled terminal thought may be discarded.
    with pytest.raises(ValueError, match="Samples differ"):
        compare.verify_row(save_record(tmp_path, row, original), tokenizer, [task], ref, m, sha)


def test_rejects_changed_manifest_and_conflicting_duplicate(tmp_path, tokenizer, journal):
    task, ref, m, sha = journal
    second = tmp_path / "resume"
    for sub in ("rows", "traces"):
        (second / sub).mkdir(parents=True)
    dump(second / "manifest.json", {**m, "probe_period": 19})
    with pytest.raises(ValueError, match="probe_period"):
        compare.comparison([tmp_path, second], [task], ref, tokenizer, 1)
    dump(second / "manifest.json", m)
    row, trace = make_record("direct", tokenizer, task, sha)
    save_record(second, {**row, "elapsed_seconds": 2.0}, trace)
    with pytest.raises(ValueError, match="Duplicate predictions differ"):
        compare.comparison([tmp_path, second], [task], ref, tokenizer, 1)


def terminal_flush_record(tok, task, sha, mode):
    row, trace = make_record("evidence-turns", tok, task, sha)
    physical = trace["trace"]
    last = physical["batches"][-1]
    ts, ws = last["requests"]
    trace["samples"] = [
        e
        for e in trace["samples"]
        if not (e["batch_step"] == last["step"] and e["role"] == "thinker")
    ]
    row["thinker_samples"] -= 1
    if mode != "joint":
        last["requests"] = [ts]
        last["kind"] = mode
        ws["cache_view"][1][1] += 1
        physical["batches"].append(
            {"step": len(physical["batches"]), "kind": "decode", "requests": [ws]}
        )
        physical["steps"] += 1
        trace["samples"][-1]["batch_step"] += 1
        row["joint_decode_batches"] -= 1
    row["forward_batches"] = len(physical["batches"])
    row["decode_batches"] = sum(b["kind"] == "decode" for b in physical["batches"])
    return row, trace


@pytest.mark.parametrize("mode", ["joint", "decode", "prefill"])
def test_budget_terminal_thinker_forward_counts_work_without_a_new_sample(
    tmp_path, tokenizer, journal, mode
):
    task, ref, m, sha = journal
    m = {**m, "budget": 2}
    sha = digest(encode_json(m))
    row, trace = terminal_flush_record(tokenizer, task, sha, mode)
    out = compare.verify_row(save_record(tmp_path, row, trace), tokenizer, [task], ref, m, sha)
    assert out["thinker_samples"] == out["writer_samples"] == 2
    assert out["joint_sampled_steps"] == 1
    assert out["joint_decode_verified"] == (2 if mode == "joint" else 1)
    assert out["forward_batches"] == (5 if mode == "joint" else 6)


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_token",
        "duplicate",
        "early",
        "stopped",
        "wrong_position",
        "fake_budget",
        "unbacked_writer",
    ],
)
def test_terminal_flush_must_match_last_earned_thought(tmp_path, tokenizer, journal, fault):
    task, ref, m, _ = journal
    m = {**m, "budget": 2}
    sha = digest(encode_json(m))
    row, trace = terminal_flush_record(tokenizer, task, sha, "joint")
    b = trace["trace"]["batches"]
    terminal = b[-1]["requests"][0]
    if fault == "wrong_token":
        terminal["input_ids"][0] += 1
    elif fault == "duplicate":
        duplicate = deepcopy(terminal)
        duplicate["write_start"] += 1
        duplicate["cache_view"][1][1] += 1
        b.append({"step": len(b), "kind": "decode", "requests": [duplicate]})
        trace["trace"]["steps"] += 1
        row["decode_batches"] += 1
        row["forward_batches"] += 1
    elif fault == "early":
        next(e for e in trace["samples"] if e["role"] == "thinker" and e["batch_step"] == 3)[
            "batch_step"
        ] = 4
    elif fault == "stopped":
        row["thinker_stop_id"] = tokenizer.eos_token_id
    elif fault == "wrong_position":
        terminal["write_start"] -= 1
    elif fault == "fake_budget":
        m = {**m, "budget": 3}
        sha = digest(encode_json(m))
        row["settings_sha256"] = sha
    else:
        trace["samples"] = [
            e for e in trace["samples"] if not (e["role"] == "writer" and e["batch_step"] == 4)
        ]
    with pytest.raises(
        ValueError, match="terminal thinker flush|Invalid block write|Writer decode has no sample"
    ):
        compare.verify_row(save_record(tmp_path, row, trace), tokenizer, [task], ref, m, sha)
