"""Record, replay, overfit, and freely reproduce one streaming-video response.

The observation protocol follows vovanbart/AsyncLLM-demo: rolling pairs of
images, a persistent thinker, a read-only yes/no gate and an answer branch.
Current AsyncLLM APIs record every mutation. No teacher actions are replayed
at evaluation: only the predetermined frame-arrival schedule is supplied.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import gc
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")
SYSTEM = (
    "You monitor an incoming video stream. Each observation contains two frames: "
    "a previous frame followed by the current frame. Describe only visible changes "
    "relevant to the user. Do not predict future events or invent details. "
    "When there is no clear evidence to answer the user, keep watching. "
    "Do not treat camera movement, scene introductions, ordinary traffic or parked "
    "cars as evidence of a harmful event."
)
THINKER = "<|im_start|>assistant\n<think>\n"
GATE = (
    "<|im_start|>user\nBased only on the frames currently visible, is there clear "
    "visual evidence of a suspicious or harmful activity or destructive accident "
    "that warrants answering the user now? If no such event is visible, wait. "
    "Answer with one word, yes or no.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
WRITER = (
    "<|im_start|>user\nAnswer the original question in one short factual sentence. "
    "Describe the harmful event and its visible evidence. Do not speculate about "
    "people, causes, dates or locations that the frames do not show.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


@dataclass(frozen=True)
class Setup:
    seed: int = 7
    tokens_per_frame: int = 16
    writer_tokens: int = 96
    max_pixels: int = 65536
    kv_tokens: int = 8192
    prefill_rows: int = 128
    dtype: str = "float16"
    replay_attention_backend: str = "torch"
    replay_query_chunk_size: int = 64


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    return value


def replay_options(setup):
    """Pin replay independently of inference and Hugging Face attention settings."""
    backend = setup.get("replay_attention_backend", "torch")
    chunk_size = setup.get("replay_query_chunk_size", 64)
    if backend not in {"torch", "auto", "torch_optimized", "fa4"}:
        raise ValueError("Unknown replay attention backend")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Replay query chunk size must be a positive integer")
    return {"attention_backend": backend, "query_chunk_size": chunk_size}


def require_replay_options(report, setup):
    """Reject cached results measured with different or unrecorded settings."""
    if report.get("replay_options") != replay_options(setup):
        raise ValueError("Saved replay settings differ; use a fresh output directory")
    return report


def task():
    data = load(ASSETS / "task.json")
    for frame in data["frames"]:
        if hashlib.sha256((ASSETS / frame["path"]).read_bytes()).hexdigest() != frame["sha256"]:
            raise ValueError("Frame does not match its pinned manifest")
    return data


def frame_indices(index, control):
    if control not in {"normal", "freeze"}:
        raise ValueError("Use normal or freeze control")
    current = min(index, 1) if control == "freeze" else index
    previous = min(max(index - 1, 0), 1) if control == "freeze" else max(index - 1, 0)
    return previous, current


def prepare_observation(processor, index, cfg, control):
    import torch
    from PIL import Image

    frames = task()["frames"]
    a, b = frame_indices(index, control)
    images = [Image.open(ASSETS / frames[j]["path"]).convert("RGB") for j in (a, b)]
    marker = "<|vision_start|>" + processor.image_token + "<|vision_end|>"
    # Only acquisition timestamps enter the prompt, never filenames, labels or event times.
    text = (
        f"<|im_start|>user\nPrevious frame at {frames[max(index - 1, 0)]['time']:.1f}s: "
        f"{marker}\nCurrent frame at {frames[index]['time']:.1f}s: {marker}\n<|im_end|>\n"
    )
    built = processor(
        text=text,
        images=images,
        return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": 65536, "longest_edge": cfg.max_pixels}},
    )
    ids = built["input_ids"].flatten().to(torch.int32)
    image_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    payload = {
        "input_ids": ids,
        "pixel_values": built["pixel_values"],
        "image_grid_thw": built["image_grid_thw"],
        "mm_token_type_ids": built.get("mm_token_type_ids", (ids == image_id).long()).flatten(),
    }
    return payload, (a, b), text


def word_ids(tokenizer, word):
    ids = {
        encoded[0]
        for form in (word, word.capitalize(), " " + word, " " + word.capitalize())
        if len(encoded := tokenizer.encode(form, add_special_tokens=False)) == 1
    }
    if not ids:
        raise ValueError(f"No one-token spelling of {word}")
    return sorted(ids)


def assess(result):
    manifest = result["task"]
    answer = result["answer_text"]
    expected_frame = manifest["expected_first_reply_frame"]
    reply_time = result["reply_time"]
    lo, hi = manifest["annotation"]["answer"][0]["reply_timespan"]
    facts = {
        "smoke": bool(re.search(r"\bsmok\w*\b", answer, re.I)),
        "building": bool(re.search(r"\b(building|station|tower|structure)\b", answer, re.I)),
        "explosion_or_fire": bool(
            re.search(r"\b(explos\w*|blast\w*|fire|burn\w*)\b", answer, re.I)
        ),
    }
    timing = reply_time is not None and lo <= reply_time <= hi
    expected = [False] * expected_frame + [True]
    decisions = [g["reply"] for g in result["gates"]]
    if result["control"] == "freeze":
        return {
            "correct_wait": len(decisions) == len(manifest["frames"])
            and not any(decisions)
            and not answer,
            "behavior_reproduced": False,
            "reply_time": reply_time,
            "facts": facts,
        }
    return {
        "reply_time": reply_time,
        "in_reference_window": timing,
        "correct_gate_sequence": decisions == expected,
        "facts": facts,
        "answer_grounded": facts["smoke"] and facts["building"],
        "behavior_reproduced": decisions == expected
        and timing
        and facts["smoke"]
        and facts["building"]
        and result["hit_eos"],
    }


async def collect(model, output, setup=None, *, control="normal"):
    import torch
    from minisgl import AsyncLLM
    from minisgl.training import RecordedAsyncLLM

    from minisgl_examples.training.recording import record_sample

    cfg = Setup(**(setup or {}))
    replay_options(asdict(cfg))
    if cfg.tokens_per_frame < 1 or cfg.writer_tokens < 1:
        raise ValueError("Token quotas must be positive")
    frame_indices(0, control)
    manifest = task()
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    llm = AsyncLLM(
        str(model),
        dtype=getattr(torch, cfg.dtype),
        attention_backend="fi",
        cuda_graph_max_bs=0,
        num_page_override=cfg.kv_tokens,
        max_seq_len_override=4096,
        max_running_req=4,
        max_prefill_rows=cfg.prefill_rows,
    )
    started = time.monotonic()
    try:
        tok = llm.tokenizer
        encode = lambda s: tok.encode(s, add_special_tokens=False)
        prompt_text = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
            + manifest["annotation"]["conversation"][0]["content"]
            + "<|im_end|>\n"
        )
        prompt_ids = encode(prompt_text)
        yes, no = word_ids(tok, "yes"), word_ids(tok, "no")
        eos = tok.eos_token_id
        forbidden = [
            tok.convert_tokens_to_ids(s)
            for s in ("</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")
        ]
        forbidden = sorted({i for i in [*forbidden, eos] if i is not None and i >= 0})
        events, gates, samples, records, thinker, writer = [], [], {}, [], [], []
        hit_eos = False
        reply_time = None
        live = []
        with RecordedAsyncLLM(llm) as recorded:
            prompt, thought = [await recorded.create_block() for _ in range(2)]
            live.extend([prompt, thought])
            obs = None

            def event(name, frame, **details):
                trace = recorded.get_trace()
                events.append(
                    {
                        "event": name,
                        "frame": frame,
                        "video_time": manifest["frames"][frame]["time"],
                        "thinker_tokens": len(thinker),
                        "writer_tokens": len(writer),
                        "step": trace.steps,
                        "trace_rows": trace.num_tokens,
                        "blocks": [
                            {"id": b.block_id, "tokens": b.num_tokens, "mrope_span": b.mrope_span}
                            for b in live
                        ],
                        **details,
                    }
                )

            async def forward(ids, view, block, *, role=None, frame=None):
                trace = recorded.get_trace()
                row_cursor, batch_cursor = trace.num_tokens, len(trace.batches)
                out = await recorded.forward(
                    ids, cache_view=view, write_to=block, return_logits=role is not None
                )
                if role is None:
                    return None
                raw = out.logits.float()
                if role == "gate":
                    choices = yes + no
                    token = max(choices, key=lambda i: float(raw[i]))
                    selected = token in yes
                    margin = float(raw[yes].max() - raw[no].max())
                else:
                    scores = raw.clone()
                    if role == "thinker":
                        scores[forbidden] = -float("inf")
                    token = int(scores.argmax())
                record_sample(
                    recorded,
                    samples,
                    block_id=block.block_id,
                    write_end=block.num_tokens,
                    batch_cursor=batch_cursor,
                    row_cursor=row_cursor,
                    token=token,
                    logits=raw,
                )
                row = max(samples)
                records.append({"row": row, "role": role, "frame": frame, "token": token})
                return (token, selected, margin) if role == "gate" else token

            await forward(prompt_ids, [], prompt)
            for index, frame in enumerate(manifest["frames"]):
                # All decode calls have finished. Flush tokens before changing the view.
                if obs is not None:
                    event("frame_barrier", index)
                    await recorded.free_block(obs)
                    live.remove(obs)
                payload, indices, observation_text = prepare_observation(
                    llm.processor, index, cfg, control
                )
                obs = await recorded.create_block()
                live.insert(1, obs)
                await recorded.prefill_block(**payload, cache_view=[prompt], write_to=obs)
                event(
                    "frame_arrival", index, pixel_frames=list(indices), input_text=observation_text
                )
                print(
                    f"frame={index} time={frame['time']}s thinker_tokens={len(thinker)}", flush=True
                )
                view = [prompt, obs, thought]
                initial = encode(THINKER if index == 0 else "\n")
                pending = await forward(initial, view, thought, role="thinker", frame=index)
                for turn in range(cfg.tokens_per_frame):
                    thinker.append(pending)
                    if turn + 1 < cfg.tokens_per_frame:
                        pending = await forward(
                            [pending], view, thought, role="thinker", frame=index
                        )
                # Last emitted token belongs to this observation; discarded logits have no loss.
                await forward([pending], view, thought)
                gate = await recorded.create_block()
                live.append(gate)
                token, selected, margin = await forward(
                    encode(GATE), [prompt, obs, gate], gate, role="gate", frame=index
                )
                gates.append(
                    {
                        "frame": index,
                        "video_time": frame["time"],
                        "reply": selected,
                        "token": token,
                        "margin": margin,
                    }
                )
                event("gate", index, reply=selected, margin=margin)
                await recorded.free_block(gate)
                live.remove(gate)
                print(f"gate={selected} margin={margin:.4f}", flush=True)
                if not selected:
                    continue
                reply_time = frame["time"]
                answer = await recorded.create_block()
                live.append(answer)
                event("reply_start", index)
                answer_view = [prompt, obs, thought, answer]
                pending = await forward(
                    encode(WRITER), answer_view, answer, role="writer", frame=index
                )
                for turn in range(cfg.writer_tokens):
                    writer.append(pending)
                    if pending == eos:
                        hit_eos = True
                        break
                    if turn + 1 < cfg.writer_tokens:
                        pending = await forward(
                            [pending], answer_view, answer, role="writer", frame=index
                        )
                event("reply_end" if hit_eos else "writer_budget", index)
                break
            trace = recorded.get_trace()
            for block in reversed(live):
                await recorded.free_block(block)
            targets, logprobs = [-100] * trace.num_tokens, [None] * trace.num_tokens
            for row, (token, lp) in samples.items():
                targets[row], logprobs[row] = token, lp
        result = {
            "model": str(model),
            "setup": asdict(cfg),
            "control": control,
            "task": manifest,
            "prompt_ids": prompt_ids,
            "gate_ids": encode(GATE),
            "writer_ids": encode(WRITER),
            "thinker_ids": thinker,
            "writer_token_ids": writer,
            "gates": gates,
            "thinker_text": tok.decode(thinker, skip_special_tokens=False),
            "answer_text": tok.decode(writer, skip_special_tokens=True),
            "reply_time": reply_time,
            "hit_eos": hit_eos,
            "events": events,
            "trace": asdict(trace),
            "targets": targets,
            "logprobs": logprobs,
            "samples": records,
            "elapsed_seconds": time.monotonic() - started,
            "runtime": {
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "gpu": torch.cuda.get_device_name(),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            },
        }
        result["replay_options"] = replay_options(asdict(cfg))
        result["assessment"] = assess(result)
        save(output, result)
        print(json.dumps(result["assessment"]), result["answer_text"], flush=True)
        return result["assessment"]
    finally:
        await llm.close()


def prepare_trace(data, device="cuda"):
    from minisgl.training import Trace

    return Trace.from_dict(data["trace"]).to(device, **replay_options(data["setup"]))


def load_for_replay(source, model=None, dtype="float32"):
    import torch

    from minisgl_examples.training.student import load_student

    data = load(source)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, trace, targets, processor = load_student(
        {**data, "trace": prepare_trace(data, "cpu")},
        str(model or data["model"]),
        dtype=getattr(torch, dtype),
    )
    model.set_attn_implementation("eager")
    return data, model.eval(), trace.to("cuda"), targets, processor


def check_replay(source, output, *, dtype="float32"):
    import torch

    from minisgl_examples.training.loss import evaluate

    data, model, trace, targets, _ = load_for_replay(source, dtype=dtype)
    measured = evaluate(model, [(trace, targets)], 32)
    errors = [
        abs(nll + lp)
        for nll, lp in zip(
            measured["examples"][0]["nll"],
            (x for x in data["logprobs"] if x is not None),
            strict=True,
        )
    ]
    phases = {}
    for sample, error in zip(data["samples"], errors, strict=True):
        phases.setdefault(f"frame_{sample['frame']}/{sample['role']}", []).append(error)
    parity = {
        "max_abs_logprob_error": max(errors),
        "mean_abs_logprob_error": sum(errors) / len(errors),
        "by_phase": {k: {"count": len(v), "max_abs": max(v)} for k, v in phases.items()},
        "argmax_target_accuracy": measured["accuracy"],
        "tolerance": 0.1,
        "passed": max(errors) <= 0.1,
        "errors": errors,
    }
    save(
        output,
        {
            "dtype": dtype,
            "parity": parity,
            "replay": measured,
            "replay_options": replay_options(data["setup"]),
        },
    )
    print(json.dumps({k: v for k, v in parity.items() if k != "errors"}), flush=True)
    del model, trace, targets
    gc.collect()
    torch.cuda.empty_cache()
    if not parity["passed"]:
        raise RuntimeError("Inference/replay mismatch; do not start training")
    return parity


def image_signatures(data):
    fields = ("input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids", "mrope_positions")
    return [
        hashlib.sha256(json.dumps({k: r[k] for k in fields}, sort_keys=True).encode()).hexdigest()
        for batch in data["trace"]["batches"]
        for r in batch["requests"]
        if r.get("pixel_values") is not None
    ]


def pixel_signatures(data):
    """Fingerprint each real image separately, independent of timestamp text."""
    signatures = []
    for batch in data["trace"]["batches"]:
        for request in batch["requests"]:
            values = request.get("pixel_values")
            if values is None:
                continue
            grids = request["image_grid_thw"]["data"]
            width = values["shape"][1]
            start, pair = 0, []
            for i in range(0, len(grids), 3):
                grid = grids[i : i + 3]
                count = math.prod(grid) * width
                pair.append(
                    hashlib.sha256(
                        json.dumps(
                            {
                                "grid": grid,
                                "dtype": values["dtype"],
                                "pixels": values["data"][start : start + count],
                            }
                        ).encode()
                    ).hexdigest()
                )
                start += count
            if start != len(values["data"]):
                raise ValueError("Image pixel slices do not match their grids")
            signatures.append(pair)
    return signatures


def train(teacher_path, student, baseline_path, output, *, steps=80, lr=3e-6):
    import torch

    from minisgl_examples.training.loss import compute_supervised_loss, evaluate

    teacher, baseline = load(teacher_path), load(baseline_path)
    if not teacher["assessment"]["behavior_reproduced"]:
        raise ValueError("Teacher must pass answer/timing checks before training")
    if baseline["model"] != str(student) or baseline["setup"] != teacher["setup"]:
        raise ValueError("Student baseline or async configuration differs")
    for key in ("prompt_ids", "gate_ids", "writer_ids", "task"):
        if teacher[key] != baseline[key]:
            raise ValueError(f"Mismatched conditioning: {key}")
    # A baseline may answer early; compare exactly its overlapping observations.
    a, b = image_signatures(teacher), image_signatures(baseline)
    if a[: min(len(a), len(b))] != b[: min(len(a), len(b))]:
        raise ValueError("Actual processed pixels differ")
    check_replay(baseline_path, Path(output) / "student_initial_parity.json")
    data, model, trace, targets, processor = load_for_replay(teacher_path, student)
    torch.manual_seed(data["setup"]["seed"])
    model.model.visual.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    probe = next(p for n, p in model.named_parameters() if p.requires_grad and "q_proj" in n)
    original = probe.detach().clone()
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    examples = [(trace, targets)]
    initial = final = evaluate(model.eval(), examples, 32)
    history = [{"step": 0, "loss": initial["loss"], "accuracy": initial["accuracy"]}]
    count = int((targets != -100).sum())
    started = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = compute_supervised_loss(model, trace, targets, 32) / count
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if norm == 0:
            raise RuntimeError("No gradient")
        optimizer.step()
        if step % 5 == 0 or step == steps:
            final = evaluate(model.eval(), examples, 32)
            row = {
                "step": step,
                "loss": final["loss"],
                "accuracy": final["accuracy"],
                "sequence_probability": math.exp(-sum(final["examples"][0]["nll"])),
                "grad_norm": float(norm),
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if row["accuracy"] == 1.0 and row["sequence_probability"] >= 0.95:
                break
    delta = (probe.detach() - original).abs().max().item()
    if delta == 0:
        raise RuntimeError("Parameters did not change")
    checkpoint = Path(output) / "checkpoint"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    model.save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)
    report = {
        "student": str(student),
        "teacher": str(teacher_path),
        "steps": step,
        "lr": lr,
        "dtype": "float32",
        "vision_frozen": True,
        "replay_options": replay_options(data["setup"]),
        "initial": initial,
        "final": final,
        "history": history,
        "tracked_parameter_max_delta": delta,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint": str(checkpoint),
    }
    save(Path(output) / "training.json", report)
    return report


def verify_checkpoint(teacher_path, checkpoint, training_path, output):
    from minisgl_examples.training.loss import evaluate

    data = load(teacher_path)
    training = require_replay_options(load(training_path), data["setup"])
    _, model, trace, targets, _ = load_for_replay(teacher_path, checkpoint)
    measured = evaluate(model, [(trace, targets)], 32)
    error = abs(measured["loss"] - training["final"]["loss"])
    result = {
        "reloaded": measured,
        "saved_loss_difference": error,
        "passed": error < 1e-5,
        "replay_options": replay_options(data["setup"]),
    }
    save(output, result)
    if not result["passed"]:
        raise RuntimeError("Checkpoint reload changed training replay")
    return result


def inspect_rollout(source):
    from minisgl.training import Trace

    data = load(source)
    trace = Trace.from_dict(data["trace"])
    rows = [s["row"] for s in data["samples"]]
    assert len(rows) == len(set(rows))
    assert rows == sorted(rows)
    assert {i for i, t in enumerate(data["targets"]) if t != -100} == set(rows)
    assert all(data["targets"][s["row"]] == s["token"] for s in data["samples"])
    assert all(
        (p is not None) == (t != -100)
        for t, p in zip(data["targets"], data["logprobs"], strict=True)
    )
    arrivals = [e for e in data["events"] if e["event"] == "frame_arrival"]
    assert [e["thinker_tokens"] for e in arrivals] == [
        i * data["setup"]["tokens_per_frame"] for i in range(len(arrivals))
    ]
    images = [r for b in trace.batches for r in b.requests if r.pixel_values is not None]
    assert len(images) == len(arrivals)
    for image in images:
        assert image.mrope_positions is not None
        assert image.image_grid_thw.to_tensor().shape == (2, 3)
    for old, new in zip(arrivals, arrivals[1:]):
        old_obs = old["blocks"][1]["id"]
        assert any(
            u.block == old_obs and u.refs == () and old["step"] <= u.step <= new["step"]
            for u in trace.updates
        )
        assert old["blocks"][-1]["id"] == new["blocks"][-1]["id"]
    summary = {
        "trace_rows": trace.num_tokens,
        "supervised_rows": len(rows),
        "image_requests": len(images),
        "images_per_request": [2] * len(images),
        "views": len({tuple(t.view) for t in trace.tokens}),
        "updates": [asdict(u) for u in trace.updates],
        "image_mrope_shapes": [r.mrope_positions.shape for r in images],
        "events": data["events"],
    }
    return summary


def compare(teacher_path, before_path, after_path, control_path, output):
    teacher = load(teacher_path)
    reports = {}
    for name, path in (("before", before_path), ("after", after_path), ("freeze", control_path)):
        run = load(path)
        if run["setup"] != teacher["setup"] or any(
            run[k] != teacher[k] for k in ("prompt_ids", "gate_ids", "writer_ids", "task")
        ):
            raise ValueError("Different async inputs")
        t, r = image_signatures(teacher), image_signatures(run)
        if name != "freeze" and t[: min(len(t), len(r))] != r[: min(len(t), len(r))]:
            raise ValueError("Different actual pixels")
        if name == "freeze":
            pre_event = pixel_signatures(teacher)
            known = {0: pre_event[0][0], 1: pre_event[1][1]}
            arrivals = [e for e in run["events"] if e["event"] == "frame_arrival"]
            for pair, event in zip(pixel_signatures(run), arrivals, strict=True):
                expected = [known[i] for i in frame_indices(event["frame"], "freeze")]
                if pair != expected:
                    raise ValueError("Freeze control contains different or future pixels")
            # Timestamp text and update schedule stay identical; verify actual input pixels too.
            assert all(
                max(e["pixel_frames"]) <= 1 for e in run["events"] if e["event"] == "frame_arrival"
            )
        ids = lambda d, role: d["thinker_ids"] if role == "thinker" else d["writer_token_ids"]
        metrics = {}
        for role in ("thinker", "writer"):
            a, b = ids(teacher, role), ids(run, role)
            metrics[role] = {
                "exact": a == b,
                "common_prefix": next(
                    (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
                ),
                "positional_agreement": sum(x == y for x, y in zip(a, b)) / max(len(a), len(b), 1),
                "sequence_similarity": difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(),
            }
        gate_ids = [g["token"] for g in run["gates"]]
        expected_gate_ids = [g["token"] for g in teacher["gates"]]
        gates_equal = gate_ids == expected_gate_ids
        events = lambda d: [
            (e["event"], e["frame"], e["thinker_tokens"])
            for e in d["events"]
            if e["event"] in {"frame_arrival", "reply_start"}
        ]
        reports[name] = {
            "assessment": run["assessment"],
            "tokens": metrics,
            "exact_gate_tokens": gates_equal,
            "same_events": events(run) == events(teacher),
            "exact_trajectory": gates_equal and all(v["exact"] for v in metrics.values()),
            "reply_time": run["reply_time"],
            "answer": run["answer_text"],
        }
    result = {
        "replay_options": replay_options(teacher["setup"]),
        "runs": reports,
        "free_running_success": reports["after"]["assessment"]["behavior_reproduced"]
        and reports["after"]["same_events"],
        "exact_trajectory": reports["after"]["exact_trajectory"],
        "visual_control_passed": reports["freeze"]["assessment"]["correct_wait"],
    }
    save(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--control", choices=["normal", "freeze"], default="normal")
    c.add_argument("--setup", type=Path)
    c.add_argument(
        "--replay-attention-backend", choices=["torch", "auto", "torch_optimized", "fa4"]
    )
    c.add_argument("--replay-query-chunk-size", type=int)
    c = sub.add_parser("check")
    c.add_argument("--source", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    c = sub.add_parser("train")
    c.add_argument("--teacher", required=True)
    c.add_argument("--student", required=True)
    c.add_argument("--baseline", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--steps", type=int, default=80)
    c.add_argument("--lr", type=float, default=3e-6)
    a = p.parse_args()
    if a.command == "collect":
        setup = load(a.setup) if a.setup else {}
        if a.replay_attention_backend is not None:
            setup["replay_attention_backend"] = a.replay_attention_backend
        if a.replay_query_chunk_size is not None:
            setup["replay_query_chunk_size"] = a.replay_query_chunk_size
        replay_options(setup)
        asyncio.run(collect(a.model, a.output, setup, control=a.control))
    elif a.command == "check":
        check_replay(a.source, a.output, dtype=a.dtype)
    else:
        train(a.teacher, a.student, a.baseline, a.output, steps=a.steps, lr=a.lr)


if __name__ == "__main__":
    main()
