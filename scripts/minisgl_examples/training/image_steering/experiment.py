"""A real AsyncLLM image replacement, trace replay, and free-running overfit check.

Run stages in fresh processes (as in the text notebooks). No generated token is
injected during evaluation. Only the fixed, answer-free user notification is replayed.
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
QUESTION = (
    "An administrator at the Department of Motor Vehicles (DMV) tracked the average "
    "wait time from month to month. According to the table, what was the rate of "
    "change between August and September? (Unit: minutes per month)"
)
NOTICE = (
    "</think><|im_end|>\n<|im_start|>user\n"
    "The earlier picture contained an error. It has now been corrected. "
    "Recheck your reasoning using the current picture."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n"
)
SYSTEM = (
    "Solve the visual question. Read the relevant table entries, explain the "
    "calculation briefly, and put the final answer in \\boxed{}. "
    "If the user corrects the picture, use the current picture and revise any "
    "earlier calculation. Do not assume the numbers in advance."
)


@dataclass(frozen=True)
class Setup:
    seed: int = 7
    update_after: int = 96
    max_tokens: int = 640
    max_pixels: int = 65536
    kv_tokens: int = 4096
    prefill_rows: int = 128
    dtype: str = "float16"
    replay_attention_backend: str = "torch"
    replay_query_chunk_size: int = 64


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each stage owns its output; never overwrite an existing run silently.
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
    return data


def load(path):
    return json.loads(Path(path).read_text())


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
    return {
        "question": QUESTION,
        "answer_before": "3",
        "answer_after": "-3",
        "images": {
            s: hashlib.sha256((ASSETS / f"{s}.png").read_bytes()).hexdigest()
            for s in ("before", "after")
        },
        "provenance": load(ASSETS / "provenance.json"),
    }


def image_inputs(processor, side, max_pixels):
    import torch
    from PIL import Image

    with Image.open(ASSETS / f"{side}.png") as im:
        values = processor(
            text="<|vision_start|>" + processor.image_token + "<|vision_end|>\n",
            images=im.convert("RGB"),
            return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": 65536, "longest_edge": max_pixels}},
        )
    ids = values["input_ids"].flatten().to(torch.int32)
    image_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    return {
        "input_ids": ids,
        "pixel_values": values["pixel_values"],
        "image_grid_thw": values["image_grid_thw"],
        "mm_token_type_ids": values.get("mm_token_type_ids", (ids == image_id).long()).flatten(),
    }


def image_signatures(data):
    """Bind comparisons to the actual processed tensors, not only image filenames."""
    fields = ("input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids", "mrope_positions")
    return [
        hashlib.sha256(json.dumps({k: r[k] for k in fields}, sort_keys=True).encode()).hexdigest()
        for batch in data["trace"]["batches"]
        for r in batch["requests"]
        if r.get("pixel_values") is not None
    ]


def stages(result):
    """Task-specific evidence checks; full text is retained for human inspection."""
    before, after = result["before_update_text"], result["after_update_text"]
    boxes = re.findall(r"\\boxed\{\s*([+-]?\s*\d+)\s*\}", result["text"])
    answer = boxes[-1].replace(" ", "") if boxes else None
    changed = result["image_replaced"]
    old_seen = bool(re.search(r"\b20\b", before))
    new_seen = bool(re.search(r"\b14\b", after))
    revision = bool(
        re.search(
            r"\b(?:correct\w*|updat\w*|revis\w*|instead|now|error|recheck|however|wait)\b",
            after,
            re.I,
        )
    )
    expected = "-3" if changed and result.get("replacement_image", "after") == "after" else "3"
    return {
        "final_answer": answer,
        "correct_answer": answer == expected,
        "old_20_before_event": old_seen,
        "new_14_after_event": new_seen,
        "revision_language_after_event": revision,
        "behavior_reproduced": changed
        and old_seen
        and new_seen
        and answer == "-3"
        and result["hit_eos"],
    }


async def collect(model, output, setup=None, *, update=True, replacement_image="after"):
    """Generate from logits through AsyncLLM; retain real views and row-aligned labels."""
    import torch
    from minisgl import AsyncLLM
    from minisgl.training import RecordedAsyncLLM

    from minisgl_examples.training.recording import record_sample

    if replacement_image not in {"before", "after"}:
        raise ValueError("replacement_image must be before/after")
    cfg = Setup(**(setup or {}))
    replay_options(asdict(cfg))
    if cfg.update_after < 1 or cfg.max_tokens <= cfg.update_after:
        raise ValueError("Require 0 < update_after < max_tokens")
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
        pair = {s: image_inputs(llm.processor, s, cfg.max_pixels) for s in ("before", "after")}
        for field in ("input_ids", "image_grid_thw", "mm_token_type_ids"):
            if not torch.equal(pair["before"][field], pair["after"][field]):
                raise ValueError(f"Image replacement changes processor layout: {field}")
        tok = llm.tokenizer
        encode = lambda text: tok.encode(text, add_special_tokens=False)
        prompt_ids = encode(
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{QUESTION}\n"
        )
        prefix_ids = encode("<|im_end|>\n<|im_start|>assistant\n<think>\n")
        notice_ids = encode(NOTICE)
        generated, samples, events = [], {}, []
        sampled_rows, phases = [], []
        image_replaced = False
        hit_eos = False
        with RecordedAsyncLLM(llm) as recorded:
            prompt, picture, reasoning = [await recorded.create_block() for _ in range(3)]
            live = [prompt, picture, reasoning]

            def event(name, **extra):
                trace = recorded.get_trace()
                events.append(
                    {
                        "event": name,
                        "generated_tokens": len(generated),
                        "trace_rows": trace.num_tokens,
                        "step": trace.steps,
                        "blocks": [
                            {"id": b.block_id, "tokens": b.num_tokens, "mrope_span": b.mrope_span}
                            for b in live
                        ],
                        **extra,
                    }
                )

            async def run(ids, view, block, *, supervise=False, multimodal=None):
                trace = recorded.get_trace()
                cursor, batch_cursor = trace.num_tokens, len(trace.batches)
                out = await recorded.forward(
                    ids,
                    cache_view=view,
                    write_to=block,
                    return_logits=supervise,
                    **(multimodal or {}),
                )
                if not supervise:
                    return None
                logits = out.logits.float()
                token = int(logits.argmax().item())  # identical greedy policy for every model
                record_sample(
                    recorded,
                    samples,
                    block_id=block.block_id,
                    write_end=block.num_tokens,
                    batch_cursor=batch_cursor,
                    row_cursor=cursor,
                    token=token,
                    logits=logits,
                )
                row = max(samples)
                sampled_rows.append(row)
                phases.append("after" if image_replaced else "before")
                return token

            await run(prompt_ids, [], prompt)
            await recorded.prefill_block(**pair["before"], cache_view=[prompt], write_to=picture)
            event("image_A")
            view = [prompt, picture, reasoning]
            pending = await run(prefix_ids, view, reasoning, supervise=True)
            while len(generated) < cfg.max_tokens:
                generated.append(pending)
                if pending == tok.eos_token_id:
                    hit_eos = True
                    break
                # Commit exactly the kth sampled token in the old view before mutation.
                # This flush has no label: its next-token logits are discarded.
                if update and len(generated) == cfg.update_after:
                    await run([pending], view, reasoning)
                    event("update_barrier")
                    old_id = picture.block_id
                    await recorded.free_block(picture)
                    live.remove(picture)
                    picture = await recorded.create_block()
                    live.insert(1, picture)
                    await recorded.prefill_block(
                        **pair[replacement_image], cache_view=[prompt], write_to=picture
                    )
                    view = [prompt, picture, reasoning]
                    image_replaced = True
                    event("image_B", replaced_block=old_id, notice=NOTICE)
                    pending = await run(notice_ids, view, reasoning, supervise=True)
                else:
                    pending = await run([pending], view, reasoning, supervise=True)
            event("eos" if hit_eos else "budget_exhausted")
            trace = recorded.get_trace()
            targets = [-100] * trace.num_tokens
            logprobs = [None] * trace.num_tokens
            for row, (token, lp) in samples.items():
                targets[row], logprobs[row] = token, lp
            # A budget-ending forward can produce an unused sample; remove its label.
            used_rows = sampled_rows[: len(generated)]
            if len(sampled_rows) > len(generated):
                for row in sampled_rows[len(generated) :]:
                    targets[row], logprobs[row] = -100, None
            event("cleanup")
            for block in reversed(live):
                await recorded.free_block(block)
        cut = cfg.update_after if image_replaced else len(generated)
        result = {
            "model": str(model),
            "setup": asdict(cfg),
            "update": update,
            "replacement_image": replacement_image,
            "task": task(),
            "prompt_ids": prompt_ids,
            "prefix_ids": prefix_ids,
            "notice_ids": notice_ids,
            "token_ids": generated,
            "sample_rows": used_rows,
            "sample_phases": phases[: len(generated)],
            "text": tok.decode(generated, skip_special_tokens=False),
            "before_update_text": tok.decode(generated[:cut], skip_special_tokens=False),
            "after_update_text": tok.decode(generated[cut:], skip_special_tokens=False),
            "image_replaced": image_replaced,
            "hit_eos": hit_eos,
            "events": events,
            "trace": asdict(trace),
            "targets": targets,
            "logprobs": logprobs,
            "elapsed_seconds": time.monotonic() - started,
            "runtime": {
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "model_revision": getattr(llm.config.hf_config, "_commit_hash", None),
                "gpu": torch.cuda.get_device_name(),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            },
            "image_grid": pair["before"]["image_grid_thw"].tolist(),
        }
        result["replay_options"] = replay_options(asdict(cfg))
        result["assessment"] = stages(result)
        save(output, result)
        print(json.dumps(result["assessment"]), flush=True)
        print(result["text"], flush=True)
        return result["assessment"]
    finally:
        await llm.close()


def prepare_trace(data, device="cuda"):
    from minisgl.training import Trace

    return Trace.from_dict(data["trace"]).to(device, **replay_options(data["setup"]))


def load_training(path, model_path=None, *, dtype="float32"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    data = load(path)
    source = str(model_path or data["model"])
    a = AutoTokenizer.from_pretrained(data["model"])
    b = AutoTokenizer.from_pretrained(source)
    if a.get_vocab() != b.get_vocab() or a.all_special_ids != b.all_special_ids:
        raise ValueError("Teacher/student token IDs differ")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModelForImageTextToText.from_pretrained(
            source,
            dtype=getattr(torch, dtype),
            attn_implementation="eager",
        )
        .cuda()
        .eval()
    )
    trace = prepare_trace(data)
    targets = torch.tensor(data["targets"], device="cuda")
    return data, model, trace, targets


def parity(data, measured, tolerance=0.1):
    errors = [
        abs(nll + lp)
        for nll, lp in zip(
            measured["examples"][0]["nll"],
            (lp for lp in data["logprobs"] if lp is not None),
            strict=True,
        )
    ]
    groups = {
        s: [e for e, phase in zip(errors, data["sample_phases"], strict=True) if phase == s]
        for s in ("before", "after")
    }
    return {
        "max_abs_logprob_error": max(errors),
        "mean_abs_logprob_error": sum(errors) / len(errors),
        "by_phase": {
            s: {"count": len(es), "max_abs": max(es, default=0)} for s, es in groups.items()
        },
        "argmax_target_accuracy": measured["accuracy"],
        "tolerance": tolerance,
        "passed": max(errors) <= tolerance,
        "errors": errors,
    }


def check_replay(source, output, *, dtype="float32"):
    from minisgl_examples.training.loss import evaluate

    data, model, trace, targets = load_training(source, dtype=dtype)
    measured = evaluate(model, [(trace, targets)], chunk_size=32)
    report = {
        "dtype": dtype,
        "parity": parity(data, measured),
        "replay": measured,
        "replay_options": replay_options(data["setup"]),
    }
    save(output, report)
    if not report["parity"]["passed"]:
        raise RuntimeError(f"Replay parity failed: {report['parity']}")
    print(json.dumps({k: v for k, v in report["parity"].items() if k != "errors"}), flush=True)
    del model, trace, targets
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return report


def train(teacher_path, student, baseline_path, output, *, steps=80, lr=3e-6):
    import torch
    from transformers import AutoProcessor

    from minisgl_examples.training.loss import compute_supervised_loss, evaluate

    teacher = load(teacher_path)
    if not teacher["assessment"]["behavior_reproduced"]:
        raise ValueError(
            "Teacher must read old image, revise using B, answer correctly, and finish"
        )
    baseline = load(baseline_path)
    if baseline["model"] != str(student) or baseline["setup"] != teacher["setup"]:
        raise ValueError("Baseline must use this student and identical async setup")
    if image_signatures(teacher) != image_signatures(baseline):
        raise ValueError("Teacher/student processed image tensors differ")
    if any(teacher[k] != baseline[k] for k in ("prompt_ids", "prefix_ids", "notice_ids", "task")):
        raise ValueError("Teacher/student conditioning differs")
    # Validate this student's own inference first; teacher logprobs have different weights.
    parity_path = Path(output) / "student_initial_parity.json"
    check_replay(baseline_path, parity_path)
    data, model, trace, targets = load_training(teacher_path, student)
    torch.manual_seed(data["setup"]["seed"])
    # Vision replay still recomputes both images; freeze it to keep this experiment small.
    model.model.visual.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    probe = next(
        (p for n, p in model.named_parameters() if p.requires_grad and "q_proj" in n), params[0]
    )
    original = probe.detach().clone()
    optimizer = torch.optim.AdamW(params, lr=lr)
    examples = [(trace, targets)]
    initial = evaluate(model, examples, 32)
    history = [{"step": 0, "loss": initial["loss"], "accuracy": initial["accuracy"]}]
    count = int((targets != -100).sum())
    started = time.monotonic()
    final = initial
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = compute_supervised_loss(model, trace, targets, 32) / count
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm) or norm == 0:
            raise RuntimeError("Missing or nonfinite gradients")
        optimizer.step()
        if step % 5 == 0 or step == steps:
            final = evaluate(model.eval(), examples, 32)
            nll = final["examples"][0]["nll"]
            row = {
                "step": step,
                "loss": final["loss"],
                "accuracy": final["accuracy"],
                "sequence_probability": math.exp(-sum(nll)),
                "grad_norm": float(norm),
                "elapsed_seconds": time.monotonic() - started,
                "runtime": {
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "model_revision": getattr(model.config, "_commit_hash", None),
                    "gpu": torch.cuda.get_device_name(),
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                },
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if final["accuracy"] == 1.0 and row["sequence_probability"] >= 0.95:
                break
    delta = (probe.detach() - original).abs().max().item()
    if delta == 0:
        raise RuntimeError("Optimizer did not change the tracked parameter")
    checkpoint = Path(output) / "checkpoint"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    model.save_pretrained(checkpoint)
    AutoProcessor.from_pretrained(student).save_pretrained(checkpoint)
    report = {
        "teacher": str(teacher_path),
        "student": str(student),
        "steps": step,
        "lr": lr,
        "dtype": "float32",
        "vision_frozen": True,
        "replay_options": replay_options(data["setup"]),
        "initial": initial,
        "final": final,
        "history": history,
        "tracked_parameter_max_delta": delta,
        "checkpoint": str(checkpoint),
    }
    save(Path(output) / "training.json", report)
    return {"initial_loss": initial["loss"], "final_loss": final["loss"], "parameter_delta": delta}


def compare(teacher_path, before_path, after_path, control_path, output):
    teacher = load(teacher_path)
    runs = {
        "before": load(before_path),
        "after": load(after_path),
        "without_update": load(control_path),
    }
    reports = {}
    for name, run in runs.items():
        expected_images = (
            image_signatures(teacher) if run["image_replaced"] else image_signatures(teacher)[:1]
        )
        if image_signatures(run) != expected_images:
            raise ValueError(f"Mismatched processed image tensors: {name}")
        if run["setup"] != teacher["setup"] or any(
            run[k] != teacher[k] for k in ("prompt_ids", "prefix_ids", "notice_ids", "task")
        ):
            raise ValueError(f"Mismatched input conditions: {name}")
        a, b = teacher["token_ids"], run["token_ids"]
        prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        signature = lambda r: [
            (e["event"], e["generated_tokens"])
            for e in r["events"]
            if e["event"] in {"image_A", "update_barrier", "image_B"}
        ]
        reports[name] = {
            **run["assessment"],
            "exact_tokens": a == b,
            "matching_prefix_tokens": prefix,
            "positional_token_agreement": sum(x == y for x, y in zip(a, b)) / max(len(a), len(b)),
            "token_sequence_similarity": difflib.SequenceMatcher(
                None, a, b, autojunk=False
            ).ratio(),
            "teacher_tokens": len(a),
            "student_tokens": len(b),
            "same_external_events": signature(run) == signature(teacher),
            "events": run["events"],
        }
    result = {
        "replay_options": replay_options(teacher["setup"]),
        "teacher": teacher["assessment"],
        "runs": reports,
        "free_running_success": reports["after"]["behavior_reproduced"]
        and reports["after"]["same_external_events"],
        "exact_trajectory_reproduction": reports["after"]["exact_tokens"],
        "visual_counterfactual_passed": reports["after"]["correct_answer"]
        and reports["without_update"]["correct_answer"]
        and reports["after"]["final_answer"] != reports["without_update"]["final_answer"],
        "semantic_metric_note": "Task-specific evidence flags, not a general semantic reasoning judge; inspect both phase texts.",
    }
    save(output, result)
    return result


def compare_pixel_control(teacher_path, after_path, control_path, output):
    """Same event/notice with A replacing A: isolate the new pixels from the cue."""
    teacher, after, control = (load(p) for p in (teacher_path, after_path, control_path))
    if any(
        after[k] != control[k] for k in ("setup", "prompt_ids", "prefix_ids", "notice_ids", "task")
    ):
        raise ValueError("Pixel control has different nonvisual conditioning")
    if image_signatures(control) != image_signatures(teacher)[:1] * 2:
        raise ValueError("Pixel control must contain the exact original image twice")
    signature = lambda d: [
        (e["event"], e["generated_tokens"])
        for e in d["events"]
        if e["event"] in {"image_A", "update_barrier", "image_B"}
    ]
    if signature(after) != signature(control):
        raise ValueError("Pixel control has a different external event schedule")
    report = {
        "replay_options": replay_options(teacher["setup"]),
        "same_nonvisual_inputs": True,
        "same_external_events": True,
        "identical_A_pixels_reused": True,
        "with_B": after["assessment"],
        "with_A_again": control["assessment"],
        "passed": after["assessment"]["final_answer"] == "-3"
        and control["assessment"]["final_answer"] == "3"
        and after["hit_eos"]
        and control["hit_eos"],
    }
    save(output, report)
    return report


def inspect_rollout(path):
    """Audit saved targets, image order, clear events, and actual attention views."""
    from minisgl.training import Trace

    data = load(path)
    trace = Trace.from_dict(data["trace"])
    tokens = trace.tokens
    rows = data["sample_rows"]
    assert len(rows) == len(data["token_ids"]) == len(set(rows))
    assert [data["targets"][i] for i in rows] == data["token_ids"]
    assert {i for i, t in enumerate(data["targets"]) if t != -100} == set(rows)
    assert all(
        (lp is not None) == (t != -100)
        for t, lp in zip(data["targets"], data["logprobs"], strict=True)
    )
    pictures, start = [], 0
    for batch in trace.batches:
        for request in batch.requests:
            if request.pixel_values is not None:
                assert request.mrope_positions is not None
                pictures.append(
                    {
                        "step": batch.step,
                        "block": request.write_to,
                        "rows": [start, start + len(request.input_ids)],
                        "pixels_shape": request.pixel_values.shape,
                        "grid": request.image_grid_thw.to_tensor().tolist(),
                        "mrope_shape": request.mrope_positions.shape,
                        "cache_view": request.cache_view,
                    }
                )
            start += len(request.input_ids)
    assert len(pictures) == (2 if data["image_replaced"] else 1)
    if data["image_replaced"]:
        boundary = next(e for e in data["events"] if e["event"] == "update_barrier")
        event = next(e for e in data["events"] if e["event"] == "image_B")
        assert event["generated_tokens"] == data["setup"]["update_after"]
        assert any(
            u.block == event["replaced_block"]
            and u.refs == ()
            and boundary["step"] <= u.step <= event["step"]
            for u in trace.updates
        )
        # Existing reasoning blocks and their lengths survive the image replacement.
        assert boundary["blocks"][-1] == event["blocks"][-1]
        assert all(
            tokens[r].step >= event["step"]
            for r, phase in zip(rows, data["sample_phases"], strict=True)
            if phase == "after"
        )
    selected = sorted(
        set(
            rows[:3]
            + rows[max(0, data["setup"]["update_after"] - 2) : data["setup"]["update_after"] + 3]
            + rows[-3:]
        )
    )
    return {
        "trace_rows": trace.num_tokens,
        "supervised_rows": len(rows),
        "batches": len(trace.batches),
        "images": pictures,
        "updates": [asdict(u) for u in trace.updates],
        "sample_rows": [
            {
                "row": r,
                "input": tokens[r].token_id,
                "target": data["targets"][r],
                "step": tokens[r].step,
                "position": tokens[r].position,
                "view": tokens[r].view,
            }
            for r in selected
        ],
    }


def verify_checkpoint(teacher_path, checkpoint, training_path, output):
    from minisgl_examples.training.loss import evaluate

    data = load(teacher_path)
    training = require_replay_options(load(training_path), data["setup"])
    _, model, trace, targets = load_training(teacher_path, checkpoint)
    measured = evaluate(model, [(trace, targets)], 32)
    trained = training["final"]
    error = abs(measured["loss"] - trained["loss"])
    report = {
        "reloaded": measured,
        "saved_loss_difference": error,
        "passed": error < 1e-5,
        "replay_options": replay_options(data["setup"]),
    }
    save(output, report)
    if not report["passed"]:
        raise RuntimeError("Saved checkpoint changed replay loss")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--setup", type=Path)
    c.add_argument(
        "--replay-attention-backend", choices=["torch", "auto", "torch_optimized", "fa4"]
    )
    c.add_argument("--replay-query-chunk-size", type=int)
    c.add_argument("--no-update", action="store_true")
    c.add_argument("--unchanged-image", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--source", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--dtype", default="float32", choices=["float16", "float32"])
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
        asyncio.run(
            collect(
                a.model,
                a.output,
                setup,
                update=not a.no_update,
                replacement_image="before" if a.unchanged_image else "after",
            )
        )
    elif a.command == "check":
        check_replay(a.source, a.output, dtype=a.dtype)
    else:
        train(a.teacher, a.student, a.baseline, a.output, steps=a.steps, lr=a.lr)


if __name__ == "__main__":
    main()
