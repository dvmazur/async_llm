"""Record and freely reproduce a real closed-loop Health Gathering H+Delta run."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .environment import ACTIONS, Setup, frame_hash, make_env, provenance, save_frame, snapshot
from .gpu import apply_memory_limit, memory_limit_gib, require_free_gpu


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


def assess(data):
    steps = data["steps"]
    recoveries = sum(row["health_after"] > row["health_before"] for row in steps)
    consolidated = sum(row["consolidated"] for row in steps)
    h_used = any(row["h_before"] > data["initial_h_tokens"] for row in steps)
    alive = bool(steps) and not steps[-1]["dead"]
    # Living reward alone is insufficient: require actual healing and use of consolidated H.
    return {
        "health_recoveries": recoveries,
        "alive": alive,
        "consolidations": consolidated,
        "consolidated_h_read": h_used,
        "goal_achieved": recoveries >= 1 and alive and consolidated >= 1 and h_used,
        "actions": len(steps),
        "return": steps[-1]["return"] if steps else 0.0,
        "final_health": steps[-1]["health_after"] if steps else None,
    }


async def run_episode(port, cfg, environment, output, *, control="normal"):
    """Same agent routine for CUDA inference and explicitly labeled CPU scheduler tests."""
    from .agent import choose_action, commit_history, free_blocks, make_prompting, parallel

    if control not in {"normal", "mirror", "blank"}:
        raise ValueError("Use normal, mirror, or blank control")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    observation, _ = environment.reset(seed=cfg.seed)
    previous = current = observation["screen"].copy()
    save_frame(current, output / "frames/000.png")
    initial_state = snapshot(environment)
    prompting = make_prompting(port, cfg)
    for name, token in zip(prompting.decision_names, prompting.decision_token_ids, strict=True):
        if port.tokenizer.encode(name, add_special_tokens=False) != [token]:
            raise ValueError(f"Action must have exactly one token: {name}")
    prompt, history, images, delta = await parallel(*(port.create_block() for _ in range(4)))
    persistent = [prompt, history, images, delta]
    steps = []
    events = []

    def encode(text):
        return port.tokenizer(text, return_tensors="pt", add_special_tokens=False)

    def event(name, step, **extra):
        trace = port.recorded.get_trace()
        events.append(
            {
                "event": name,
                "step": step,
                "trace_step": trace.steps,
                "trace_rows": trace.num_tokens,
                "blocks": [
                    {
                        "id": b.block_id,
                        "tokens": b.num_tokens,
                        "mrope_span": getattr(b, "mrope_span", b.num_tokens),
                    }
                    for b in persistent
                ],
                **extra,
            }
        )

    try:
        await port(**encode(prompting.common_prompt), cache_view=[prompt], return_logits=False)
        await port(
            **encode(prompting.history_init), cache_view=[prompt, history], return_logits=False
        )
        initial_h_tokens = history.num_tokens
        for step in range(1, cfg.max_actions + 1):
            port.step = step
            before = snapshot(environment)
            h_before = history.num_tokens
            delta_before = delta.num_tokens
            await port.free_block(images)
            pair = (previous, current)
            if control == "mirror":
                pair = tuple(frame[:, ::-1].copy() for frame in pair)
            elif control == "blank":
                pair = tuple(frame.copy() for frame in pair)
                for frame in pair:
                    frame.fill(0)
            await port(
                **prompting.get_image_dict(*pair, without_eot=True, end_previous_turn=True),
                cache_view=[prompt, history, delta, images],
                return_logits=False,
            )
            event(
                "observation",
                step,
                previous_frame_hash=frame_hash(previous),
                current_frame_hash=frame_hash(current),
                model_pixel_hashes=[frame_hash(f) for f in pair],
            )
            temporary = None
            try:
                action_id, description, thought, probes, temporary = await choose_action(
                    port, prompting, prompt, history, delta, images, cfg
                )
                # This is the student's own action; evaluation never replays teacher actions.
                observation, reward, terminated, truncated, _ = environment.step(action_id)
                after = snapshot(environment)
                action = ACTIONS[action_id]
                event("env_action", step, action=action, probes=probes, health=after["health"])
                consolidated = await commit_history(
                    port,
                    prompting,
                    prompt,
                    history,
                    delta,
                    step,
                    description,
                    thought,
                    action,
                    f"reward={reward:+.2f}, done={terminated or truncated}",
                    cfg,
                )
                if consolidated:
                    event("delta_to_h", step)
                save_frame(observation["screen"], output / f"frames/{step:03d}.png")
                row = {
                    "step": step,
                    "action": action,
                    "action_id": int(action_id),
                    "description": description,
                    "thought": thought,
                    "probes": probes,
                    "health_before": before["health"],
                    "health_after": after["health"],
                    "game_tick": after["game_tick"],
                    "return": after["return"],
                    "reward": float(reward),
                    "dead": after["dead"],
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "previous_frame_hash": frame_hash(previous),
                    "current_frame_hash": frame_hash(current),
                    "next_frame_hash": frame_hash(observation["screen"]),
                    "h_before": h_before,
                    "delta_before": delta_before,
                    "h_after": history.num_tokens,
                    "delta_after": delta.num_tokens,
                    "consolidated": consolidated,
                }
                steps.append(row)
                print(
                    f"step={step} action={action} health={before['health']}->{after['health']} "
                    f"H={history.num_tokens} Delta={delta.num_tokens}",
                    flush=True,
                )
            finally:
                if temporary is not None:
                    await free_blocks(port, temporary)
            if terminated or truncated:
                stop_reason = "death" if after["dead"] else "episode_end"
                break
            partial = {"steps": steps, "initial_h_tokens": initial_h_tokens}
            if step >= cfg.min_actions and assess(partial)["goal_achieved"]:
                stop_reason = "health_recovered"
                break
            if history.num_tokens + delta.num_tokens >= cfg.history_token_limit:
                stop_reason = "history_limit"
                break
            previous, current = current, observation["screen"].copy()
        else:
            stop_reason = "action_budget"
        prompts = {
            name: getattr(prompting, name)
            for name in (
                "common_prompt",
                "history_init",
                "describer_init",
                "actor_init",
                "mailbox_bridge",
                "actor_probe_prompt",
            )
        }
        data = {
            "setup": asdict(cfg),
            "control": control,
            "steps": steps,
            "events": events,
            "initial_state": initial_state,
            "initial_h_tokens": initial_h_tokens,
            "prompts": prompts,
            "stop_reason": stop_reason,
            "provenance": provenance(),
        }
        data["assessment"] = assess(data)
        return data
    finally:
        await free_blocks(port, persistent)


async def collect(model, output, setup=None, *, control="normal"):
    # Check headroom before loading; shared use requires explicit authorization.
    require_free_gpu()
    import torch
    from minisgl import AsyncLLM
    from minisgl.training import RecordedAsyncLLM

    from .recording import TargetRecorder

    cfg = Setup(**(setup or {}))
    replay_options(asdict(cfg))
    cfg.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(cfg.agent_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    llm = AsyncLLM(
        str(model),
        dtype=getattr(torch, cfg.dtype),
        attention_backend="fi",
        cuda_graph_max_bs=0,
        num_page_override=cfg.kv_tokens,
        max_seq_len_override=cfg.max_seq_len,
        max_running_req=4,
        max_prefill_rows=cfg.prefill_rows,
    )
    apply_memory_limit()
    env = None
    try:
        env = make_env(cfg)
        with RecordedAsyncLLM(llm) as recorded:
            port = TargetRecorder(recorded)
            data = await run_episode(port, cfg, env, output.with_suffix(""), control=control)
            data.update(port.export())
        data.update(
            model=str(model),
            replay_options=replay_options(asdict(cfg)),
            runtime={
                "kind": "async_llm_cuda",
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "gpu": torch.cuda.get_device_name(),
                "allocator_limit_gib": memory_limit_gib(),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            },
        )
        save(output, data)
        print(json.dumps(data["assessment"]), flush=True)
        return data["assessment"]
    finally:
        if env is not None:
            env.close()
        await llm.close()


def assert_real(data):
    if data.get("runtime", {}).get("kind") != "async_llm_cuda":
        raise ValueError("A CPU stub/preflight is not a teacher/student inference rollout")


def prepare_replay_source(data):
    """Preserve the same explicit settings through shared training and reload helpers."""
    from minisgl.training import Trace

    prepared = dict(data)
    prepared["trace"] = Trace.from_dict(data["trace"]).to("cpu", **replay_options(data["setup"]))
    return prepared


def load_replay(source, model=None, dtype="float32"):
    import torch

    from minisgl_examples.training.student import load_student

    data = load(source)
    assert_real(data)
    apply_memory_limit()
    torch.backends.cuda.matmul.allow_tf32 = False
    net, trace, targets, processor = load_student(
        prepare_replay_source(data), str(model or data["model"]), dtype=getattr(torch, dtype)
    )
    return data, net.eval(), trace.to("cuda"), targets, processor


def check_replay(source, output, *, dtype="float32"):
    require_free_gpu()
    import torch

    from minisgl_examples.training.loss import evaluate

    data, model, trace, targets, _ = load_replay(source, dtype=dtype)
    measured = evaluate(model, [(trace, targets)], 32)
    errors = [
        abs(lp + nll)
        for lp, nll in zip(
            (p for p in data["logprobs"] if p is not None),
            measured["examples"][0]["nll"],
            strict=True,
        )
    ]
    groups = {}
    for sample, error in zip(data["samples"], errors, strict=True):
        groups.setdefault(f"step_{sample['step']}/{sample['role']}", []).append(error)
    parity = {
        "max_abs_logprob_error": max(errors),
        "mean_abs_logprob_error": sum(errors) / len(errors),
        "by_phase": {k: {"count": len(v), "max_abs": max(v)} for k, v in groups.items()},
        "argmax_target_accuracy": measured["accuracy"],
        "passed": max(errors) <= 0.1,
        "tolerance": 0.1,
        "errors": errors,
    }
    save(
        output,
        {
            "parity": parity,
            "replay": measured,
            "dtype": dtype,
            "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "replay_options": replay_options(data["setup"]),
        },
    )
    del model, trace, targets
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    if not parity["passed"]:
        raise RuntimeError("Replay mismatch: do not train")
    return parity


def parameter_delta(base, checkpoint):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    base = (
        Path(base)
        if Path(base).is_dir()
        else Path(hf_hub_download(str(base), "config.json", local_files_only=True)).parent
    )

    def first_qproj(folder, key=None):
        for file in sorted(Path(folder).glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as f:
                match = (
                    key
                    if key in f.keys()
                    else (
                        next((k for k in f.keys() if k.endswith("q_proj.weight")), None)
                        if key is None
                        else None
                    )
                )
                if match is not None:
                    return match, f.get_tensor(match).float()
        raise ValueError(f"Missing q_proj weight in {folder}")

    key, before = first_qproj(base)
    _, after = first_qproj(checkpoint, key)
    return {
        "parameter": key,
        "max_abs_delta": float((after - before).abs().max()),
        "before_f32_sha256": hashlib.sha256(before.numpy().tobytes()).hexdigest(),
        "after_f32_sha256": hashlib.sha256(after.numpy().tobytes()).hexdigest(),
    }


def train(teacher_path, student, baseline_path, output, *, steps=80, lr=3e-6):
    require_free_gpu()
    import torch

    apply_memory_limit()

    from minisgl_examples.training.student import train_student

    teacher, baseline = load(teacher_path), load(baseline_path)
    assert_real(teacher)
    assert_real(baseline)
    if not teacher["assessment"]["goal_achieved"]:
        raise ValueError("Teacher did not actually recover health/use H+Delta")
    if teacher["setup"] != baseline["setup"] or baseline["model"] != str(student):
        raise ValueError("Different student/async setup")
    if teacher["prompts"] != baseline["prompts"] or teacher["provenance"] != baseline["provenance"]:
        raise ValueError("Different conditions")
    if teacher["steps"][0]["current_frame_hash"] != baseline["steps"][0]["current_frame_hash"]:
        raise ValueError("Different initial state")
    output = Path(output)
    # CLI training has the same mandatory parity gates as the notebook.
    for source, parity_path, dtype in (
        (teacher_path, output / "teacher_initial_parity.json", teacher["setup"]["dtype"]),
        (baseline_path, output / "student_initial_parity.json", "float32"),
    ):
        if not parity_path.exists():
            check_replay(source, parity_path, dtype=dtype)
        parity = require_replay_options(load(parity_path), teacher["setup"])
        digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if not parity["parity"]["passed"] or parity.get("source_sha256") != digest:
            raise RuntimeError("Saved parity failed or belongs to a different rollout")
    result = train_student(
        prepare_replay_source(teacher),
        str(student),
        output / "checkpoint",
        steps=steps,
        lr=lr,
        dtype=torch.float32,
        chunk_size=32,
    )
    result["replay_options"] = replay_options(teacher["setup"])
    result["parameter_update"] = parameter_delta(student, result["checkpoint"])
    result["cuda_memory"] = {
        "allocator_limit_gib": memory_limit_gib(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    if result["parameter_update"]["max_abs_delta"] == 0:
        raise RuntimeError("No actual parameter update")
    save(output / "training.json", result)
    return result


def verify(teacher_path, checkpoint, training_path, output):
    require_free_gpu()
    import torch

    apply_memory_limit()

    from minisgl_examples.training.student import verify_checkpoint

    data = load(teacher_path)
    training = require_replay_options(load(training_path), data["setup"])
    result = verify_checkpoint(
        prepare_replay_source(data), checkpoint, dtype=torch.float32, chunk_size=32
    )
    error = abs(result["loss"] - training["final"]["loss"])
    save(
        output,
        {
            "reloaded": result,
            "saved_loss_difference": error,
            "passed": error < 1e-5,
            "replay_options": replay_options(data["setup"]),
        },
    )
    if error >= 1e-5:
        raise RuntimeError("Checkpoint reload changed replay loss")
    return result


def inspect_rollout(source):
    from minisgl.training import Trace

    data = load(source)
    assert_real(data)
    trace = Trace.from_dict(data["trace"])
    rows = [s["row"] for s in data["samples"]]
    assert rows == sorted(set(rows))
    assert set(rows) == {i for i, t in enumerate(data["targets"]) if t != -100}
    assert all(data["targets"][s["row"]] == s["token"] for s in data["samples"])
    assert all(
        (p is not None) == (t != -100)
        for t, p in zip(data["targets"], data["logprobs"], strict=True)
    )
    images = [r for b in trace.batches for r in b.requests if r.pixel_values is not None]
    assert len(images) == len(data["steps"])
    assert all(
        r.mrope_positions is not None and r.image_grid_thw.to_tensor().shape == (2, 3)
        for r in images
    )
    assert any(u.refs and len(u.refs) == 2 for u in trace.updates), "Missing append operations"
    assert all(row["delta_after"] == 0 for row in data["steps"] if row["consolidated"])
    return {
        "rows": trace.num_tokens,
        "targets": len(rows),
        "image_requests": len(images),
        "decode_batch_sizes": [len(b.requests) for b in trace.batches if b.kind == "decode"],
        "append_operations": sum(bool(u.refs) for u in trace.updates),
        "clear_operations": sum(not u.refs for u in trace.updates),
        "events": data["events"],
        "assessment": data["assessment"],
    }


def compare(teacher_path, before_path, after_path, output):
    import difflib

    teacher = load(teacher_path)
    assert_real(teacher)
    results = {}
    for name, path in (("before", before_path), ("after", after_path)):
        data = load(path)
        assert_real(data)
        if any(
            data[k] != teacher[k]
            for k in ("setup", "prompts", "provenance", "initial_state", "control")
        ):
            raise ValueError("Different closed-loop initial conditions")
        role = {}
        for kind in ("descriptor", "thinker", "actor"):
            ids = lambda d: [
                s["token"]
                for s in sorted(d["samples"], key=lambda s: s["emission"])
                if s["role"] == kind
            ]
            a, b = ids(teacher), ids(data)
            role[kind] = {
                "exact": a == b,
                "teacher_tokens": len(a),
                "student_tokens": len(b),
                "sequence_similarity": difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(),
            }
        actions = [s["action"] for s in data["steps"]]
        expected = [s["action"] for s in teacher["steps"]]
        outcome = lambda d: [
            (s["health_after"], s["next_frame_hash"], s["game_tick"]) for s in d["steps"]
        ]
        equal_outcomes = outcome(data) == outcome(teacher)
        event_sig = lambda d: [
            (s["action"], len(s["probes"]), s["consolidated"]) for s in d["steps"]
        ]
        results[name] = {
            "assessment": data["assessment"],
            "roles": role,
            "actions": actions,
            "exact_actions": actions == expected,
            "exact_outcomes_and_pixels": equal_outcomes,
            "same_checkpoint_schedule": event_sig(data) == event_sig(teacher),
            "exact_role_tokens": all(v["exact"] for v in role.values()),
        }
    result = {
        "replay_options": replay_options(teacher["setup"]),
        "teacher": teacher["assessment"],
        "runs": results,
        "free_running_success": results["after"]["assessment"]["goal_achieved"]
        and results["after"]["exact_actions"]
        and results["after"]["exact_outcomes_and_pixels"],
        "exact_trajectory": results["after"]["exact_role_tokens"]
        and results["after"]["same_checkpoint_schedule"]
        and results["after"]["exact_outcomes_and_pixels"],
    }
    save(output, result)
    return result


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
    c.add_argument("--control", default="normal", choices=["normal", "mirror", "blank"])
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
        asyncio.run(collect(a.model, a.output, setup, control=a.control))
    elif a.command == "check":
        check_replay(a.source, a.output, dtype=a.dtype)
    else:
        train(a.teacher, a.student, a.baseline, a.output, steps=a.steps, lr=a.lr)


if __name__ == "__main__":
    main()
