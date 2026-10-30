"""Opt-in MiniSGL engine versus parallel Torch replay, using a local Qwen3.5 checkpoint."""

import argparse
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import training_engine_reference as reference

ROOT = None
MODEL = None
LAYERS = (0, 1, 2, 3, 23)


def build_engine(args):
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig

    original_init = Engine._init_communication

    def initialize(self, config):
        reference.install()
        from minisgl.models import qwen3_5_delta

        qwen3_5_delta._fla_chunk = qwen3_5_delta._fla_recurrent = None
        return original_init(self, config)

    Engine._init_communication = initialize

    engine = Engine(
        EngineConfig(
            model_path=str(MODEL),
            tp_info=DistributedInfo(0, 1),
            dtype=getattr(torch, args.dtype),
            attention_backend="fi",
            cuda_graph_max_bs=0,
            num_page_override=16384,
            max_seq_len_override=8192,
            max_running_req=64,
            max_prefill_rows=args.prefill_rows,
            distributed_addr=(ROOT / f"{args.name}.rendezvous").as_uri(),
        )
    )
    reference.check()
    return engine


def record(args):
    from minisgl.shared_cache import AsyncContext, PrefillJob, SharedCacheSession, WorkerGroup
    from minisgl.training.stress import workload
    from minisgl.training.trace import RecordingSession, Trace

    engine = build_engine(args)
    dtype = getattr(torch, args.dtype)
    weights = fingerprints(
        {
            key: value
            for key, value in engine.model.state_dict().items()
            if not key.startswith("model.visual.")
        }
    )
    raw = SharedCacheSession(engine)
    if args.unprepared:
        raw.sc_gdn.prepare_prefill = lambda *a, **kw: None
        raw.sc_gdn.prepare_decode = lambda *a, **kw: None
        raw.sc_gdn.state_dtype = getattr(torch, args.dtype)

    class Capture(RecordingSession):
        def __init__(self, session):
            super().__init__(session)
            self.rows, self.logits, self.phases = [], [], []
            self.pending, self.active = {}, []
            self.features = defaultdict(list)
            self.phase = "stress"

        def save(self, rows, logits):
            for row, value in zip(rows, logits, strict=True):
                assert value.dtype == dtype, value.dtype
                assert torch.isfinite(value).all()
                self.rows.append(row)
                self.logits.append(value.detach().cpu().flatten())
                self.phases.append(self.phase)

        def prefill_batch(self, jobs):
            jobs = list(jobs)
            cursor, endpoints = self.trace.num_tokens, []
            self.pending = {}
            for job in jobs:
                for offset in range(job.input_ids.numel()):
                    self.pending[job.block.block_id, job.block.num_tokens + offset] = (
                        cursor + offset
                    )
                cursor += job.input_ids.numel()
                endpoints.append(cursor - 1)
            result = super().prefill_batch(jobs)
            self.save(endpoints, result)
            return result

        def decode_step(self, group, input_ids):
            cursor = self.trace.num_tokens
            self.pending = {
                (ctx.output_block.block_id, ctx.output_block.num_tokens): cursor + i
                for i, ctx in enumerate(group)
            }
            result = super().decode_step(group, input_ids)
            self.save(range(cursor, cursor + len(group)), result)
            return result

    from contextlib import nullcontext

    from training_engine_reference import DoubleMath

    math_mode = DoubleMath() if args.dtype == "float64" else nullcontext()
    if args.dtype == "float64":
        for layer in engine.model.model.layers.op_list:
            if not layer._is_linear:
                layer.self_attn._ensure_inv_freq(engine.device)
    math_mode.__enter__()
    capture = Capture(raw)
    history = reference.RuntimeHistory(raw)
    reference.topology = history
    original_forward = raw._forward

    def session_forward(batch, *a, **kw):
        targets = kw["write_to"]
        lengths = (
            [1] * len(targets)
            if batch.is_decode
            else kw.get("prefill_segments") or [batch.input_ids.numel()]
        )
        positions = iter(batch.positions.tolist())
        capture.active = [
            capture.pending[target.block_id, next(positions)]
            for target, length in zip(targets, lengths, strict=True)
            for _ in range(length)
        ]
        history.begin(batch, capture.active, kw["cache_structure"], targets, lengths)
        result = original_forward(batch, *a, **kw)
        for target in targets:
            states = [v for pair in target.linear_affine.values() for v in pair]
            states.extend(target.linear_conv_state.values())
            assert states and all(value.dtype == dtype for value in states)
        return result

    raw._forward = session_forward

    def intercept(module, key, tuple_output=False):
        original = module.forward

        def call(*a, **kw):
            result = original(*a, **kw)
            value = result[0] if tuple_output else result
            assert value.dtype == dtype, (key, value.dtype)
            assert torch.isfinite(value).all(), key
            capture.features[key].append((list(capture.active), value.detach().cpu().clone()))
            return result

        module.forward = call

    for index in LAYERS:
        layer = engine.model.model.layers.op_list[index]
        intercept(layer.input_layernorm, f"{index}.input", True)
        intercept(layer.linear_attn if layer._is_linear else layer.self_attn, f"{index}.attention")

    rng = random.Random(917)
    vocab = engine.config.model_config.vocab_size

    def ids(n):
        return torch.tensor([rng.randrange(100, vocab - 100) for _ in range(n)])

    def prefill(block, n, context=(), phase="prefill"):
        capture.phase = phase
        capture.prefill_batch([PrefillJob(block, ids(n), list(context))])

    def decode(outputs, views, phase):
        capture.phase = phase
        capture.decode_step(
            WorkerGroup(
                [
                    AsyncContext(cache_view=view, output_block=block)
                    for block, view in zip(outputs, views, strict=True)
                ]
            ),
            ids(len(outputs)),
        )

    started = time.perf_counter()
    try:
        workload(capture, vocab, large_size=256)
        prompt = capture.create_block()
        prefill(prompt, 2048, phase="long_prefix")
        a, b, c = [capture.create_block() for _ in range(3)]
        for block in [a, b, c]:
            prefill(block, 128, [prompt], "fork_prefill")
        copied = None
        for turn in range(32):
            if turn == 8:
                copied = capture.merge_blocks(a, b)
                capture.append_block(copied, c)
            if turn == 12:
                capture.append_block(copied, copied)
                capture.free_block(b)
                prefill(b, 64, [prompt, copied], "clear_reuse_b")
            if turn == 16:
                capture.append_block(copied, a)
                capture.free_block(a)
                prefill(a, 64, [copied, prompt], "clear_reuse_a")
            extra = [] if copied is None else [copied]
            views = [
                (
                    [prompt, b, a, c]
                    if turn % 4 == 0
                    else (
                        [c, prompt, b]
                        if turn % 4 == 1
                        else [prompt, b, prompt, a] if turn % 4 == 2 else [b, a, c, prompt]
                    )
                ),
                [prompt, c, *extra, a, b],
                [a, prompt, c, b, *extra],
            ]
            decode([a, b, c], views, f"long_decode_{turn:02d}")
        torch.cuda.synchronize()
        tokens = capture.trace.tokens
        serialized = asdict(capture.trace)
        assert serialized["format_version"] == 2 and "tokens" not in serialized
        assert Trace.from_dict(json.loads(json.dumps(serialized))) == capture.trace
        legacy = Trace.from_dict(
            json.loads(
                json.dumps(
                    {
                        "tokens": [asdict(token) for token in tokens],
                        "steps": capture.trace.steps,
                        "updates": serialized["updates"],
                    }
                )
            )
        )
        assert legacy == capture.trace and legacy.tokens == tokens
        long_requests = [
            request
            for batch in capture.trace.batches
            for request in batch.requests
            if len(request.input_ids) == 2048
        ]
        assert len(long_requests) == 1
        metadata = {
            "dtype": args.dtype,
            "prefill_rows": args.prefill_rows,
            "seconds": time.perf_counter() - started,
            "tokens": capture.trace.num_tokens,
            "format_version": serialized["format_version"],
            "requests": sum(len(batch.requests) for batch in capture.trace.batches),
            "long_prefill_tokens": len(long_requests[0].input_ids),
            "output_rows": len(capture.rows),
            "steps": capture.trace.steps,
            "updates": len(capture.trace.updates),
            "max_visible": max(sum(n for _, n in token.view) for token in tokens),
            "max_depth": max(len(token.view) for token in tokens),
            "ordered_views": len({tuple(b for b, _ in token.view) for token in tokens}),
            "decode_batch_sizes": dict(
                Counter(Counter(t.step for t in tokens if t.kind == "decode").values())
            ),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        }
        from minisgl.training.replay import Replay

        metadata["provenance"] = history.assert_replay(Replay.from_trace(capture.trace))
        assert metadata["tokens"] == 4099 and metadata["updates"] == 39
        assert metadata["output_rows"] == 213 and metadata["max_depth"] == 12
        features = {}
        for key, chunks in capture.features.items():
            value = torch.empty(
                capture.trace.num_tokens, chunks[0][1].shape[-1], dtype=chunks[0][1].dtype
            )
            seen = []
            for rows, chunk in chunks:
                value[rows] = chunk
                seen.extend(rows)
            assert sorted(seen) == list(range(capture.trace.num_tokens)), key
            features[key] = value
        torch.save(
            {
                "rows": torch.tensor(capture.rows),
                "logits": torch.stack(capture.logits),
                "phases": capture.phases,
                "features": features,
            },
            ROOT / f"{args.name}.pt",
        )
        (ROOT / f"{args.name}.json").write_text(
            json.dumps(
                {
                    "model": str(MODEL),
                    "trace": serialized,
                    "metadata": metadata,
                    "weights": weights,
                }
            )
        )
        print(json.dumps(metadata), flush=True)
    finally:
        reference.topology = None
        engine.shutdown()
        math_mode.__exit__(None, None, None)


def record_image(args):
    from minisgl.shared_cache import SharedCacheSession, WorkerGroup
    from minisgl.training.gdn import GDNReplay
    from minisgl.training.replay import Replay
    from minisgl.training.trace import RecordingSession, Trace

    engine = build_engine(args)
    try:
        weights = fingerprints(engine.model.state_dict())
        raw = SharedCacheSession(engine)
        recorder = RecordingSession(raw)
        rows, logits, phases = [], [], []

        def save(value, phase):
            assert value.dtype == getattr(torch, args.dtype) and torch.isfinite(value).all()
            rows.append(recorder.trace.num_tokens - 1)
            logits.append(value.detach().cpu().flatten())
            phases.append(phase)

        config = engine.config.model_config
        vision = config.vision_config
        patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
        grid = torch.tensor([[1, 8, 8]])
        pixels = (
            torch.sin(torch.arange(64 * patch_dim).float().reshape(64, patch_dim) * 7.7e-4) * 0.5
        )
        original_pixels = pixels.clone()
        ids = torch.tensor(
            [config.vision_start_token_id]
            + [config.image_token_id] * (64 // vision.spatial_merge_size**2)
            + [config.vision_end_token_id],
            dtype=torch.int32,
        )
        types = (ids == config.image_token_id).long()
        image = {
            "pixel_values": pixels,
            "image_grid_thw": grid,
            "mm_token_type_ids": types,
        }
        baseline, block = raw.create_block(), recorder.create_block()
        expected = raw.prefill_block(baseline, ids, **image)
        image_logits = expected.clone()
        calls = []
        original_forward = raw._forward

        def capture(batch, *a, **kw):
            calls.append((batch.input_ids.cpu().clone(), batch.mrope_positions.cpu().clone()))
            return original_forward(batch, *a, **kw)

        raw._forward = capture
        actual = recorder.prefill_block(block, ids, **image)
        save(actual, "image_prefill")
        raw._forward = original_forward
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(calls) > 1 and all(len(tokens) <= args.prefill_rows for tokens, _ in calls)
        image_chunks = len(calls)
        batch = recorder.trace.batches[0]
        request = batch.requests[0]
        assert batch.kind == "prefill" and len(batch.requests) == 1
        assert request.input_ids == tuple(ids.tolist()) and request.write_start == 0
        assert request.write_span == 0 and request.cache_spans == (0,)
        assert torch.equal(torch.cat([tokens for tokens, _ in calls]), ids)
        assert torch.equal(
            torch.cat([positions for _, positions in calls], dim=1),
            request.mrope_positions.tensor(),
        )
        for field, tensor in image.items():
            assert torch.equal(getattr(request, field).tensor(), tensor), field
        pixels.add_(1)
        assert torch.equal(request.pixel_values.tensor(), original_pixels)
        span = block.mrope_span
        assert span < block.num_tokens == len(ids)
        text = torch.tensor([42, 43, 44], dtype=torch.int32)
        expected = raw.prefill_block(baseline, text)
        actual = recorder.prefill_block(block, text)
        save(actual, "text_extension")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        request = recorder.trace.batches[-1].requests[0]
        assert request.write_span == span and request.cache_spans == (span,)
        assert torch.equal(
            request.mrope_positions.tensor(), torch.arange(span, span + 3).expand(3, -1)
        )
        assert block.mrope_span == baseline.mrope_span == span + 3
        token = torch.tensor([45], dtype=torch.int32)

        def group(target):
            return WorkerGroup(cache_structure=[[target]], write_to=[target])

        expected = raw.decode_step(group(baseline), token)
        actual = recorder.decode_step(group(block), token)
        save(actual, "decode")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        request = recorder.trace.batches[-1].requests[0]
        assert request.write_span == span + 3 and request.cache_spans == (span + 3,)
        span, length = block.mrope_span, block.num_tokens
        raw.append_block(baseline, baseline)
        recorder.append_block(block, block)
        assert block.mrope_span == 2 * span and block.num_tokens == 2 * length
        expected = raw.prefill_block(baseline, token)
        actual = recorder.prefill_block(block, token)
        save(actual, "self_append")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert recorder.trace.batches[-1].requests[0].write_span == 2 * span
        recorder.free_block(block)
        assert block.num_tokens == block.mrope_span == 0
        actual = recorder.prefill_block(block, ids, **{**image, "pixel_values": original_pixels})
        save(actual, "clear_reuse")
        torch.testing.assert_close(actual, image_logits, rtol=0, atol=0)
        request = recorder.trace.batches[-1].requests[0]
        assert request.write_start == request.write_span == 0 and request.cache_spans == (0,)
        data = json.loads(json.dumps(asdict(recorder.trace)))
        restored = Trace.from_dict(data)
        assert restored == recorder.trace
        assert data["format_version"] == 2 and "tokens" not in data
        assert len(restored.batches) == 5 and len(restored.updates) == 2
        assert sum(r.pixel_values is not None for b in restored.batches for r in b.requests) == 2
        replay = Replay.from_trace(restored)
        assert replay.input_ids.numel() == restored.num_tokens
        gdn = GDNReplay.from_trace(restored, replay.order)
        assert gdn.conv_sources.shape[0] == restored.num_tokens
        metadata = {
            "image_tokens": len(ids),
            "image_chunks": image_chunks,
            "batches": len(restored.batches),
            "tokens": restored.num_tokens,
            "updates": len(restored.updates),
            "output_rows": len(rows),
            "max_abs": (actual - image_logits).abs().max().item(),
        }
        torch.save(
            {
                "rows": torch.tensor(rows),
                "logits": torch.stack(logits),
                "phases": phases,
                "features": {},
            },
            ROOT / f"{args.name}.pt",
        )
        (ROOT / f"{args.name}.json").write_text(
            json.dumps({"trace": data, "metadata": metadata, "weights": weights})
        )
        print(json.dumps(metadata), flush=True)
    finally:
        engine.shutdown()


def record_image_long(args):
    from training_image_workload import record_image_workload

    assert args.dtype == "float32"
    engine = build_engine(args)
    try:
        print("Fingerprinting native model weights", flush=True)
        weights = fingerprints(engine.model.state_dict())
        print("Recording long multimodal workload", flush=True)
        started = time.perf_counter()
        trace, metadata, saved = record_image_workload(engine, args.prefill_rows)
        torch.cuda.synchronize()
        metadata.update(dtype=args.dtype, seconds=time.perf_counter() - started)
        torch.save(saved, ROOT / f"{args.name}.pt")
        (ROOT / f"{args.name}.json").write_text(
            json.dumps({"trace": trace, "metadata": metadata, "weights": weights})
        )
        print(json.dumps(metadata), flush=True)
    finally:
        engine.shutdown()


def replay(args):
    from minisgl.training import replay
    from minisgl.training.replay import Replay
    from minisgl.training.sft import restore_trace
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    for name in (
        "FusedRMSNormGated",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        setattr(modeling, name, None)
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        setattr(modeling, name, inspect.unwrap(getattr(modeling, name)))
    data = json.loads((ROOT / f"{args.source}.json").read_text())
    saved = torch.load(ROOT / f"{args.source}.pt", weights_only=True)
    trace = restore_trace(data["trace"])
    dtype = getattr(torch, args.dtype)
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    model_class = AutoModelForImageTextToText if trace.multimodal else AutoModelForCausalLM
    if not trace.multimodal:
        config = config.get_text_config()
    model = model_class.from_pretrained(
        MODEL,
        config=config,
        dtype=dtype,
        device_map="cuda",
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    actual_weights = fingerprints(canonical_weights(model))
    differences = {
        key: (data["weights"].get(key), actual_weights.get(key))
        for key in data["weights"].keys() | actual_weights.keys()
        if data["weights"].get(key) != actual_weights.get(key)
    }
    assert not differences, differences
    from contextlib import nullcontext

    from training_engine_reference import DoubleMath

    math_mode = DoubleMath() if args.dtype == "float64" else nullcontext()
    math_mode.__enter__()
    plan = Replay.from_trace(
        trace,
        device="cuda",
        query_chunk_size=32,
        gdn_segment_size=args.segment,
        gdn_read_mode=args.read_mode,
        gdn_read_backend="torch",
    )
    features = {}
    inverse = plan.order.argsort()

    def hook(key):
        def capture(module, a, output):
            value = output[0] if isinstance(output, tuple) else output
            assert value.dtype == dtype, (key, value.dtype)
            assert torch.isfinite(value).all(), key
            features[key] = value[0, inverse].detach().cpu()

        return capture

    backbone = model.model.language_model if trace.multimodal else model.model
    for index in LAYERS if saved["features"] else ():
        layer = backbone.layers[index]
        layer.input_layernorm.register_forward_hook(hook(f"{index}.input"))
        getattr(layer, "linear_attn", getattr(layer, "self_attn", None)).register_forward_hook(
            hook(f"{index}.attention")
        )
    started = time.perf_counter()
    with torch.no_grad():
        logits = replay.forward(model, plan, rows=saved["rows"].to("cuda")).cpu()
        assert logits.dtype == dtype and logits.shape == saved["logits"].shape
        assert torch.isfinite(logits).all()
    torch.cuda.synchronize()
    torch.save(
        {"rows": saved["rows"], "logits": logits, "phases": saved["phases"], "features": features},
        ROOT / f"{args.name}.pt",
    )
    meta = {
        "source": args.source,
        "dtype": args.dtype,
        "segment": args.segment,
        "read_mode": args.read_mode,
        "seconds": time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    }
    (ROOT / f"{args.name}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    math_mode.__exit__(None, None, None)


def compare(args):
    reference = torch.load(ROOT / f"{args.source}.pt", weights_only=True)
    actual = torch.load(ROOT / f"{args.name}.pt", weights_only=True)
    assert torch.equal(reference["rows"], actual["rows"])

    def metrics(a, b):
        a, b = a.double(), b.double()
        d = a - b
        per_row = d.abs().amax(-1)
        return {
            "finite": bool(a.isfinite().all() and b.isfinite().all()),
            "max_abs": d.abs().max().item(),
            "rms": d.square().mean().sqrt().item(),
            "relative_l2": (d.norm() / b.norm()).item(),
            "max_row": int(per_row.argmax()),
            "row_max_p50": per_row.median().item(),
            "row_max_p95": per_row.quantile(0.95).item(),
        }

    result = {
        "reference": args.source,
        "actual": args.name,
        "logits": metrics(actual["logits"], reference["logits"]),
    }
    a, b = actual["logits"], reference["logits"]
    result["argmax_mismatches"] = int((a.argmax(-1) != b.argmax(-1)).sum())
    result["rows"] = len(a)
    result["logprob"] = metrics(a.log_softmax(-1), b.log_softmax(-1))
    result["features"] = {
        key: metrics(value, reference["features"][key]) for key, value in actual["features"].items()
    }
    phases = defaultdict(list)
    for i, phase in enumerate(reference["phases"]):
        phases[phase].append(i)
    result["phases"] = {key: metrics(a[rows], b[rows]) for key, rows in phases.items()}
    (ROOT / f"{args.source}--{args.name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "phases"}, indent=2), flush=True)


def fingerprints(state):
    return {
        name: [
            list(value.shape),
            str(value.dtype),
            hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        ]
        for name, value in state.items()
    }


def canonical_weights(model):
    state = {
        name.replace("model.language_model.", "model.", 1): value
        for name, value in model.state_dict().items()
    }
    if model.config.get_text_config().tie_word_embeddings:
        assert torch.equal(state.pop("lm_head.weight"), state["model.embed_tokens.weight"])
    for key in list(state):
        for source, target, parts in (
            (".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight", ("q", "k", "v")),
            (".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight", ("gate", "up")),
        ):
            if key.endswith(source):
                prefix = key[: -len(source)]
                family = ".self_attn." if len(parts) == 3 else ".mlp."
                state[prefix + target] = torch.cat(
                    [state.pop(prefix + family + part + "_proj.weight") for part in parts]
                )
    return state


@pytest.fixture(scope="module")
def engine_runner(tmp_path_factory):
    if not os.environ.get("MINISGL_TRAINING_LOGITS_MODEL"):
        pytest.skip(
            "Set MINISGL_TRAINING_LOGITS_MODEL to a local dense Qwen3.5 checkpoint directory"
        )
    assert Path(os.environ["MINISGL_TRAINING_LOGITS_MODEL"]).is_dir()
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    directory = tmp_path_factory.mktemp("engine_replay")
    repository = Path(__file__).resolve().parents[2]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    env["PYTHONPATH"] = str(repository / "python") + os.pathsep + env.get("PYTHONPATH", "")

    def run(mode, name, *extra):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            mode,
            "--name",
            name,
            "--directory",
            str(directory),
            *extra,
        ]
        log = directory / f"{mode}-{name}.log"
        with log.open("w") as output:
            result = subprocess.run(
                command,
                cwd=repository,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        assert result.returncode == 0, f"{log}\n{log.read_text()[-8000:]}"

    return directory, run


@pytest.fixture(scope="module")
def engine_runs(engine_runner):
    directory, run = engine_runner
    for dtype, unprepared in [("float32", False), ("float32", True), ("float64", True)]:
        name = dtype + ("-torch-state" if unprepared else "-prepared")
        run(
            "record",
            name,
            "--dtype",
            dtype,
            *(["--unprepared"] if unprepared else []),
        )
        if name != "float32-torch-state":
            for read_mode in ("expanded", "factorized"):
                replay_name = name + "-" + read_mode
                run(
                    "replay",
                    replay_name,
                    "--source",
                    name,
                    "--dtype",
                    dtype,
                    "--read-mode",
                    read_mode,
                )
                run("compare", replay_name, "--source", name)
    run("compare", "float32-torch-state", "--source", "float32-prepared")
    return directory


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
def test_sequential_minisgl_matches_parallel_torch(engine_runs, dtype, read_mode, record_property):
    name = dtype + ("-prepared" if dtype == "float32" else "-torch-state")
    comparison = json.loads((engine_runs / f"{name}--{name}-{read_mode}.json").read_text())
    metadata = json.loads((engine_runs / f"{name}.json").read_text())["metadata"]
    assert metadata["provenance"]["gdn_rows"] == 4099
    assert metadata["provenance"]["attention_segments"] > 4099
    assert metadata["provenance"]["partition_modes"] == 2
    errors = comparison["logits"]
    assert errors["finite"] and comparison["rows"] == 213
    assert comparison["argmax_mismatches"] == 0
    assert errors["max_abs"] < (2e-4 if dtype == "float32" else 1e-9), errors
    assert errors["rms"] < (2e-5 if dtype == "float32" else 1e-10), errors
    for key, value in comparison["features"].items():
        assert value["finite"], key
        assert value["max_abs"] < (2e-4 if dtype == "float32" else 1e-9), (key, value)
    for key, value in errors.items():
        record_property(key, value)
    record_property("report", str(engine_runs / f"{name}--{name}-{read_mode}.json"))


def test_prepared_gdn_matches_torch_state_path(engine_runs, record_property):
    names = ["float32-prepared", "float32-torch-state", "float64-torch-state"]
    traces = [json.loads((engine_runs / f"{name}.json").read_text())["trace"] for name in names]
    assert traces[0] == traces[1] == traces[2]
    result = json.loads((engine_runs / "float32-prepared--float32-torch-state.json").read_text())
    assert result["logits"]["finite"] and result["argmax_mismatches"] == 0
    assert result["logits"]["max_abs"] < 2e-4, result["logits"]
    assert result["logits"]["rms"] < 2e-5, result["logits"]
    for key, value in result["features"].items():
        assert value["finite"] and value["max_abs"] < 2e-4, (key, value)
    record_property("prepared_vs_torch_max_abs", result["logits"]["max_abs"])


@pytest.fixture(scope="module")
def image_engine_run(engine_runner):
    directory, run = engine_runner
    run("record_image", "image-recording", "--prefill-rows", "8")
    return directory, run


def test_image_recording_preserves_request_payload(image_engine_run, record_property):
    directory, _ = image_engine_run
    result = json.loads((directory / "image-recording.json").read_text())["metadata"]
    assert result["image_chunks"] > 1 and result["batches"] == 5
    assert result["max_abs"] == 0
    record_property("report", str(directory / "image-recording.json"))


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
def test_sequential_image_minisgl_matches_parallel_torch(
    image_engine_run, read_mode, record_property
):
    directory, run = image_engine_run
    source, name = "image-recording", "image-" + read_mode
    run("replay", name, "--source", source, "--read-mode", read_mode)
    run("compare", name, "--source", source)
    report = directory / f"{source}--{name}.json"
    comparison = json.loads(report.read_text())
    assert comparison["rows"] == 5 and comparison["argmax_mismatches"] == 0
    assert set(comparison["phases"]) == {
        "image_prefill",
        "text_extension",
        "decode",
        "self_append",
        "clear_reuse",
    }
    errors = comparison["logits"]
    assert errors["finite"] and errors["max_abs"] < 2e-4, errors
    assert errors["rms"] < 2e-5, errors
    record_property("report", str(report))


@pytest.fixture(scope="module")
def long_image_engine_run(engine_runner):
    directory, run = engine_runner
    run("record_image_long", "long-image-recording")
    return directory, run


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
def test_long_image_cache_views_minisgl_matches_parallel_torch(
    long_image_engine_run, read_mode, record_property
):
    directory, run = long_image_engine_run
    source, name = "long-image-recording", "long-image-" + read_mode
    metadata = json.loads((directory / f"{source}.json").read_text())["metadata"]
    assert metadata["tokens"] == 4560 and metadata["output_rows"] == 154
    assert metadata["images_per_request"] == [2, 1, 1, 1]
    assert len({tuple(grid) for grid in metadata["image_grids"]}) == 5
    assert metadata["operations"] == {
        "prefill": 10,
        "decode_batch": 48,
        "copy": 1,
        "merge": 1,
        "self_append": 1,
        "clear": 2,
        "clear_reuse": 2,
        "saved_copy_continuation": 1,
    }
    assert metadata["saved_copies_survive_source_clear"]
    assert metadata["repeated_views"] == metadata["omit_self_views"] == 12
    assert metadata["decode_batch_sizes"] == {"3": 48}
    assert metadata["max_visible"] < 8192 and metadata["peak_live_tokens"] < 16384
    assert all(
        metadata["chunks"][phase] > 1
        for phase in ("two_images", "image_b", "image_c", "clear_reuse_image")
    )
    run("replay", name, "--source", source, "--read-mode", read_mode)
    run("compare", name, "--source", source)
    report = directory / f"{source}--{name}.json"
    comparison = json.loads(report.read_text())
    assert comparison["rows"] == 154 and comparison["argmax_mismatches"] == 0
    errors = comparison["logits"]
    assert errors["finite"] and errors["max_abs"] < 2e-4, errors
    assert errors["rms"] < 2e-5, errors
    record_property("report", str(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["record", "record_image", "record_image_long", "replay", "compare"]
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--source")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--unprepared", action="store_true")
    parser.add_argument("--prefill-rows", type=int, default=64)
    parser.add_argument("--segment", type=int, default=8)
    parser.add_argument("--read-mode", choices=["expanded", "factorized"], default="expanded")
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.directory.resolve()
    MODEL = Path(os.environ["MINISGL_TRAINING_LOGITS_MODEL"]).resolve()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    globals()[args.mode](args)
