"""Deterministic multimodal cache mutations for real-engine replay parity."""

import json
import random
from collections import Counter
from dataclasses import asdict

import torch


def record_image_workload(engine, prefill_rows):
    from minisgl.shared_cache import PrefillJob, SharedCacheSession, WorkerGroup
    from minisgl.training.trace import RecordingSession, Trace

    raw = SharedCacheSession(engine)
    recorder = RecordingSession(raw)
    config = engine.config.model_config
    vision = config.vision_config
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    rng = random.Random(917)
    rows, logits, phases, blocks = [], [], [], []
    operations, chunks = Counter(), Counter()
    phase, calls, peak_live = "initial", [], 0
    original_forward = raw._forward

    def capture(batch, *args, **kwargs):
        chunks[phase] += 1
        if not batch.is_decode:
            positions = batch.mrope_positions
            calls.append(
                (
                    batch.input_ids.cpu().clone(),
                    None if positions is None else positions.cpu().clone(),
                )
            )
        if chunks[phase] % 16 == 0:
            print(json.dumps({"phase": phase, "chunks": chunks[phase]}), flush=True)
        return original_forward(batch, *args, **kwargs)

    raw._forward = capture

    def ids(count):
        return torch.tensor(
            [rng.randrange(100, min(30000, config.vocab_size - 100)) for _ in range(count)],
            dtype=torch.int32,
        )

    def create():
        block = recorder.create_block()
        blocks.append(block)
        return block

    def progress():
        nonlocal peak_live
        peak_live = max(peak_live, sum(block.num_tokens for block in blocks))
        assert peak_live < 16384
        print(
            json.dumps(
                {"phase": phase, "tokens": recorder.trace.num_tokens, "output_rows": len(rows)}
            ),
            flush=True,
        )

    def save(indices, values):
        for index, value in zip(indices, values, strict=True):
            assert value.dtype == torch.float32 and torch.isfinite(value).all()
            rows.append(index)
            logits.append(value.detach().cpu().flatten())
            phases.append(phase)
        progress()

    def image_request(length, grids):
        tokens, pixels = ids(73).tolist(), []
        for index, (t, h, w) in enumerate(grids):
            count = t * h * w
            tokens += [config.vision_start_token_id]
            tokens += [config.image_token_id] * (count // vision.spatial_merge_size**2)
            tokens += [config.vision_end_token_id, *ids(17).tolist()]
            values = torch.arange(count * patch_dim).float().reshape(count, patch_dim)
            pixels.append(torch.sin(values * (7.7e-4 + index * 1e-4) + h + w) * 0.5)
        assert len(tokens) <= length
        tokens = torch.tensor(tokens + ids(length - len(tokens)).tolist(), dtype=torch.int32)
        return tokens, {
            "pixel_values": torch.cat(pixels),
            "image_grid_thw": torch.tensor(grids),
            "mm_token_type_ids": (tokens == config.image_token_id).long(),
        }

    def prefill(block, length, context, name, grids=()):
        nonlocal phase, calls
        phase, calls = name, []
        tokens, payload = image_request(length, grids) if grids else (ids(length), {})
        assert sum(b.num_tokens for b in [*context, block]) + length < 8192
        result = recorder.prefill_batch([PrefillJob(block, tokens, list(context), **payload)])
        request = recorder.trace.batches[-1].requests[0]
        assert torch.equal(torch.cat([tokens for tokens, _ in calls]), tokens)
        assert all(len(tokens) <= prefill_rows for tokens, _ in calls)
        if request.mrope_positions is not None:
            assert torch.equal(
                torch.cat([positions for _, positions in calls], dim=1),
                request.mrope_positions.tensor(),
            )
        operations["prefill"] += 1
        save([recorder.trace.num_tokens - 1], result)

    prompt, a, b, c = [create() for _ in range(4)]
    prefill(prompt, 1536, [], "long_prefix")
    prefill(a, 512, [prompt], "two_images", [(1, 8, 8), (1, 8, 12)])
    prefill(b, 384, [prompt, a], "image_b", [(1, 12, 8)])
    prefill(c, 384, [b, prompt, a], "image_c", [(1, 12, 12)])
    for block, context, name in (
        (a, [c, prompt, b], "a"),
        (b, [prompt, c, a], "b"),
        (c, [b, a, prompt], "c"),
    ):
        prefill(block, 384, context, "extend_" + name)
    copied = merged = None
    copied_snapshot = merged_snapshot = None
    for turn in range(48):
        phase = f"decode_{turn:02d}"
        if turn == 8:
            copied = create()
            recorder.append_block(copied, a)
            copied_snapshot = (list(a.token_ids), a.mrope_span)
            assert (copied.token_ids, copied.mrope_span) == copied_snapshot
            merged = recorder.merge_blocks(b, c)
            blocks.append(merged)
            merged_snapshot = (list(b.token_ids) + list(c.token_ids), b.mrope_span + c.mrope_span)
            assert (merged.token_ids, merged.mrope_span) == merged_snapshot
            operations.update(copy=1, merge=1)
            progress()
        if turn == 12:
            recorder.append_block(copied, copied)
            copied_snapshot = (copied_snapshot[0] * 2, copied_snapshot[1] * 2)
            assert (copied.token_ids, copied.mrope_span) == copied_snapshot
            operations["self_append"] += 1
            progress()
        if turn == 16:
            recorder.free_block(a)
            assert a.num_tokens == a.mrope_span == 0
            assert (copied.token_ids, copied.mrope_span) == copied_snapshot
            operations.update(clear=1, clear_reuse=1)
            prefill(a, 256, [copied, prompt, b], "clear_reuse_image", [(1, 6, 10)])
        if turn == 24:
            recorder.free_block(c)
            assert c.num_tokens == c.mrope_span == 0
            assert (merged.token_ids, merged.mrope_span) == merged_snapshot
            operations.update(clear=1, clear_reuse=1)
            prefill(c, 128, [merged, b, prompt], "clear_reuse_text")
        phase = f"decode_{turn:02d}"
        extra_copy = [] if copied is None else [copied]
        extra_merge = [] if merged is None else [merged]
        views = [
            ([prompt, b, a, c], [c, prompt, b], [prompt, b, prompt, a], [b, a, c, prompt])[
                turn % 4
            ],
            [prompt, c, *extra_copy, a, b],
            [*extra_merge, a, prompt, c, b],
        ]
        assert (
            max(sum(block.num_tokens + int(block in [a, b, c]) for block in view) for view in views)
            < 8192
        )
        cursor = recorder.trace.num_tokens
        result = recorder.decode_step(
            WorkerGroup(cache_structure=views, write_to=[a, b, c]), ids(3)
        )
        operations["decode_batch"] += 1
        save(range(cursor, cursor + 3), result)
    assert (copied.token_ids, copied.mrope_span) == copied_snapshot
    assert (merged.token_ids, merged.mrope_span) == merged_snapshot
    prefill(copied, 64, [merged, prompt, b, c], "saved_copy_continuation")
    operations["saved_copy_continuation"] += 1
    data = json.loads(json.dumps(asdict(recorder.trace)))
    trace = Trace.from_dict(data)
    assert trace == recorder.trace and data["format_version"] == 2 and "tokens" not in data
    tokens = trace.tokens
    requests = [request for batch in trace.batches for request in batch.requests]
    images = [request for request in requests if request.pixel_values is not None]
    metadata = {
        "tokens": trace.num_tokens,
        "output_rows": len(rows),
        "batches": len(trace.batches),
        "updates": len(trace.updates),
        "image_requests": len(images),
        "images_per_request": [len(request.image_grid_thw.tensor()) for request in images],
        "image_grids": [
            grid for request in images for grid in request.image_grid_thw.tensor().tolist()
        ],
        "operations": dict(operations),
        "chunks": dict(chunks),
        "forward_calls": sum(chunks.values()),
        "prefill_rows": prefill_rows,
        "peak_live_tokens": peak_live,
        "max_visible": max(sum(n for _, n in token.view) for token in tokens),
        "max_depth": max(len(token.view) for token in tokens),
        "ordered_views": len({tuple(b for b, _ in token.view) for token in tokens}),
        "repeated_views": sum(
            len({b for b, _ in r.cache_view}) < len(r.cache_view) for r in requests
        ),
        "omit_self_views": sum(all(b != r.write_to for b, _ in r.cache_view) for r in requests),
        "decode_batch_sizes": dict(
            Counter(len(batch.requests) for batch in trace.batches if batch.kind == "decode")
        ),
        "saved_copies_survive_source_clear": True,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    }
    assert metadata["tokens"] == 4560 and metadata["output_rows"] == 154
    saved = {
        "rows": torch.tensor(rows),
        "logits": torch.stack(logits),
        "phases": phases,
        "features": {},
    }
    return data, metadata, saved
