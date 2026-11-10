"""Exact image/window geometry and causal block layouts for external parity."""
import hashlib

import torch

LAYOUTS = ("shared-cache", "image-split", "image-append", "image-chunked")
CASES = ("image_cross_window", "image_evict_first", "image_evict_last", "image_after_old_image")


def image_runs(types):
    values = types.flatten().tolist()
    starts = [i for i, value in enumerate(values) if value == 1 and (i == 0 or values[i-1] != 1)]
    ends = [i+1 for i, value in enumerate(values) if value == 1 and (i+1 == len(values) or values[i+1] != 1)]
    return list(zip(starts, ends))


def window_cases(base, data, filler, window, output):
    """Preserve processor tensors; only add ordinary text and its zero modality IDs."""
    ids = base["prompt_ids"]
    types = data["mm_token_type_ids"]
    [(first, end)] = image_runs(types)
    assert len(filler) > 0 and window > end
    def text(count):
        assert count >= 0
        return (filler * (count // len(filler) + 1))[:count]
    cases = []
    for name in CASES:
        mm = dict(data)
        if name == "image_cross_window":
            # Insert BEFORE the image's opening delimiter, not inside its patch run.
            insert, count = max(1, first-1), window-2-first
            prompt = ids[:insert] + text(count) + ids[insert:]
            mm["mm_token_type_ids"] = torch.cat((types[:, :insert], types.new_zeros(1, count), types[:, insert:]), 1)
            cuts = [first + count, end + count]
        elif name in ("image_evict_first", "image_evict_last"):
            target = first if name.endswith("first") else end-1
            # Last prefill query still sees target; the very next decode query does not.
            count = window + target - len(ids)
            prompt = ids + text(count)
            mm["mm_token_type_ids"] = torch.cat((types, types.new_zeros(1, count)), 1)
            cuts = [end]
        else:
            count = window-len(ids)
            prompt = ids + text(count) + ids
            mm["mm_token_type_ids"] = torch.cat((types, types.new_zeros(1, count), types), 1)
            for key in ("pixel_values", "image_position_ids"):
                mm[key] = torch.cat((data[key], data[key]), 0)
            cuts = [len(ids), window]
        runs = image_runs(mm["mm_token_type_ids"])
        assert len(prompt) + len(base["teacher_tokens"]) < 2048
        assert mm["mm_token_type_ids"].numel() == len(prompt)
        assert all(not (start < cut < stop) for start, stop in runs for cut in cuts)
        path = output / (name + ".pt")
        torch.save(mm, path)
        cases.append(dict(base, name=name, prompt_ids=prompt, image_tensors=str(path),
                          image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          image_window=dict(window=window, runs=runs, cuts=cuts)))
    validate_geometry(cases)
    return cases


def validate_geometry(cases):
    assert [c["name"] for c in cases] == list(CASES)
    for case in cases:
        w, runs = case["image_window"]["window"], case["image_window"]["runs"]
        q = len(case["prompt_ids"])-1
        if case["name"] == "image_cross_window":
            assert runs[0][0] == w-2 and runs[0][1] > w
        elif case["name"].startswith("image_evict_"):
            target = runs[0][0] if case["name"].endswith("first") else runs[0][1]-1
            assert target > q-w and not target > (q+1)-w
        else:
            assert len(runs) == 2 and runs[1][0]-runs[0][0] == w


def materialize(session, case, tokens, mm, layout, *, chunk_rows=128):
    """Same token history, using the public Session operations without math patches."""
    assert layout in LAYOUTS
    cuts = case["image_window"]["cuts"] if layout in ("image-split", "image-append") else []
    runs = image_runs(mm["mm_token_type_ids"])
    blocks, records, forwards = [], [], []
    previous_budget = session.max_prefill_rows
    original = session._prefill_batch_fused
    def observed(jobs):
        forwards.extend(dict(rows=j.input_ids.numel(), images=int((j.mm_token_type_ids == 1).sum())
                             if j.mm_token_type_ids is not None else 0) for j in jobs)
        return original(jobs)
    session._prefill_batch_fused = observed
    session.max_prefill_rows = chunk_rows if layout == "image-chunked" else None
    try:
        for start, stop in zip([0, *cuts], [*cuts, len(tokens)]):
            if not blocks or layout != "image-append":
                blocks.append(session.create_block())
            indices = [i for i, (a, b) in enumerate(runs) if start <= a and b <= stop]
            kwargs = {}
            if indices:
                kwargs = dict(mm_token_type_ids=mm["mm_token_type_ids"][start:stop],
                              pixel_values=mm["pixel_values"][indices],
                              image_position_ids=mm["image_position_ids"][indices])
            logits = session.prefill_block(blocks[-1], torch.tensor(tokens[start:stop], dtype=torch.int32),
                                           context=blocks[:-1], **kwargs)
            records.append(dict(start=start, stop=stop, images=indices, blocks=len(blocks)))
        assert [token for block in blocks for token in block.token_ids] == tokens
        return logits, blocks, dict(parts=records, forwards=forwards, tokens=len(tokens))
    finally:
        session.max_prefill_rows = previous_budget
        session._prefill_batch_fused = original
