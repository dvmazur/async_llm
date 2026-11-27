"""Reconstruct token views at reasoning cutoffs without loading a model or KV cache."""

from __future__ import annotations


def reconstruct_cutoff(rollout: dict, cutoff: int, *, include_pending: bool = False) -> dict:
    """Return the first worker's post-publication, pre-final-flush token view.

    ``cutoff`` counts sampled reasoning rounds and is clamped to the saved run.
    With ``include_pending=False``, the view contains only already encoded tokens.
    With ``include_pending=True``, each worker's latest non-EOS sample is appended
    to its own residual before concatenation. This latter view includes pending
    peer samples that were unavailable to the next ordinary worker query.

    These are token sequences and block roles. Reconstructing this view does not
    reconstruct KV values or perform the original model computations.
    """
    if type(cutoff) is not int or cutoff < 1:
        raise ValueError("The cutoff must be a positive integer number of sampled rounds")
    if type(include_pending) is not bool:
        raise ValueError("include_pending must be a boolean")
    layout = rollout.get("layout")
    if layout not in {"native_workers", "native", "causal"}:
        raise ValueError(f"Unsupported cutoff layout: {layout!r}")
    if layout == "native_workers" and rollout.get("publication_policy") != "encoded_separator_v1":
        raise ValueError("Native workers require the encoded_separator_v1 publication policy")

    workers = list(rollout["workers"])
    streams = rollout["worker_token_ids"]
    prompt = list(rollout["prompt_ids"])
    available = rollout["decode_rounds"]
    if not workers or len(set(workers)) != len(workers) or set(workers) != set(streams):
        raise ValueError("Worker identities and sampled streams must match")
    if layout != "native_workers" and len(workers) != 1:
        raise ValueError("Native and causal layouts require one worker")
    if type(available) is not int or available < 1:
        raise ValueError("The saved rollout must contain at least one sampled reasoning round")
    if any(len(streams[name]) != available for name in workers):
        raise ValueError("Every worker must have one sample per saved reasoning round")
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("The saved prompt must contain valid token IDs")
    if any(type(token) is not int or token < 0 for ids in streams.values() for token in ids):
        raise ValueError("Sampled streams must contain valid token IDs")
    if (rollout.get("other_header_ids") or rollout.get("own_header_ids")
            or any(rollout.get("initial_worker_prefix_ids", {}).values())):
        raise ValueError("Scaffold headers and worker prefixes are unsupported")

    eos_ids = set(rollout.get("eos_token_ids", ()))
    if rollout.get("eos_token_id") is not None:
        eos_ids.add(rollout["eos_token_id"])
    if any(token in eos_ids for ids in streams.values() for token in ids[:-1]):
        raise ValueError("A sampled worker stream continued after EOS")
    effective = min(cutoff, available)
    pending = {
        name: [] if streams[name][effective - 1] in eos_ids else [streams[name][effective - 1]]
        for name in workers
    }
    encoded = {name: list(streams[name][:effective - 1]) for name in workers}
    common = prompt.copy() if layout == "native_workers" else []
    if layout != "native_workers":
        encoded[workers[0]] = prompt + encoded[workers[0]]

    publications = rollout["publications"]
    if layout == "native" and publications:
        raise ValueError("The native baseline must not contain publications")
    previous_round = 0
    committed = dict.fromkeys(workers, 0)
    selected = []
    for event in publications:
        round_number, worker, count = event["round"], event["worker"], event["committed_token_count"]
        if (type(round_number) is not int or not previous_round <= round_number <= available
                or round_number < 1 or worker not in committed or type(count) is not int or count < 1
                or event.get("next_prefix_ids")):
            raise ValueError("Invalid publication event for an unscaffolded rollout")
        previous_round = round_number
        end = committed[worker] + count
        expected_end = round_number - 1
        if layout == "causal":
            expected_end += len(prompt)
        if end != expected_end:
            raise ValueError("Publication count does not match the encoded prefix at that round")
        if round_number <= effective:
            common.extend(encoded[worker][committed[worker]:end])
            selected.append(dict(event))
        committed[worker] = end

    # Future publications validate the saved trajectory but do not affect this cutoff.
    cutoff_committed = dict.fromkeys(workers, 0)
    for event in selected:
        cutoff_committed[event["worker"]] += event["committed_token_count"]
    residuals = {name: encoded[name][cutoff_committed[name]:] for name in workers}
    view_order = [*workers[1:], workers[0]]
    view_ids = common.copy()
    for name in view_order:
        view_ids.extend(residuals[name])
        if include_pending:
            view_ids.extend(pending[name])

    return {
        "layout": layout,
        "workers": workers,
        "view_worker": workers[0],
        "cutoff": cutoff,
        "effective_rounds": effective,
        "available_rounds": available,
        "clamped_to_saved_run": effective < cutoff,
        "include_pending": include_pending,
        "view_semantics": "all_non_eos_samples" if include_pending else "encoded_before_final_flush",
        "cutoff_phase": "after_publications_for_the_sampled_round",
        "publication_count": len(selected),
        "publications": selected,
        "common_history_ids": common,
        "encoded_residual_ids": residuals,
        "pending_token_ids": pending,
        "view_order": ["common_history", *view_order],
        "view_token_ids": view_ids,
    }
