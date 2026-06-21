"""Thin wrappers around minisgl's engine + shared cache.

Everything in here is a helper that minisgl doesn't expose directly: engine
construction, tokenizer encoding, shared-cache block bookkeeping, the
in-context prefill, and the mode-switching probe.
"""

from __future__ import annotations

from typing import List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.shared_cache import SharedBlock, SharedCacheSession, WorkerGroup
from transformers import AutoTokenizer


def build_engine(
    model_path: str,
    memory_ratio: float,
    page_size: int = 1,
) -> Engine:
    """Build a single-GPU engine sized for 1- or 2-worker decode."""
    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=4,
        cuda_graph_bs=[1, 2],  # we only ever run 1- or 2-worker decode
        cuda_graph_max_bs=2,
        memory_ratio=memory_ratio,
        max_seq_len_override=4096,
        page_size=page_size,
    )
    return Engine(config)


def encode(text: str, tokenizer: AutoTokenizer) -> torch.Tensor:
    """1-D int32 CPU tensor.

    ``add_special_tokens=False`` because all the boundary markers are encoded
    literally in the prompt strings.
    """
    return (
        tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        .view(-1)
        .to(torch.int32)
    )


def single_token_id(text: str, tokenizer: AutoTokenizer) -> int:
    """Resolve a string we expect to be exactly one token to its vocab id.

    Used both for the literal ``"\\n\\n"`` separator and for the ``yes``/``no``
    probe tokens.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"Expected {text!r} to tokenize to 1 token, got {len(ids)}: {ids}")
    return int(ids[0])


def vocab_id_or_none(tokenizer: AutoTokenizer, text: str) -> int | None:
    """Return ``tokenizer.vocab[text]`` if present, else ``None``.

    Used to map special-token strings (``</think>``, ``<|im_start|>``, ...) to
    ids, gracefully skipping markers a given tokenizer doesn't have.
    """
    return tokenizer.vocab.get(text)


def ends_with_double_newline(token_ids: List[int], tokenizer: AutoTokenizer) -> bool:
    """Mirror AR's ``is_end_of_step``: decode the last two tokens, check tail."""
    if len(token_ids) < 2:
        return False
    return tokenizer.decode(token_ids[-2:]).endswith("\n\n")


def free_block(session: SharedCacheSession, block: SharedBlock) -> None:
    """Return a block's pages to the engine's page allocator and reset it."""
    session.free_block(block)


def prefill_block_in_context(
    session: SharedCacheSession,
    write_to: SharedBlock,
    context: List[SharedBlock],
    token_ids: torch.Tensor,
) -> None:
    """Append ``token_ids`` to ``write_to`` one token at a time, with the rest
    of ``context`` visible as past KV.

    This emulates AsyncReasoning's multi-token in-context prefill, which
    minisgl's ``prefill_block`` does not expose directly (it only supports
    cached_len=0).  The cost is N forward passes for N tokens, but it runs once
    at setup so the overhead is small.
    """
    group = WorkerGroup(
        cache_structure=[list(context) + [write_to]],
        write_to=[write_to],
    )
    for tok_id in token_ids.tolist():
        session.decode_step(group, torch.tensor([int(tok_id)], dtype=torch.int32))


def check_continue_writing(
    session: SharedCacheSession,
    tokenizer: AutoTokenizer,
    mode_switching_prompt: str,
    mode_switching_question: str,
    thinker_tokens: List[int],
    writer_tokens: List[int],
    yes_id: int,
    no_id: int,
) -> tuple[bool, float, float]:
    """Monolithically prefill the mode-switching probe and compare yes/no.

    Returns ``(should_continue_writing, yes_logit, no_logit)``.  The two raw
    logit values are useful for debug logging since they show how confident the
    probe was at any given step.

    This re-encodes the whole probe context each call (no caching across
    iterations).  That's a small fixed cost: the probe runs every ~20-30
    main-loop steps, the probe text is ~100-300 tokens, so each call costs one
    prefill of that length.
    """
    probe_text = (
        mode_switching_prompt
        + tokenizer.decode(thinker_tokens, skip_special_tokens=False)
        + tokenizer.decode(writer_tokens, skip_special_tokens=False)
        + mode_switching_question
    )
    ids = encode(probe_text, tokenizer)
    blk = session.create_block()
    try:
        logits = session.prefill_block(blk, ids)[0].float().cpu()
        yes_logit = float(logits[yes_id])
        no_logit = float(logits[no_id])
        return yes_logit > no_logit, yes_logit, no_logit
    finally:
        free_block(session, blk)
