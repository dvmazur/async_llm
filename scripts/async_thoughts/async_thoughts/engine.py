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
from minisgl.shared_cache import SharedBlock, SharedCacheSession
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
        max_seq_len_override=8192 * 2,
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

    The probe still runs a fresh prefill of the whole context (correct: reusing
    the *live* thinker/writer blocks as context under the new
    ``mode_switching_prompt`` prefix would attend to stale KV/state).  But the
    growing thinker/writer parts are already token-id lists, so we concatenate
    them directly instead of decoding to text and re-encoding every call — that
    avoids O(context) CPU re-encoding and any decode/encode token drift.  Only
    the constant prompt/question fragments are encoded.
    """
    mode_ids = encode(mode_switching_prompt, tokenizer)
    question_ids = encode(mode_switching_question, tokenizer)
    ids = torch.cat(
        [
            mode_ids,
            torch.tensor(thinker_tokens, dtype=torch.int32),
            torch.tensor(writer_tokens, dtype=torch.int32),
            question_ids,
        ]
    )
    blk = session.create_block()
    try:
        # The probe block is read once (for yes/no logits) then freed, so skip
        # the GDN affine capture — on hybrid (Qwen3.5) models that O(seq) capture
        # over the growing probe context is the dominant per-probe cost.
        logits = session.prefill_block(blk, ids, capture_affine=False)[0].float().cpu()
        yes_logit = float(logits[yes_id])
        no_logit = float(logits[no_id])
        return yes_logit > no_logit, yes_logit, no_logit
    finally:
        free_block(session, blk)
