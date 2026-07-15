"""Thin wrappers around minisgl's AsyncLLM.

Everything in here is a helper that minisgl doesn't expose directly: AsyncLLM
construction sized for the demo, tokenizer encoding, and the mode-switching
probe.  All reasoning *policy* (prompts, forbidden sets, state machine) stays
in ``demo.py``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from minisgl.llm import AsyncLLM
from transformers import AutoTokenizer


def build_async_llm(
    model_path: str,
    memory_ratio: float,
    page_size: int = 1,
) -> AsyncLLM:
    """Build a single-GPU AsyncLLM sized for 1- or 2-worker decode."""
    return AsyncLLM(
        model_path,
        dtype=torch.bfloat16,
        max_running_req=4,
        cuda_graph_bs=[1, 2],  # we only ever run 1- or 2-worker decode
        cuda_graph_max_bs=2,
        memory_ratio=memory_ratio,
        max_seq_len_override=8192 * 2,
        page_size=page_size,
    )


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


async def check_continue_writing(
    llm: AsyncLLM,
    tokenizer: AutoTokenizer,
    mode_switching_prompt: str,
    mode_switching_question: str,
    thinker_tokens: List[int],
    writer_tokens: List[int],
    yes_id: int,
    no_id: int,
) -> Tuple[bool, float, float]:
    """Prefill the mode-switching probe and compare the yes/no logits.

    Returns ``(should_continue_writing, yes_logit, no_logit)``.  The two raw
    logit values are useful for debug logging since they show how confident the
    probe was at any given step.

    The probe runs a fresh, throwaway prefill of the whole context (correct:
    reusing the *live* thinker/writer blocks as context under the new
    ``mode_switching_prompt`` prefix would attend to stale KV/state).  The
    growing thinker/writer parts arrive as token-id lists, so we concatenate
    them directly instead of decoding to text and re-encoding every call.
    Runs concurrently with the decode streams: the scheduler slots the prefill
    between decode ticks.
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
    # The probe block is read once (for yes/no logits) then freed, so skip
    # the GDN affine capture — on hybrid (Qwen3.5) models that O(seq) capture
    # over the growing probe context is the dominant per-probe cost.
    res = await llm.prefill_block(ids, capture_affine=False, return_logits=True)
    try:
        logits = res.logits.float().cpu()
        yes_logit = float(logits[yes_id])
        no_logit = float(logits[no_id])
        return yes_logit > no_logit, yes_logit, no_logit
    finally:
        await llm.free_block(res.block)
