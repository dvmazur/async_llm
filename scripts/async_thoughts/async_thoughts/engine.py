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
from minisgl.shared_cache import CacheBlock
from transformers import AutoTokenizer


def build_async_llm(
    model_path: str,
    memory_ratio: float,
    page_size: int = 1,
    num_pages: int | None = None,
    attention_backend: str = "auto",
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
        num_page_override=num_pages,
        attention_backend=attention_backend,
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


class ModeSwitchProbe:
    """The mode-switching probe, with everything static cached across calls.

    Each probe is the design doc's "action choice" use of ``AsyncLLM.forward``:
    a fresh, throwaway prefill of the probe context (correct: reusing the
    *live* thinker/writer blocks as context under the new
    ``mode_switching_prompt`` prefix would attend to stale KV/state), read for
    one yes/no logits row and freed.  The growing thinker/writer parts arrive
    as token-id lists, so we concatenate them directly instead of decoding to
    text and re-encoding every call.  Runs concurrently with the decode
    streams: the scheduler slots the prefill between decode ticks.

    Cached here, computed once instead of per call:

    * the encoded ``mode_switching_prompt`` / ``mode_switching_question`` ids
      and the yes/no vocab ids;
    * on standard-attention models, the KV of the static prompt prefix — it is
      prefilled (lazily) into a persistent block, and each probe then only
      prefills the growing thinker/writer tail in context of it (``forward``'s
      conditional-prefill mode).

    Call ``close()`` when done to release the cached prefix block.
    """

    def __init__(
        self,
        llm: AsyncLLM,
        tokenizer: AutoTokenizer,
        mode_switching_prompt: str,
        mode_switching_question: str,
        yes_token: str,
        no_token: str,
    ):
        self.llm = llm
        self.yes_id = single_token_id(yes_token, tokenizer)
        self.no_id = single_token_id(no_token, tokenizer)
        self._prompt_ids = encode(mode_switching_prompt, tokenizer)
        self._question_ids = encode(mode_switching_question, tokenizer)
        self._reuse_prompt_kv = getattr(llm.async_engine.session, "sc_gdn", None) is None
        self._prompt_block: CacheBlock | None = None

    async def check_continue_writing(
        self, thinker_tokens: List[int], writer_tokens: List[int]
    ) -> Tuple[bool, float, float]:
        """Prefill the probe and compare the yes/no logits.

        Returns ``(should_continue_writing, yes_logit, no_logit)``.  The two
        raw logit values are useful for debug logging since they show how
        confident the probe was at any given step.
        """
        tail = torch.cat(
            [
                torch.tensor(thinker_tokens, dtype=torch.int32),
                torch.tensor(writer_tokens, dtype=torch.int32),
                self._question_ids,
            ]
        )
        if self._reuse_prompt_kv:
            if self._prompt_block is None:
                self._prompt_block = await self.llm.create_block()
                await self.llm.forward(
                    self._prompt_ids, write_to=self._prompt_block, return_logits=False
                )
            context, ids = [self._prompt_block], tail
        else:
            context, ids = [], torch.cat([self._prompt_ids, tail])
        block = await self.llm.create_block()
        try:
            # The probe block is read once then freed, so skip the GDN affine
            # capture on the flat (hybrid) path; ignored with a context.
            out = await self.llm.forward(ids, context, write_to=block)
            logits = out.logits.float().cpu()
            yes_logit = float(logits[self.yes_id])
            no_logit = float(logits[self.no_id])
            return yes_logit > no_logit, yes_logit, no_logit
        finally:
            await self.llm.free_block(block)

    async def close(self) -> None:
        if self._prompt_block is not None:
            await self.llm.free_block(self._prompt_block)
            self._prompt_block = None
