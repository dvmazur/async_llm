"""AsyncLLM setup, tokenizer helpers, and the live-cache mode-switching probe."""

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
) -> AsyncLLM:
    """Build a single-GPU engine for two decode streams and a four-block probe."""
    return AsyncLLM(
        model_path,
        dtype=torch.bfloat16,
        max_running_req=4,
        cuda_graph_bs=[1, 2],
        cuda_graph_max_bs=2,
        shared_cuda_graph_max_depth=4,
        shared_cuda_graph_prefill_rows=[32, 256],
        max_prefill_rows=256,
        memory_ratio=memory_ratio,
        max_seq_len_override=8192,
        page_size=page_size,
        distributed_addr="tcp://127.0.0.1:0",
    )


def encode(text: str, tokenizer: AutoTokenizer) -> torch.Tensor:
    """Encode literal prompt markers into a flat int32 CPU tensor."""
    return (
        tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        .view(-1)
        .to(torch.int32)
    )


def single_token_id(text: str, tokenizer: AutoTokenizer) -> int:
    """Resolve a string that must encode to exactly one token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"Expected {text!r} to tokenize to 1 token, got {len(ids)}: {ids}")
    return int(ids[0])


def vocab_id_or_none(tokenizer: AutoTokenizer, text: str) -> int | None:
    """Look up an optional special token."""
    return tokenizer.vocab.get(text)


def ends_with_double_newline(token_ids: List[int], tokenizer: AutoTokenizer) -> bool:
    """Detect a reasoning step boundary in the last two tokens."""
    if len(token_ids) < 2:
        return False
    return tokenizer.decode(token_ids[-2:]).endswith("\n\n")


class ModeSwitchProbe:
    """Ask a fixed question using the live thinker and writer cache blocks.

    The static prompt is cached until ``close()``. Each question uses a fresh
    output block, so its tokens never enter either generation stream.
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
        self._prompt_block: CacheBlock | None = None

    async def check_continue_writing(
        self, thinker: CacheBlock, writer: CacheBlock
    ) -> Tuple[bool, float, float]:
        """Return ``(yes_logit > no_logit, yes_logit, no_logit)``."""
        if self._prompt_block is None:
            prompt = await self.llm.create_block()
            try:
                await self.llm.forward(self._prompt_ids, write_to=prompt, return_logits=False)
            except BaseException:
                await self.llm.free_block(prompt)
                raise
            self._prompt_block = prompt

        question = await self.llm.create_block()
        try:
            out = await self.llm.forward(
                self._question_ids, [self._prompt_block, thinker, writer, question]
            )
            yes_logit, no_logit = out.logits[[self.yes_id, self.no_id]].float().cpu().tolist()
            return yes_logit > no_logit, yes_logit, no_logit
        finally:
            await self.llm.free_block(question)

    async def close(self) -> None:
        if self._prompt_block is not None:
            await self.llm.free_block(self._prompt_block)
            self._prompt_block = None
