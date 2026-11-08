"""AsyncLLM setup, tokenizer helpers, and the live-cache mode-switching probe."""

from __future__ import annotations

from typing import List, Tuple

import torch
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import CacheBlock
from transformers import AutoTokenizer, GenerationConfig


def build_async_llm(
    model_path: str,
    memory_ratio: float,
    page_size: int = 1,
    *,
    max_context_len: int = 8192,
    generation_config: GenerationConfig | None = None,
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
        max_seq_len_override=max_context_len,
        **({"generation_config": generation_config} if generation_config is not None else {}),
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


def _single_token_spellings(label: str, tokenizer: AutoTokenizer) -> list[int]:
    """Candidate IDs in spelling order, starting with the supplied unspaced label."""
    ids = []
    unknown = getattr(tokenizer, "unk_token_id", None)
    for prefix in ("", " "):
        for spelling in (label.lower(), label.title(), label.upper()):
            tokens = tokenizer.encode(prefix + spelling, add_special_tokens=False)
            if len(tokens) == 1 and tokens[0] != unknown and tokens[0] not in ids:
                ids.append(int(tokens[0]))
    if not ids:
        raise RuntimeError(f"No single-token spellings for probe label {label!r}")
    return ids


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
        mode_switching_prompt: str | List[int],
        mode_switching_question: str,
        yes_token: str,
        no_token: str,
        *,
        label_variants: bool = False,
    ) -> None:
        self.llm = llm
        self._label_indices = None
        if label_variants:
            yes_ids = _single_token_spellings(yes_token, tokenizer)
            no_ids = _single_token_spellings(no_token, tokenizer)
            if set(yes_ids).intersection(no_ids):
                raise ValueError("The yes and no probe labels share a token ID")
            self.yes_id, self.no_id = yes_ids[0], no_ids[0]
            self._yes_ids, self._no_ids = sorted(yes_ids), sorted(no_ids)
            self._label_indices = torch.tensor(self._yes_ids + self._no_ids, dtype=torch.long)
        else:
            self.yes_id = single_token_id(yes_token, tokenizer)
            self.no_id = single_token_id(no_token, tokenizer)
        self._prompt_ids = (
            encode(mode_switching_prompt, tokenizer)
            if isinstance(mode_switching_prompt, str)
            else torch.tensor(mode_switching_prompt, dtype=torch.int32)
        )
        self._question_ids = encode(mode_switching_question, tokenizer)
        self._prompt_block: CacheBlock | None = None

    def can_fit_start_question(self, length: int) -> bool:
        """Leave room for the thinker's pending decode while scoring a private copy of the notes."""
        engine = getattr(self.llm, "engine", None)
        allocator = getattr(engine, "page_allocator", None)
        return length <= getattr(engine, "max_seq_len", length) and (
            allocator is None or length <= (allocator.num_free_pages - 1) * allocator.page_size
        )

    async def check_start_writing(self, input_ids: List[int]) -> Tuple[bool, float, float]:
        """Score an independent native chat without writing into either role's cache."""
        scratch = await self.llm.create_block()
        try:
            out = await self.llm.forward(input_ids, write_to=scratch, return_logits=True)
            return self._vote(out.logits)
        finally:
            await self.llm.free_block(scratch)

    async def check_continue_writing(
        self, thinker: CacheBlock, writer: CacheBlock
    ) -> Tuple[bool, float, float]:
        """Return the vote and the best logit for each of the two probe labels."""
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
            return self._vote(out.logits)
        finally:
            await self.llm.free_block(question)

    def _vote(self, logits: torch.Tensor) -> Tuple[bool, float, float]:
        if self._label_indices is None:
            yes, no = logits[[self.yes_id, self.no_id]].float().cpu().tolist()
            return yes > no, yes, no

        self._label_indices = self._label_indices.to(logits.device)
        scores = logits[self._label_indices].float().cpu().tolist()
        split = len(self._yes_ids)
        yes_scores, no_scores = scores[:split], scores[split:]
        yes, no = max(yes_scores), max(no_scores)
        # Sorted IDs make a tie match torch.argmax's vocabulary order.
        vote = yes > no or (
            yes == no and self._yes_ids[yes_scores.index(yes)] < self._no_ids[no_scores.index(no)]
        )
        return vote, yes, no

    async def close(self) -> None:
        if self._prompt_block is not None:
            await self.llm.free_block(self._prompt_block)
            self._prompt_block = None
