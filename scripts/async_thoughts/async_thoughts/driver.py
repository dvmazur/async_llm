"""Async-reasoning policy driver on the scheduler's shared-cache service.

Drives the arXiv:2512.10931 thinker/writer state machine against minisgl's
scheduler-owned ``SharedCacheService`` (``llm.shared_cache_service``), whose
``WorkerGroup``/block forwards return **logprobs**. All reasoning *policy* — the
state machine,
the mode-switch probe, forbidden-token masking + argmax, and streaming — lives
here, *around* the engine; the scheduler owns only the mechanism. This is
application/harness code, hence it lives in the scripts project rather than
the ``minisgl`` library.

Cache layout (mirrors AsyncReasoning's AsyncReasoningCache):

  input_prompt block  : chat-formatted user prompt ending with <|im_end|>.
  thinker_output block: pre-filled with "<|im_start|>assistant\\n<think>\\n",
                        grows as the thinker decodes.
  writer_output block : pre-filled with " ... [SYSTEM: thoughts will continue
                        here]\\n</think>\\n", grows as the writer decodes.
                        Both prefixes are prefilled IN CONTEXT of the blocks
                        before them (matters numerically -- prefilling
                        standalone gives the writer prefix the wrong attention
                        outputs at deeper layers).

Masking is done on logprobs with the same ``-= 100`` the original demo applies
to raw logits; because ``log_softmax`` is a per-row constant shift, masked
argmax and the probe's yes/no comparison are token-identical to the
logit-space versions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch
from minisgl.shared_cache import WorkerGroup

if TYPE_CHECKING:
    from minisgl.scheduler import SharedCacheService

# Default math problem from the original AsyncReasoning notebook.
DEFAULT_PROBLEM = "Calculate x - x^2 + x^3 for x = 5, 6, 7, 8. Return all 4 answers in \\boxed{ }."

# Qwen3-32B at bf16 fits on a single 80 GiB GPU and keeps the writer alive;
# smaller models starve the writer because the probe margin is too small.
DEFAULT_MODEL = os.environ.get("MINISGL_DEMO_MODEL", "Qwen/Qwen3-32B")
DEFAULT_MAX_STEPS = 800
DEFAULT_PROBE_PERIOD = 30
DEFAULT_MEMORY_RATIO = 0.9
DEFAULT_PAGE_SIZE = 1

# Boundary markers neither stream may emit (skipped if a tokenizer lacks them).
_WRITER_FORBID_NAMES = ["</think>", "<|im_start|>", "<|endoftext|>"]
_THINKER_FORBID_NAMES = ["</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]


class Prompting:
    """Prompt fragments + the mode-switching probe text.

    Verbatim port of AsyncReasoning's ``AsyncReasoningPrompting``.
    """

    def __init__(self, problem: str):
        self.input_prompt = f"<|im_start|>user\n{problem}\n"
        self.thinker_output_prefix = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        self.writer_output_prefix = " ... [SYSTEM: thoughts will continue here]\n</think>\n"
        self.mode_switching_prompt = (
            "<|im_start|>user\n"
            "You are an AI assistant that can think and write responses concurrently, "
            "and you must decide whether or not you should pause writing and think more.\n"
            "Read the current partial thoughts and response below, then decide whether "
            "you can continue writing the response without pausing (yes/no):\n"
            ' - Answer "yes" if your thoughts have enough information to write the next '
            "response paragraph, even if the full task is not solved yet.\n"
            ' - Answer "no" if your thoughts aren\'t enough to write the next response '
            "paragraph, i.e. if your response ran out of thoughts.\n"
        )
        self.mode_switching_question = (
            "...\n\nWait, are my current thoughts enough to write the next paragraph "
            "or formula? (yes/no): "
        )
        self.yes_token = "yes"
        self.no_token = "no"


def encode(text: str, tokenizer) -> torch.Tensor:
    """1-D int32 CPU tensor.

    ``add_special_tokens=False`` because all the boundary markers are encoded
    literally in the prompt strings.
    """
    return (
        tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        .view(-1)
        .to(torch.int32)
    )


def single_token_id(text: str, tokenizer) -> int:
    """Resolve a string we expect to be exactly one token to its vocab id.

    Used both for the literal ``"\\n\\n"`` separator and for the ``yes``/``no``
    probe tokens.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"Expected {text!r} to tokenize to 1 token, got {len(ids)}: {ids}")
    return int(ids[0])


def vocab_id_or_none(tokenizer, text: str) -> int | None:
    """Return ``tokenizer.vocab[text]`` if present, else ``None``.

    Used to map special-token strings (``</think>``, ``<|im_start|>``, ...) to
    ids, gracefully skipping markers a given tokenizer doesn't have.
    """
    return tokenizer.vocab.get(text)


def ends_with_double_newline(token_ids: List[int], tokenizer) -> bool:
    """Mirror AR's ``is_end_of_step``: decode the last two tokens, check tail."""
    if len(token_ids) < 2:
        return False
    return tokenizer.decode(token_ids[-2:]).endswith("\n\n")


@dataclass
class ReasoningConfig:
    max_steps: int = DEFAULT_MAX_STEPS
    probe_period: int = DEFAULT_PROBE_PERIOD


class ReasoningDriver:
    """Drives one async-reasoning chain against a shared-cache backend.

    The optional callbacks stream progress (e.g. to a colored CLI): the token
    callbacks receive each newly-decided token's decoded text, and
    ``on_state_change`` receives a human-readable description of every
    state-machine transition.
    """

    def __init__(
        self,
        backend: "SharedCacheService",
        tokenizer,
        problem: str = DEFAULT_PROBLEM,
        config: Optional[ReasoningConfig] = None,
        device: Optional[torch.device] = None,
        on_thinker_token: Optional[Callable[[str], None]] = None,
        on_writer_token: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[str], None]] = None,
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.prompting = Prompting(problem)
        self.config = config or ReasoningConfig()
        self.device = device or torch.device("cpu")
        self.on_thinker_token = on_thinker_token
        self.on_writer_token = on_writer_token
        self.on_state_change = on_state_change

    def _mask_argmax(self, dist_row: torch.Tensor, forbid_ids: torch.Tensor) -> int:
        # -= 100 on the fresh logprob row is argmax-identical to the demo's logit
        # mask (log_softmax is a per-row constant shift).
        dist_row[forbid_ids] -= 100.0
        return int(dist_row.argmax().item())

    def _probe(
        self, thinker_tokens: List[int], writer_tokens: List[int], yes_id: int, no_id: int
    ) -> Tuple[bool, float, float]:
        """Monolithically prefill the mode-switching probe and compare yes/no.

        This re-encodes the whole probe context each call (no caching across
        iterations).  That's a small fixed cost: the probe runs every ~20-30
        main-loop steps, the probe text is ~100-300 tokens, so each call costs
        one prefill of that length.
        """
        tok, p = self.tokenizer, self.prompting
        probe_text = (
            p.mode_switching_prompt
            + tok.decode(thinker_tokens, skip_special_tokens=False)
            + tok.decode(writer_tokens, skip_special_tokens=False)
            + p.mode_switching_question
        )
        blk = self.backend.create_block()
        try:
            dist = self.backend.prefill_block(blk, encode(probe_text, tok), return_logprobs=True)[0]
            yes_lp, no_lp = float(dist[yes_id]), float(dist[no_id])
            return yes_lp > no_lp, yes_lp, no_lp
        finally:
            self.backend.free_block(blk)

    def _state_change(self, message: str) -> None:
        if self.on_state_change is not None:
            self.on_state_change(message)

    @torch.inference_mode()
    def run(self) -> Dict[str, object]:
        tok, p, backend = self.tokenizer, self.prompting, self.backend

        writer_forbid = torch.tensor(
            [i for i in (vocab_id_or_none(tok, n) for n in _WRITER_FORBID_NAMES) if i is not None],
            dtype=torch.long,
            device=self.device,
        )
        thinker_forbid = torch.tensor(
            [i for i in (vocab_id_or_none(tok, n) for n in _THINKER_FORBID_NAMES) if i is not None],
            dtype=torch.long,
            device=self.device,
        )
        yes_id = single_token_id(p.yes_token, tok)
        no_id = single_token_id(p.no_token, tok)
        nn_id = single_token_id("\n\n", tok)
        eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else -1

        prompt_blk = backend.create_block()
        thinker_blk = backend.create_block()
        writer_blk = backend.create_block()

        try:
            # input_prompt: prefilled standalone (matches AR -- input_prompt has
            # no context to attend to). The prefixes are prefilled IN CONTEXT of
            # the blocks before them, in a single batched prefill pass each.
            backend.prefill_block(prompt_blk, encode(p.input_prompt, tok))

            thinker_prefix_ids = encode(p.thinker_output_prefix, tok)
            backend.prefill_block(thinker_blk, thinker_prefix_ids, context=[prompt_blk])

            writer_prefix_ids = encode(p.writer_output_prefix, tok)
            backend.prefill_block(writer_blk, writer_prefix_ids, context=[prompt_blk, thinker_blk])

            # Token-sequence bookkeeping for display + the mode-switching probe.
            # Includes the "\n\n" separator that AR appends but does NOT
            # prefill; it gets sent as the first decode-step input below.
            thinker_tokens: List[int] = thinker_prefix_ids.tolist() + [nn_id]
            writer_tokens: List[int] = writer_prefix_ids.tolist() + [nn_id]

            # Index of the next token to stream to the callbacks. We skip the
            # prefix since it's boilerplate, but the model internally sees it.
            next_thinker = len(thinker_prefix_ids)
            next_writer = len(writer_prefix_ids)
            writer_stream: List[int] = []
            probe_decisions: List[bool] = []

            thinker_only = WorkerGroup(
                cache_structure=[[prompt_blk, thinker_blk]],
                write_to=[thinker_blk],
            )
            thinker_and_writer = WorkerGroup(
                cache_structure=[
                    [prompt_blk, thinker_blk],
                    [prompt_blk, thinker_blk, writer_blk],
                ],
                write_to=[thinker_blk, writer_blk],
            )

            state = "thinker_only"
            for step in range(self.config.max_steps):
                # decode one (or two) tokens
                if state == "thinker_only":
                    inp = torch.tensor([thinker_tokens[-1]], dtype=torch.int32)
                    lp = backend.decode_group(thinker_only, inp, return_logprobs=True)
                    thinker_tokens.append(self._mask_argmax(lp[0], thinker_forbid))
                else:
                    inp = torch.tensor([thinker_tokens[-1], writer_tokens[-1]], dtype=torch.int32)
                    lp = backend.decode_group(thinker_and_writer, inp, return_logprobs=True)
                    thinker_tokens.append(self._mask_argmax(lp[0], thinker_forbid))
                    writer_tokens.append(self._mask_argmax(lp[1], writer_forbid))

                    # Writer hit \n\n -> end of a writer step; back to thinker_only.
                    if writer_tokens[-1] == nn_id or ends_with_double_newline(writer_tokens, tok):
                        state = "thinker_only"
                        self._state_change("writer end-of-step -> thinker_only")

                # stream newly-decided tokens
                while next_thinker < len(thinker_tokens):
                    if self.on_thinker_token is not None:
                        self.on_thinker_token(tok.decode([thinker_tokens[next_thinker]]))
                    next_thinker += 1
                while next_writer < len(writer_tokens):
                    writer_stream.append(writer_tokens[next_writer])
                    if self.on_writer_token is not None:
                        self.on_writer_token(tok.decode([writer_tokens[next_writer]]))
                    next_writer += 1

                # mode-switching probe
                if (step + 1) % self.config.probe_period == 0 or ends_with_double_newline(
                    thinker_tokens, tok
                ):
                    should_write, yes_lp, no_lp = self._probe(
                        thinker_tokens, writer_tokens, yes_id, no_id
                    )
                    probe_decisions.append(should_write)
                    new_state = "thinker_and_writer" if should_write else "thinker_only"
                    if new_state != state:
                        self._state_change(
                            f"step {step + 1}: probe yes={yes_lp:.2f} no={no_lp:.2f} -> {new_state}"
                        )
                    state = new_state

                # termination
                if writer_tokens[-1] == eos_id:
                    self._state_change("writer hit EOS -- terminating")
                    break
        finally:
            backend.free_block(prompt_blk)
            backend.free_block(thinker_blk)
            backend.free_block(writer_blk)

        return {
            "writer_text": tok.decode(writer_tokens, skip_special_tokens=True),
            "thinker_text": tok.decode(thinker_tokens, skip_special_tokens=True),
            "writer_tokens": writer_tokens,
            "thinker_tokens": thinker_tokens,
            "writer_stream": writer_stream,
            "probe_decisions": probe_decisions,
        }
