#!/usr/bin/env python3
"""
Async Thoughts demo on minisgl's shared-cache.

Port of the notebook
    https://github.com/yandex-research/AsyncReasoning/blob/main/notebooks/demo_async_thoughts.ipynb
to a standalone Python file driven by minisgl's in-process ``SharedCacheSession``
(no HTTP server, no notebook).

Two streams run concurrently against the same model:
  * thinker -- internal chain-of-thought reasoning between <think>...</think>
  * writer  -- user-facing answer, sees the thinker's partial reasoning

A periodic mode-switching probe asks the model whether the thinker has produced
enough new reasoning to let the writer continue.  Based on yes/no the demo
switches between two states:

  thinker_only        : single-worker decode of just the thinker.
  thinker_and_writer  : batched two-worker decode of thinker + writer.

Cache layout (mirrors AsyncReasoning's AsyncReasoningCache):

  input_prompt block  : chat-formatted user prompt ending with <|im_end|>.
  thinker_output block: pre-filled with "<|im_start|>assistant\\n<think>\\n",
                        grows as the thinker decodes.
  writer_output block : pre-filled with " ... [SYSTEM: thoughts will continue
                        here]\\n</think>\\n", grows as the writer decodes.
                        This block is prefilled IN CONTEXT of the prompt +
                        thinker prefix (matters numerically -- prefilling
                        standalone gives the writer prefix the wrong attention
                        outputs at deeper layers).

Mode-switching probe:
  Build the concatenated probe text
    "<|im_start|>user\\n<mode-switching-system-prompt>"
    + current thinker tokens
    + current writer tokens
    + "\\n\\nWait, are my current thoughts enough...? (yes/no): "
  Monolithically prefill into a scratch block and compare logits at "yes" vs
  "no".  This is simpler than threading the probe through the shared-cache
  abstraction and only runs every N steps -- the extra prefill cost is small.

Usage:
    MINISGL_DEMO_MODEL=Qwen/Qwen3-8B \\
        uv run python scripts/demo_async_thoughts.py

    # Override the math problem:
    uv run python scripts/demo_async_thoughts.py \\
        --problem "What is 17 * 23?"

    # Limit total steps for a quick smoke test:
    uv run python scripts/demo_async_thoughts.py --max-steps 60
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from typing import List

import torch
from transformers import AutoTokenizer

from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.shared_cache import SharedBlock, SharedCacheSession, WorkerGroup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("MINISGL_DEMO_MODEL", "Qwen/Qwen3-8B")
DEFAULT_PROBLEM = (
    "Calculate x - x^2 + x^3 for x = 5, 6, 7, 8. "
    "Return all 4 answers in \\boxed{ }."
)


class Prompting:
    """Verbatim port of AsyncReasoning's AsyncReasoningPrompting."""

    def __init__(self, problem: str):
        self.input_prompt = (
            f"<|im_start|>user\n{problem}\n"
        )
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


# ANSI for streaming output.
_C = {
    "thinker": "\033[2;36m",   # dim cyan
    "writer":  "\033[1;32m",   # bold green
    "state":   "\033[1;33m",   # bold yellow
    "dim":     "\033[2m",
    "reset":   "\033[0m",
}


def _ansi(text: str, *keys: str) -> str:
    return "".join(_C[k] for k in keys) + text + _C["reset"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode(text: str, tokenizer) -> torch.Tensor:
    """1-D int32 CPU tensor.  add_special_tokens=False because all the
    boundary markers are encoded literally in the prompt strings."""
    return tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    ).view(-1).to(torch.int32)


def _encode_single_token(text: str, tokenizer) -> int:
    """Encode a string that we expect to be one token (used for the literal
    ``\\n\\n`` separator between prefix and first generation)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(
            f"Expected {text!r} to tokenize to 1 token, got {len(ids)}: {ids}"
        )
    return int(ids[0])


def _single_token_id(text: str, tokenizer) -> int:
    """Resolve a literal token (e.g. ``"yes"``) to its single vocab id."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(
            f"Expected {text!r} to be a single token; got {ids}"
        )
    return int(ids[0])


def _vocab_id_or_none(tok: AutoTokenizer, text: str) -> int | None:
    """Return tokenizer.vocab[text] if present, else None.  Used to map
    special-token strings (``</think>``, ``<|im_start|>``, etc) to ids."""
    return tok.vocab.get(text)


def _free_block(session: SharedCacheSession, block: SharedBlock) -> None:
    """Return a block's pages to the session pool and reset it."""
    pages = block.get_page_indices()
    block.clear()
    if pages.numel():
        session._free_pages_back(pages)  # noqa: SLF001 -- internal but stable


def _prefill_block_in_context(
    session: SharedCacheSession,
    write_to: SharedBlock,
    context: List[SharedBlock],
    token_ids: torch.Tensor,
) -> None:
    """Append ``token_ids`` to ``write_to`` one token at a time, with the
    rest of ``context`` visible as past KV.  This emulates AsyncReasoning's
    multi-token in-context prefill, which minisgl's ``prefill_block`` does
    not expose directly (it only supports cached_len=0).

    The cost is N forward passes for N tokens, but it runs once at setup
    so the overhead is small.
    """
    group = WorkerGroup(
        cache_structure=[list(context) + [write_to]],
        write_to=[write_to],
    )
    for tok_id in token_ids.tolist():
        session.decode_step(
            group, torch.tensor([int(tok_id)], dtype=torch.int32)
        )


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------


def _build_engine(model_path: str, memory_ratio: float) -> Engine:
    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=4,
        cuda_graph_bs=[1, 2],   # we only ever run 1- or 2-worker decode
        cuda_graph_max_bs=2,
        memory_ratio=memory_ratio,
        max_seq_len_override=4096,
    )
    return Engine(config)


# ---------------------------------------------------------------------------
# Mode-switching probe
# ---------------------------------------------------------------------------


def _check_continue_writing(
    session: SharedCacheSession,
    tokenizer: AutoTokenizer,
    prompting: Prompting,
    thinker_tokens: List[int],
    writer_tokens: List[int],
    yes_id: int,
    no_id: int,
) -> tuple[bool, float, float]:
    """Monolithically prefill the mode-switching probe and compare yes/no.

    Returns ``(should_continue_writing, yes_logit, no_logit)``.  The two
    raw logit values are useful for debug logging since they show how
    confident the probe was at any given step.

    This re-encodes the whole probe context each call (no caching across
    iterations).  That's a small fixed cost: probe runs every ~20 main-loop
    steps, the probe text is ~100-300 tokens, so each call costs one
    prefill of that length.
    """
    probe_text = (
        prompting.mode_switching_prompt
        + tokenizer.decode(thinker_tokens, skip_special_tokens=False)
        + tokenizer.decode(writer_tokens, skip_special_tokens=False)
        + prompting.mode_switching_question
    )
    ids = _encode(probe_text, tokenizer)
    blk = session.create_block()
    try:
        logits = session.prefill_block(blk, ids)[0].float().cpu()
        yes_logit = float(logits[yes_id])
        no_logit = float(logits[no_id])
        return yes_logit > no_logit, yes_logit, no_logit
    finally:
        _free_block(session, blk)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_header(text: str) -> None:
    bar = "─" * 64
    print(f"\n{bar}")
    print(_ansi(f"  {text}", "state"))
    print(f"{bar}\n", flush=True)


def _print_state_change(text: str) -> None:
    print(_ansi(f"\n  [{text}]", "state"), flush=True)


def _stream_token(text: str, role: str) -> None:
    sys.stdout.write(_ansi(text, role))
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model path (default: {DEFAULT_MODEL}).")
    p.add_argument("--problem", default=DEFAULT_PROBLEM,
                   help="The user problem the assistant should solve.")
    p.add_argument("--max-steps", type=int, default=1024,
                   help="Hard cap on total decode steps.")
    p.add_argument("--probe-period", type=int, default=20,
                   help="Run the mode-switching probe every N decode steps "
                        "(plus on every thinker end-of-step).")
    p.add_argument("--memory-ratio", type=float, default=0.3,
                   help="Fraction of free GPU memory to reserve for KV cache.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(2)

    _print_header("Async Thoughts Demo (minisgl)")
    print(f"  model    : {args.model}")
    print(f"  problem  : {args.problem}")
    print(f"  max steps: {args.max_steps}")

    print("\nLoading tokenizer & engine...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine = _build_engine(args.model, memory_ratio=args.memory_ratio)
    session = SharedCacheSession(engine)
    prompting = Prompting(args.problem)

    try:
        # ------------------------------------------------------------------
        # Forbidden token ids: never let the streams emit boundary markers
        # ------------------------------------------------------------------
        # Match AR's two forbidden sets, gracefully skipping any markers the
        # tokenizer doesn't have (e.g. for non-Qwen models without <think>).
        writer_forbid_names = ["</think>", "<|im_start|>", "<|endoftext|>"]
        thinker_forbid_names = [
            "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        ]
        writer_forbid_ids = [
            i for i in (_vocab_id_or_none(tokenizer, n) for n in writer_forbid_names)
            if i is not None
        ]
        thinker_forbid_ids = [
            i for i in (_vocab_id_or_none(tokenizer, n) for n in thinker_forbid_names)
            if i is not None
        ]

        yes_id = _single_token_id(prompting.yes_token, tokenizer)
        no_id = _single_token_id(prompting.no_token, tokenizer)

        # ------------------------------------------------------------------
        # Prefill the three persistent blocks
        # ------------------------------------------------------------------
        print("Prefilling blocks...")
        prompt_blk = session.create_block()
        thinker_blk = session.create_block()
        writer_blk = session.create_block()

        # input_prompt: prefilled standalone (matches AR -- input_prompt has
        # no context to attend to).
        prompt_ids = _encode(prompting.input_prompt, tokenizer)
        session.prefill_block(prompt_blk, prompt_ids)

        # thinker_output_prefix: prefilled IN CONTEXT of [prompt].  AR does
        # this via SharedCacheManager(view=[input_prompt, thinker_output]).
        # minisgl has no public multi-token in-context prefill, so we feed
        # the prefix tokens one at a time via decode_step.
        thinker_prefix_ids = _encode(prompting.thinker_output_prefix, tokenizer)
        _prefill_block_in_context(
            session, write_to=thinker_blk, context=[prompt_blk],
            token_ids=thinker_prefix_ids,
        )

        # writer_output_prefix: prefilled IN CONTEXT of [prompt, thinker].
        writer_prefix_ids = _encode(prompting.writer_output_prefix, tokenizer)
        _prefill_block_in_context(
            session, write_to=writer_blk, context=[prompt_blk, thinker_blk],
            token_ids=writer_prefix_ids,
        )

        # ------------------------------------------------------------------
        # Token-sequence bookkeeping for display + the mode-switching probe.
        # Includes the "\n\n" separator that AR appends but does NOT prefill;
        # it gets sent as the first decode-step input below.
        # ------------------------------------------------------------------
        nn_id = _encode_single_token("\n\n", tokenizer)
        thinker_tokens: List[int] = thinker_prefix_ids.tolist() + [nn_id]
        writer_tokens:  List[int] = writer_prefix_ids.tolist()  + [nn_id]

        # Index into *_tokens of the next token to stream-print.  We skip the
        # prefix when displaying since it's boilerplate, but the model
        # internally sees it.
        next_print_thinker = len(thinker_prefix_ids)
        next_print_writer  = len(writer_prefix_ids)

        # ------------------------------------------------------------------
        # WorkerGroup factories.  Recreate per step -- they're just
        # references; the blocks themselves grow in-place.
        # ------------------------------------------------------------------
        def thinker_only_group() -> WorkerGroup:
            return WorkerGroup(
                cache_structure=[[prompt_blk, thinker_blk]],
                write_to=[thinker_blk],
            )

        def thinker_and_writer_group() -> WorkerGroup:
            return WorkerGroup(
                cache_structure=[
                    [prompt_blk, thinker_blk],
                    [prompt_blk, thinker_blk, writer_blk],
                ],
                write_to=[thinker_blk, writer_blk],
            )

        # ------------------------------------------------------------------
        # Main decode loop
        # ------------------------------------------------------------------
        state = "thinker_only"
        _print_header("Generation")
        print(_ansi("  thinker (dim cyan) | writer (bold green)\n", "dim"))

        eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1

        for step in range(args.max_steps):
            # --- decode one (or two) tokens ---
            if state == "thinker_only":
                inp = torch.tensor([thinker_tokens[-1]], dtype=torch.int32)
                logits = session.decode_step(thinker_only_group(), inp)[0].float()
                logits[thinker_forbid_ids] -= 100.0
                t_next = int(logits.argmax().item())
                thinker_tokens.append(t_next)

            elif state == "thinker_and_writer":
                inp = torch.tensor(
                    [thinker_tokens[-1], writer_tokens[-1]], dtype=torch.int32
                )
                logits = session.decode_step(thinker_and_writer_group(), inp).float()
                logits[0, thinker_forbid_ids] -= 100.0
                logits[1, writer_forbid_ids] -= 100.0
                t_next = int(logits[0].argmax().item())
                w_next = int(logits[1].argmax().item())
                thinker_tokens.append(t_next)
                writer_tokens.append(w_next)

                # Writer hit \n\n -> end of a writer step; back to thinker_only.
                if writer_tokens[-1] == nn_id or _ends_with_double_newline(
                    writer_tokens, tokenizer
                ):
                    state = "thinker_only"
                    _print_state_change("writer end-of-step -> thinker_only")

            else:
                raise RuntimeError(f"unexpected state {state!r}")

            # --- stream newly-decided tokens to stdout ---
            while next_print_thinker < len(thinker_tokens):
                tok = thinker_tokens[next_print_thinker]
                _stream_token(tokenizer.decode([tok]), "thinker")
                next_print_thinker += 1
            while next_print_writer < len(writer_tokens):
                tok = writer_tokens[next_print_writer]
                _stream_token(tokenizer.decode([tok]), "writer")
                next_print_writer += 1

            # --- mode-switching probe ---
            if (
                (step + 1) % args.probe_period == 0
                or _ends_with_double_newline(thinker_tokens, tokenizer)
            ):
                should_write, yes_logit, no_logit = _check_continue_writing(
                    session, tokenizer, prompting,
                    thinker_tokens, writer_tokens,
                    yes_id=yes_id, no_id=no_id,
                )
                new_state = "thinker_and_writer" if should_write else "thinker_only"
                if new_state != state:
                    _print_state_change(
                        f"step {step + 1}: probe "
                        f"yes={yes_logit:.2f} no={no_logit:.2f} "
                        f"-> {new_state}"
                    )
                    state = new_state

            # --- termination ---
            if writer_tokens[-1] == eos_id:
                _print_state_change("writer hit EOS -- terminating")
                break

        # ------------------------------------------------------------------
        # Final output
        # ------------------------------------------------------------------
        _print_header("Final")
        print(_ansi("  Thinker:", "thinker"))
        print(_ansi(
            tokenizer.decode(thinker_tokens, skip_special_tokens=True),
            "thinker",
        ))
        print()
        print(_ansi("  Writer:", "writer"))
        print(_ansi(
            tokenizer.decode(writer_tokens, skip_special_tokens=True),
            "writer",
        ))
        print()

    finally:
        engine.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


def _ends_with_double_newline(token_ids: List[int], tokenizer: AutoTokenizer) -> bool:
    """Mirror AR's is_end_of_step: decode the last two tokens, check tail."""
    if len(token_ids) < 2:
        return False
    return tokenizer.decode(token_ids[-2:]).endswith("\n\n")


if __name__ == "__main__":
    main()
