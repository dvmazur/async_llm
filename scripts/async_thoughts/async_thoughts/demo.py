"""Prompting, display scaffolding, and the main decode loop.

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
"""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass
from typing import List

import torch
from minisgl.shared_cache import SharedCacheSession, WorkerGroup
from transformers import AutoTokenizer

from .engine import (
    build_engine,
    check_continue_writing,
    encode,
    ends_with_double_newline,
    single_token_id,
    vocab_id_or_none,
)

# Default math problem from the original AsyncReasoning notebook.
DEFAULT_PROBLEM = "Calculate x - x^2 + x^3 for x = 5, 6, 7, 8. Return all 4 answers in \\boxed{ }."

# Qwen3-32B at bf16 fits on a single 80 GiB GPU and keeps the writer alive;
# smaller Qwen3 models starve the writer because the probe margin is too small.
#
# Qwen3.5 (hybrid Gated-DeltaNet) models are also supported and run through the
# same shared-cache session — their linear-attention layers compose via the GDN
# affine cache (see minisgl.shared_cache.gdn).  Hybrid models decode eagerly
# (CUDA graph is auto-disabled for them).  Even small Qwen3.5 variants
# (e.g. Qwen/Qwen3.5-0.8B) keep the writer active, so they make handy quick demos;
# for Qwen/Qwen3.5-27B lower --memory-ratio (~0.4) so the forward has headroom.
DEFAULT_MODEL = os.environ.get("MINISGL_DEMO_MODEL", "Qwen/Qwen3-32B")
DEFAULT_MAX_STEPS = 800
DEFAULT_PROBE_PERIOD = 30
DEFAULT_MEMORY_RATIO = 0.9
DEFAULT_PAGE_SIZE = 1


@dataclass
class DemoConfig:
    """Everything the demo needs to run, with sane defaults baked in."""

    model: str = DEFAULT_MODEL
    problem: str = DEFAULT_PROBLEM
    max_steps: int = DEFAULT_MAX_STEPS
    probe_period: int = DEFAULT_PROBE_PERIOD
    memory_ratio: float = DEFAULT_MEMORY_RATIO
    page_size: int = DEFAULT_PAGE_SIZE


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


# ANSI colours keyed by role.
_C = {
    "thinker": "\033[2;36m",  # dim cyan
    "writer": "\033[1;32m",  # bold green
    "state": "\033[1;33m",  # bold yellow
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def _ansi(text: str, *keys: str) -> str:
    return "".join(_C[k] for k in keys) + text + _C["reset"]


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


def _run_loop(
    config: DemoConfig,
    tokenizer: AutoTokenizer,
    session: SharedCacheSession,
    prompting: Prompting,
) -> None:
    # Forbidden token ids: never let the streams emit boundary markers.
    # Match AR's two forbidden sets, gracefully skipping any markers the
    # tokenizer doesn't have (e.g. for non-Qwen models without <think>).
    writer_forbid_names = ["</think>", "<|im_start|>", "<|endoftext|>"]
    thinker_forbid_names = ["</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    writer_forbid_ids = [
        i for i in (vocab_id_or_none(tokenizer, n) for n in writer_forbid_names) if i is not None
    ]
    thinker_forbid_ids = [
        i for i in (vocab_id_or_none(tokenizer, n) for n in thinker_forbid_names) if i is not None
    ]

    yes_id = single_token_id(prompting.yes_token, tokenizer)
    no_id = single_token_id(prompting.no_token, tokenizer)

    # Prefill the three persistent blocks.
    print("Prefilling blocks...")
    prompt_blk = session.create_block()
    thinker_blk = session.create_block()
    writer_blk = session.create_block()

    # input_prompt: prefilled standalone (matches AR -- input_prompt has no
    # context to attend to).
    prompt_ids = encode(prompting.input_prompt, tokenizer)
    session.prefill_block(prompt_blk, prompt_ids)

    # thinker_output_prefix: prefilled IN CONTEXT of [prompt].  AR does this via
    # SharedCacheManager(view=[input_prompt, thinker_output]); minisgl's
    # prefill_block(..., context=[...]) does the same in a single batched
    # prefill pass (the new tokens attend causally to themselves and fully to
    # the context), instead of one decode step per token.
    thinker_prefix_ids = encode(prompting.thinker_output_prefix, tokenizer)
    session.prefill_block(thinker_blk, thinker_prefix_ids, context=[prompt_blk])

    # writer_output_prefix: prefilled IN CONTEXT of [prompt, thinker].
    writer_prefix_ids = encode(prompting.writer_output_prefix, tokenizer)
    session.prefill_block(writer_blk, writer_prefix_ids, context=[prompt_blk, thinker_blk])

    # Token-sequence bookkeeping for display + the mode-switching probe.
    # Includes the "\n\n" separator that AR appends but does NOT prefill; it
    # gets sent as the first decode-step input below.
    nn_id = single_token_id("\n\n", tokenizer)
    thinker_tokens: List[int] = thinker_prefix_ids.tolist() + [nn_id]
    writer_tokens: List[int] = writer_prefix_ids.tolist() + [nn_id]

    # Index into *_tokens of the next token to stream-print.  We skip the prefix
    # when displaying since it's boilerplate, but the model internally sees it.
    next_print_thinker = len(thinker_prefix_ids)
    next_print_writer = len(writer_prefix_ids)

    thinker_only_group = WorkerGroup(
        cache_structure=[[prompt_blk, thinker_blk]],
        write_to=[thinker_blk],
    )

    thinker_and_writer_group = WorkerGroup(
        cache_structure=[
            [prompt_blk, thinker_blk],
            [prompt_blk, thinker_blk, writer_blk],
        ],
        write_to=[thinker_blk, writer_blk],
    )

    # Main decode loop.
    state = "thinker_only"
    _print_header("Generation")
    print(_ansi("  thinker (dim cyan) | writer (bold green)\n", "dim"))

    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1

    for step in range(config.max_steps):
        # decode one (or two) tokens
        if state == "thinker_only":
            inp = torch.tensor([thinker_tokens[-1]], dtype=torch.int32)
            logits = session.decode_step(thinker_only_group, inp)[0].float()
            logits[thinker_forbid_ids] -= 100.0
            t_next = int(logits.argmax().item())
            thinker_tokens.append(t_next)

        elif state == "thinker_and_writer":
            inp = torch.tensor([thinker_tokens[-1], writer_tokens[-1]], dtype=torch.int32)
            logits = session.decode_step(thinker_and_writer_group, inp).float()
            logits[0, thinker_forbid_ids] -= 100.0
            logits[1, writer_forbid_ids] -= 100.0
            t_next = int(logits[0].argmax().item())
            w_next = int(logits[1].argmax().item())
            thinker_tokens.append(t_next)
            writer_tokens.append(w_next)

            # Writer hit \n\n -> end of a writer step; back to thinker_only.
            if writer_tokens[-1] == nn_id or ends_with_double_newline(writer_tokens, tokenizer):
                state = "thinker_only"
                _print_state_change("writer end-of-step -> thinker_only")

        else:
            raise RuntimeError(f"unexpected state {state!r}")

        # stream newly-decided tokens to stdout
        while next_print_thinker < len(thinker_tokens):
            tok = thinker_tokens[next_print_thinker]
            _stream_token(tokenizer.decode([tok]), "thinker")
            next_print_thinker += 1
        while next_print_writer < len(writer_tokens):
            tok = writer_tokens[next_print_writer]
            _stream_token(tokenizer.decode([tok]), "writer")
            next_print_writer += 1

        # mode-switching probe
        if (step + 1) % config.probe_period == 0 or ends_with_double_newline(
            thinker_tokens, tokenizer
        ):
            should_write, yes_logit, no_logit = check_continue_writing(
                session,
                tokenizer,
                prompting.mode_switching_prompt,
                prompting.mode_switching_question,
                thinker_tokens,
                writer_tokens,
                yes_id=yes_id,
                no_id=no_id,
            )
            new_state = "thinker_and_writer" if should_write else "thinker_only"
            if new_state != state:
                _print_state_change(
                    f"step {step + 1}: probe yes={yes_logit:.2f} no={no_logit:.2f} -> {new_state}"
                )
                state = new_state

        # termination
        if writer_tokens[-1] == eos_id:
            _print_state_change("writer hit EOS -- terminating")
            break

    # Final output.
    _print_header("Final")
    print(_ansi("  Thinker:", "thinker"))
    print(_ansi(tokenizer.decode(thinker_tokens, skip_special_tokens=True), "thinker"))
    print()
    print(_ansi("  Writer:", "writer"))
    print(_ansi(tokenizer.decode(writer_tokens, skip_special_tokens=True), "writer"))
    print()


def run(config: DemoConfig) -> None:
    """Run the demo end-to-end against a freshly-built engine."""
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(2)

    _print_header("Async Thoughts Demo (minisgl)")
    print(f"  model    : {config.model}")
    print(f"  problem  : {config.problem}")
    print(f"  max steps: {config.max_steps}")

    print("\nLoading tokenizer & engine...")
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    engine = build_engine(
        config.model, memory_ratio=config.memory_ratio, page_size=config.page_size
    )
    session = SharedCacheSession(engine)
    prompting = Prompting(config.problem)

    try:
        _run_loop(config, tokenizer, session, prompting)
    finally:
        engine.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
