#!/usr/bin/env python3
"""
Async Reasoning demo using the shared-cache API.

A 'thinker' worker generates chain-of-thought reasoning tokens inside
<think> tags while a 'writer' worker simultaneously generates the final
answer.  The writer sees the thinker's growing partial reasoning via
shared KV pages — it does not wait for the thinker to finish.

This demo implements the core idea from:
  https://github.com/yandex-research/AsyncReasoning/blob/main/demo_simple.ipynb

Cache structure each round:

  Thinker : [prompt_block | thinker_block]
  Writer  : [prompt_block | thinker_block | close_block | writer_block]

prompt_block  — chat-formatted prompt ending just after <think>\\n
thinker_block — grows in place as the thinker writes reasoning tokens
close_block   — prefilled with \\n</think>\\n
writer_block  — grows in place as the writer generates its answer

Server must be running with --shared-cache-page-budget > 0:
    MODEL=Qwen/Qwen3-9B ./scripts/launch_demo.sh

Usage:
    uv run python scripts/async_reasoning_demo.py --model Qwen/Qwen3-9B
    uv run python scripts/async_reasoning_demo.py --model Qwen/Qwen3-9B \\
        --think-budget 256 --write-budget 128 --tokens-per-round 32
"""
from __future__ import annotations

import argparse
import json
import textwrap
from typing import List, Tuple

import requests
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROBLEM = "Calculate x - x^2 + x^3 for x = 5, 6, 7, 8.  Return all 4 answers in \\boxed{}."
EXPECTED = "105, 186, 301, 456"   # x=5→105, x=6→186, x=7→301, x=8→456

SYSTEM_PROMPT = "You are a helpful math assistant."

# Injected between thinker block and writer block to close the <think> section
THINK_CLOSE = "\n</think>\n"

_COLORS = {
    "thinker": "\033[2;36m",   # dim cyan   — internal reasoning
    "writer":  "\033[1;32m",   # bold green — polished answer
    "header":  "\033[1;33m",   # bold yellow
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "reset":   "\033[0m",
}


def _c(text: str, *keys: str) -> str:
    return "".join(_COLORS[k] for k in keys) + text + _COLORS["reset"]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(tok: AutoTokenizer, problem: str) -> str:
    """Return chat-formatted text ending just after the opening <think>\\n tag."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Qwen3: <|im_start|>assistant\n is the generation prompt; we open the thinking block.
    text += "<think>\n"
    return text


# ---------------------------------------------------------------------------
# API helpers  (identical signature to hogwild_demo.py)
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def create_block(
    base_url: str, text: str | None = None, context: List[str] | None = None
) -> str:
    payload: dict = {"text": text} if text is not None else {}
    if context is not None:
        payload["context"] = context
    return _post(f"{base_url}/v1/shared-cache/blocks", payload)["block_id"]


def delete_block(base_url: str, block_id: str) -> None:
    requests.delete(f"{base_url}/v1/shared-cache/blocks/{block_id}", timeout=10)


def run_generate(
    base_url: str,
    cache_structure: List[List[str]],
    write_to: List[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    first_tokens: List[int] | None = None,
) -> List[List[int]]:
    """Stream one generate call; return accumulated token IDs per worker.

    *first_tokens* (one per worker) continues generation from a previous
    call's last reported token; without it the server seeds from prefill
    logits and reports the seed as the first chunk.
    """
    payload = {
        "cache_structure": cache_structure,
        "write_to": write_to,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if first_tokens is not None:
        payload["first_tokens"] = first_tokens
    tokens: List[List[int]] = [[] for _ in range(len(write_to))]
    with requests.post(
        f"{base_url}/v1/shared-cache/generate",
        json=payload, stream=True, timeout=600,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            data = json.loads(raw)
            for i, tok_id in enumerate(data["worker_tokens"]):
                tokens[i].append(tok_id)
            if data.get("finished"):
                break
    return tokens


# ---------------------------------------------------------------------------
# Cache-structure helpers
# ---------------------------------------------------------------------------

def thinker_cache(prompt: str, thinker: str) -> List[str]:
    return [prompt, thinker]


def writer_cache(prompt: str, thinker: str, close: str, writer: str) -> List[str]:
    return [prompt, thinker, close, writer]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_banner(text: str) -> None:
    bar = "─" * 64
    print(f"\n{bar}")
    print(_c(f"  {text}", "header"))
    print(f"{bar}")


def print_worker_output(role: str, color: str, text: str) -> None:
    if not text:
        return
    print(f"\n  {_c(f'[{role}]', color, 'bold')}")
    for line in text.split("\n"):
        print(_c(textwrap.fill(line, width=72, subsequent_indent="    ") if line.strip() else "", color))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-9B",
                        help="HF model path (tokenizer only; weights run on server)")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--problem", default=PROBLEM)
    parser.add_argument("--think-budget", type=int, default=512,
                        help="Max tokens the thinker may generate in total")
    parser.add_argument("--write-budget", type=int, default=256,
                        help="Max tokens the writer may generate in total")
    parser.add_argument("--tokens-per-round", type=int, default=64,
                        help="Max tokens each worker generates per round")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    print_banner("Async Reasoning Demo")
    print(f"  model         : {args.model}")
    print(f"  server        : {args.base_url}")
    print(f"  think budget  : {args.think_budget} tokens")
    print(f"  write budget  : {args.write_budget} tokens")
    print(f"  tokens/round  : {args.tokens_per_round}")

    print("\nLoading tokenizer…")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompt_text = build_prompt(tok, args.problem)

    # ------------------------------------------------------------------
    # Create shared blocks
    # ------------------------------------------------------------------
    print("Prefilling shared blocks…")
    prompt_blk  = create_block(args.base_url, prompt_text)
    thinker_blk = create_block(args.base_url)   # empty; grows as thinker writes
    writer_blk  = create_block(args.base_url)   # empty; grows as writer writes
    # Prefill the </think> close tag IN CONTEXT so its KV (and the writer's
    # round-1 seed logits) are conditioned on the prompt rather than computed
    # from the bare tag.  thinker_blk is still empty here — same as the
    # reference, which prefills the writer prefix once at start.
    close_blk   = create_block(args.base_url, THINK_CLOSE,
                               context=[prompt_blk, thinker_blk])

    print(f"  prompt  = {prompt_blk}")
    print(f"  close   = {close_blk}")
    print(f"  thinker = {thinker_blk}")
    print(f"  writer  = {writer_blk}")

    all_blocks = [prompt_blk, close_blk, thinker_blk, writer_blk]

    print()
    print(_c("Problem : ", "bold") + args.problem)
    print(_c("Expected: ", "dim") + EXPECTED)

    thinker_all: List[int] = []
    writer_all:  List[int] = []
    think_remaining = args.think_budget
    write_remaining = args.write_budget
    thinker_done = False
    writer_done  = False

    print_banner("Generation")

    try:
        round_idx = 0
        while not writer_done:
            round_idx += 1

            if thinker_done:
                cs  = [writer_cache(prompt_blk, thinker_blk, close_blk, writer_blk)]
                wto = [writer_blk]
                first = [writer_all[-1]] if writer_all else None
            else:
                cs  = [
                    thinker_cache(prompt_blk, thinker_blk),
                    writer_cache(prompt_blk, thinker_blk, close_blk, writer_blk),
                ]
                wto = [thinker_blk, writer_blk]
                # Round 1: no history yet -> let the server seed (and report it).
                first = [thinker_all[-1], writer_all[-1]] if thinker_all else None

            new_tokens = run_generate(
                args.base_url,
                cache_structure=cs,
                write_to=wto,
                max_tokens=args.tokens_per_round,
                temperature=args.temperature,
                top_p=args.top_p,
                first_tokens=first,
            )

            if thinker_done:
                new_think: List[int] = []
                new_write: List[int] = new_tokens[0]
            else:
                new_think = new_tokens[0]
                new_write = new_tokens[1]

            thinker_all.extend(new_think)
            writer_all.extend(new_write)
            think_remaining -= len(new_think)
            write_remaining -= len(new_write)

            # A worker is done when its budget runs out or it generates fewer
            # tokens than requested (natural EOS before hitting max_tokens).
            if think_remaining <= 0 or len(new_think) < args.tokens_per_round:
                thinker_done = True
            if write_remaining <= 0 or len(new_write) < args.tokens_per_round:
                writer_done = True

            t_str = tok.decode(new_think, skip_special_tokens=True) if new_think else ""
            w_str = tok.decode(new_write, skip_special_tokens=True) if new_write else ""

            print(f"\n  {_c(f'[ Round {round_idx}  ·  thinker: {len(thinker_all)} tok  ·  writer: {len(writer_all)} tok ]', 'dim')}")
            print_worker_output("Thinker", "thinker", t_str)
            print_worker_output("Writer ", "writer",  w_str)

    finally:
        print("\n" + "─" * 64)
        print(_c("Cleaning up server blocks…", "dim"))
        for bid in all_blocks:
            delete_block(args.base_url, bid)

    # ------------------------------------------------------------------
    # Full outputs
    # ------------------------------------------------------------------
    print_banner("Final Output")
    print(_c("  Problem  : ", "bold") + args.problem)
    print(_c("  Expected : ", "dim") + EXPECTED)

    print()
    print(_c("  Thinker's reasoning:", "dim"))
    full_think = tok.decode(thinker_all, skip_special_tokens=True)
    for line in full_think.splitlines():
        print(_c(f"    {line}", "thinker"))

    print()
    print(_c("  Writer's answer:", "bold"))
    full_write = tok.decode(writer_all, skip_special_tokens=True)
    for line in full_write.splitlines():
        print(_c(f"    {line}", "writer"))
    print()


if __name__ == "__main__":
    main()
