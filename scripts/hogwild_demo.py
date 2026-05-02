#!/usr/bin/env python3
"""
Hogwild! collaborative reasoning demo using the shared-cache API.

Two workers (Alice and Bob) reason in parallel about the same problem.
The cache structure mirrors the Hogwild! paper:

  [common_history | "### Work in progress (others)" | other_current |
   "<...>\\n\\n### Work in progress (own)" | self_current]

Each worker sees the other's current in-progress step, enabling real-time
collaboration via shared KV pages.

Server must be running with --shared-cache-page-budget > 0:
    MODEL=Qwen/Qwen3-9B ./scripts/launch_demo.sh

Usage:
    uv run python scripts/hogwild_demo.py --model Qwen/Qwen3-9B
    uv run python scripts/hogwild_demo.py --model Qwen/Qwen3-9B --rounds 4 --tokens-per-round 128
"""

from __future__ import annotations

import argparse
import json
import textwrap
from typing import List, Tuple

import requests
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Prompt constants — mirroring the Hogwild! colab notebook
# ---------------------------------------------------------------------------

_WORKERS = ("Alice", "Bob")

# Section headers (each becomes a separate shared prefill block)
HISTORY_HEADER = "\n\n### Past steps\n\n"
STEP_HEADER    = "### Work in progress (others)\n\n"
SEPARATOR      = "<...>\n\n### Work in progress (own)\n\n"

PROBLEM = "Calculate x - x^2 + x * (1 - x) for x = 4, 5, 6, 7."

# System prompt adapted from hogwild_llm/formatting.py
SYSTEM_PROMPT = textwrap.dedent(f"""\
    # Collaborative Reasoning

    You will collaborate on this problem with another assistant. You will write your thoughts \
simultaneously with them and collaborate without redundant work. You can collaborate by doing \
different parts of the problem, double-checking each other's results, trying different approaches, \
or any other means.

    There are 2 assistants, including yourself: Alice and Bob.

    You will solve the problem together, writing your thoughts in parallel. You will be able to \
see each other's past and current thoughts as you write them. You will see each other's previous \
steps as **AssistantName [step]:** <...> .

    In the '{STEP_HEADER.strip()}' section, you will see the other assistant's unfinished steps. \
They write those steps concurrently with you. Take into account what they are doing. If another \
assistant gives you suggestions, address them.

    You will always see *other* assistants' incomplete thoughts first, and then, after \
'### Work in progress (own)', your own current step.

    If what you are currently doing is the same thing another assistant has already done or is \
doing, stop and change to a different task right away to avoid redundant work.

    # How to collaborate

    - **Strategizing:** divide work between yourselves (if Alice says "Bob, please do X", Bob should)
    - **Splitting:** split the problem into subtasks and assign them
    - **Communicating:** ask questions, give corrections (e.g. "Hey Bob! You have a mistake in step 2")
    - **Announcing:** say what you'll do next; if another assistant says this, factor it in
    - **Reacting:** if you see the other assistant doing the same thing as you, stop and switch

    # Solve the following problem

    Alice and Bob, solve the next problem together. Keep track of who does what and avoid doing \
the same work twice.\
""")

# Shared prefix: system prompt + problem + history section header
COMMON_TEXT = SYSTEM_PROMPT + "\n\n" + PROBLEM + HISTORY_HEADER

# Seed texts for each worker's first step (matches the Hogwild notebook)
def _step_seed(worker: str, step: int) -> str:
    if step == 1:
        if worker == "Alice":
            return f"**Alice [1]:** Hi, I'm Alice. Here's how we can"
        else:
            return f"**Bob [1]:** Hi, I'm Bob."
    return f"**{worker} [{step}]:** "


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def create_block(base_url: str, text: str | None = None) -> str:
    payload = {"text": text} if text is not None else {}
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
) -> List[List[int]]:
    """Stream one generate call; return accumulated token IDs per worker."""
    payload = {
        "cache_structure": cache_structure,
        "write_to": write_to,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
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
            for i, tok in enumerate(data["worker_tokens"]):
                tokens[i].append(tok)
            if data.get("finished"):
                break
    return tokens


# ---------------------------------------------------------------------------
# Cache-structure builders — Hogwild layout
#
# Alice sees: [common, history..., step_header, bob_current, separator, alice_current]
# Bob  sees: [common, history..., step_header, alice_current, separator, bob_current]
# ---------------------------------------------------------------------------

def alice_cache(
    common: str,
    history: List[Tuple[str, str]],   # [(alice_id, bob_id), ...] completed rounds
    step_header: str,
    separator: str,
    bob_current: str,                  # Bob's in-progress block this round
    alice_current: str,                # Alice's in-progress block this round
) -> List[str]:
    seq = [common]
    for alice_b, bob_b in history:
        seq.append(alice_b)
        seq.append(bob_b)
    seq.extend([step_header, bob_current, separator, alice_current])
    return seq


def bob_cache(
    common: str,
    history: List[Tuple[str, str]],
    step_header: str,
    separator: str,
    alice_current: str,
    bob_current: str,
) -> List[str]:
    seq = [common]
    for alice_b, bob_b in history:
        seq.append(alice_b)
        seq.append(bob_b)
    seq.extend([step_header, alice_current, separator, bob_current])
    return seq


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_COLORS = {"alice": "\033[94m", "bob": "\033[92m", "bold": "\033[1m",
           "dim": "\033[2m", "reset": "\033[0m"}


def _c(text: str, *keys: str) -> str:
    return "".join(_COLORS[k] for k in keys) + text + _COLORS["reset"]


def print_round_header(round_idx: int, total: int) -> None:
    bar = "─" * 64
    print(f"\n{bar}")
    print(_c(f"  Round {round_idx + 1} / {total}", "bold"))
    print(f"{bar}")


def print_worker_output(name: str, step: int, seed: str, body: str) -> None:
    color = "alice" if name == "Alice" else "bob"
    print(f"\n{_c(f'[{name} — step {step}]', color, 'bold')}")
    full = seed + body
    for line in full.split("\n"):
        print(textwrap.fill(line, width=72, subsequent_indent="  ") if line.strip() else "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-9B",
                        help="HF model path (tokenizer only; weights on server)")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of collaborative rounds")
    parser.add_argument("--tokens-per-round", type=int, default=96,
                        help="Max tokens each worker generates per round")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    print(_c("Hogwild! Collaborative Reasoning Demo", "bold"))
    print(f"  model    : {args.model}")
    print(f"  server   : {args.base_url}")
    print(f"  rounds   : {args.rounds}  ×  {args.tokens_per_round} tokens/worker\n")

    print("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ------------------------------------------------------------------
    # Prefill shared structural blocks (created once, never mutated)
    # ------------------------------------------------------------------
    print("Prefilling shared blocks…")
    common_block      = create_block(args.base_url, COMMON_TEXT)
    step_header_block = create_block(args.base_url, STEP_HEADER)
    separator_block   = create_block(args.base_url, SEPARATOR)
    print(f"  common      = {common_block}")
    print(f"  step_header = {step_header_block}")
    print(f"  separator   = {separator_block}")

    print(_c("\nProblem: ", "bold") + PROBLEM)

    # accumulated history: list of (alice_block_id, bob_block_id) per completed round
    history: List[Tuple[str, str]] = []
    # seed text used per round (for display)
    alice_seeds: List[str] = []
    bob_seeds:   List[str] = []
    alice_bodies: List[str] = []
    bob_bodies:   List[str] = []

    all_blocks: List[str] = [common_block, step_header_block, separator_block]

    try:
        for round_idx in range(args.rounds):
            print_round_header(round_idx, args.rounds)
            step = round_idx + 1

            # Seed each worker's block with their step label (prefill → provides first-token logits)
            a_seed = _step_seed("Alice", step)
            b_seed = _step_seed("Bob",   step)
            alice_block = create_block(args.base_url, a_seed)
            bob_block   = create_block(args.base_url, b_seed)
            all_blocks.extend([alice_block, bob_block])

            cache_str = [
                alice_cache(common_block, history, step_header_block,
                            separator_block, bob_block, alice_block),
                bob_cache(common_block, history, step_header_block,
                          separator_block, alice_block, bob_block),
            ]

            print(_c("  Generating…", "dim"), flush=True)
            token_lists = run_generate(
                args.base_url,
                cache_structure=cache_str,
                write_to=[alice_block, bob_block],
                max_tokens=args.tokens_per_round,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            a_body = tok.decode(token_lists[0], skip_special_tokens=True)
            b_body = tok.decode(token_lists[1], skip_special_tokens=True)

            alice_seeds.append(a_seed)
            bob_seeds.append(b_seed)
            alice_bodies.append(a_body)
            bob_bodies.append(b_body)
            history.append((alice_block, bob_block))

            print_worker_output("Alice", step, a_seed, a_body)
            print_worker_output("Bob",   step, b_seed, b_body)

    finally:
        # Free all server-side blocks regardless of errors
        print("\n" + "─" * 64)
        print(_c("Cleaning up server blocks…", "dim"))
        for bid in all_blocks:
            delete_block(args.base_url, bid)

    # ------------------------------------------------------------------
    # Full conversation summary
    # ------------------------------------------------------------------
    print(_c("\n=== Full conversation ===", "bold"))
    print(_c("Problem: ", "bold") + PROBLEM)
    for i, (a_seed, a_body, b_seed, b_body) in enumerate(
            zip(alice_seeds, alice_bodies, bob_seeds, bob_bodies)):
        print_worker_output("Alice", i + 1, a_seed, a_body)
        print_worker_output("Bob",   i + 1, b_seed, b_body)
    print()


if __name__ == "__main__":
    main()
