"""Async vision demo: reason over an image that is swapped *in context* mid-stream.

Same principles as ``demo.py`` (async coroutines over ``AsyncLLM``, one forward
per engine tick, coordinated by ``asyncio.Event``s) but for the browser-agent
pattern: a single agent keeps reasoning while its image is updated in place.

Cache layout (single worker):

  image block     : "<|im_start|>user\\n" + <vision_start> IMG* <vision_end>.
                    RE-PREFILLABLE via ``AsyncLLM.refresh_block`` — swapped to a
                    new (possibly different-sized) image without touching the rest.
  prompt block    : the question + "<|im_end|>\\n<|im_start|>assistant\\n",
                    prefilled ONCE in context of the image and never re-encoded.
  reasoning block : grows as the agent decodes.

Concurrency: a ``reason`` coroutine streams tokens via ``async_generate``; an
``image_feed`` coroutine refreshes the image block via ``await
llm.refresh_block``.  The engine tick serializes the in-place refresh between
decode steps, so the stream attends to the new image on its next token; the
prompt and the reasoning so far are kept (the fixed-trajectory AR reuse — the
model's memory of the previous frame).  A short note is spliced into the
thoughts on each swap so the model re-inspects instead of trusting stale reasoning.

Run (from scripts/async_thoughts, after ``pip install -e .``)::

    HF_HOME=/mnt/LLM CUDA_VISIBLE_DEVICES=<free-gpu> python -m async_thoughts.vision_demo
    ... --image a.png --image b.png     # real images (needs pillow); default: 2 synthetic
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from minisgl.llm import AsyncLLM
from minisgl.models.qwen2vl_image import preprocess_image
from minisgl.models.qwen3_5_mrope import get_rope_index
from minisgl.shared_cache import AsyncContext
from transformers import AutoTokenizer

from .engine import build_async_llm, encode

DEFAULT_MODEL = os.environ.get("MINISGL_DEMO_MODEL", "Qwen/Qwen3.5-0.8B")
DEFAULT_QUESTION = "Describe what you see in this image."
DEFAULT_MAX_STEPS = 200
DEFAULT_SWAP_EVERY = 60  # tokens between background image swaps
DEFAULT_HINT = "\n\n[The image has been updated.]\n\n"
DEFAULT_MEMORY_RATIO = 0.9
DEFAULT_PAGE_SIZE = 1

# Qwen3.5 vision special ids + spatial-merge (see minisgl.models config).
IMG_TOK, VSTART, VEND, MERGE = 248056, 248053, 248054, 2


@dataclass
class VisionConfig:
    """Everything the vision demo needs, with sane defaults baked in."""

    model: str = DEFAULT_MODEL
    # Image paths, one per frame; ``None`` entries become synthetic images.
    images: List[Optional[str]] = field(default_factory=lambda: [None, None])
    question: str = DEFAULT_QUESTION
    max_steps: int = DEFAULT_MAX_STEPS
    swap_every: int = DEFAULT_SWAP_EVERY
    hint: str = DEFAULT_HINT  # empty -> no note injected on swap
    memory_ratio: float = DEFAULT_MEMORY_RATIO
    page_size: int = DEFAULT_PAGE_SIZE


# ANSI colours keyed by role.
_C = {
    "gen": "\033[1;32m",  # bold green (generated tokens)
    "hint": "\033[1;35m",  # bold magenta (injected note)
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


def _stream_token(text: str, role: str = "gen") -> None:
    sys.stdout.write(_ansi(text, role))
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def _synthetic_rgb(h: int, w: int, seed: int) -> np.ndarray:
    """Deterministic H×W×3 uint8 image (a stand-in for a real screenshot)."""
    yy, xx = np.mgrid[0:h, 0:w]
    a = (yy + xx + seed * 37) % 256
    b = (yy * 3 + seed) % 256
    c = (xx * 5 + seed) % 256
    return np.stack([a, b, c], -1).astype("uint8")


def _load_rgb(path: str) -> np.ndarray:
    """Decode a real image file to an H×W×3 uint8 array (RGB); needs pillow."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


ImageInputs = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _image_inputs(arr: np.ndarray, tokenizer: AutoTokenizer) -> ImageInputs:
    """Build the image block from an RGB array via the real Qwen2-VL preprocessor.

    Returns (input_ids, pixel_values, image_grid_thw[1,3], mrope[3,L]).
    """
    pixel_values, grid = preprocess_image(arr, merge_size=MERGE)
    n_img = int(grid.prod().item()) // (MERGE**2)
    pre = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    ids = torch.tensor(pre + [VSTART] + [IMG_TOK] * n_img + [VEND], dtype=torch.int32)
    mrope = get_rope_index(ids.long(), IMG_TOK, MERGE, grid)
    return ids, pixel_values, grid, mrope


def _frame(config: VisionConfig, i: int, tokenizer: AutoTokenizer) -> Tuple[np.ndarray, ImageInputs]:
    """The i-th frame's RGB array + model inputs (synthetic when no path given).

    Synthetic frames use different sizes so each swap changes the patch grid and
    the mRoPE span (exercising the changing-mRoPE path)."""
    path = config.images[i] if i < len(config.images) else None
    arr = _load_rgb(path) if path else _synthetic_rgb(64 + 32 * i, 64, seed=i + 1)
    return arr, _image_inputs(arr, tokenizer)


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


async def _inject_hint(llm: AsyncLLM, ctx: AsyncContext, text: str, tokenizer: AutoTokenizer) -> None:
    """Splice *text* into the reasoning: force-feed its tokens into the output
    block (ignoring the model's own predictions), echoing them in the hint colour.
    Leaves ``ctx.next_input_id`` at the model's prediction after the note, so the
    stream resumes cleanly."""
    for tid in tokenizer.encode(text, add_special_tokens=False):
        ctx.next_input_id = tid
        async for _ in llm.async_generate(ctx, max_steps=1):
            pass
        _stream_token(tokenizer.decode([tid]), "hint")


async def _run_loop(config: VisionConfig, llm: AsyncLLM, tokenizer: AutoTokenizer) -> List[int]:
    forbid_ids = [
        i
        for i in (tokenizer.vocab.get(n) for n in ("<|im_start|>", "<|endoftext|>"))
        if i is not None
    ]
    frames = [_frame(config, i, tokenizer) for i in range(len(config.images))]

    # Prefill [image] then [prompt in context of image]; reasoning writes to G.
    print("Prefilling blocks...")
    arr0, (ids0, pv0, grid0, mr0) = frames[0]
    R = (
        await llm.prefill_block(ids0, pixel_values=pv0, image_grid_thw=grid0, mrope_positions=mr0)
    ).block
    prompt_ids = encode(f"{config.question}<|im_end|>\n<|im_start|>assistant\n", tokenizer)
    P = await llm.prefill_block(prompt_ids, context=[R], return_logits=True)
    first = int(P.logits.argmax())
    G = await llm.create_block()
    ctx = AsyncContext(cache_view=[R, P.block, G], output_block=G, next_input_id=first)

    n_swaps = len(frames) - 1
    _print_header("Generation")
    print(_ansi(f"  image #1 {arr0.shape} grid={grid0.tolist()[0]} span={R.mrope_span}; "
                f"{n_swaps} background swap(s) every {config.swap_every} tokens\n", "dim"))

    done = asyncio.Event()
    swap_now = asyncio.Event()  # reason -> feeder: a swap boundary was reached
    swapped = asyncio.Event()  # feeder -> reason: the image is refreshed

    async def reason() -> None:
        _stream_token(tokenizer.decode([first]))
        emitted = 1
        swaps_left = n_swaps
        try:
            while not done.is_set() and emitted < config.max_steps:
                broke = False
                async for token in llm.async_generate(ctx, forbid_ids=forbid_ids):
                    _stream_token(tokenizer.decode([token]))
                    emitted += 1
                    if emitted >= config.max_steps:
                        break
                    if swaps_left > 0 and emitted % config.swap_every == 0:
                        broke = True
                        break
                if not broke or done.is_set():
                    break
                swaps_left -= 1
                swap_now.set()
                await swapped.wait()
                swapped.clear()
                if config.hint:
                    await _inject_hint(llm, ctx, config.hint, tokenizer)
        finally:
            done.set()
            swap_now.set()

    async def image_feed() -> None:
        for idx in range(1, len(frames)):
            await swap_now.wait()
            swap_now.clear()
            if done.is_set():
                return
            _, (ids, pv, grid, mr) = frames[idx]
            await llm.refresh_block(R, ids, pixel_values=pv, image_grid_thw=grid, mrope_positions=mr)
            _print_state_change(
                f"background: image #{idx + 1} loaded -> grid={grid.tolist()[0]} span={R.mrope_span}"
            )
            swapped.set()

    await asyncio.gather(reason(), image_feed())

    _print_header("Final")
    tokens = list(G.token_ids) + ([ctx.next_input_id] if ctx.next_input_id is not None else [])
    print(_ansi(tokenizer.decode(tokens, skip_special_tokens=True), "gen"))
    print()
    for blk in (R, P.block, G):
        await llm.free_block(blk)
    return tokens


async def _run_demo(config: VisionConfig, llm: AsyncLLM) -> List[int]:
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    try:
        return await _run_loop(config, llm, tokenizer)
    finally:
        await llm.close()


def run(config: VisionConfig) -> None:
    """Run the vision demo end-to-end against a freshly-built AsyncLLM."""
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(2)

    _print_header("Async Vision Demo (minisgl)")
    print(f"  model     : {config.model}")
    print(f"  images    : {[p or 'synthetic' for p in config.images]}")
    print(f"  question  : {config.question}")
    print(f"  max steps : {config.max_steps}   swap every: {config.swap_every}")

    print("\nLoading tokenizer & engine...")
    llm = build_async_llm(config.model, memory_ratio=config.memory_ratio, page_size=config.page_size)
    try:
        asyncio.run(_run_demo(config, llm))
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> VisionConfig:
    p = argparse.ArgumentParser(
        prog="vision-demo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model path (default: {DEFAULT_MODEL}; MINISGL_DEMO_MODEL).")
    p.add_argument("--image", action="append", dest="images", default=None,
                   help="Image path; repeat for multiple frames (default: 2 synthetic).")
    p.add_argument("--question", default=DEFAULT_QUESTION, help="What to ask about the image.")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--swap-every", type=int, default=DEFAULT_SWAP_EVERY,
                   help="Tokens between background image swaps.")
    p.add_argument("--hint", default=DEFAULT_HINT, help="Note spliced into the thoughts on each swap.")
    p.add_argument("--no-hint", action="store_true", help="Do not inject a note on swap.")
    p.add_argument("--memory-ratio", type=float, default=DEFAULT_MEMORY_RATIO)
    p.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    a = p.parse_args(argv)
    return VisionConfig(
        model=a.model,
        images=a.images if a.images else [None, None],
        question=a.question,
        max_steps=a.max_steps,
        swap_every=a.swap_every,
        hint="" if a.no_hint else a.hint,
        memory_ratio=a.memory_ratio,
        page_size=a.page_size,
    )


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
