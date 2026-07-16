"""Async vision demo: reason over an image while it is swapped in the background.

Uses the asyncio frontend (``AsyncLLM``) to show the browser-agent pattern — a
single agent reasons continuously, and a *separate* coroutine refreshes the image
block on its own schedule (``await llm.refresh_block``).  The engine tick
serializes the in-place refresh between decode steps, so the ongoing generation
simply starts attending to the new image on its next token; the surrounding
prompt and the reasoning so far are kept (fixed-trajectory AR reuse).

Layout (single worker):  [ R image | P prompt | G reasoning ]
R is re-prefilled in the background; P and G are never re-encoded.

Run (check nvidia-smi first)::

    HF_HOME=/mnt/LLM CUDA_VISIBLE_DEVICES=<free-gpu> .venv/bin/python scripts/async_vision_demo.py

Env: MINISGL_DEMO_MODEL / MINISGL_MEMORY_RATIO / MINISGL_DEMO_IMAGE /
MINISGL_DEMO_IMAGE2 (same as scripts/vision_ar_demo.py; real images need pillow).
"""

from __future__ import annotations

import asyncio
import glob
import os
import sys

import numpy as np
import torch
from minisgl.llm.async_llm import AsyncLLM
from minisgl.models.qwen2vl_image import preprocess_image
from minisgl.models.qwen3_5_mrope import get_rope_index
from minisgl.shared_cache import AsyncContext

MODEL = os.environ.get("MINISGL_DEMO_MODEL", "Qwen/Qwen3.5-0.8B")
MEMORY_RATIO = os.environ.get("MINISGL_MEMORY_RATIO")
IMAGE1 = os.environ.get("MINISGL_DEMO_IMAGE")
IMAGE2 = os.environ.get("MINISGL_DEMO_IMAGE2")

IMG_TOK, VSTART, VEND, MERGE = 248056, 248053, 248054, 2
PROMPT_TEXT = "Describe what you see.<|im_end|>\n<|im_start|>assistant\n"
TOTAL_STEPS = 160
SWAP_AFTER = 60  # tokens emitted before the background feeder swaps the image

_C = {"gen": "\033[1;32m", "sys": "\033[1;33m", "dim": "\033[2m", "z": "\033[0m"}


def _c(t: str, k: str) -> str:
    return _C[k] + t + _C["z"]


def _synthetic_rgb(h: int, w: int, seed: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([(yy + xx + seed * 37) % 256, (yy * 3 + seed) % 256, (xx * 5 + seed) % 256], -1).astype("uint8")


def _load_rgb(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _image_inputs(arr: np.ndarray, tokenizer):
    """(input_ids, pixel_values, grid_thw, mrope) for the image block from an RGB array."""
    pixel_values, grid = preprocess_image(arr, merge_size=MERGE)
    n_img = int(grid.prod().item()) // (MERGE**2)
    pre = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    ids = torch.tensor(pre + [VSTART] + [IMG_TOK] * n_img + [VEND], dtype=torch.int32)
    mrope = get_rope_index(ids.long(), IMG_TOK, MERGE, grid)
    return ids, pixel_values, grid, mrope


def _resolve_ckpt(model: str) -> str:
    if os.path.isdir(model):
        return model
    cache = "models--" + model.replace("/", "--")
    for root in (os.path.join(os.environ.get("HF_HOME", ""), "hub", cache), f"/mnt/LLM/hub/{cache}"):
        for snap in sorted(glob.glob(os.path.join(root, "snapshots", "*", ""))):
            if glob.glob(os.path.join(snap, "*.safetensors")):
                return snap
    print(f"no snapshot with weights for {model!r}", file=sys.stderr)
    sys.exit(2)


async def run() -> None:
    ckpt = _resolve_ckpt(MODEL)
    print(_c(f"\n  Async vision demo — background image swap while reasoning\n  model: {MODEL}\n", "sys"))
    extra = {"memory_ratio": float(MEMORY_RATIO)} if MEMORY_RATIO else {}
    llm = AsyncLLM(model_path=ckpt, num_page_override=8192, max_seq_len_override=8192, **extra)
    tok = llm.tokenizer

    arr1 = _load_rgb(IMAGE1) if IMAGE1 else _synthetic_rgb(64, 64, seed=1)
    arr2 = _load_rgb(IMAGE2) if IMAGE2 else _synthetic_rgb(96, 64, seed=2)
    ids1, pv1, grid1, mr1 = _image_inputs(arr1, tok)
    ids2, pv2, grid2, mr2 = _image_inputs(arr2, tok)
    prompt_ids = tok.encode(PROMPT_TEXT, add_special_tokens=False)

    # Prefill [R image] then [P prompt in context of R]; reasoning writes to G.
    R = (await llm.prefill_block(ids1, pixel_values=pv1, image_grid_thw=grid1, mrope_positions=mr1)).block
    P = await llm.prefill_block(prompt_ids, context=[R], return_logits=True)
    first = int(P.logits.argmax())
    G = await llm.create_block()
    ctx = AsyncContext(cache_view=[R, P.block, G], output_block=G)
    print(_c(f"  image #1 {arr1.shape} grid={grid1.tolist()[0]} span={R.mrope_span}; will swap to "
             f"#2 {arr2.shape} grid={grid2.tolist()[0]} after {SWAP_AFTER} tokens\n", "dim"))

    emitted = [0]

    async def image_feeder() -> None:
        # A separate coroutine refreshing the image on its own schedule, like a
        # screenshot feed arriving while the agent keeps thinking.
        while emitted[0] < SWAP_AFTER:
            await asyncio.sleep(0.002)
        await llm.refresh_block(R, ids2, pixel_values=pv2, image_grid_thw=grid2, mrope_positions=mr2)
        print(_c(f"\n  [background: image updated -> grid={grid2.tolist()[0]} span={R.mrope_span}]\n", "sys"), end="")
        sys.stdout.flush()

    feeder = asyncio.create_task(image_feeder())
    sys.stdout.write(_c(tok.decode([first]), "gen"))
    async for token in llm.async_generate(ctx, first_token_id=first, max_steps=TOTAL_STEPS):
        sys.stdout.write(_c(tok.decode([token]), "gen"))
        sys.stdout.flush()
        emitted[0] += 1
    await feeder
    print()
    await llm.close()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(2)
    asyncio.run(run())
