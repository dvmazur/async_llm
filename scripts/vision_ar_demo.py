"""Single-worker async-reasoning demo with an **updatable image in context**.

The point of this demo is the shared-cache block layout for a live-updating image
(think: a browser screenshot that refreshes while the agent keeps reasoning):

    R  image block   -- re-prefillable via ``session.refresh_block`` (swap the
                        image any time, any size -> the mRoPE span may change)
    P  prompt block  -- the fixed instruction text; prefilled ONCE in context of R
                        and never re-prefilled
    G  reasoning     -- grown one token per decode step (single worker)

Decoding reads the view ``[R, P, G]``.  When the image is swapped mid-generation
(``refresh_block(R, image2)``) only R's KV + GDN affine + mRoPE span change; P and G
keep their cached state (the fixed-trajectory AR reuse -- i.e. the reasoning so far
is the agent's memory of the *previous* frame), and every downstream rotation offset
adapts to R's new span automatically.  The block ORDER here is a choice: R is the
root so its image is encoded exactly; put a text block first instead and the image
becomes a standalone mid-sequence block (encoded without attending to that text).

Run (check nvidia-smi first; the checkpoint lives under HF_HOME)::

    HF_HOME=/mnt/LLM CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/vision_ar_demo.py

By default it uses two synthetic images; point MINISGL_DEMO_IMAGE / MINISGL_DEMO_IMAGE2
at real image files to run those instead (needs pillow: ``uv pip install pillow``)::

    MINISGL_DEMO_IMAGE=a.png MINISGL_DEMO_IMAGE2=b.png .venv/bin/python scripts/vision_ar_demo.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.models.qwen2vl_image import preprocess_image
from minisgl.models.qwen3_5_mrope import get_rope_index
from minisgl.shared_cache import SharedCacheSession, WorkerGroup
from transformers import AutoTokenizer

# Env-configurable so the same demo runs on 0.8B or 27B, with real or synthetic images:
#   MINISGL_DEMO_MODEL   HF name ("Qwen/Qwen3.5-27B") or a local snapshot dir
#   MINISGL_MEMORY_RATIO fraction of GPU memory the engine may use (lower for big
#                        models, e.g. 0.5 for 27B; unset -> engine default)
#   MINISGL_DEMO_IMAGE   path to a real image for frame 1 (unset -> synthetic)
#   MINISGL_DEMO_IMAGE2  path to a real image for frame 2 (unset -> synthetic)
MODEL = os.environ.get("MINISGL_DEMO_MODEL", "Qwen/Qwen3.5-0.8B")
MEMORY_RATIO = os.environ.get("MINISGL_MEMORY_RATIO")
IMAGE1 = os.environ.get("MINISGL_DEMO_IMAGE")
IMAGE2 = os.environ.get("MINISGL_DEMO_IMAGE2")

# Qwen3.5 special ids (see tmp/vis_ref.py / config).
IMG_TOK, VSTART, VEND = 248056, 248053, 248054
PROMPT_TEXT = "What is in this image? Answer in one short sentence.<|im_end|>\n<|im_start|>assistant\n"

_C = {"img": "\033[1;35m", "gen": "\033[1;32m", "sys": "\033[1;33m", "dim": "\033[2m", "z": "\033[0m"}


def _c(text: str, k: str) -> str:
    return _C[k] + text + _C["z"]


def _synthetic_rgb(h_px: int, w_px: int, seed: int) -> np.ndarray:
    """Deterministic H×W×3 uint8 image (a stand-in for a real screenshot)."""
    yy, xx = np.mgrid[0:h_px, 0:w_px]
    a = (yy + xx + seed * 37) % 256
    b = (yy * 3 + seed) % 256
    c = (xx * 5 + seed) % 256
    return np.stack([a, b, c], -1).astype("uint8")


def _load_rgb(path: str) -> np.ndarray:
    """Decode a real image file to an H×W×3 uint8 array (RGB)."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _image_block(arr: np.ndarray, merge: int, tokenizer):
    """Build a re-prefillable image block from an RGB uint8 array, run through the
    real Qwen2-VL preprocessor (smart-resize + normalize + patchify):
    '<|im_start|>user\\n' + <vision_start> + IMG*n + <vision_end>.

    Different image sizes yield different patch grids -> different token counts and
    mRoPE spans (the changing-mRoPE path).  Returns
    (input_ids[int32], pixel_values[f32], grid_thw[long,[1,3]], mrope[3,L]).
    """
    pixel_values, grid = preprocess_image(arr, merge_size=merge)
    n_img = int(grid.prod().item()) // (merge**2)
    pre = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    ids = torch.tensor(pre + [VSTART] + [IMG_TOK] * n_img + [VEND], dtype=torch.int32)
    mrope = get_rope_index(ids.long(), IMG_TOK, merge, grid)
    return ids, pixel_values, grid, mrope


def _decode_emit(session, group, feed_id: int, tokenizer) -> int:
    """One decode step: feed ``feed_id`` (written into the write block), stream the
    predicted next token, and return it (pending -- not yet written)."""
    nxt = int(session.decode_step(group, torch.tensor([feed_id], dtype=torch.int32)).argmax(-1)[0])
    sys.stdout.write(_c(tokenizer.decode([nxt]), "gen"))
    sys.stdout.flush()
    return nxt


def _stream(session, group, pending: int, n_steps: int, tokenizer) -> int:
    """Stream ``n_steps`` tokens, feeding ``pending`` first; return the new pending
    token (last generated, not yet written into the block)."""
    for _ in range(n_steps):
        pending = _decode_emit(session, group, pending, tokenizer)
    print()
    return pending


def _resolve_ckpt(model: str) -> str:
    """A local snapshot dir with weights, from an explicit path or an HF-cache name.

    Passing the local snapshot path (not the repo id) avoids a transformers offline
    ``model_info`` network call; we also skip snapshot dirs that hold no safetensors.
    """
    if os.path.isdir(model):
        return model
    cache = "models--" + model.replace("/", "--")
    roots = [os.path.join(os.environ.get("HF_HOME", ""), "hub", cache), f"/mnt/LLM/hub/{cache}"]
    for root in roots:
        for snap in sorted(glob.glob(os.path.join(root, "snapshots", "*", ""))):
            if glob.glob(os.path.join(snap, "*.safetensors")):
                return snap
    print(f"no snapshot with weights for {model!r} (set MINISGL_DEMO_MODEL)", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(2)
    ckpt = _resolve_ckpt(MODEL)

    print(_c(f"\n  Vision AR demo — updatable image in context (single worker)\n  model: {MODEL}\n", "sys"))
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    extra = {"memory_ratio": float(MEMORY_RATIO)} if MEMORY_RATIO else {}
    cfg = EngineConfig(
        model_path=ckpt, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=4, num_page_override=8192, max_seq_len_override=8192, **extra,
    )
    engine = Engine(cfg)
    session = SharedCacheSession(engine)

    merge = cfg.model_config.vision_config.spatial_merge_size

    # Real image files if provided, else two different-sized synthetic images
    # (different patch grids -> different mRoPE spans -> the changing-mRoPE path).
    arr1 = _load_rgb(IMAGE1) if IMAGE1 else _synthetic_rgb(64, 64, seed=1)
    arr2 = _load_rgb(IMAGE2) if IMAGE2 else _synthetic_rgb(96, 64, seed=2)
    print(_c(f"  image #1: {IMAGE1 or 'synthetic'} {arr1.shape}   image #2: {IMAGE2 or 'synthetic'} {arr2.shape}", "dim"))

    prompt_ids = torch.tensor(tokenizer.encode(PROMPT_TEXT, add_special_tokens=False), dtype=torch.int32)

    try:
        # --- Block layout (programmer's choice): [R image] [P prompt] [G reasoning] ---
        R = session.create_block()  # re-prefillable image block (root)
        P = session.create_block()  # fixed prompt text (non-re-prefillable)
        G = session.create_block()  # reasoning (single worker)

        ids1, pv1, grid1, mrope1 = _image_block(arr1, merge, tokenizer)
        session.prefill_block(R, ids1, pixel_values=pv1, image_grid_thw=grid1, mrope_positions=mrope1)
        first = int(session.prefill_block(P, prompt_ids, context=[R])[0].argmax().item())
        print(_c(f"  image #1 grid={grid1.tolist()[0]}  R.tokens={R.num_tokens}  R.mrope_span={R.mrope_span}", "dim"))

        group = WorkerGroup(cache_structure=[[R, P, G]], write_to=[G])
        print(_c("  reasoning (frame 1): ", "img"), end="")
        sys.stdout.write(_c(tokenizer.decode([first]), "gen"))
        pending = _stream(session, group, first, 24, tokenizer)

        # --- Swap the image IN PLACE (different size => changing mRoPE) ---
        ids2, pv2, grid2, mrope2 = _image_block(arr2, merge, tokenizer)
        session.refresh_block(R, ids2, pixel_values=pv2, image_grid_thw=grid2, mrope_positions=mrope2)
        print(_c(f"  [refresh_block] image #2 grid={grid2.tolist()[0]}  R.tokens={R.num_tokens}  R.mrope_span={R.mrope_span}", "sys"))
        print(_c(f"  P and G kept ({P.num_tokens} + {G.num_tokens} tokens); only R re-encoded", "dim"))

        # Continue the SAME reasoning block, now attending to image #2.
        print(_c("  reasoning (frame 2): ", "img"), end="")
        _stream(session, group, pending, 24, tokenizer)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
