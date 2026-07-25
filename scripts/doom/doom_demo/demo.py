"""Qwen3.5 vision plays ViZDoom through the AsyncLLM frontend.

This module holds the episode loop; the command line lives in ``__main__``.

The doer, each tick, re-encodes the current frame into an updatable image block
(session.refresh_block) and, crucially, *reasons* about it before acting: it
generates a short frame-grounded analysis, then presses a key-name action
(LEFT / RIGHT / SPACE) constrained + arg-maxed from that reasoning.  This is the
configuration that actually clears VizdoomBasic on Qwen3.5-27B (~+69 mean reward
over seeds 0-3).  A purely reactive single-token doer (--no-reason) is a constant
"fire" regardless of frame or prompt; see scripts/doom/doom_sweep*.py for the
sweep that established this and settled the winning prompts.

Cache layout:
  sys block   : system role + corrected mechanics + "<|im_start|>user\\n"  (once)
  user block  : the user request text                                     (once)
  frame queue : the last K frames, each <vision_start> IMG* <vision_end>
                (grow to K, then the oldest block is recycled via refresh_block)
  reason block: per action, a short analysis of the current frame (throwaway)
  thinker     : OPTIONAL persistent planner (--thinker) the doer also reads

Two things made it work: a system prompt that corrects the mechanics the model
gets wrong on its own (no forward/back; strafe to centre the monster under the
fixed crosshair, then fire), and high enough resolution to localize the monster.

Run (headless)::

    HF_HOME=/mnt/LLM CUDA_VISIBLE_DEVICES=<gpu> doom-demo --steps 60 --out doom.gif
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless ViZDoom

import gymnasium  # noqa: E402
import vizdoom as vzd  # noqa: E402
from async_thoughts.engine import build_async_llm, encode  # noqa: E402
from async_thoughts.vision_demo import _warm_sampler  # noqa: E402
from minisgl.core import SamplingParams  # noqa: E402
from minisgl.models.qwen2vl_image import preprocess_image  # noqa: E402
from minisgl.models.qwen3_5_mrope import get_rope_index  # noqa: E402
from minisgl.shared_cache import AsyncContext  # noqa: E402
from vizdoom import gymnasium_wrapper  # noqa: E402,F401

from .config import ACTIONS, IMG_TOK, KEYNAMES, MERGE, VEND, VSTART, DoomConfig  # noqa: E402
from .prompt import (  # noqa: E402
    DOER_ACT_QUERY,
    DOER_QUERY_BASE,
    DOER_REASON_PREFIX,
    FRAME_HINT,
    SYS_TEXT,
    THINK_PREFIX,
    USER_TEXT,
)

_C = {"doer": "\033[1;32m", "think": "\033[2;36m", "state": "\033[1;33m", "dim": "\033[2m",
      "reset": "\033[0m"}


def _c(t: str, k: str) -> str:
    return _C[k] + t + _C["reset"]


def _frame_inputs(arr: np.ndarray, max_pixels: int):
    """(input_ids, pixel_values, grid_thw, mrope) for one RGB frame's image block."""
    pixel_values, grid = preprocess_image(arr, merge_size=MERGE, max_pixels=max_pixels)
    n_img = int(grid.prod().item()) // (MERGE**2)
    ids = torch.tensor([VSTART] + [IMG_TOK] * n_img + [VEND], dtype=torch.int32)
    mrope = get_rope_index(ids.long(), IMG_TOK, MERGE, grid)
    return {"token_ids": ids, "pixel_values": pixel_values, "image_grid_thw": grid,
            "mrope_positions": mrope}


def _action_token_ids(tokenizer, n_actions: int) -> list[int]:
    """Digit token id per action (for the reactive --no-reason baseline)."""
    return [tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(n_actions)]


def _key_action_ids(tokenizer) -> dict[int, list[int]]:
    """action -> candidate first-token ids for its key word (with/without leading space)."""
    return {
        a: sorted({tokenizer.encode(s, add_special_tokens=False)[0] for s in (name, " " + name)})
        for a, name in KEYNAMES.items()
    }


def _pick_key(logits: torch.Tensor, key_ids: dict[int, list[int]]) -> int:
    """The action whose key word has the highest logit (max over its candidate tokens)."""
    return max(key_ids, key=lambda a: max(float(logits[t]) for t in key_ids[a]))


async def _push_frame(llm, frame_blks: list, k: int, inputs: dict) -> None:
    """Append the newest frame to the queue (chronological, oldest first), keeping
    at most ``k`` image blocks: grow until full, then recycle the oldest block in
    place (refresh_block) and move it to the end.  Each frame is a standalone
    image block; the doer/thinker views read the whole queue so the model sees
    motion across the last k frames."""
    if len(frame_blks) < k:
        frame_blks.append((await llm.prefill_block(**inputs)).block)
    else:
        oldest = frame_blks.pop(0)
        await llm.refresh_block(oldest, **inputs)
        frame_blks.append(oldest)


def _save_gif(frames: list[np.ndarray], path: str) -> None:
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    if imgs:
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=120, loop=0)
        print(_c(f"  saved {len(imgs)} frames -> {path}", "state"))


async def _inject_thinker(llm, ctx, text: str, tokenizer) -> None:
    """Splice *text* into the thinker's block: force-feed its tokens (ignoring the
    model's own predictions), echoing them in the state colour, and leave
    ``ctx.next_input_id`` at the model's prediction after the note so thinking
    resumes cleanly.  Used to tell the thinker the screen just changed."""
    for tid in tokenizer.encode(text, add_special_tokens=False):
        ctx.next_input_id = tid
        async for _ in llm.async_generate(ctx, max_steps=1):
            pass
        sys.stdout.write(_c(tokenizer.decode([tid]), "state"))
        sys.stdout.flush()


async def run(cfg: DoomConfig) -> None:
    env = gymnasium.make(
        cfg.env_id, render_mode="rgb_array", frame_skip=cfg.frame_skip,
        screen_resolution=vzd.ScreenResolution.RES_640X480,
    )
    obs, _ = env.reset(seed=cfg.seed)
    n_actions = int(env.action_space.n)

    mode = "reason-then-act" if cfg.reason else "reactive"
    print(_c(f"\n  Doom demo — {cfg.model} plays {cfg.env_id} "
             f"({mode} doer{' + thinker' if cfg.thinker else ''}, k={cfg.k_frames})\n", "state"))
    llm = build_async_llm(cfg.model, memory_ratio=cfg.memory_ratio)
    tok = llm.tokenizer
    act_ids = _action_token_ids(tok, n_actions)

    # Fixed prompt blocks: system role, then the user request (prefilled in context
    # of the system).  Both are encoded once and never re-prefilled.
    sys_blk = (await llm.prefill_block(encode(SYS_TEXT, tok))).block
    user_text = cfg.user_prompt or USER_TEXT.format(k=cfg.k_frames)
    user_blk = (await llm.prefill_block(encode(user_text, tok), context=[sys_blk])).block

    # Frame queue: up to k image blocks, oldest first, seeded with the first frame.
    frame_blks: list = []
    await _push_frame(llm, frame_blks, cfg.k_frames, _frame_inputs(obs["screen"], cfg.max_pixels))

    def view(tail: list) -> list:
        return [sys_blk, user_blk, *frame_blks, *tail]

    thinker_ctx = None
    thinker_blk = None
    sampling = SamplingParams(temperature=cfg.thinker_temp, top_p=0.95, max_tokens=1)
    if cfg.thinker:
        tres = await llm.prefill_block(
            encode(THINK_PREFIX, tok), context=view([]), return_logits=True
        )
        thinker_blk = tres.block
        thinker_ctx = AsyncContext(
            cache_view=view([thinker_blk]), output_block=thinker_blk,
            next_input_id=int(tres.logits.argmax()),
        )
        _warm_sampler(llm, sampling)

    # Doer: reason-then-act with key-name actions (the config that clears the level).
    key_ids = _key_action_ids(tok)
    reason_ids, act_ids_q = encode(DOER_REASON_PREFIX, tok), encode(DOER_ACT_QUERY, tok)
    base_query = encode(DOER_QUERY_BASE, tok)  # reactive baseline (--no-reason)
    reason_sampling = (
        None if cfg.reason_temp <= 0
        else SamplingParams(temperature=cfg.reason_temp, top_p=0.95, max_tokens=1)
    )
    turn_forbid = [
        i for i in (tok.vocab.get(n) for n in
                    ("<|im_start|>", "<|im_end|>", "</think>", "<|endoftext|>"))
        if i is not None
    ]
    if not cfg.thinker and reason_sampling is not None:  # thinker already warmed above
        _warm_sampler(llm, reason_sampling)

    frames: list[np.ndarray] = []
    total = 0.0
    final_plan = ""
    try:
        for t in range(cfg.steps):
            frame = obs["screen"]
            frames.append(frame)
            if t > 0:  # the queue already holds frame 0 from the pre-loop push
                await _push_frame(llm, frame_blks, cfg.k_frames,
                                  _frame_inputs(frame, cfg.max_pixels))
                if cfg.thinker:  # the queue rotated -> refresh the thinker's view
                    thinker_ctx.cache_view[:] = view([thinker_blk])
                    if cfg.frame_hint:
                        await _inject_thinker(llm, thinker_ctx, FRAME_HINT, tok)

            # optional persistent thinker: a short chunk the doer will also read.
            if cfg.thinker:
                async for token in llm.async_generate(
                    thinker_ctx, max_steps=cfg.thinker_tokens,
                    forbid_ids=turn_forbid, sampling_params=sampling,
                ):
                    sys.stdout.write(_c(tok.decode([token]), "think"))
                    sys.stdout.flush()

            base_ctx = view([thinker_blk]) if cfg.thinker else view([])
            print(_c(f"\n  [{t:>3}] ", "doer"), end="", flush=True)
            if cfg.reason:
                # reason about the frame, then pick a key-name action from that reasoning.
                rp = await llm.prefill_block(reason_ids, context=base_ctx, return_logits=True)
                rblk = rp.block
                rctx = AsyncContext(cache_view=[*base_ctx, rblk], output_block=rblk,
                                    next_input_id=int(rp.logits.argmax()))
                async for token in llm.async_generate(
                    rctx, max_steps=cfg.reason_tokens, forbid_ids=turn_forbid,
                    sampling_params=reason_sampling,
                ):
                    sys.stdout.write(_c(tok.decode([token]), "dim"))
                    sys.stdout.flush()
                aq = await llm.prefill_block(act_ids_q, context=[*base_ctx, rblk], return_logits=True)
                logits = aq.logits.float().cpu()
                await llm.free_block(aq.block)
                await llm.free_block(rblk)
                action = _pick_key(logits, key_ids)
            else:  # reactive baseline: single constrained digit
                q = await llm.prefill_block(base_query, context=base_ctx, return_logits=True)
                logits = q.logits.float().cpu()
                await llm.free_block(q.block)
                action = int(torch.tensor([logits[i] for i in act_ids]).argmax())

            obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            print(_c(f"  -> {ACTIONS.get(action, '?')}  r={reward:+.0f} total={total:+.0f}",
                     "doer"), flush=True)
            if term or trunc:
                obs, _ = env.reset()
        if cfg.thinker and thinker_blk is not None:  # capture before finally frees the block
            final_plan = tok.decode(thinker_blk.token_ids, skip_special_tokens=True)
    finally:
        for blk in (sys_blk, user_blk, *frame_blks, thinker_blk):
            if blk is not None:
                await llm.free_block(blk)
        await llm.close()
        env.close()

    print(_c(f"\n\n  total reward: {total:+.1f} over {len(frames)} steps", "state"))
    if final_plan:
        print(_c(f"  thinker (final): ...{final_plan[-300:]}", "dim"))
    if cfg.out:
        _save_gif(frames, cfg.out)
