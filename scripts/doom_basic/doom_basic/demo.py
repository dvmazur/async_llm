"""Qwen3.5 vision plays ViZDoom: reason abstractly about the frame, then probe
the logits for the action.

The loop is the plain VLM agent loop, and the action never has to be parsed out
of prose — it is read off the model's own next-token distribution::

    def action_probe(reasoning_prompt, reasoning_trace) -> action:
        logits = model.encode(cat(reasoning_prompt, reasoning_trace, PROBE_QUERY))
        valid  = logits[action_token_ids]         # FIRE / RIGHT / LEFT only
        return action_of[action_token_ids[valid.argmax()]]

    for each step:
        frame  = env screen
        prompt = reasoning_system + user + <the frame> + "what has to happen?"
        trace  = model.generate(prompt, max_steps=cfg.reason_tokens)   # brief!
        action = action_probe(prompt, trace)
        obs    = env.step(action)

Splitting it this way means the *reasoning* prompt can ask for free abstract
thought (and is told not to name a key), while the *action* is always legal and
always available: the probe is one extra prefill whose logits are masked down to
the action words, so there is no format for the model to get wrong and no
fallback path to reason about.

Every step builds a fresh, self-contained prompt and frees it afterwards: no KV
is reused across steps, no frame queue, no in-place image refresh, no background
thinker.  The only thing the model remembers is the short list of its own recent
actions, rendered into the prompt as text.  See ``scripts/doom`` for the
shared-cache version that keeps the frames and a running plan in the cache.

Each prompt is a single standalone block (text + image + text), because an image
can only be encoded in a context-free prefill; the trace is generated into a
second block that reads it, and the probe is a third, throwaway one on top.

Run (headless)::

    HF_HOME=/mnt/LLM CUDA_VISIBLE_DEVICES=<gpu> \\
      uv run python -m doom_basic --steps 40 --out doom_basic.gif
"""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless ViZDoom

import gymnasium  # noqa: E402
import vizdoom as vzd  # noqa: E402
from minisgl.core import SamplingParams  # noqa: E402
from minisgl.llm import AsyncLLM  # noqa: E402
from minisgl.models.qwen2vl_image import preprocess_image  # noqa: E402
from minisgl.models.qwen3_5_mrope import get_rope_index  # noqa: E402
from minisgl.shared_cache import AsyncContext, CacheView  # noqa: E402
from vizdoom import gymnasium_wrapper  # noqa: E402,F401

from .config import ACTIONS, IMG_TOK, KEYNAMES, MERGE, VEND, VSTART, BasicConfig
from .prompt import (
    HISTORY_TEXT,
    PROBE_QUERY,
    REASON_SEED,
    REASON_SYS_TEXT,
    USER_PREFIX,
    USER_SUFFIX,
)

_C = {"say": "\033[2;37m", "act": "\033[1;32m", "state": "\033[1;33m", "probe": "\033[1;36m",
      "reset": "\033[0m"}


def _c(text: str, key: str) -> str:
    return _C[key] + text + _C["reset"]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def build_llm(model_path: str, memory_ratio: float) -> AsyncLLM:
    """A single-GPU AsyncLLM sized for this demo (one prompt in flight at a time)."""
    return AsyncLLM(
        model_path,
        dtype=torch.bfloat16,
        max_running_req=2,
        cuda_graph_bs=[1],
        cuda_graph_max_bs=1,
        memory_ratio=memory_ratio,
        max_seq_len_override=8192,
        page_size=1,
    )


def _warm_sampler(llm: AsyncLLM, sampling: SamplingParams) -> None:
    """Trigger flashinfer's one-time JIT compile of the sampling kernel up front.

    The first *sampled* (non-greedy) decode compiles the softmax/top-p kernels
    with ninja (~1-2 min, cached afterwards); doing it here keeps that pause from
    looking like a hang mid-episode.  Greedy decode is argmax and needs none."""
    if sampling.is_greedy:
        return
    print("Warming up sampling kernel (first run JIT-compiles with ninja, ~1-2 min)...", flush=True)
    sampler = llm.engine.sampler
    dummy = torch.zeros((1, sampler.vocab_size), dtype=torch.float32, device=sampler.device)
    with torch.inference_mode():
        sampler.sample(dummy, sampler.prepare_params([sampling]))


# ---------------------------------------------------------------------------
# The reasoning prompt
# ---------------------------------------------------------------------------


def reasoning_prompt(tok, frame: np.ndarray, cfg: BasicConfig, history: deque) -> dict:
    """The whole reasoning turn as one multimodal prefill: text, the frame, text.

    ``get_rope_index`` lays out interleaved mRoPE over the mixed sequence (image
    tokens take grid positions and compress the running position), so the frame
    can sit in the middle of the prompt rather than having to lead it.
    """
    pixel_values, grid = preprocess_image(frame, merge_size=MERGE, max_pixels=cfg.max_pixels)
    n_img = int(grid.prod().item()) // (MERGE**2)

    hist = HISTORY_TEXT.format(actions=", ".join(history)) if history else ""
    head = tok.encode(REASON_SYS_TEXT + USER_PREFIX, add_special_tokens=False)
    tail = tok.encode(USER_SUFFIX.format(history=hist) + REASON_SEED, add_special_tokens=False)

    ids = torch.tensor(head + [VSTART] + [IMG_TOK] * n_img + [VEND] + tail, dtype=torch.int32)
    return {
        "token_ids": ids,
        "pixel_values": pixel_values,
        "image_grid_thw": grid,
        "mrope_positions": get_rope_index(ids.long(), IMG_TOK, MERGE, grid),
    }


# ---------------------------------------------------------------------------
# The action probe
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    """The probe's tokenized query plus the action-token mask it argmaxes over.

    ``token_ids`` are the candidate *first* tokens of every action word (with and
    without a leading space, since which one the model wants depends on the
    query's last character), and ``action_of`` maps each back to its env action.
    """

    query_ids: List[int]
    token_ids: torch.Tensor  # the "possible actions mask", as ids
    action_of: Dict[int, int]


def build_probe(tok) -> Probe:
    """Resolve PROBE_QUERY and the action words against *tok*'s vocabulary."""
    action_of: Dict[int, int] = {}
    for action, name in KEYNAMES.items():
        for text in (name, " " + name):
            action_of.setdefault(tok.encode(text, add_special_tokens=False)[0], action)
    return Probe(
        query_ids=tok.encode(PROBE_QUERY, add_special_tokens=False),
        token_ids=torch.tensor(list(action_of), dtype=torch.long),
        action_of=action_of,
    )


async def action_probe(
    llm: AsyncLLM, probe: Probe, context: CacheView
) -> tuple[int, Dict[int, float]]:
    """Prefill the probe query on top of *context* and argmax over the action tokens.

    *context* is ``[reasoning prompt, reasoning trace]``, so the probe reads the
    frame and the trace the model just wrote.  The query's block is read once and
    freed, hence ``capture_affine=False`` (skips the O(seq) GDN affine capture).

    Returns the chosen env action plus the probe's distribution over actions
    (softmax of the per-action best logit), which is only there to be printed.
    """
    query = await llm.prefill_block(
        probe.query_ids, context=context, capture_affine=False, return_logits=True
    )
    try:
        logits = query.logits.float().cpu()
        valid = logits[probe.token_ids]  # the possible-actions mask
        winner = int(probe.token_ids[int(valid.argmax())])

        # Per-action score = its best candidate token; softmax for display only.
        best: Dict[int, float] = {}
        for token_id, action in probe.action_of.items():
            best[action] = max(best.get(action, -float("inf")), float(logits[token_id]))
        probs = torch.softmax(torch.tensor(list(best.values())), dim=0).tolist()
        return probe.action_of[winner], dict(zip(best, probs))
    finally:
        await llm.free_block(query.block)


# ---------------------------------------------------------------------------
# One agent step
# ---------------------------------------------------------------------------


async def reason_and_act(
    llm: AsyncLLM, cfg: BasicConfig, probe: Probe, frame: np.ndarray, history: deque
) -> tuple[str, int, Dict[int, float]]:
    """One agent step: reason about *frame*, then probe for the action.

    Returns ``(trace, action, probs)``; the trace is streamed to stdout as it is
    generated.  All three blocks (prompt, trace, probe query) are freed before
    returning, so nothing survives the step.
    """
    tok = llm.tokenizer
    eot = tok.vocab.get("<|im_end|>")
    # <think> is forbidden so a thinking-mode model reasons in the open instead of
    # spending the whole trace budget inside a block we then cut off.
    forbid = [
        i for i in (tok.vocab.get(n) for n in
                    ("<|im_start|>", "<think>", "</think>", "<|endoftext|>"))
        if i is not None
    ]
    sampling = (
        None if cfg.temperature <= 0
        else SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=1)
    )

    prompt = await llm.prefill_block(
        **reasoning_prompt(tok, frame, cfg, history), return_logits=True
    )
    trace = await llm.create_block()  # the reasoning is generated in here
    # The trace's first token comes from the prompt's own logits (async_generate
    # needs a token to feed), so the forbidden set is applied here by hand — the
    # generator only masks the steps it samples itself, and "<think>" is the
    # model's favourite opener.
    first = prompt.logits.float().clone()  # clone: the engine owns the logits buffer
    first[forbid] = float("-inf")
    ctx = AsyncContext(
        cache_view=[prompt.block, trace], output_block=trace, next_input_id=int(first.argmax()),
    )
    out = [] if ctx.next_input_id == eot else [ctx.next_input_id]
    try:
        at_turn_end = not out
        if out:
            sys.stdout.write(_c(tok.decode(out), "say"))
            async for token in llm.async_generate(
                ctx, max_steps=cfg.reason_tokens - 1, forbid_ids=forbid, sampling_params=sampling,
            ):
                if at_turn_end := (token == eot):  # the model is done reasoning
                    break
                out.append(token)
                sys.stdout.write(_c(tok.decode([token]), "say"))
                sys.stdout.flush()
            if not at_turn_end:
                # A decode step writes the token it is *fed*, so the one sampled
                # last is still pending: one throwaway step flushes it into the
                # trace the probe reads.  Skipped at turn end, where the pending
                # token is <|im_end|> and would close the turn.
                async for _ in llm.async_generate(ctx, max_steps=1, forbid_ids=forbid):
                    pass

        action, probs = await action_probe(llm, probe, [prompt.block, trace])
        return tok.decode(out, skip_special_tokens=True), action, probs
    finally:
        await llm.free_block(trace)
        await llm.free_block(prompt.block)


def _save_gif(frames: list[np.ndarray], path: str) -> None:
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    if imgs:
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=120, loop=0)
        print(_c(f"  saved {len(imgs)} frames -> {path}", "state"))


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


async def run(cfg: BasicConfig) -> None:
    env = gymnasium.make(
        cfg.env_id, render_mode="rgb_array", frame_skip=cfg.frame_skip,
        screen_resolution=vzd.ScreenResolution.RES_640X480,
    )
    obs, _ = env.reset(seed=cfg.seed)

    print(_c(f"\n  Basic Doom demo — {cfg.model} plays {cfg.env_id} "
             f"(reason -> probe, {cfg.reason_tokens} reason tokens, history={cfg.history})\n",
             "state"))
    llm = build_llm(cfg.model, cfg.memory_ratio)
    if cfg.temperature > 0:
        _warm_sampler(llm, SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p,
                                          max_tokens=1))
    probe = build_probe(llm.tokenizer)

    history: deque[str] = deque(maxlen=max(cfg.history, 0))
    frames: list[np.ndarray] = []
    total = 0.0
    confidence = 0.0
    try:
        for t in range(cfg.steps):
            frame = obs["screen"]
            frames.append(frame)
            print(_c(f"\n  [{t:>3}] {REASON_SEED}", "act"), end="", flush=True)
            _trace, action, probs = await reason_and_act(llm, cfg, probe, frame, history)
            history.append(KEYNAMES.get(action, ACTIONS[action]))
            confidence += probs[action]

            obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            ranked = " ".join(f"{KEYNAMES[a]}={p:.2f}"
                              for a, p in sorted(probs.items(), key=lambda kv: -kv[1]))
            print(_c(f"\n        probe: {ranked}", "probe")
                  + _c(f"  -> {ACTIONS.get(action, '?')}  r={reward:+.0f} total={total:+.0f}",
                       "act"), flush=True)
            if term or trunc:
                obs, _ = env.reset()
                history.clear()
    finally:
        await llm.close()
        env.close()

    steps = len(frames)
    print(_c(f"\n  total reward: {total:+.1f} over {steps} steps "
             f"(mean probe confidence {confidence / max(steps, 1):.2f})", "state"))
    if cfg.history == 0:
        print(_c("  (history=0: every step was decided from the frame alone)", "state"))
    if cfg.out:
        _save_gif(frames, cfg.out)
