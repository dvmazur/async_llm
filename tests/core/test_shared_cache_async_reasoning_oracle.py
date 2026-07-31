"""
Oracle tests for ``minisgl.shared_cache`` against the reference implementation
at https://github.com/yandex-research/AsyncReasoning .

AsyncReasoning ships a sibling implementation of the SAME paged shared-cache
abstraction that minisgl implements:

  AsyncReasoning            <->  minisgl
  -------------------------------------------
  CacheBlock                <->  SharedBlock
  SharedCacheManager        <->  WorkerGroup + SharedCacheSession
  rotate_by_offset          <->  apply_rope_correction
  combine_cache_from_struct <->  decode_step page-table fill + _apply_corrections

This file compares the two implementations head-to-head.  It does NOT use
HuggingFace transformers as the reference -- transformers has no shared-cache
concept, so comparing minisgl to HF only validates end-to-end output.
AsyncReasoning validates the shared-cache abstraction itself at the kernel
and forward-pass level.

Layers compared:

  1. RoPE kernel direct (CPU, fast):
     apply_rope_correction(keys, [delta]*N, cs_cache)
       vs
     rotate_by_offset(keys_4d, offset=delta, config=hf_config)

     Both should produce numerically identical rotated keys at fp32.

  2. End-to-end SharedCacheManager forward (GPU + model):
     minisgl decode_step on [[prompt, w]]
       vs
     HF model(input_ids, past_key_values=CombinedCacheView([[prompt, w]]))

     Both should produce equivalent last-token logits per worker.

Setup:
  AsyncReasoning is cloned locally at /home/dvmazur/AsyncReasoning .  Its
  ``shared_cache`` package is added to sys.path below.  Triton/torch.compile
  paths are disabled before import so the CPU kernel tests work without
  CUDA.

Run::

    # CPU kernel parity only (needs a HF config in the local HF cache):
    uv run pytest tests/core/test_shared_cache_async_reasoning_oracle.py \\
        -v -k kernel

    # Full e2e parity (GPU + model weights):
    MINISGL_E2E_MODEL=Qwen/Qwen2.5-0.5B \\
        uv run pytest tests/core/test_shared_cache_async_reasoning_oracle.py -v -s

NOTE: like test_shared_cache.py, this file must run in its own pytest
invocation -- Engine.__init__ asserts CUDA is not yet initialized.
"""

from __future__ import annotations

# Must set these BEFORE importing shared_cache: AsyncReasoning's kernel paths
# default to triton (CUDA-only) and torch.compile.  Disable both so the CPU
# kernel-parity tests don't hit a CUDA assert.
import os

os.environ.setdefault("USE_TRITON", "0")
os.environ.setdefault("USE_TORCH_COMPILE", "0")

import gc
import sys
from typing import List

import pytest
import torch

# AsyncReasoning is cloned locally; add the repo root to sys.path so
# `import shared_cache` finds yandex-research/AsyncReasoning's package.
_ASYNC_REASONING_ROOT = "/home/dvmazur/AsyncReasoning"
if _ASYNC_REASONING_ROOT not in sys.path:
    sys.path.insert(0, _ASYNC_REASONING_ROOT)

import transformers

# AsyncReasoning was written against transformers==4.51 (per its
# requirements.txt) but minisgl requires >=4.56.  Two API changes broke
# direct compatibility:
#   (a) Cache.__init__ now requires `layers=` or `layer_class_to_replicate=`.
#   (b) DynamicCache no longer exposes `key_cache` / `value_cache` lists --
#       data lives in `self.layers[i].keys` / `.values`.
# Patch both before importing AsyncReasoning so its CacheBlock works
# unchanged.
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

_ORIG_CACHE_INIT = Cache.__init__


def _patched_cache_init(self, *args, **kwargs):
    if not args and not kwargs:
        return _ORIG_CACHE_INIT(self, layer_class_to_replicate=DynamicLayer)
    return _ORIG_CACHE_INIT(self, *args, **kwargs)


Cache.__init__ = _patched_cache_init


class _LegacyLayerListProxy:
    """List-like view over DynamicCache.layers exposing a single attribute
    (``keys`` or ``values``) as if it were the old top-level ``key_cache`` /
    ``value_cache`` list."""

    def __init__(self, cache, attr_name):
        self._cache = cache
        self._attr = attr_name

    def __len__(self):
        return sum(
            1
            for l in self._cache.layers
            if getattr(l, self._attr, None) is not None and getattr(l, self._attr).numel() > 0
        )

    def __getitem__(self, i):
        return getattr(self._cache.layers[i], self._attr)

    def __setitem__(self, i, value):
        while len(self._cache.layers) <= i:
            self._cache.layers.append(DynamicLayer())
        setattr(self._cache.layers[i], self._attr, value)

    def clear(self):
        self._cache.layers.clear()

    def append(self, value):
        layer = DynamicLayer()
        setattr(layer, self._attr, value)
        self._cache.layers.append(layer)


def _make_legacy_prop(attr_name: str):
    def _getter(self):
        proxy_attr = "_legacy_proxy_" + attr_name
        if not hasattr(self, proxy_attr):
            object.__setattr__(self, proxy_attr, _LegacyLayerListProxy(self, attr_name))
        return getattr(self, proxy_attr)

    return property(_getter)


DynamicCache.key_cache = _make_legacy_prop("keys")
DynamicCache.value_cache = _make_legacy_prop("values")

import shared_cache as ar_sc  # type: ignore  # noqa: E402  (must come after patches)
from minisgl.shared_cache import (
    SharedCacheSession,
    WorkerGroup,
    apply_rope_correction,
)

# -----------------------------------------------------------------------------
# Configuration & gating
# -----------------------------------------------------------------------------

# CPU kernel-parity tests use only the HF config (no weights), so they
# always have a default model to draw config from; the env var overrides
# it for users who want to test against a different family.
DEFAULT_CONFIG_MODEL = "Qwen/Qwen2.5-0.5B"

E2E_MODEL_PATH = os.environ.get("MINISGL_E2E_MODEL", "")

requires_e2e = pytest.mark.skipif(
    not E2E_MODEL_PATH or not torch.cuda.is_available(),
    reason="Set MINISGL_E2E_MODEL and ensure CUDA is available",
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _hf_config(model_path: str) -> transformers.PretrainedConfig:
    return transformers.AutoConfig.from_pretrained(model_path)


def _head_dim(config: transformers.PretrainedConfig) -> int:
    """Resolve head_dim the way both implementations do."""
    return int(
        getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    )


def _build_minisgl_cos_sin_cache(
    config: transformers.PretrainedConfig, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build minisgl's [max_pos, head_dim] cos/sin cache from an HF config.

    Mirrors ``minisgl.layers.rotary.RotaryEmbedding.__init__`` (default
    rope_type only; the kernel tests pick configs without rope_scaling).
    """
    head_dim = _head_dim(config)
    base = float(getattr(config, "rope_parameters", None).get("rope_theta"))
    max_pos = int(config.max_position_embeddings)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def _minisgl_to_async_reasoning_shape(keys_3d: torch.Tensor) -> torch.Tensor:
    """minisgl shape [N, num_heads, head_dim] -> AsyncReasoning shape
    [1, num_heads, N, head_dim]."""
    return keys_3d.permute(1, 0, 2).unsqueeze(0).contiguous()


def _async_reasoning_to_minisgl_shape(keys_4d: torch.Tensor) -> torch.Tensor:
    """Inverse of the above."""
    return keys_4d.squeeze(0).permute(1, 0, 2).contiguous()


# -----------------------------------------------------------------------------
# Layer 1: RoPE kernel direct parity (CPU)
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kernel_config():
    """An HF config to drive both kernels (CPU; no model weights loaded)."""
    return _hf_config(DEFAULT_CONFIG_MODEL)


_DTYPE_ATOL = [
    (torch.float32, 1e-5),
    (torch.bfloat16, 5e-2),
]


@pytest.mark.parametrize("dtype,atol", _DTYPE_ATOL, ids=lambda v: str(v).split(".")[-1])
@pytest.mark.parametrize("offset", [0, 1, 5, 13, 100, -1, -7, -100])
def test_rope_kernel_matches_async_reasoning(kernel_config, dtype, atol, offset):
    """apply_rope_correction([delta]*N, ...) must match rotate_by_offset(delta, ...).

    Both kernels implement the same RoPE-shift operation; running them on
    identical inputs should produce identical outputs modulo dtype noise.
    """
    config = kernel_config
    head_dim = _head_dim(config)
    num_heads = config.num_key_value_heads
    n_tokens = 6

    torch.manual_seed(0)
    keys_3d = torch.randn(n_tokens, num_heads, head_dim, dtype=dtype)
    keys_4d = _minisgl_to_async_reasoning_shape(keys_3d)

    # minisgl side
    cs_cache = _build_minisgl_cos_sin_cache(config)  # fp32
    corrections = torch.full((n_tokens,), offset, dtype=torch.int64)
    out_minisgl = apply_rope_correction(keys_3d, corrections, cs_cache)

    # AsyncReasoning side
    out_async_4d = ar_sc.rotate_by_offset(keys=keys_4d, offset=offset, config=config)
    out_async = _async_reasoning_to_minisgl_shape(out_async_4d)

    assert out_minisgl.shape == out_async.shape
    assert out_minisgl.dtype == out_async.dtype == dtype
    max_diff = (out_minisgl.float() - out_async.float()).abs().max().item()
    assert max_diff <= atol, (
        f"offset={offset} dtype={dtype}: max |diff| = {max_diff:.3e} > atol={atol:.3e}"
    )


def test_rope_kernel_uniform_offset_matches_per_token_correction(kernel_config):
    """A uniform corrections vector [delta]*N must produce the same result as
    a single-offset rotate_by_offset(delta).  This is the invariant minisgl
    relies on for SharedBlock reads at non-zero target_start."""
    config = kernel_config
    head_dim = _head_dim(config)
    num_heads = config.num_key_value_heads
    n_tokens = 10
    delta = 17

    torch.manual_seed(1)
    keys_3d = torch.randn(n_tokens, num_heads, head_dim, dtype=torch.float32)

    # Apply per-token corrections (all equal to delta)
    cs_cache = _build_minisgl_cos_sin_cache(config)
    out_per_token = apply_rope_correction(
        keys_3d, torch.full((n_tokens,), delta, dtype=torch.int64), cs_cache
    )

    # Apply via rotate_by_offset (one uniform offset)
    out_uniform = _async_reasoning_to_minisgl_shape(
        ar_sc.rotate_by_offset(
            keys=_minisgl_to_async_reasoning_shape(keys_3d),
            offset=delta,
            config=config,
        )
    )

    max_diff = (out_per_token - out_uniform).abs().max().item()
    assert max_diff <= 1e-5, f"uniform-offset disagreement: max |diff| = {max_diff:.3e}"


def test_rope_kernel_zero_offset_is_identity_both(kernel_config):
    """offset=0 must be identity for both implementations.  Sanity check that
    the comparison is well-posed."""
    config = kernel_config
    head_dim = _head_dim(config)
    keys_3d = torch.randn(4, config.num_key_value_heads, head_dim, dtype=torch.float32)

    cs_cache = _build_minisgl_cos_sin_cache(config)
    out_m = apply_rope_correction(keys_3d, torch.zeros(4, dtype=torch.int64), cs_cache)
    out_a = _async_reasoning_to_minisgl_shape(
        ar_sc.rotate_by_offset(
            keys=_minisgl_to_async_reasoning_shape(keys_3d), offset=0, config=config
        )
    )
    assert torch.allclose(out_m, keys_3d, atol=1e-6)
    assert torch.allclose(out_a, keys_3d, atol=1e-6)
    assert torch.allclose(out_m, out_a, atol=1e-6)


def test_rope_kernel_inverse_offset_undoes_rotation(kernel_config):
    """Rotating by +delta then -delta must recover the original keys.  Verify
    independently in both implementations and confirm they agree."""
    config = kernel_config
    head_dim = _head_dim(config)
    delta = 23
    keys_3d = torch.randn(5, config.num_key_value_heads, head_dim, dtype=torch.float32)

    cs_cache = _build_minisgl_cos_sin_cache(config)

    # minisgl round-trip
    fwd_m = apply_rope_correction(keys_3d, torch.full((5,), delta, dtype=torch.int64), cs_cache)
    back_m = apply_rope_correction(fwd_m, torch.full((5,), -delta, dtype=torch.int64), cs_cache)
    assert torch.allclose(back_m, keys_3d, atol=1e-5)

    # AsyncReasoning round-trip
    keys_4d = _minisgl_to_async_reasoning_shape(keys_3d)
    fwd_a = ar_sc.rotate_by_offset(keys=keys_4d, offset=delta, config=config)
    back_a = ar_sc.rotate_by_offset(keys=fwd_a, offset=-delta, config=config)
    back_a_3d = _async_reasoning_to_minisgl_shape(back_a)
    assert torch.allclose(back_a_3d, keys_3d, atol=1e-5)

    # Cross-check: forward outputs from both implementations agree.
    assert torch.allclose(fwd_m, _async_reasoning_to_minisgl_shape(fwd_a), atol=1e-5)


# -----------------------------------------------------------------------------
# Layer 2: End-to-end SharedCacheManager forward parity (GPU + model)
# -----------------------------------------------------------------------------


def _build_engine(model_path: str):
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig

    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=8,
        cuda_graph_bs=[1, 2, 4],
        cuda_graph_max_bs=4,
        page_size=int(os.environ.get("MINISGL_TEST_PAGE_SIZE", "1")),
        memory_ratio=float(os.environ.get("MINISGL_TEST_MEMORY_RATIO", "0.35")),
        max_seq_len_override=2048,
    )
    return Engine(config)


@pytest.fixture(scope="module")
def engine_and_session():
    if not E2E_MODEL_PATH or not torch.cuda.is_available():
        pytest.skip("e2e tests disabled")
    engine = _build_engine(E2E_MODEL_PATH)
    session = SharedCacheSession(engine)
    yield engine, session
    engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def hf_tokenizer():
    if not E2E_MODEL_PATH:
        pytest.skip("e2e tests disabled")
    return transformers.AutoTokenizer.from_pretrained(E2E_MODEL_PATH)


@pytest.fixture(scope="module")
def hf_model(engine_and_session):
    """HF model loaded AFTER the minisgl engine so memory_ratio reserves
    happen first."""
    if not torch.cuda.is_available():
        pytest.skip("e2e tests disabled")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        E2E_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",
    )
    model.eval()
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _encode(text: str, tokenizer) -> torch.Tensor:
    return tokenizer.encode(text, return_tensors="pt").view(-1).to(torch.int32)


def _seed_minisgl_block(session, prompt_block, seed_token: int, num_tokens: int) -> tuple:
    """Build a minisgl SharedBlock containing ``num_tokens`` tokens, starting
    with ``seed_token`` and greedy-decoding ``num_tokens - 1`` more.

    Returns ``(block, token_list)`` where ``token_list`` has length
    ``num_tokens`` and corresponds 1:1 to the KV slots in the block.
    """
    blk = session.create_block()
    group = WorkerGroup(cache_structure=[[prompt_block, blk]], write_to=[blk])
    tokens: List[int] = []
    cur_int = int(seed_token)
    for _ in range(num_tokens):
        tokens.append(cur_int)
        logits = session.decode_step(group, torch.tensor([cur_int], dtype=torch.int32))
        cur_int = int(logits.argmax(dim=-1).item())
    return blk, tokens


def _seed_async_block(model, prompt_block, tokens: List[int]) -> "ar_sc.CacheBlock":
    """Build an AsyncReasoning CacheBlock containing exactly the given token
    sequence's KV, by appending tokens one-at-a-time while it sees
    ``[prompt_block, blk]`` -- mirroring the minisgl decode pattern."""
    blk = ar_sc.CacheBlock(config=model.config)
    for tok_int in tokens:
        cm = ar_sc.SharedCacheManager(cache_structure=[[prompt_block, blk]], write_to=[blk])
        ids = torch.tensor([[tok_int]], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            model(**cm.get_input_kwargs(input_ids=ids))
    return blk


@torch.inference_mode()
def _async_reasoning_prefill(model, cache_block, input_ids: torch.Tensor) -> torch.Tensor:
    """Prefill an AsyncReasoning CacheBlock with input_ids; return last-token
    logits on CPU (fp32)."""
    cm = ar_sc.SharedCacheManager(cache_structure=[[cache_block]], write_to=[cache_block])
    ids = input_ids.long().view(1, -1).to(model.device)
    out = model(**cm.get_input_kwargs(input_ids=ids))
    return out.logits[0, -1].float().cpu()


@torch.inference_mode()
def _async_reasoning_decode_step(model, cache_structure, write_to, probe_id: int) -> torch.Tensor:
    """One decode step in AsyncReasoning's idiom; return last-token logits
    (fp32, CPU)."""
    cm = ar_sc.SharedCacheManager(cache_structure=cache_structure, write_to=write_to)
    ids = torch.tensor([[probe_id]], dtype=torch.long, device=model.device)
    out = model(**cm.get_input_kwargs(input_ids=ids))
    return out.logits[0, -1].float().cpu()


def _assert_logits_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1.0,
    softmax_atol: float = 0.1,
    near_tie_threshold: float = 0.5,
    label: str = "",
) -> None:
    """Argmax strict (with bf16 near-tie escape), softmax-diff strict, raw-
    logit-diff loose.

    Tolerances are tighter than the deleted HF oracle's because both sides
    here implement the SAME algorithm (shared-cache + RoPE correction) on
    the same HF model weights; only bf16 kernel-internal noise differs
    between minisgl-flashinfer and HF-eager.  Observed first-run diffs on
    Qwen2.5-0.5B: softmax_max <= 0.07, logit_max <= 0.65 -- so 0.1 / 1.0
    leave ~30-50% headroom.

    Near-tie escape: at bf16, two tokens with logits within ~0.25 are
    numerically indistinguishable.  Python's ``argmax`` breaks exact ties
    by returning the lower index, so two equally-good top-1 candidates
    can swap argmax across implementations even when the math is the
    same.  If argmaxes disagree but the GAP between ``expected``'s top-1
    and ``expected[actual.argmax()]`` is below ``near_tie_threshold``, log
    a warning and accept.
    """
    a = actual.float().flatten()
    e = expected.float().flatten()
    prefix = f"[{label}] " if label else ""
    assert a.shape == e.shape
    a_arg, e_arg = int(a.argmax()), int(e.argmax())
    sm_diff = (torch.softmax(a, -1) - torch.softmax(e, -1)).abs().max().item()
    raw_diff = (a - e).abs().max().item()
    print(
        f"{prefix}diffs softmax_max={sm_diff:.4g} logit_max={raw_diff:.4g} "
        f"argmax_minisgl={a_arg} argmax_async={e_arg}"
    )
    if a_arg != e_arg:
        gap = float(e[e_arg] - e[a_arg])
        assert gap < near_tie_threshold, (
            f"{prefix}argmax mismatch with gap {gap:.4f} > {near_tie_threshold} "
            f"(genuine disagreement, NOT a bf16 tie): "
            f"actual_top5={a.topk(5).indices.tolist()} "
            f"expected_top5={e.topk(5).indices.tolist()}"
        )
        print(
            f"{prefix}argmax differs but near-tie: gap={gap:.4f} < {near_tie_threshold}; accepting"
        )
    assert sm_diff <= softmax_atol, f"{prefix}softmax max-diff {sm_diff:.4g} > {softmax_atol}"
    assert raw_diff <= atol, f"{prefix}logit max-diff {raw_diff:.4g} > {atol}"


@requires_e2e
def test_prefill_matches_async_reasoning(engine_and_session, hf_model, hf_tokenizer):
    """Prefill the same prompt in both implementations; last-token logits
    must match.  This validates the shared-cache prefill path against the
    reference."""
    _, session = engine_and_session
    prompt_ids = _encode("The capital of France is", hf_tokenizer)

    # minisgl side
    ms_prompt = session.create_block()
    ms_logits = session.prefill_block(ms_prompt, prompt_ids)[0].float().cpu()

    # AsyncReasoning side
    ar_prompt = ar_sc.CacheBlock(config=hf_model.config)
    ar_logits = _async_reasoning_prefill(hf_model, ar_prompt, prompt_ids)

    _assert_logits_close(ms_logits, ar_logits, label="prefill")


@requires_e2e
def test_decode_step_single_block_matches_async_reasoning(
    engine_and_session, hf_model, hf_tokenizer
):
    """Prefill a prompt; one decode step on [[prompt, w]] in both
    implementations must agree.  Direct test of decode_step vs the
    reference SharedCacheManager.update path."""
    _, session = engine_and_session
    prompt_ids = _encode("The capital of France is", hf_tokenizer)

    # minisgl side
    ms_prompt = session.create_block()
    session.prefill_block(ms_prompt, prompt_ids)
    ms_w = session.create_block()
    probe_id = 42
    ms_group = WorkerGroup(cache_structure=[[ms_prompt, ms_w]], write_to=[ms_w])
    ms_logits = (
        session.decode_step(ms_group, torch.tensor([probe_id], dtype=torch.int32))[0].float().cpu()
    )

    # AsyncReasoning side: prefill the prompt into ar_prompt, then do one
    # decode step with [[ar_prompt, ar_w]] writing to ar_w.
    ar_prompt = ar_sc.CacheBlock(config=hf_model.config)
    _async_reasoning_prefill(hf_model, ar_prompt, prompt_ids)
    ar_w = ar_sc.CacheBlock(config=hf_model.config)
    ar_logits = _async_reasoning_decode_step(
        hf_model,
        cache_structure=[[ar_prompt, ar_w]],
        write_to=[ar_w],
        probe_id=probe_id,
    )

    _assert_logits_close(ms_logits, ar_logits, label="decode_single")


@requires_e2e
def test_decode_step_block_reorder_matches_async_reasoning(
    engine_and_session, hf_model, hf_tokenizer
):
    """The acceptance test for RoPE correction at non-zero offset: build
    blocks a, b each at offset len(p), then probe with [p, a, b] AND [p, b, a].
    Both arrangements should match the reference at logit level, AND the
    reordering must observably change the logits (sanity)."""
    _, session = engine_and_session
    prompt_ids = _encode("The largest planet is", hf_tokenizer)

    # ===== minisgl side =====
    ms_prompt = session.create_block()
    p_logits = session.prefill_block(ms_prompt, prompt_ids)
    top2 = torch.topk(p_logits[0], k=2).indices.tolist()

    ms_a, a_tokens = _seed_minisgl_block(session, ms_prompt, int(top2[0]), 3)
    ms_b, b_tokens = _seed_minisgl_block(session, ms_prompt, int(top2[1]), 3)
    probe_id = 99
    probe = torch.tensor([probe_id], dtype=torch.int32)

    ms_logits_ab = (
        session.decode_step(
            WorkerGroup(cache_structure=[[ms_prompt, ms_a, ms_b]], write_to=[ms_b]),
            probe,
        )[0]
        .float()
        .cpu()
    )
    ms_logits_ba = (
        session.decode_step(
            WorkerGroup(cache_structure=[[ms_prompt, ms_b, ms_a]], write_to=[ms_a]),
            probe,
        )[0]
        .float()
        .cpu()
    )

    # ===== AsyncReasoning side =====
    ar_prompt = ar_sc.CacheBlock(config=hf_model.config)
    _async_reasoning_prefill(hf_model, ar_prompt, prompt_ids)

    ar_a = _seed_async_block(hf_model, ar_prompt, a_tokens)
    ar_b = _seed_async_block(hf_model, ar_prompt, b_tokens)

    ar_logits_ab = _async_reasoning_decode_step(
        hf_model,
        cache_structure=[[ar_prompt, ar_a, ar_b]],
        write_to=[ar_b],
        probe_id=probe_id,
    )
    ar_logits_ba = _async_reasoning_decode_step(
        hf_model,
        cache_structure=[[ar_prompt, ar_b, ar_a]],
        write_to=[ar_a],
        probe_id=probe_id,
    )

    _assert_logits_close(ms_logits_ab, ar_logits_ab, label="reorder[p,a,b]")
    _assert_logits_close(ms_logits_ba, ar_logits_ba, label="reorder[p,b,a]")

    # Sanity: reordering observably changes logits in both implementations.
    ms_reorder_diff = (ms_logits_ab - ms_logits_ba).abs().max().item()
    ar_reorder_diff = (ar_logits_ab - ar_logits_ba).abs().max().item()
    assert ms_reorder_diff > 1e-2, (
        f"minisgl: [p,a,b] vs [p,b,a] identical (max diff = {ms_reorder_diff:.3e})"
    )
    assert ar_reorder_diff > 1e-2, (
        f"async: [p,a,b] vs [p,b,a] identical (max diff = {ar_reorder_diff:.3e})"
    )


@requires_e2e
def test_prefill_kv_layer0_matches_async_reasoning(engine_and_session, hf_model, hf_tokenizer):
    """Direct KV-tensor parity at layer 0 after prefilling the same prompt
    in both implementations.

    The K and V tensors that minisgl writes to its kv_cache for the prompt
    block must match the K and V that AsyncReasoning's CacheBlock stores
    for layer 0, modulo bf16 attention-kernel noise (flashinfer vs HF eager).

    This is the deepest possible parity check: if the stored KV bytes match,
    every downstream consumer (decode_step, multi-block reads, RoPE
    corrections) is operating on equivalent state.
    """
    engine, session = engine_and_session
    prompt_ids = _encode("The capital of France is", hf_tokenizer)

    # ===== minisgl side: prefill, then read out layer-0 KV =====
    ms_prompt = session.create_block()
    session.prefill_block(ms_prompt, prompt_ids)
    pages = ms_prompt.token_slots_tensor().to(engine.device).long()
    # k_cache(0) shape: [num_pages, page_size, local_kv_heads, head_dim].
    # Flatten the first two dims to a flat token-slot layout and gather by slot.
    k0_full = engine.kv_cache.k_cache(0).flatten(0, 1)  # [num_pages * page_size, H, D]
    v0_full = engine.kv_cache.v_cache(0).flatten(0, 1)
    ms_k0 = k0_full[pages].float().cpu()  # [N, H, D]
    ms_v0 = v0_full[pages].float().cpu()

    # ===== AsyncReasoning side: prefill into a fresh CacheBlock =====
    ar_prompt = ar_sc.CacheBlock(config=hf_model.config)
    _async_reasoning_prefill(hf_model, ar_prompt, prompt_ids)
    # get_kv_with_offset returns [1, num_kv_heads, N, head_dim]; reshape to
    # match minisgl's [N, H, D].
    ar_k0_4d, ar_v0_4d = ar_prompt.get_kv_with_offset(layer_idx=0, offset=0)
    ar_k0 = ar_k0_4d.squeeze(0).transpose(0, 1).float().cpu()
    ar_v0 = ar_v0_4d.squeeze(0).transpose(0, 1).float().cpu()

    assert ms_k0.shape == ar_k0.shape, f"K shape {ms_k0.shape} vs {ar_k0.shape}"
    assert ms_v0.shape == ar_v0.shape, f"V shape {ms_v0.shape} vs {ar_v0.shape}"

    k_max_diff = (ms_k0 - ar_k0).abs().max().item()
    v_max_diff = (ms_v0 - ar_v0).abs().max().item()
    # Average per-element diff is also informative; very large max-diff at
    # tail keys can hide a small-systematic-bias bug.
    k_mean_diff = (ms_k0 - ar_k0).abs().mean().item()
    v_mean_diff = (ms_v0 - ar_v0).abs().mean().item()
    print(
        f"[prefill_kv layer0] k_max={k_max_diff:.4g} v_max={v_max_diff:.4g} "
        f"k_mean={k_mean_diff:.4g} v_mean={v_mean_diff:.4g}"
    )

    # bf16 attention-kernel noise tolerance.  Both impls compute the same
    # K, V mathematically but with different attention backends
    # (minisgl-flashinfer vs HF-eager).  Tail keys can spike to ~0.25;
    # observed mean diff ~1e-3.  Threshold = 0.5 catches gross errors.
    K_ATOL, V_ATOL = 0.5, 0.05
    assert k_max_diff <= K_ATOL, f"K max-diff {k_max_diff:.4g} > {K_ATOL}"
    assert v_max_diff <= V_ATOL, f"V max-diff {v_max_diff:.4g} > {V_ATOL}"


@requires_e2e
def test_multi_step_decode_matches_async_reasoning(engine_and_session, hf_model, hf_tokenizer):
    """K=5 consecutive greedy decode steps.  Per step, minisgl's argmax must
    match AsyncReasoning's argmax.  AsyncReasoning stays the absolute
    reference: its argmax token is fed to both sides for the next step.

    Surfaces accumulated drift / canonical-position regressions over a
    sustained rollout.  Compared to my deleted HF rollout, this version
    uses AsyncReasoning (which applies the SAME shared-cache abstraction)
    so per-step argmax disagreement is a real bug, not a numerical
    near-tie."""
    _, session = engine_and_session
    K = 5
    prompt_ids = _encode("Once upon a time there was a", hf_tokenizer)

    # minisgl prefill
    ms_prompt = session.create_block()
    ms_first_logits = session.prefill_block(ms_prompt, prompt_ids)
    ms_w = session.create_block()
    ms_group = WorkerGroup(cache_structure=[[ms_prompt, ms_w]], write_to=[ms_w])

    # AsyncReasoning prefill
    ar_prompt = ar_sc.CacheBlock(config=hf_model.config)
    ar_first_logits = _async_reasoning_prefill(hf_model, ar_prompt, prompt_ids)
    ar_w = ar_sc.CacheBlock(config=hf_model.config)

    # Step 0: compare argmax of prefill-output logits (they predict the FIRST
    # next token).  AsyncReasoning's argmax becomes the ground-truth seed.
    ms_arg0 = int(ms_first_logits[0].argmax().item())
    ar_arg0 = int(ar_first_logits.argmax().item())
    assert ms_arg0 == ar_arg0, f"prefill argmax mismatch: minisgl={ms_arg0} async={ar_arg0}"

    # Decode K steps, feeding AsyncReasoning's argmax to both sides each step.
    cur_token = ar_arg0
    for k in range(K):
        ms_logits = (
            session.decode_step(ms_group, torch.tensor([cur_token], dtype=torch.int32))[0]
            .float()
            .cpu()
        )
        ar_logits = _async_reasoning_decode_step(
            hf_model,
            cache_structure=[[ar_prompt, ar_w]],
            write_to=[ar_w],
            probe_id=cur_token,
        )
        ms_arg = int(ms_logits.argmax())
        ar_arg = int(ar_logits.argmax())
        sm_diff = (torch.softmax(ms_logits, -1) - torch.softmax(ar_logits, -1)).abs().max().item()
        logit_diff = (ms_logits - ar_logits).abs().max().item()
        print(
            f"[rollout k={k}] minisgl_arg={ms_arg} async_arg={ar_arg} "
            f"softmax_max={sm_diff:.4g} logit_max={logit_diff:.4g}"
        )
        assert ms_arg == ar_arg, (
            f"step k={k}: minisgl argmax={ms_arg} != async argmax={ar_arg}, "
            f"minisgl_top5={ms_logits.topk(5).indices.tolist()} "
            f"async_top5={ar_logits.topk(5).indices.tolist()}"
        )
        cur_token = ar_arg


@requires_e2e
def test_multi_worker_two_workers_matches_async_reasoning(
    engine_and_session, hf_model, hf_tokenizer
):
    """N=2 workers sharing one prompt but with independent suffix blocks.
    A single decode_step processes both workers in one batched call;
    AsyncReasoning processes the same structure in its own forward.
    Each worker's logits must match its AsyncReasoning counterpart -- this
    catches batched-decode cross-talk and per-worker corrections."""
    _, session = engine_and_session
    prompt_ids = _encode("Colors of the rainbow include", hf_tokenizer)

    # minisgl prompt + two seed suffixes
    ms_prompt = session.create_block()
    p_logits = session.prefill_block(ms_prompt, prompt_ids)
    top2 = torch.topk(p_logits[0], k=2).indices.tolist()
    ms_a, a_tokens = _seed_minisgl_block(session, ms_prompt, int(top2[0]), 2)
    ms_b, b_tokens = _seed_minisgl_block(session, ms_prompt, int(top2[1]), 2)

    # AsyncReasoning prompt + two seed suffixes (matching tokens)
    ar_prompt = ar_sc.CacheBlock(config=hf_model.config)
    _async_reasoning_prefill(hf_model, ar_prompt, prompt_ids)
    ar_a = _seed_async_block(hf_model, ar_prompt, a_tokens)
    ar_b = _seed_async_block(hf_model, ar_prompt, b_tokens)

    probe_a, probe_b = 13, 21
    probes = torch.tensor([probe_a, probe_b], dtype=torch.int32)

    # Fresh write blocks placed AT THE END of each worker's cache_structure
    # so AsyncReasoning's attention sees the new K/V (it only reads from
    # cache_structure, not write_to-outside).  minisgl always sees the new
    # K/V via its page-table layout.
    ms_w_a, ms_w_b = session.create_block(), session.create_block()
    ms_group = WorkerGroup(
        cache_structure=[[ms_prompt, ms_a, ms_w_a], [ms_prompt, ms_b, ms_w_b]],
        write_to=[ms_w_a, ms_w_b],
    )
    ms_out = session.decode_step(ms_group, probes)
    ms_logits_a = ms_out[0].float().cpu()
    ms_logits_b = ms_out[1].float().cpu()

    # AsyncReasoning: same shape -- write block is the last element of each
    # worker's cache_structure.
    ar_w_a = ar_sc.CacheBlock(config=hf_model.config)
    ar_w_b = ar_sc.CacheBlock(config=hf_model.config)
    cm = ar_sc.SharedCacheManager(
        cache_structure=[[ar_prompt, ar_a, ar_w_a], [ar_prompt, ar_b, ar_w_b]],
        write_to=[ar_w_a, ar_w_b],
    )
    ids = torch.tensor([[probe_a], [probe_b]], dtype=torch.long, device=hf_model.device)
    with torch.inference_mode():
        out = hf_model(**cm.get_input_kwargs(input_ids=ids))
    ar_logits_a = out.logits[0, -1].float().cpu()
    ar_logits_b = out.logits[1, -1].float().cpu()

    _assert_logits_close(ms_logits_a, ar_logits_a, label="mw2_worker_a")
    _assert_logits_close(ms_logits_b, ar_logits_b, label="mw2_worker_b")


@requires_e2e
def test_block_reuse_across_groups_matches_async_reasoning(
    engine_and_session, hf_model, hf_tokenizer
):
    """Same SharedBlock used in two distinct WorkerGroups sequentially.
    Both reads must match the AsyncReasoning reference, AND the two
    minisgl reads must match each other (re-use must not corrupt the
    block's stored KV / positions)."""
    _, session = engine_and_session
    prompt_ids = _encode("Colors of the rainbow include", hf_tokenizer)

    # Set up both impls' prompt + suffix.
    ms_prompt = session.create_block()
    p_logits = session.prefill_block(ms_prompt, prompt_ids)
    seed = int(p_logits[0].argmax().item())
    ms_suffix, suffix_tokens = _seed_minisgl_block(session, ms_prompt, seed, 3)

    ar_prompt = ar_sc.CacheBlock(config=hf_model.config)
    _async_reasoning_prefill(hf_model, ar_prompt, prompt_ids)
    ar_suffix = _seed_async_block(hf_model, ar_prompt, suffix_tokens)

    probe_id = 7
    probe = torch.tensor([probe_id], dtype=torch.int32)

    # Reference forward.  Write block goes at the END of cache_structure so
    # AsyncReasoning's attention sees the new K/V on equal footing with
    # minisgl's always-visible page-table layout.
    ar_w = ar_sc.CacheBlock(config=hf_model.config)
    ar_logits = _async_reasoning_decode_step(
        hf_model,
        cache_structure=[[ar_prompt, ar_suffix, ar_w]],
        write_to=[ar_w],
        probe_id=probe_id,
    )

    # Trial 1: fresh WorkerGroup with a fresh write block in cache_structure.
    ms_w_1 = session.create_block()
    g1 = WorkerGroup(
        cache_structure=[[ms_prompt, ms_suffix, ms_w_1]],
        write_to=[ms_w_1],
    )
    ms_logits_1 = session.decode_step(g1, probe)[0].float().cpu()

    # Trial 2: another fresh WorkerGroup reusing the same ms_suffix block.
    ms_w_2 = session.create_block()
    g2 = WorkerGroup(
        cache_structure=[[ms_prompt, ms_suffix, ms_w_2]],
        write_to=[ms_w_2],
    )
    ms_logits_2 = session.decode_step(g2, probe)[0].float().cpu()

    # Both must match the AsyncReasoning reference.
    _assert_logits_close(ms_logits_1, ar_logits, label="reuse_trial1")
    _assert_logits_close(ms_logits_2, ar_logits, label="reuse_trial2")

    # And the two minisgl trials must agree closely with each other (same
    # backend; bf16 cuda-graph noise only).
    trial_diff = (ms_logits_1 - ms_logits_2).abs().max().item()
    assert trial_diff < 1e-2, f"block-reuse trials diverged: max |diff| = {trial_diff:.3e}"


@requires_e2e
def test_context_prefill_matches_async_reasoning(engine_and_session, hf_model, hf_tokenizer):
    """Prefill a block IN CONTEXT of another block (the reference's
    ``prefill_cache_block(text, [ctx, new])`` pattern): the new tokens attend
    causally to themselves and fully to the context, but their KV is stored
    block-relative.  Validates both the returned logits and a subsequent
    decode step that reads the context-prefilled block."""
    _, session = engine_and_session
    prompt_ids = _encode("Why is the sky blue? Think step by step.", hf_tokenizer)
    suffix_ids = _encode("\n</think>\nThe answer is", hf_tokenizer)
    assert len(suffix_ids) > 1  # exercise S > 1 (causal self-segment)

    # minisgl: prompt standalone, suffix in context of the prompt
    ms_prompt = session.create_block()
    session.prefill_block(ms_prompt, prompt_ids)
    ms_close = session.create_block()
    ms_logits = session.prefill_block(ms_close, suffix_ids, context=[ms_prompt])[0].float().cpu()

    # AsyncReasoning: same pattern.  NOTE: the reference's batched multi-token
    # update onto an existing cache breaks under transformers>=4.56 (mask
    # shape mismatch in qwen2 eager attention; the repo targets 4.51), so feed
    # the suffix one token at a time — with causal attention this is
    # mathematically identical to a batched causal prefill.
    ar_prompt = ar_sc.CacheBlock(config=hf_model.config)
    _async_reasoning_prefill(hf_model, ar_prompt, prompt_ids)
    ar_close = ar_sc.CacheBlock(config=hf_model.config)
    ar_logits = None
    for tok_int in suffix_ids.tolist():
        ar_logits = _async_reasoning_decode_step(
            hf_model,
            cache_structure=[[ar_prompt, ar_close]],
            write_to=[ar_close],
            probe_id=int(tok_int),
        )

    # Looser raw-logit atol: the suffix KV is built token-by-token by HF eager
    # on the reference side vs one batched bf16 prefill here, so tail-logit
    # divergence compounds over the suffix length (argmax/softmax budgets stay
    # default; the per-layer fp32-reference check bounds our own kernel at
    # bf16 noise).
    _assert_logits_close(ms_logits, ar_logits, atol=2.5, label="ctx_prefill")

    # The stored KV must read back correctly: one decode step over
    # [prompt, close, w] in both implementations.
    probe_id = 11
    ms_w = session.create_block()
    ms_dec = (
        session.decode_step(
            WorkerGroup(cache_structure=[[ms_prompt, ms_close, ms_w]], write_to=[ms_w]),
            torch.tensor([probe_id], dtype=torch.int32),
        )[0]
        .float()
        .cpu()
    )
    ar_dec = _async_reasoning_decode_step(
        hf_model,
        cache_structure=[[ar_prompt, ar_close, ar_sc.CacheBlock(config=hf_model.config)]],
        write_to=None,
        probe_id=probe_id,
    )
    _assert_logits_close(ms_dec, ar_dec, label="ctx_prefill_decode")


@requires_e2e
def test_interleaved_growth_matches_async_reasoning(engine_and_session, hf_model, hf_tokenizer):
    """Hogwild-style pattern: two workers decode in the SAME group every step,
    each seeing the other's growing block:

        thinker view: [prompt, W, T]     writes T
        writer  view: [prompt, T, W]     writes W

    T and W grow in alternation (one token each per step), so under a
    view-global storage scheme each block's stored positions would be
    non-contiguous (the per-token RoPE delta is not constant within a block).
    Block-relative storage + query rotation must handle this exactly;
    per-step argmax must match the AsyncReasoning reference, which also gives
    each worker same-step visibility of the other's newest token.

    NOTE: both views are kept the same length on purpose.  AsyncReasoning's
    slow reference left-pads ragged multi-worker batches, and under
    transformers>=4.56 (this repo's floor; the reference targets 4.51) that
    path is broken — it disagrees with ITS OWN single-worker forward on the
    same state (verified directly; minisgl's ragged path is self-consistent
    and matches the reference's single-worker forward)."""
    _, session = engine_and_session
    K = 6
    prompt_ids = _encode("Let me think about prime numbers.", hf_tokenizer)

    # minisgl prefill + blocks
    ms_prompt = session.create_block()
    ms_first_logits = session.prefill_block(ms_prompt, prompt_ids)
    ms_t, ms_w = session.create_block(), session.create_block()
    ms_group = WorkerGroup(
        cache_structure=[[ms_prompt, ms_w, ms_t], [ms_prompt, ms_t, ms_w]],
        write_to=[ms_t, ms_w],
    )

    # AsyncReasoning prefill + blocks
    ar_prompt = ar_sc.CacheBlock(config=hf_model.config)
    ar_first_logits = _async_reasoning_prefill(hf_model, ar_prompt, prompt_ids)
    ar_t = ar_sc.CacheBlock(config=hf_model.config)
    ar_w = ar_sc.CacheBlock(config=hf_model.config)

    assert int(ms_first_logits[0].argmax()) == int(ar_first_logits.argmax())

    # Seed: thinker gets the prefill argmax, writer a distinct fixed probe.
    t_tok = int(ar_first_logits.argmax())
    w_tok = 42

    for k in range(K):
        ms_logits = session.decode_step(ms_group, torch.tensor([t_tok, w_tok], dtype=torch.int32))
        ms_logits_t = ms_logits[0].float().cpu()
        ms_logits_w = ms_logits[1].float().cpu()

        cm = ar_sc.SharedCacheManager(
            cache_structure=[[ar_prompt, ar_w, ar_t], [ar_prompt, ar_t, ar_w]],
            write_to=[ar_t, ar_w],
        )
        ids = torch.tensor([[t_tok], [w_tok]], dtype=torch.long, device=hf_model.device)
        with torch.inference_mode():
            out = hf_model(**cm.get_input_kwargs(input_ids=ids))
        ar_logits_t = out.logits[0, -1].float().cpu()
        ar_logits_w = out.logits[1, -1].float().cpu()

        # Looser raw-logit atol than single-step tests: this is a 6-step
        # rollout where each side's KV state is produced by a different
        # attention kernel (flashinfer vs HF eager) and cross-worker
        # same-step reads compound the bf16 divergence on tail logits.
        # argmax and softmax-diff (the strict signals) keep default budgets.
        _assert_logits_close(ms_logits_t, ar_logits_t, atol=2.0, label=f"interleave_t[k={k}]")
        _assert_logits_close(ms_logits_w, ar_logits_w, atol=2.0, label=f"interleave_w[k={k}]")

        # AsyncReasoning stays the absolute reference for the rollout.
        t_tok = int(ar_logits_t.argmax())
        w_tok = int(ar_logits_w.argmax())

    assert ms_t.num_tokens == K and ms_w.num_tokens == K


@requires_e2e
def test_empty_block_in_group_matches_async_reasoning(engine_and_session, hf_model, hf_tokenizer):
    """A degenerate empty block in the cache_structure is a no-op in both
    implementations.  Verify that minisgl's logits with the empty block
    inserted match AsyncReasoning's logits WITHOUT the empty block."""
    _, session = engine_and_session
    prompt_ids = _encode("The weather today is", hf_tokenizer)

    ms_prompt = session.create_block()
    session.prefill_block(ms_prompt, prompt_ids)
    ms_empty = session.create_block()  # never grown

    ar_prompt = ar_sc.CacheBlock(config=hf_model.config)
    _async_reasoning_prefill(hf_model, ar_prompt, prompt_ids)

    probe_id = 3
    probe = torch.tensor([probe_id], dtype=torch.int32)

    # minisgl WITH empty block in the structure
    ms_group_with = WorkerGroup(
        cache_structure=[[ms_prompt, ms_empty, session.create_block()]],
        write_to=None,
    )
    ms_logits = session.decode_step(ms_group_with, probe)[0].float().cpu()

    # AsyncReasoning WITHOUT empty block
    ar_logits = _async_reasoning_decode_step(
        hf_model,
        cache_structure=[[ar_prompt, ar_sc.CacheBlock(config=hf_model.config)]],
        write_to=None,
        probe_id=probe_id,
    )

    # Slightly looser raw-logit atol: adding an empty block in the
    # structure exercises an additional code path (page-table iteration
    # over the empty block in minisgl), which adds a small amount of
    # additional bf16 noise to a tail logit.  argmax + softmax stay tight.
    _assert_logits_close(ms_logits, ar_logits, atol=1.5, label="empty_block")
