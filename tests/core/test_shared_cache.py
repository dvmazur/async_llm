"""
Tests for ``minisgl.shared_cache``.

This file contains two groups of tests:

* **Unit tests** (always run) that exercise pure-Python/torch logic:
  ``SharedBlock``, ``WorkerGroup`` and the ``apply_rope_correction`` math.

* **End-to-end tests** (require ``MINISGL_E2E_MODEL`` env var pointing to an
  HF model path and a working CUDA device) that spin up a real ``Engine`` and
  drive it through ``SharedCacheSession`` to verify the full pipeline works.

Run unit tests only::

    pytest tests/core/test_shared_cache.py -v

Run everything (needs GPU + model weights)::

    MINISGL_E2E_MODEL=meta-llama/Llama-3.2-1B \
        pytest tests/core/test_shared_cache.py -v -s

Run only the standalone demo (no pytest) — preferred wrapper::

    MINISGL_E2E_MODEL=Qwen/Qwen2.5-0.5B python scripts/run_shared_cache_demo.py

Or run the test module directly (``__name__ == "__main__"`` runs pytest then the demo)::

    MINISGL_E2E_MODEL=Qwen/Qwen2.5-0.5B python tests/core/test_shared_cache.py
"""

from __future__ import annotations

import os
from typing import List

import pytest
import torch
from minisgl.shared_cache import (
    SharedBlock,
    SharedCacheSession,
    WorkerGroup,
    apply_rope_correction,
)

# =============================================================================
# Unit tests — no model/GPU needed
# =============================================================================


class TestSharedBlock:
    def test_empty_block(self):
        block = SharedBlock(torch.device("cpu"))
        assert block.num_tokens == 0
        assert block.num_pages == 0
        assert block.last_page_len == 0
        assert not block.has_capacity

    def test_grow_pages_page_size_1(self):
        block = SharedBlock(torch.device("cpu"), page_size=1)
        block.grow_pages(torch.tensor([10, 11, 12], dtype=torch.int32), 3)
        assert block.num_tokens == 3
        assert block.num_pages == 3
        assert block.page_starts == [10, 11, 12]
        # at page_size=1, token slots == page starts == page numbers
        assert block.token_slots_tensor().tolist() == [10, 11, 12]
        assert block.page_numbers_tensor().tolist() == [10, 11, 12]

    def test_grow_pages_page_size_4(self):
        block = SharedBlock(torch.device("cpu"), page_size=4)
        # 5 tokens packed into 2 pages starting at slots 0 and 4
        block.grow_pages(torch.tensor([0, 4], dtype=torch.int32), 5)
        assert block.num_tokens == 5
        assert block.num_pages == 2
        assert block.last_page_len == 1
        assert block.has_capacity  # room for 3 more in page 1
        assert block.token_slots_tensor().tolist() == [0, 1, 2, 3, 4]
        assert block.page_numbers_tensor().tolist() == [0, 1]

    def test_append_token_decode_growth(self):
        block = SharedBlock(torch.device("cpu"), page_size=4)
        block.grow_pages(torch.tensor([0, 4], dtype=torch.int32), 5)  # last page has room
        block.append_token(None)  # fits in page 1 (offset 1)
        assert block.num_tokens == 6
        assert block.last_page_len == 2
        block.append_token(None)
        block.append_token(None)  # now page 1 is full (8 tokens)
        assert block.num_tokens == 8 and not block.has_capacity
        block.append_token(8)  # new page at slot 8
        assert block.num_tokens == 9
        assert block.page_starts == [0, 4, 8]
        assert block.last_page_len == 1
        assert block.token_slots_tensor().tolist() == list(range(9))

    def test_clear(self):
        block = SharedBlock(torch.device("cpu"), page_size=4)
        block.grow_pages(torch.tensor([8, 12], dtype=torch.int32), 6)
        pages = block.clear()
        assert pages == [8, 12]
        assert block.num_tokens == 0 and block.num_pages == 0

    def test_unique_block_ids(self):
        b1 = SharedBlock(torch.device("cpu"))
        b2 = SharedBlock(torch.device("cpu"))
        assert b1.block_id != b2.block_id


class TestWorkerGroup:
    def _make_group(self, device: torch.device) -> tuple:
        prompt = SharedBlock(device)
        w1 = SharedBlock(device)
        w2 = SharedBlock(device)
        prompt.grow_pages(torch.arange(5, dtype=torch.int32), 5)
        w1.grow_pages(torch.arange(5, 8, dtype=torch.int32), 3)
        w2.grow_pages(torch.arange(8, 10, dtype=torch.int32), 2)
        group = WorkerGroup(
            cache_structure=[
                [prompt, w2, w1],
                [prompt, w1, w2],
            ],
            write_to=[w1, w2],
        )
        return prompt, w1, w2, group

    def test_num_workers(self):
        _, _, _, group = self._make_group(torch.device("cpu"))
        assert group.num_workers == 2

    def test_worker_cache_length(self):
        _, _, _, group = self._make_group(torch.device("cpu"))
        # worker 0: prompt(5) + w2(2) + w1(3) = 10
        # worker 1: prompt(5) + w1(3) + w2(2) = 10
        assert group.worker_cache_length(0) == 10
        assert group.worker_cache_length(1) == 10

    def test_max_cache_length(self):
        _, _, _, group = self._make_group(torch.device("cpu"))
        assert group.max_cache_length() == 10

    def test_default_write_to(self):
        a, b, c = [SharedBlock(torch.device("cpu")) for _ in range(3)]
        group = WorkerGroup(cache_structure=[[a, b], [a, c]])
        # write_to defaults to last block per worker
        assert group.write_to[0] is b
        assert group.write_to[1] is c


class TestApplyRopeCorrection:
    """Verify the RoPE correction kernel numerically."""

    def _build_cos_sin_cache(self, head_dim: int, max_pos: int, base: float = 10000.0):
        """Builds the cos_sin_cache tensor used by minisgl.layers.rotary."""
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        t = torch.arange(max_pos, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    def _reference_rope(self, keys: torch.Tensor, positions: torch.Tensor, base: float = 10000.0):
        """Reference RoPE application: apply RoPE at `positions` to `keys`.

        keys: [N, num_heads, head_dim]
        positions: [N]
        """
        head_dim = keys.shape[-1]
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # [N, half]
        cos = freqs.cos().unsqueeze(1)  # [N, 1, half]
        sin = freqs.sin().unsqueeze(1)
        k1 = keys[..., :half]
        k2 = keys[..., half:]
        return torch.cat([cos * k1 - sin * k2, cos * k2 + sin * k1], dim=-1)

    def test_zero_correction_is_identity(self):
        head_dim = 64
        keys = torch.randn(4, 2, head_dim)
        cs = self._build_cos_sin_cache(head_dim, 128)
        corrections = torch.zeros(4, dtype=torch.int64)
        out = apply_rope_correction(keys, corrections, cs)
        # cos(0)=1, sin(0)=0 → identity
        assert torch.allclose(out, keys, atol=1e-5)

    def test_positive_correction_matches_reference(self):
        """Applying correction by Δ should equal reference RoPE at position Δ
        when the input keys are un-rotated (stored_pos = 0)."""
        head_dim = 64
        base = 10000.0
        max_pos = 256
        keys = torch.randn(5, 3, head_dim)
        corrections = torch.tensor([0, 7, 13, 42, 100], dtype=torch.int64)

        cs = self._build_cos_sin_cache(head_dim, max_pos, base)
        got = apply_rope_correction(keys, corrections, cs)
        expected = self._reference_rope(keys, corrections, base)

        assert torch.allclose(got, expected, atol=1e-5), (
            f"max diff = {(got - expected).abs().max().item()}"
        )

    def test_negative_correction_matches_reference(self):
        head_dim = 64
        base = 10000.0
        max_pos = 256
        keys = torch.randn(3, 2, head_dim)
        corrections = torch.tensor([-5, -20, -1], dtype=torch.int64)

        cs = self._build_cos_sin_cache(head_dim, max_pos, base)
        got = apply_rope_correction(keys, corrections, cs)
        expected = self._reference_rope(keys, corrections, base)

        assert torch.allclose(got, expected, atol=1e-5), (
            f"max diff = {(got - expected).abs().max().item()}"
        )

    def test_positive_then_negative_roundtrip(self):
        """Rotate by +k then by -k should recover the original."""
        head_dim = 64
        max_pos = 128
        keys = torch.randn(4, 2, head_dim)
        k = 23
        cs = self._build_cos_sin_cache(head_dim, max_pos)

        forward = apply_rope_correction(keys, torch.full((4,), k, dtype=torch.int64), cs)
        back = apply_rope_correction(forward, torch.full((4,), -k, dtype=torch.int64), cs)

        assert torch.allclose(back, keys, atol=1e-5), (
            f"roundtrip failed, max diff = {(back - keys).abs().max().item()}"
        )


# =============================================================================
# End-to-end tests — require GPU + model
# =============================================================================

E2E_MODEL_PATH = os.environ.get("MINISGL_E2E_MODEL", "")

requires_e2e = pytest.mark.skipif(
    not E2E_MODEL_PATH or not torch.cuda.is_available(),
    reason="Set MINISGL_E2E_MODEL to an HF model path and ensure CUDA is available",
)


def _build_engine(model_path: str):
    """Lazy import — importing Engine eagerly pulls in CUDA-only deps."""
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig

    # Init torch.distributed with a TCP store if not already initialised
    if not torch.distributed.is_initialized():
        # Engine.__init__ will call init_process_group itself; nothing to do
        pass

    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=8,
        cuda_graph_bs=[2, 4],
        cuda_graph_max_bs=4,
        page_size=int(os.environ.get("MINISGL_TEST_PAGE_SIZE", "1")),
        memory_ratio=0.7,
        max_seq_len_override=2048,
    )
    return Engine(config)


def _encode(text: str, model_path: str) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    ids = tok.encode(text, return_tensors="pt").view(-1).to(torch.int32)
    return ids


@pytest.fixture(scope="module")
def engine_and_session():
    """Shared fixture for all e2e tests."""
    if not E2E_MODEL_PATH or not torch.cuda.is_available():
        pytest.skip("e2e tests disabled")
    engine = _build_engine(E2E_MODEL_PATH)
    session = SharedCacheSession(engine)
    yield engine, session
    engine.shutdown()


@requires_e2e
class TestSharedCacheE2E:
    def test_prefill_then_decode_shape(self, engine_and_session):
        engine, session = engine_and_session
        prompt_ids = _encode("The capital of France is", E2E_MODEL_PATH)

        prompt = session.create_block()
        logits = session.prefill_block(prompt, prompt_ids)

        vocab_size = engine.model.lm_head.num_embeddings
        assert logits.shape == (1, vocab_size)
        assert prompt.num_tokens == len(prompt_ids)

    def test_identical_workers_give_identical_logits(self, engine_and_session):
        """Two workers with identical cache structures must produce identical logits."""
        engine, session = engine_and_session
        prompt_ids = _encode("Once upon a time", E2E_MODEL_PATH)

        prompt = session.create_block()
        session.prefill_block(prompt, prompt_ids)

        w1 = session.create_block()
        w2 = session.create_block()

        # Both workers see only the shared prompt → identical attention results
        group = WorkerGroup(
            cache_structure=[[prompt, w1], [prompt, w2]],
            write_to=[w1, w2],
        )

        # Feed the same token to both workers
        next_tok = torch.tensor([42, 42], dtype=torch.int32)
        logits = session.decode_step(group, next_tok)

        max_diff = (logits[0] - logits[1]).abs().max().item()
        assert max_diff < 1e-2, f"Identical workers diverged: max |logit diff| = {max_diff}"

    def test_multi_step_decode_generates_tokens(self, engine_and_session):
        """Run a greedy decode loop and verify we generate a plausible continuation."""
        engine, session = engine_and_session
        prompt_ids = _encode("The largest planet in our solar system is", E2E_MODEL_PATH)

        prompt = session.create_block()
        prompt_logits = session.prefill_block(prompt, prompt_ids)
        first_token = prompt_logits[0].argmax(dim=-1)

        w1 = session.create_block()
        group = WorkerGroup(
            cache_structure=[[prompt, w1]],
            write_to=[w1],
        )

        generated: List[int] = [int(first_token.item())]
        current = first_token.view(1).to(torch.int32)

        for _ in range(20):
            logits = session.decode_step(group, current)
            nxt = logits[0].argmax(dim=-1)
            generated.append(int(nxt.item()))
            current = nxt.view(1).to(torch.int32)

        assert len(generated) == 21
        # Tokens should not all collapse to the same id (that would indicate bad attention)
        assert len(set(generated)) > 3, f"Suspicious generation: {generated}"

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(E2E_MODEL_PATH)
        text = tok.decode(generated)
        print(f"\n[multi-step decode] Prompt continuation: ...{text!r}")

    def test_two_workers_share_prompt(self, engine_and_session):
        """Two workers, different first tokens, shared prompt — both should decode."""
        engine, session = engine_and_session
        prompt_ids = _encode("The weather today is", E2E_MODEL_PATH)

        prompt = session.create_block()
        prompt_logits = session.prefill_block(prompt, prompt_ids)
        # Pick top-2 tokens as different seeds for the two workers
        top2 = torch.topk(prompt_logits[0], k=2).indices
        seeds = top2.to(torch.int32)

        w1 = session.create_block()
        w2 = session.create_block()
        group = WorkerGroup(
            cache_structure=[[prompt, w1], [prompt, w2]],
            write_to=[w1, w2],
        )

        current = seeds.clone()
        for _ in range(10):
            logits = session.decode_step(group, current)
            current = logits.argmax(dim=-1).to(torch.int32)

        # After 10 decode steps, each worker should have 10 tokens
        assert w1.num_tokens == 10
        assert w2.num_tokens == 10

    def test_reordered_structure_changes_logits(self, engine_and_session):
        """Worker with [prompt, A, B] vs [prompt, B, A] must produce different logits
        (unless A or B is empty), confirming that ordering actually matters
        and the RoPE correction path is active."""
        engine, session = engine_and_session
        prompt_ids = _encode("Colors of the rainbow include", E2E_MODEL_PATH)

        prompt = session.create_block()
        prompt_logits = session.prefill_block(prompt, prompt_ids)
        # Use top-2 different tokens so the two workers diverge and write
        # distinct KV into blocks a and b.
        top2 = torch.topk(prompt_logits[0], k=2).indices.to(torch.int32)

        a = session.create_block()
        b = session.create_block()

        # Seed a and b with three decode steps each, via a group where they share
        # the prompt.  Workers start with different seed tokens, so blocks a and b
        # will contain distinct KV content.
        seed_group = WorkerGroup(
            cache_structure=[[prompt, a], [prompt, b]],
            write_to=[a, b],
        )
        current = top2.clone()
        for _ in range(3):
            logits = session.decode_step(seed_group, current)
            current = logits.argmax(dim=-1).to(torch.int32)

        assert a.num_tokens == 3 and b.num_tokens == 3

        # Now compare [prompt, a, b] vs [prompt, b, a] at the same next-token
        probe = torch.tensor([7, 7], dtype=torch.int32)
        compare_group = WorkerGroup(
            cache_structure=[[prompt, a, b], [prompt, b, a]],
            write_to=[a, b],  # writes go somewhere we'll discard
        )
        logits_cmp = session.decode_step(compare_group, probe)

        diff = (logits_cmp[0] - logits_cmp[1]).abs().max().item()
        print(f"\n[reorder test] max |logit diff| between [p,a,b] and [p,b,a] = {diff}")
        # Different orderings must give different logits (not identical).
        assert diff > 1e-3, (
            f"Reordering had no effect on logits (max diff = {diff}). "
            "This suggests RoPE correction is not being applied."
        )


# =============================================================================
# Standalone entrypoint
# =============================================================================


def _run_standalone_demo():
    """Standalone demo: prefill a prompt, decode two diverging workers, print output."""
    if not E2E_MODEL_PATH:
        print("Set MINISGL_E2E_MODEL=<hf_model_path> to run the demo.")
        return
    if not torch.cuda.is_available():
        print("Demo requires CUDA.")
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(E2E_MODEL_PATH)
    engine = _build_engine(E2E_MODEL_PATH)
    session = SharedCacheSession(engine)

    prompt_text = "The three largest planets in our solar system are"
    prompt_ids = _encode(prompt_text, E2E_MODEL_PATH)
    print(f"\n=== Prompt: {prompt_text!r} ({len(prompt_ids)} tokens) ===")

    prompt = session.create_block()
    prompt_logits = session.prefill_block(prompt, prompt_ids)

    # Seed two workers with the top-2 next tokens
    top2 = torch.topk(prompt_logits[0], k=2).indices.to(torch.int32)
    print(f"Top-2 continuations: {[tok.decode([int(t)]) for t in top2]}")

    w1 = session.create_block()
    w2 = session.create_block()
    group = WorkerGroup(
        cache_structure=[[prompt, w1], [prompt, w2]],
        write_to=[w1, w2],
    )

    seq1: List[int] = [int(top2[0].item())]
    seq2: List[int] = [int(top2[1].item())]
    current = top2.clone()

    for _ in range(30):
        logits = session.decode_step(group, current)
        nxt = logits.argmax(dim=-1).to(torch.int32)
        seq1.append(int(nxt[0].item()))
        seq2.append(int(nxt[1].item()))
        current = nxt

    print("\n--- Worker 1 ---")
    print(prompt_text + tok.decode(seq1))
    print("\n--- Worker 2 ---")
    print(prompt_text + tok.decode(seq2))

    engine.shutdown()


if __name__ == "__main__":
    # Run unit tests synchronously, then the e2e demo if configured.
    import sys

    unit_argv = [sys.argv[0], "-v", "-k", "not E2E"]
    exit_code = pytest.main(unit_argv)
    if exit_code != 0:
        sys.exit(exit_code)

    _run_standalone_demo()
