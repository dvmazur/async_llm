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
from types import SimpleNamespace
from typing import List

import pytest
import torch
from minisgl.shared_cache import (
    SharedBlock,
    SharedCacheSession,
    WorkerGroup,
    apply_rope_correction,
)
from minisgl.shared_cache.gdn_affine import compose_gdn_affines

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

    def test_pages_needed(self):
        block = SharedBlock(torch.device("cpu"), page_size=4)
        assert block.free_tail == 0
        assert block.pages_needed(1) == 1 and block.pages_needed(9) == 3
        block.grow_pages(torch.tensor([0, 4], dtype=torch.int32), 5)
        assert block.free_tail == 3
        # the 3 free slots of the last page come first
        assert block.pages_needed(3) == 0
        assert block.pages_needed(4) == 1
        assert block.pages_needed(12) == 3

    def test_grow_pages_extends_non_empty_block(self):
        block = SharedBlock(torch.device("cpu"), page_size=4)
        block.grow_pages(torch.tensor([0, 4], dtype=torch.int32), 5)
        # 3 more tokens fit in the last page's free slots -> no new pages
        block.grow_pages(torch.tensor([], dtype=torch.int32), 3)
        assert block.num_tokens == 8 and block.page_starts == [0, 4]
        assert not block.has_capacity
        # 5 more need 2 fresh pages
        block.grow_pages(torch.tensor([8, 12], dtype=torch.int32), 5)
        assert block.num_tokens == 13
        assert block.page_starts == [0, 4, 8, 12]
        assert block.last_page_len == 1
        assert block.token_slots_tensor().tolist() == list(range(13))

    def test_grow_pages_rejects_wrong_page_count(self):
        block = SharedBlock(torch.device("cpu"), page_size=4)
        block.grow_pages(torch.tensor([0], dtype=torch.int32), 2)
        with pytest.raises(AssertionError, match="grow_pages got"):
            block.grow_pages(torch.tensor([4], dtype=torch.int32), 2)  # fits in the last page

    def test_clear(self):
        block = SharedBlock(torch.device("cpu"), page_size=4)
        block.grow_pages(torch.tensor([8, 12], dtype=torch.int32), 6)
        block.affine_storage_pair(
            0, num_layers=2, num_heads=2, d_k=4, d_v=4
        )
        pages = block.clear()
        assert pages == [8, 12]
        assert block.num_tokens == 0 and block.num_pages == 0
        assert block.linear_affine_storage is None

    def test_unique_block_ids(self):
        b1 = SharedBlock(torch.device("cpu"))
        b2 = SharedBlock(torch.device("cpu"))
        assert b1.block_id != b2.block_id

    def test_append_affines_remain_in_destination_owned_slab(self):
        layers, heads, dim = 2, 2, 4
        session = SharedCacheSession.__new__(SharedCacheSession)
        session.sc_gdn = SimpleNamespace(
            num_linear_layers=layers,
            num_heads=heads,
            head_k_dim=dim,
            head_v_dim=dim,
        )
        left = SharedBlock(torch.device("cpu"))
        right = SharedBlock(torch.device("cpu"))
        left.num_tokens = right.num_tokens = 1
        left.token_ids = [11]
        right.token_ids = [22]

        generator = torch.Generator().manual_seed(711)
        for layer_idx in range(layers):
            for block in (left, right):
                A, B = block.affine_storage_pair(
                    layer_idx,
                    num_layers=layers,
                    num_heads=heads,
                    d_k=dim,
                    d_v=dim,
                )
                A.copy_(torch.randn(A.shape, generator=generator))
                B.copy_(torch.randn(B.shape, generator=generator))
                block.set_linear_affine(layer_idx, (A, B))

        expected = {
            layer_idx: compose_gdn_affines(
                A_first=left.linear_affine[layer_idx][0].clone(),
                B_first=left.linear_affine[layer_idx][1].clone(),
                A_second=right.linear_affine[layer_idx][0],
                B_second=right.linear_affine[layer_idx][1],
            )
            for layer_idx in range(layers)
        }
        left_storage = left.linear_affine_storage
        assert left_storage is not None

        session._finish_block_merge(
            destination=left,
            left=left,
            right=right,
            left_span=1,
            right_span=1,
            keep_left_state=True,
        )

        assert left.linear_affine_storage is left_storage
        for layer_idx, (expected_A, expected_B) in expected.items():
            actual_A, actual_B = left.linear_affine[layer_idx]
            assert actual_A.untyped_storage().data_ptr() == left_storage.untyped_storage().data_ptr()
            assert actual_B.untyped_storage().data_ptr() == left_storage.untyped_storage().data_ptr()
            torch.testing.assert_close(actual_A, expected_A)
            torch.testing.assert_close(actual_B, expected_B)


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

    def test_prefill_extends_non_empty_block(self, engine_and_session):
        """Prefilling a non-empty block appends: two prefills of a split prompt
        must land the same tokens, KV and logits as one prefill of the whole."""
        engine, session = engine_and_session
        ids = _encode("The capital of France is a city that", E2E_MODEL_PATH)
        cut = len(ids) // 2
        assert cut > 0 and cut < len(ids)

        whole = session.create_block()
        ref_logits = session.prefill_block(whole, ids)

        split = session.create_block()
        session.prefill_block(split, ids[:cut])
        assert split.num_tokens == cut
        logits = session.prefill_block(split, ids[cut:])

        assert split.num_tokens == len(ids)
        assert split.token_ids == ids.tolist()

        # Layer-0 key parity: the appended keys are rotated at the same
        # block-relative positions the one-shot prefill used.
        k_pool = engine.kv_cache.k_cache(0)
        k_flat = k_pool.reshape(-1, *k_pool.shape[2:])

        def keys_of(blk):
            return k_flat[blk.token_slots_tensor().to(torch.int64)].float()

        kv_diff = (keys_of(split) - keys_of(whole)).abs().max().item()
        assert kv_diff < 1e-2, f"extended keys differ from one-shot prefill: {kv_diff}"

        diff = (logits[0] - ref_logits[0]).abs().max().item()
        print(f"\n[extend prefill] max |logit diff| vs one-shot prefill = {diff}")
        assert diff < 1e-2, f"extension logits differ from one-shot prefill: {diff}"

        session.free_block(whole)
        session.free_block(split)

    def test_prefill_extends_non_empty_block_in_context(self, engine_and_session):
        """Same, with a context view: extending in context must match one
        context prefill of the concatenated tokens."""
        engine, session = engine_and_session
        prompt_ids = _encode("Once upon a time", E2E_MODEL_PATH)
        tail_ids = _encode(" there was a small village near the sea", E2E_MODEL_PATH)
        cut = len(tail_ids) // 2

        prompt = session.create_block()
        session.prefill_block(prompt, prompt_ids)

        whole = session.create_block()
        ref_logits = session.prefill_block(whole, tail_ids, context=[prompt])

        split = session.create_block()
        session.prefill_block(split, tail_ids[:cut], context=[prompt])
        logits = session.prefill_block(split, tail_ids[cut:], context=[prompt])

        assert split.num_tokens == len(tail_ids)
        diff = (logits[0] - ref_logits[0]).abs().max().item()
        print(f"\n[extend context prefill] max |logit diff| vs one-shot = {diff}")
        assert diff < 1e-2, f"in-context extension differs from one-shot prefill: {diff}"

        for blk in (prompt, whole, split):
            session.free_block(blk)

    def test_prefill_rejects_write_block_in_context(self, engine_and_session):
        """A non-empty write block listed in its own context would be counted
        twice (context segment + causal self segment)."""
        engine, session = engine_and_session
        ids = _encode("The capital of France is", E2E_MODEL_PATH)

        prompt = session.create_block()
        session.prefill_block(prompt, ids)
        blk = session.create_block()
        # A fresh block in the view is filtered out (empty) -- still allowed.
        session.prefill_block(blk, ids, context=[prompt, blk])
        with pytest.raises(ValueError, match="must not appear in context"):
            session.prefill_block(blk, ids, context=[prompt, blk])

        session.free_block(prompt)
        session.free_block(blk)

    def test_prefill_extend_matches_decode_growth(self, engine_and_session):
        """An extension leaves the same block state (tokens, pages, slots) that
        feeding the tokens through ``decode_step`` would.

        Numeric parity is asserted against the *one-shot prefill* of the whole
        sequence (see ``test_prefill_extends_non_empty_block``): the two batched
        kernels agree exactly, whereas lock-step decode -- query rotation plus
        per-segment LSE merging -- differs from any prefill kernel at bf16."""
        engine, session = engine_and_session
        prompt_ids = _encode("The largest planet in our solar system is", E2E_MODEL_PATH)
        fed = [11, 13, 17, 19]

        ref = session.create_block()
        session.prefill_block(ref, prompt_ids)
        group = WorkerGroup(cache_structure=[[ref]])
        for tok in fed:
            session.decode_step(group, torch.tensor([tok], dtype=torch.int32))

        ext = session.create_block()
        session.prefill_block(ext, prompt_ids)
        session.prefill_block(ext, torch.tensor(fed, dtype=torch.int32))

        assert ext.num_tokens == ref.num_tokens == len(prompt_ids) + len(fed)
        assert ext.num_pages == ref.num_pages
        assert ext.last_page_len == ref.last_page_len
        assert ext.token_ids == ref.token_ids
        # slots are page-packed in both cases (different pages, same offsets)
        page_size = session.page_size
        assert [s % page_size for s in ext.token_slots_tensor().tolist()] == [
            s % page_size for s in ref.token_slots_tensor().tolist()
        ]

        session.free_block(ref)
        session.free_block(ext)


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
