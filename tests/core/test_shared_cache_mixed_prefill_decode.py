"""
Mixed prefill + decode behaviour of the shared-cache (async-reasoning)
attention (``minisgl.shared_cache``).

These tests answer a specific question: *does the shared-cache attention work
with a batch that mixes prefill and decode rows?*

Findings the tests encode
-------------------------
1. The shared-cache op is **single-phase per forward**.  A
   ``SharedCacheAttnMetadata`` carries one scalar ``phase``
   (``"decode"`` | ``"context_prefill"``) and ``SharedCacheAttention.forward``
   dispatches the *entire* q-tensor on it.  There is no constructor that emits
   a metadata mixing the two, so a single shared-cache forward cannot serve
   prefill rows and decode rows together.  (``test_forward_is_single_phase`` —
   no GPU needed.)

2. A mixed *workload* — some workers decoding while a fresh worker is
   prefilled in-context — is handled correctly when the two ops run as
   **separate** shared-cache passes.  Each pass reproduces, to bf16 tolerance,
   a plain monolithic causal prefill of the equivalent concatenated sequence.
   (``test_decode_matches_monolithic_oracle``,
   ``test_context_prefill_matches_monolithic_oracle``,
   ``test_mixed_workload_serialized_matches_oracle``.)

Oracle
------
The reference is the session's own plain (standard causal) prefill of the fully
concatenated token sequence — the same head-to-head philosophy as
``test_shared_cache_async_reasoning_oracle.py``, but self-contained (no HF
transformers dependency).  bf16 rounding makes raw logits differ by ~0.2 in
absolute terms, so correctness is asserted on argmax + cosine similarity, not
elementwise equality.

Run
---
    # defaults to Qwen/Qwen2.5-0.5B, downloads if absent:
    uv run pytest tests/core/test_shared_cache_mixed_prefill_decode.py -v -s

    # pick a different model / page size:
    MINISGL_E2E_MODEL=meta-llama/Llama-3.2-1B MINISGL_TEST_PAGE_SIZE=4 \
        uv run pytest tests/core/test_shared_cache_mixed_prefill_decode.py -v -s

Like the other shared-cache e2e files, this must run in its own pytest
invocation (``Engine.__init__`` asserts CUDA is not yet initialised).
"""

from __future__ import annotations

import os
from typing import List

import pytest
import torch

# ---------------------------------------------------------------------------
# Structural test — no model or GPU required.
# ---------------------------------------------------------------------------


def test_mixed_api_present():
    """A single shared-cache forward *can* now mix prefill and decode rows.

    (This replaced an earlier contract test that pinned the single-phase
    limitation; mixed support lifts it -- the e2e oracles below verify the
    numerics.)  ``prepare_mixed`` builds one metadata whose ``phase="mixed"``
    routes decode segments to the decode/aux wrappers and prefill segments to
    the prefill wrappers through a single LSE merge.
    """
    from minisgl.shared_cache import PrefillRequest
    from minisgl.shared_cache.attention import SharedCacheAttention, SharedCacheAttnMetadata

    assert hasattr(SharedCacheAttention, "prepare_mixed")
    assert "mixed" in SharedCacheAttnMetadata.__dataclass_fields__["phase"].type
    # PrefillRequest is the public descriptor consumed by mixed_step.
    assert set(PrefillRequest.__dataclass_fields__) >= {"block", "input_ids", "context"}


# ---------------------------------------------------------------------------
# End-to-end oracle tests — require a model + CUDA.
# ---------------------------------------------------------------------------

E2E_MODEL_PATH = os.environ.get("MINISGL_E2E_MODEL", "Qwen/Qwen2.5-0.5B")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="e2e shared-cache tests need CUDA",
)

# argmax must match exactly; cosine guards the whole distribution.  Raw bf16
# logits differ by ~0.2 abs, so we do NOT assert elementwise closeness.
_COS_TOL = 0.999


def _build_engine(model_path: str):
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig

    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=8,
        cuda_graph_bs=[2, 4],
        cuda_graph_max_bs=4,
        page_size=int(os.environ.get("MINISGL_TEST_PAGE_SIZE", "1")),
        memory_ratio=0.6,
        max_seq_len_override=2048,
    )
    return Engine(config)


@pytest.fixture(scope="module")
def sess():
    if not torch.cuda.is_available():
        pytest.skip("e2e tests need CUDA")
    from minisgl.shared_cache import SharedCacheSession

    engine = _build_engine(E2E_MODEL_PATH)
    session = SharedCacheSession(engine)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(E2E_MODEL_PATH)
    yield engine, session, tok
    engine.shutdown()


def _enc(tok, text: str) -> torch.Tensor:
    return tok.encode(text, return_tensors="pt").view(-1).to(torch.int32)


def _monolithic_last_logits(session, ids: torch.Tensor) -> torch.Tensor:
    """Oracle: plain causal prefill of `ids`; last-token logits [vocab] (cpu/f32)."""
    blk = session.create_block()
    try:
        return session.prefill_block(blk, ids)[0].float().cpu()
    finally:
        session.free_block(blk)


def _assert_matches(got: torch.Tensor, want: torch.Tensor, label: str) -> None:
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0).item()
    print(
        f"\n[{label}] argmax got={got.argmax().item()} want={want.argmax().item()} "
        f"cos={cos:.5f} maxabs={(got - want).abs().max().item():.4f}"
    )
    assert got.argmax().item() == want.argmax().item(), f"{label}: argmax diverged"
    assert cos > _COS_TOL, f"{label}: cosine {cos} <= {_COS_TOL}"


@requires_cuda
class TestSharedCacheMixedPrefillDecode:
    def test_decode_matches_monolithic_oracle(self, sess):
        """A decode step reproduces a monolithic prefill of prompt+prefix+token."""
        from minisgl.shared_cache import WorkerGroup

        engine, session, tok = sess
        p_ids = _enc(tok, "The capital of France is Paris. The capital of Japan is")
        a_prefix = _enc(tok, " Tokyo, and the")
        a_next = _enc(tok, " capital")
        assert a_next.numel() == 1

        prompt = session.create_block()
        session.prefill_block(prompt, p_ids)
        A = session.create_block()
        session.prefill_block(A, a_prefix, context=[prompt])
        try:
            group = WorkerGroup(cache_structure=[[prompt, A]], write_to=[A])
            got = session.decode_step(group, a_next)[0].float().cpu()
            want = _monolithic_last_logits(session, torch.cat([p_ids, a_prefix, a_next]))
            _assert_matches(got, want, "decode")
        finally:
            session.free_block(A)
            session.free_block(prompt)

    def test_context_prefill_matches_monolithic_oracle(self, sess):
        """A context-prefill reproduces a monolithic prefill of prompt+prefix."""
        engine, session, tok = sess
        p_ids = _enc(tok, "The capital of France is Paris. The capital of Japan is")
        b_prefix = _enc(tok, " Beijing is the capital of")

        prompt = session.create_block()
        session.prefill_block(prompt, p_ids)
        B = session.create_block()
        try:
            got = session.prefill_block(B, b_prefix, context=[prompt])[0].float().cpu()
            want = _monolithic_last_logits(session, torch.cat([p_ids, b_prefix]))
            _assert_matches(got, want, "context_prefill")
        finally:
            session.free_block(B)
            session.free_block(prompt)

    def test_mixed_workload_serialized_matches_oracle(self, sess):
        """The mixed scenario — worker A mid-decode while a fresh worker B is
        prefilled in-context — is correct when the two shared-cache ops run as
        separate passes.  Both rows match their monolithic oracle.

        This is the *serialized* baseline; ``TestMixedStep`` verifies the same
        workload run in a single forward via ``mixed_step`` and cross-checks the
        two against each other.
        """
        from minisgl.shared_cache import WorkerGroup

        engine, session, tok = sess
        p_ids = _enc(tok, "Water boils at 100 degrees Celsius at sea level.")
        a_prefix = _enc(tok, " The freezing point of")
        a_next = _enc(tok, " water")
        assert a_next.numel() == 1
        b_prefix = _enc(tok, " Ice is frozen")

        prompt = session.create_block()
        session.prefill_block(prompt, p_ids)
        A = session.create_block()
        session.prefill_block(A, a_prefix, context=[prompt])
        B = session.create_block()
        try:
            # Decode pass for the in-flight worker A.
            group = WorkerGroup(cache_structure=[[prompt, A]], write_to=[A])
            a_logits = session.decode_step(group, a_next)[0].float().cpu()

            # Prefill pass for the freshly-admitted worker B.
            b_logits = session.prefill_block(B, b_prefix, context=[prompt])[0].float().cpu()

            a_want = _monolithic_last_logits(session, torch.cat([p_ids, a_prefix, a_next]))
            b_want = _monolithic_last_logits(session, torch.cat([p_ids, b_prefix]))
            _assert_matches(a_logits, a_want, "mixed/decode-row")
            _assert_matches(b_logits, b_want, "mixed/prefill-row")
        finally:
            session.free_block(B)
            session.free_block(A)
            session.free_block(prompt)


@requires_cuda
class TestMixedStep:
    """The single-forward mixed path: ``SharedCacheSession.mixed_step``."""

    def _build_decoder(self, session, tok, prompt, prefix_text: str, n_steps: int):
        """Prefill *prefix_text* into a fresh block in ``[prompt]`` then greedily
        decode *n_steps*.  Returns ``(block, next_token, cached_token_ids)`` where
        ``next_token`` is the token to feed next and ``cached_token_ids`` are the
        block's tokens so far (prefix + decoded, excluding ``next_token``)."""
        from minisgl.shared_cache import WorkerGroup

        prefix = _enc(tok, prefix_text)
        blk = session.create_block()
        logits = session.prefill_block(blk, prefix, context=[prompt])[0].float().cpu()
        toks = prefix.tolist()
        group = WorkerGroup(cache_structure=[[prompt, blk]], write_to=[blk])
        nxt = int(logits.argmax())
        for _ in range(n_steps):
            toks.append(nxt)
            logits = session.decode_step(
                group, torch.tensor([nxt], dtype=torch.int32)
            )[0].float().cpu()
            nxt = int(logits.argmax())
        return blk, nxt, toks

    def test_mixed_matches_oracle(self, sess):
        """Two in-flight decoders + two fresh prefills in one ``mixed_step``;
        every returned row matches an independent monolithic causal prefill."""
        from minisgl.shared_cache import PrefillRequest, WorkerGroup

        engine, session, tok = sess
        p_ids = _enc(tok, "The quick brown fox jumps over the lazy dog.")
        prompt = session.create_block()
        session.prefill_block(prompt, p_ids)

        A, a_next, a_toks = self._build_decoder(session, tok, prompt, " Meanwhile, the cat", 2)
        B, b_next, b_toks = self._build_decoder(session, tok, prompt, " In physics, energy", 1)
        C = session.create_block()
        c_ids = _enc(tok, " A list of primary colors:")
        D = session.create_block()
        d_ids = _enc(tok, " The year was")
        try:
            group = WorkerGroup(cache_structure=[[prompt, A], [prompt, B]], write_to=[A, B])
            dec, pf = session.mixed_step(
                group,
                torch.tensor([a_next, b_next], dtype=torch.int32),
                [PrefillRequest(C, c_ids, [prompt]), PrefillRequest(D, d_ids, [prompt])],
            )
            assert dec.shape[0] == 2 and pf.shape[0] == 2
            cat = torch.cat
            i32 = lambda xs: torch.tensor(xs, dtype=torch.int32)
            _assert_matches(
                dec[0].float().cpu(),
                _monolithic_last_logits(session, cat([p_ids, i32(a_toks + [a_next])])),
                "mixed/decode-0",
            )
            _assert_matches(
                dec[1].float().cpu(),
                _monolithic_last_logits(session, cat([p_ids, i32(b_toks + [b_next])])),
                "mixed/decode-1",
            )
            _assert_matches(
                pf[0].float().cpu(), _monolithic_last_logits(session, cat([p_ids, c_ids])),
                "mixed/prefill-0",
            )
            _assert_matches(
                pf[1].float().cpu(), _monolithic_last_logits(session, cat([p_ids, d_ids])),
                "mixed/prefill-1",
            )
        finally:
            for blk in (A, B, C, D, prompt):
                session.free_block(blk)

    def test_mixed_matches_serialized(self, sess):
        """``mixed_step`` equals ``decode_step`` + ``prefill_block`` run on
        identical (independently rebuilt) block state."""
        from minisgl.shared_cache import PrefillRequest, WorkerGroup

        engine, session, tok = sess
        p_ids = _enc(tok, "Water boils at 100 degrees Celsius at sea level.")
        prompt = session.create_block()
        session.prefill_block(prompt, p_ids)
        a_prefix = _enc(tok, " The freezing point of")
        a_next = _enc(tok, " water")
        assert a_next.numel() == 1
        b_ids = _enc(tok, " Ice is frozen solid")

        blocks = [prompt]
        try:
            # serialized
            A_s = session.create_block()
            session.prefill_block(A_s, a_prefix, context=[prompt])
            dec_ser = session.decode_step(
                WorkerGroup([[prompt, A_s]], [A_s]), a_next
            )[0].float().cpu()
            B_s = session.create_block()
            pf_ser = session.prefill_block(B_s, b_ids, context=[prompt])[0].float().cpu()

            # mixed, on freshly rebuilt identical state
            A_m = session.create_block()
            session.prefill_block(A_m, a_prefix, context=[prompt])
            B_m = session.create_block()
            dec_mix, pf_mix = session.mixed_step(
                WorkerGroup([[prompt, A_m]], [A_m]),
                a_next,
                [PrefillRequest(B_m, b_ids, [prompt])],
            )
            blocks += [A_s, B_s, A_m, B_m]

            _assert_matches(dec_mix[0].float().cpu(), dec_ser, "mixed==serialized/decode")
            _assert_matches(pf_mix[0].float().cpu(), pf_ser, "mixed==serialized/prefill")
        finally:
            for blk in blocks:
                session.free_block(blk)

    @pytest.mark.parametrize("W,n_pf,S", [(0, 1, 5), (1, 1, 1), (2, 1, 7), (1, 2, 3), (2, 2, 4)])
    def test_matrix(self, sess, W, n_pf, S):
        """Sweep decoder count / prefill count / prefill length; every row of the
        single mixed forward matches its monolithic oracle."""
        from minisgl.shared_cache import PrefillRequest, WorkerGroup

        engine, session, tok = sess
        p_ids = _enc(tok, "Once upon a time in a land far away there lived")
        prompt = session.create_block()
        session.prefill_block(prompt, p_ids)
        long_ids = _enc(tok, "the cat sat on the mat and looked around the room quite slowly")
        assert long_ids.numel() >= S

        created = [prompt]
        try:
            decoders = [
                self._build_decoder(session, tok, prompt, f" chapter {w} opens with", 1)
                for w in range(W)
            ]
            created += [blk for blk, _, _ in decoders]
            group = (
                WorkerGroup([[prompt, blk] for blk, _, _ in decoders],
                            [blk for blk, _, _ in decoders])
                if W else None
            )
            dec_ids = (
                torch.tensor([nxt for _, nxt, _ in decoders], dtype=torch.int32) if W else None
            )
            reqs, pf_ids = [], []
            for _ in range(n_pf):
                blk = session.create_block()
                created.append(blk)
                ids = long_ids[:S].clone()
                reqs.append(PrefillRequest(blk, ids, [prompt]))
                pf_ids.append(ids)

            dec, pf = session.mixed_step(group, dec_ids, reqs)

            assert dec.shape[0] == W and pf.shape[0] == n_pf
            for w, (_, nxt, toks) in enumerate(decoders):
                _assert_matches(
                    dec[w].float().cpu(),
                    _monolithic_last_logits(
                        session, torch.cat([p_ids, torch.tensor(toks + [nxt], dtype=torch.int32)])
                    ),
                    f"matrix[W={W},n_pf={n_pf},S={S}]/decode-{w}",
                )
            for k, ids in enumerate(pf_ids):
                _assert_matches(
                    pf[k].float().cpu(),
                    _monolithic_last_logits(session, torch.cat([p_ids, ids])),
                    f"matrix[W={W},n_pf={n_pf},S={S}]/prefill-{k}",
                )
        finally:
            for blk in created:
                session.free_block(blk)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
