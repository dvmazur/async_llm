"""Driver smoke test for the scheduler's ``SharedCacheService``.

Runs the async-reasoning ``ReasoningDriver`` (from the ``async_thoughts``
scripts package) against ``llm.shared_cache_service`` end to end, checking the
streaming callbacks fire and the borrowed-page accounting returns to zero.

(Before the legacy ``SharedCacheSession`` was retired, this file also held the
token-parity gate that ran the same driver against both backends on one engine
and asserted identical thinker/writer token ids and probe decisions.)

Requires ``MINISGL_E2E_MODEL`` (an HF model path) + CUDA, e.g.::

    MINISGL_E2E_MODEL=Qwen/Qwen2.5-0.5B uv run pytest tests/core/test_shared_cache_service.py -v -s
"""

from __future__ import annotations

import os
from typing import List

import pytest
import torch

E2E_MODEL_PATH = os.environ.get("MINISGL_E2E_MODEL", "")

requires_e2e = pytest.mark.skipif(
    not E2E_MODEL_PATH or not torch.cuda.is_available(),
    reason="Set MINISGL_E2E_MODEL to an HF model path and ensure CUDA is available",
)


@pytest.fixture(scope="module")
def shared_llm():
    if not E2E_MODEL_PATH or not torch.cuda.is_available():
        pytest.skip("e2e tests disabled")
    from minisgl.llm import LLM

    llm = LLM(
        E2E_MODEL_PATH,
        page_size=int(os.environ.get("MINISGL_TEST_PAGE_SIZE", "1")),
        memory_ratio=0.7,
        max_running_req=4,
        cuda_graph_bs=[1, 2],
        cuda_graph_max_bs=2,
        max_seq_len_override=4096,
    )
    yield llm
    llm.engine.shutdown()


@requires_e2e
def test_driver_smoke_on_service(shared_llm):
    """Drive a short reasoning chain end to end with streaming callbacks."""
    pytest.importorskip("async_thoughts")
    llm = shared_llm

    from async_thoughts.driver import ReasoningConfig, ReasoningDriver

    thinker_chunks: List[str] = []
    state_changes: List[str] = []
    driver = ReasoningDriver(
        backend=llm.shared_cache_service,
        tokenizer=llm.tokenizer,
        config=ReasoningConfig(max_steps=120, probe_period=30),
        device=llm.engine.device,
        on_thinker_token=thinker_chunks.append,
        on_state_change=state_changes.append,
    )
    result = driver.run()

    assert len(result["thinker_tokens"]) > 10
    assert thinker_chunks, "thinker stream callback never fired"
    assert llm.cache_manager.borrowed_block_pages == 0
    llm.cache_manager.check_integrity()

    print(f"\n[smoke] writer text: {result['writer_text']!r}")


@requires_e2e
def test_driver_deterministic_across_runs(shared_llm):
    """Two identical runs on one engine must produce identical tokens."""
    pytest.importorskip("async_thoughts")
    llm = shared_llm

    from async_thoughts.driver import ReasoningConfig, ReasoningDriver

    def run_once():
        return ReasoningDriver(
            backend=llm.shared_cache_service,
            tokenizer=llm.tokenizer,
            config=ReasoningConfig(max_steps=60, probe_period=30),
            device=llm.engine.device,
        ).run()

    first, second = run_once(), run_once()
    assert first["thinker_tokens"] == second["thinker_tokens"]
    assert first["writer_tokens"] == second["writer_tokens"]
    assert first["probe_decisions"] == second["probe_decisions"]
    assert llm.cache_manager.borrowed_block_pages == 0
    llm.cache_manager.check_integrity()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
