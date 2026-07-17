"""
E2E smoke test for the async-thoughts demo (``scripts/async_thoughts``) on the
asyncio API.  Requires ``MINISGL_E2E_MODEL`` + CUDA; run in its own process
(the demo builds its own engine)::

    MINISGL_E2E_MODEL=Qwen/Qwen3-0.6B pytest tests/core/test_async_thoughts_demo.py -v

Numeric equivalence of the underlying scheduling (concurrent streams + probe
vs lock-step group stepping) is pinned in ``test_async_llm.py``; this test
checks the demo's policy loop end-to-end: it terminates, both streams produce
tokens, and no forbidden markers leak into the generated text.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest
import torch

E2E_MODEL_PATH = os.environ.get("MINISGL_E2E_MODEL", "")

requires_e2e = pytest.mark.skipif(
    not E2E_MODEL_PATH or not torch.cuda.is_available(),
    reason="Set MINISGL_E2E_MODEL to an HF model path and ensure CUDA is available",
)

_SCRIPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "async_thoughts")
)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


@requires_e2e
def test_demo_smoke():
    from async_thoughts.demo import DemoConfig, Prompting, _run_demo
    from async_thoughts.engine import build_async_llm, encode, vocab_id_or_none
    from transformers import AutoTokenizer

    config = DemoConfig(
        model=E2E_MODEL_PATH,
        max_steps=25,
        probe_period=8,
        memory_ratio=0.7,
    )
    prompting = Prompting(config.problem)
    llm = build_async_llm(config.model, memory_ratio=config.memory_ratio)
    thinker_tokens, writer_tokens = asyncio.run(_run_demo(config, llm, prompting))

    tokenizer = AutoTokenizer.from_pretrained(E2E_MODEL_PATH, trust_remote_code=True)
    # Generated parts: everything past the prefilled prefix (the first
    # generated entry is the "\n\n" seed).
    thinker_gen = thinker_tokens[len(encode(prompting.thinker_output_prefix, tokenizer)) :]
    writer_gen = writer_tokens[len(encode(prompting.writer_output_prefix, tokenizer)) :]

    # The thinker ran to max_steps: seed + one token per step.
    assert len(thinker_gen) >= config.max_steps
    assert len(writer_gen) >= 1

    # Forbidden boundary markers never leak into the generated parts.
    forbidden = {
        i
        for i in (
            vocab_id_or_none(tokenizer, n) for n in ("</think>", "<|im_start|>", "<|endoftext|>")
        )
        if i is not None
    }
    assert not forbidden & set(thinker_gen)
    assert not forbidden & set(writer_gen)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
