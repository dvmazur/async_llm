"""Native baseline sampling and metric accounting on the real CPU scheduler."""

import asyncio
from contextlib import nullcontext

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.training import RecordedAsyncLLM
from minisgl_examples.training.parallel_thinking.native import run_native

from tests.core.test_async_llm import _make_llm


@pytest.mark.parametrize(
    "budget, context_cap, eos_ids, count, stop",
    [
        (4, 12, [6, 31], 2, "eos"),
        (1, 12, [6, 31], 1, "max_steps"),
        (4, 4, [6, 31], 1, "context_limit"),
        (4, 3, [6, 31], 0, "context_limit"),
    ],
)
@pytest.mark.parametrize("capture_trace", [False, True])
def test_native_baseline_preserves_chat_sampling_and_counts(budget, context_cap, eos_ids, count, stop, monkeypatch, capture_trace):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())
    problem = "Solve the problem and box the answer.\n\nWhat is one half?"

    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            assert messages == [{"role": "user", "content": problem}]
            assert options == {
                "tokenize": False, "add_generation_prompt": True,
                "enable_thinking": True, "reasoning_effort": "xhigh",
            }
            return "native chat"

        def encode(self, text, **options):
            assert text == "native chat" and options == {"add_special_tokens": False}
            return [3, 1, 2]

        def decode(self, tokens, **options):
            assert options == {"skip_special_tokens": True}
            assert tokens == [4, 6][:count]
            return r"\boxed{\frac{1}{2}}" if count == 2 else ""

    async def check():
        llm, session = _make_llm()
        llm.tokenizer = Tokenizer()
        sampling = SamplingParams(temperature=0.0, max_tokens=1)
        generate = llm.async_generate

        def checked_generate(context, *, sampling_params, max_steps, **kwargs):
            assert sampling_params is sampling and len(context.cache_view) == 1
            return generate(context, sampling_params=sampling_params, max_steps=max_steps, **kwargs)

        llm.async_generate = checked_generate
        try:
            with (RecordedAsyncLLM(llm) if capture_trace else nullcontext(llm)) as model:
                result = await run_native(
                    model, problem=problem, max_steps=budget, sampling=sampling,
                    eos_token_ids=eos_ids, max_context_tokens=context_cap, capture_trace=capture_trace,
                )
            if capture_trace:
                trace = result["trace"]
                assert trace["prompt_ids"] == [3, 1, 2]
                assert trace["worker_token_ids"] == {"Qwen": [4, 6][:count]}
                assert trace["publications"] == [] and trace["finisher"]["sampled_token_ids"] == []
                executed = model.get_trace().tokens
                # The final sample was never fed back to the engine.
                assert [t.token_id for t in executed] == ([3, 1, 2, 4][:2 + count] if count else [])
                assert {t.block for t in executed} == ({trace["block_ids"]["workers"]["Qwen"]} if count else set())
                assert trace["eos_token_id"] == (6 if stop == "eos" else None)
            else:
                assert "trace" not in result
            assert all(call["block"].num_tokens == 0 for call in session.prefill_calls)
            assert len(session.decode_calls) == count
            assert result["stop_reason"] == stop
            assert result["eos_token_id"] == (6 if stop == "eos" else None)
            assert result["boxed_answer"] == (r"\frac{1}{2}" if count == 2 else None)
            assert result["total_generated_tokens"] == result["reasoning_generated_tokens"] == count
            assert result["generated_tokens"] == {"Qwen": count}
            assert result["decode_rounds"] == count and result["peak_context_tokens"] == 3 + count
            assert result["answer_generated_tokens"] == result["completed_steps"] == 0
            assert not result["used_finisher"] and result["finisher_stop_reason"] == "not_applicable"
        finally:
            await llm.close()

    with torch.device("cpu"):
        asyncio.run(check())
