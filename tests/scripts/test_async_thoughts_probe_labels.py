"""Exercise the Llama and Qwen readiness labels through the demo's normal loop."""

import asyncio

import pytest
from minisgl_examples.demos.async_thoughts import demo

from tests.scripts.test_async_thoughts_demo import FakeLLM


def run_probe(monkeypatch, llama, scores, multitoken=(), *, standalone=False):
    class ScriptedLLM(FakeLLM):
        async def forward(self, *args, **kwargs):
            out = await super().forward(*args, **kwargs)
            is_start = kwargs.get("write_to") is not None and kwargs.get("return_logits", True)
            if out.logits is not None:
                if is_start == standalone:
                    out.logits.fill_(-100)
                    for text, score in scores.items():
                        out.logits[self.tokenizer.vocab[text]] = score
                elif is_start:
                    # A checkpoint can lack a lowercase yes while providing a one-token Yes.
                    out.logits[self.tokenizer.vocab["Yes"]] = 10.0
            return out

    # Grant the initial admission before the script overrides the live logits.
    llm = ScriptedLLM([True] * 100, writer=("one", "two", "three", "<|im_end|>"))
    tokenizer = llm.tokenizer
    for text in scores:
        tokenizer.encode(text)
    original_encode = tokenizer.encode

    def encode(text, **kwargs):
        if text in multitoken:
            return [50, 51]
        return original_encode(text, **kwargs)

    tokenizer.encode = encode
    if llama:
        llm.use_llama("problem")

    observed = []
    method = "check_start_writing" if standalone else "check_continue_writing"
    original_check = getattr(demo.ModeSwitchProbe, method)

    async def check(self, *args):
        result = await original_check(self, *args)
        observed.append(result)
        return result

    monkeypatch.setattr(demo.ModeSwitchProbe, method, check)
    config = demo.DemoConfig(max_steps=16, max_answer_tokens=4, probe_period=1)
    asyncio.run(demo._run_loop(config, llm, tokenizer, llm.prompting))
    assert observed
    return observed[0]


@pytest.mark.parametrize(
    "llama,expected", [(True, (False, 24.25, 24.75)), (False, (True, 20.125, 18.875))]
)
def test_capitalized_no_wins_only_for_llama(monkeypatch, llama, expected):
    # Tülu MATH-500 row 7, direct first-probe calibration. The full-vocab greedy token is No.
    scores = {"yes": 20.125, "no": 18.875, "Yes": 24.25, "No": 24.75}
    assert run_probe(monkeypatch, llama, scores) == expected


def test_standalone_admission_uses_semantic_label_logits(monkeypatch):
    scores = {"yes": 20.125, "no": 18.875, "Yes": 24.25, "No": 24.75}
    assert run_probe(monkeypatch, True, scores, standalone=True) == (False, 24.25, 24.75)


@pytest.mark.parametrize("winner", ["YES", " NO", " Yes", " no"])
def test_llama_accepts_single_token_label_spellings(monkeypatch, winner):
    scores = {"yes": 0.0, "no": 4.0, winner: 6.0}
    yes = winner.strip().lower() == "yes"
    assert run_probe(monkeypatch, True, scores) == ((True, 6.0, 4.0) if yes else (False, 0.0, 6.0))


def test_llama_ignores_multitoken_labels_and_uses_surviving_spelling(monkeypatch):
    scores = {"yes": 20.0, "no": 1.0, "Yes": 3.0, " No": 30.0}
    assert run_probe(monkeypatch, True, scores, multitoken={"yes", " No"}) == (True, 3.0, 1.0)


def test_llama_rejects_missing_single_token_label(monkeypatch):
    with pytest.raises(RuntimeError, match="single-token.*yes|yes.*single-token"):
        run_probe(
            monkeypatch, True, {"no": 1.0}, multitoken={"yes", "Yes", "YES", " yes", " Yes", " YES"}
        )


@pytest.mark.parametrize("llama,expected", [(True, True), (False, False)])
def test_tie_uses_model_argmax_order_only_for_llama(monkeypatch, llama, expected):
    # Fixture vocabulary assigns yes a lower token ID than no, so torch.argmax picks yes.
    assert run_probe(monkeypatch, llama, {"yes": 2.0, "no": 2.0}) == (expected, 2.0, 2.0)


def test_long_standalone_question_is_recorded_as_an_isolated_prefill():
    from minisgl.training import RecordedAsyncLLM

    from tests.core.test_async_llm import _make_llm
    from tests.scripts.test_async_thoughts_demo import Tokenizer

    async def run():
        llm, _ = _make_llm()
        live = []
        ids = [5, 6, 7] * 90
        try:
            with RecordedAsyncLLM(llm) as recorder:
                for _ in range(3):
                    live.append(await recorder.create_block())
                for i in range(3):
                    await recorder.forward([i + 1, i + 2], live[: i + 1], return_logits=False)
                original = [b.token_ids.copy() for b in live]
                probe = demo.ModeSwitchProbe(
                    recorder, Tokenizer(), "prefix", "question", "yes", "no", label_variants=True
                )
                try:
                    await probe.check_start_writing(ids)
                finally:
                    await probe.close()
                assert [b.token_ids for b in live] == original
                trace = recorder.get_trace()
            batch = trace.batches[-1]
            assert batch.kind == "prefill" and len(batch.requests) == 1
            request = batch.requests[0]
            assert list(request.input_ids) == ids
            assert request.write_to not in {b.block_id for b in live}
            assert request.cache_view == ((request.write_to, 0),)
            assert trace.to("cpu").tokens[-1].view == ((request.write_to, len(ids)),)
        finally:
            for block in reversed(live):
                await llm.free_block(block)
            await llm.close()

    asyncio.run(run())
