"""Exercise the Llama and Qwen readiness labels through the demo's normal loop."""

import asyncio

import pytest
from minisgl_examples.demos.async_thoughts import demo

from tests.scripts.test_async_thoughts_demo import FakeLLM


def run_probe(monkeypatch, llama, scores, multitoken=()):
    class ScriptedLLM(FakeLLM):
        async def forward(self, *args, **kwargs):
            out = await super().forward(*args, **kwargs)
            if out.logits is not None:
                out.logits.fill_(-100)
                for text, score in scores.items():
                    out.logits[self.tokenizer.vocab[text]] = score
            return out

    llm = ScriptedLLM(writer=("<|im_end|>",))
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
        tokenizer.apply_chat_template = lambda messages, **kwargs: "".join(
            message["content"] for message in messages
        )
        llm.prompting = demo.LlamaPrompting("problem", tokenizer)
        llm.question_ids = tokenizer.encode(llm.prompting.mode_switching_question)

    observed = []
    original_check = demo.ModeSwitchProbe.check_continue_writing

    async def check(self, *args):
        result = await original_check(self, *args)
        observed.append(result)
        return result

    monkeypatch.setattr(demo.ModeSwitchProbe, "check_continue_writing", check)
    config = demo.DemoConfig(max_steps=12, max_answer_tokens=1, probe_period=1)
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
