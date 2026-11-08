"""Token and cache witnesses for native-turn Llama AsyncReasoning."""

import asyncio
from contextlib import nullcontext

import pytest
import torch
from minisgl.training import RecordedAsyncLLM, replay_trace
from minisgl_examples.demos.async_thoughts import demo
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from tests.core.test_async_llm import _make_llm
from tests.training.helpers import copy_operations


def native_tokenizer():
    tokens = ["[unk]", "<user>", "<assistant>", "<system>", "<end>", "yes", "no", "task"]
    tokens += [str(i) for i in range(32 - len(tokens))]
    backend = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[unk]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[unk]", eos_token="<end>")
    tok.add_special_tokens({"additional_special_tokens": ["<user>", "<assistant>", "<system>"]})
    tok.chat_template = (
        "{% for m in messages %}{{ '<' + m.role + '>' + m.content + '<end>' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    )
    return tok


def test_native_turns_preserve_complete_tokenized_history():
    tok = native_tokenizer()
    prompt = demo.LlamaPrompting("task", tok, protocol="native-chat")
    native = prompt.native_turns
    assert native.prompt_ids == tok.encode(prompt.input_prompt, add_special_tokens=False)
    # Build the independent expected continuation with the trusted tokenized template.
    followup = {
        "role": "user",
        "content": "The preceding response is a partial draft. Complete the original task, "
        "checking the draft's reasoning and following the original instructions.",
    }
    full = tok.apply_chat_template(
        [
            {
                "role": "user",
                "content": "task\n\nWork through the problem in your working notes. In the final answer, answer the original question directly.",
            },
            {"role": "assistant", "content": "yes"},
            followup,
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert native.prompt_ids + tok.encode(
        "yes", add_special_tokens=False
    ) + native.writer_ids == tok.encode(full, add_special_tokens=False)
    assert demo.LlamaPrompting("task", tok).native_turns is None
    with pytest.raises(ValueError, match="Unknown Llama protocol"):
        demo.LlamaPrompting("task", tok, protocol="typo")


def test_native_ordinary_thinker_and_concurrent_writer(monkeypatch):
    torch.manual_seed(7)
    tok = native_tokenizer()
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        eos_token_id=None,
    )
    model = LlamaForCausalLM(config).eval()
    prompt = demo.LlamaPrompting("task", tok, protocol="native-chat")
    native = prompt.native_turns
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())
    # Fix admission to YES to exercise genuine joint decodes independently of tiny model quality.
    monkeypatch.setattr(demo.ModeSwitchProbe, "_vote", lambda self, logits: (True, 1.0, 0.0))

    async def run():
        llm, session = _make_llm()
        copy_operations(session)
        llm.tokenizer = tok
        llm.eos_token_ids = frozenset()
        stub_decode = session.decode_step

        def decode(group, inputs):
            logits = stub_decode(group, inputs)
            for i, view in enumerate(group.cache_structure):
                if len(view) == 2:
                    ids = [t for block in view for t in block.token_ids]
                    logits[i] = model(torch.tensor([ids])).logits[0, -1]
                elif len(view) == 3:
                    logits[i].fill_(-100)
                    logits[i, 13] = 1
            return logits

        session.decode_step = decode
        try:
            with RecordedAsyncLLM(llm) as recorder:
                result = await demo._run_loop(
                    demo.DemoConfig(max_steps=28, max_answer_tokens=6, probe_period=3),
                    recorder,
                    tok,
                    prompt,
                )
            return result, recorder.get_trace()
        finally:
            await llm.close()

    with torch.inference_mode():
        (thoughts, answer), trace = asyncio.run(run())
        native_input = torch.tensor([native.prompt_ids])
        ordinary = model.generate(
            native_input,
            attention_mask=torch.ones_like(native_input),
            max_new_tokens=len(thoughts),
            do_sample=False,
            pad_token_id=0,
        )
        assert ordinary[0, native_input.shape[1] :].tolist() == thoughts
        requests = [(batch, req) for batch in trace.batches for req in batch.requests]
        first = requests[0][1]
        assert list(first.input_ids) == native.prompt_ids[:-1]
        # The sole two-block writer is the native thinker; compare the actual recorded
        # replay logits with the ordinary causal sequence, including the first sample.
        thinker = next(
            r.write_to for b, r in requests if b.kind == "decode" and len(r.cache_view) == 2
        )
        rows = []
        thinker_inputs = []
        row = 0
        for b, r in requests:
            if r.write_to == thinker:
                rows.append(row)
                thinker_inputs.extend(r.input_ids)
            row += len(r.input_ids)
        assert thinker_inputs[0] == native.prompt_ids[-1]
        replay = replay_trace(model, trace, rows=rows, hidden_only=False)
        serial_input = torch.tensor([native.prompt_ids[:-1] + thinker_inputs])
        serial = model(serial_input).logits[0, -len(rows) :]
        torch.testing.assert_close(replay, serial, rtol=1e-4, atol=1e-5)

    bridge = next(req for batch, req in requests if list(req.input_ids) == native.writer_ids[:-1])
    assert dict(bridge.cache_view)[thinker] > 1
    writer = bridge.write_to
    steps = [req for batch, req in requests if batch.kind == "decode" and req.write_to == writer]
    assert steps[0].input_ids == (native.writer_ids[-1],)
    assert len(steps) == len(answer) == 6
    assert dict(steps[-1].cache_view)[thinker] > dict(steps[0].cache_view)[thinker]
    assert any({r.write_to for r in b.requests} == {thinker, writer} for b in trace.batches)
