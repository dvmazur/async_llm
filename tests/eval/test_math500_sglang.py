"""Vanilla SGLang baseline uses the same dataset/prompt/grader, without mini runtime."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from eval.math500 import INSTRUCTION, read_config, solve_sglang, prepare_run
from tests.eval.test_math500 import Tokenizer as PlainTokenizer


class Tokenizer(PlainTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered<turn|>\n"

    def encode(self, text, **kwargs):
        return [{"rendered\n": 10, "<turn|>\n<|turn>model\n<|channel>thought": 11,
                 "\n": 12}[text]]


def test_sglang_defaults_are_plain_and_not_minisgl_settings():
    config = read_config(model_path="gemma", backend="sglang")
    assert config["mode"] == "plain" and not config["record_training_trace"]
    assert config["max_new_tokens"] == 16384 and config["thinking"]
    assert config["sampling"] == dict(temperature=1.0, top_p=0.95, top_k=64)
    assert config["engine"]["max_running_requests"] == 1
    assert "num_page_override" not in config["engine"]
    assert "quantization" not in config["engine"]  # SGLang reads checkpoint metadata.


def test_default_arguments_exist_in_installed_sglang_public_api():
    pytest.importorskip("sglang")
    from sglang.srt.server_args import ServerArgs
    defaults = read_config(model_path="gemma", backend="sglang")["engine"]
    assert set(defaults) <= set(ServerArgs.__struct_fields__)


@pytest.mark.parametrize("fields", [dict(mode="async_thoughts"),
    dict(record_training_trace=True), dict(engine=dict(tp_size=2)),
    dict(engine=dict(context_length=0)), dict(backend="vllm")])
def test_rejects_incompatible_configuration(tmp_path, fields):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(model_path="gemma", backend="sglang") | fields))
    with pytest.raises(ValueError):
        read_config(path)


@pytest.mark.parametrize("finish,expected", [("stop", "eos"), ("length", "length")])
def test_one_request_preserves_prompt_seed_tokens_and_grading_channel(finish, expected):
    tokenizer = Tokenizer()
    config = read_config(model_path="gemma", backend="sglang")
    calls = []
    meta = dict(prompt_tokens=3, completion_tokens=5, finish_reason=dict(type=finish),
                output_token_logprobs=[[-0.5, 23, None]])

    async def generate(**kwargs):
        calls.append(kwargs)
        return dict(output_ids=[20, 21, 22, 23, 0], meta_info=meta)

    result = asyncio.run(solve_sglang(SimpleNamespace(async_generate=generate), tokenizer,
                                     "FULL PROBLEM", config, {0, 7}, 456))
    assert tokenizer.calls[0][0] == [dict(role="user", content=INSTRUCTION + "\n\nFULL PROBLEM")]
    assert tokenizer.calls[0][1]["enable_thinking"]
    call, = calls
    assert call["input_ids"] == [10, 11, 12] and call["return_logprob"]
    assert call["sampling_params"] == dict(config["sampling"], max_new_tokens=16384,
                                           sampling_seed=456, stop_token_ids=[0, 7])
    assert result["answer_text"] == r"final \boxed{2}<eos>"
    assert result["stop_reason"] == expected and result["generated_tokens"] == 5
    assert result["sglang_meta"] is meta and "training_trace" not in result


def test_aborted_sglang_request_is_not_a_successful_answer():
    async def generate(**kwargs):
        return dict(output_ids=[], meta_info=dict(finish_reason=dict(type="abort", message="OOM")))
    with pytest.raises(RuntimeError, match="abort"):
        asyncio.run(solve_sglang(SimpleNamespace(async_generate=generate), Tokenizer(),
            "problem", read_config(model_path="gemma", backend="sglang"), {0}, 1))


def test_unfinished_thought_is_not_graded_when_opening_marker_was_in_prompt():
    async def generate(**kwargs):
        # The chat template already placed the thought opener in the INPUT.
        return dict(output_ids=[21], meta_info=dict(prompt_tokens=3, completion_tokens=1,
                                                    finish_reason=dict(type="length")))
    result = asyncio.run(solve_sglang(SimpleNamespace(async_generate=generate), Tokenizer(),
        "problem", read_config(model_path="gemma", backend="sglang"), {0}, 1))
    assert result["completion_text"] == r"guess \boxed{1}"
    assert result["answer_text"] == ""


def test_resume_cannot_mix_backends(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{}\n')
    args = SimpleNamespace(data=data, output=tmp_path / "run", resume=False)
    config = read_config(model_path="gemma", backend="sglang")
    prepare_run(args, config, [dict(id="a")])
    args.resume = True
    config["backend"] = "minisgl"
    with pytest.raises(ValueError, match="Resume refused"):
        prepare_run(args, config, [dict(id="a")])
