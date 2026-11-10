"""CPU-only checks for the ordinary, single-stream MATH-500 baseline."""
import asyncio
import io
import json
from types import SimpleNamespace

import pytest

from eval import math500 as plain


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [10, 11, 12]

    def decode(self, tokens, **kwargs):
        vocabulary = {0: "<eos>", 20: "<|channel>thought\n", 21: r"guess \boxed{1}",
                      22: "<channel|>", 23: r"final \boxed{2}"}
        return "".join(vocabulary[t] for t in tokens)


class LLM:
    def __init__(self, tokens=(20, 21, 22, 23, 0), fail=False):
        self.tokenizer = Tokenizer()
        self.tokens, self.fail = tokens, fail
        self.prefills, self.decode_inputs = [], []
        self.block = SimpleNamespace(active=False, freed=False)
        self.created = 0

    async def create_block(self):
        self.created += 1
        return self.block

    async def free_block(self, block):
        assert block is self.block and not block.active
        block.freed = True

    async def forward(self, ids, cache_view, return_logits):
        assert cache_view == [self.block] and not return_logits
        self.block.active = True
        try:
            await asyncio.sleep(0)
            self.prefills.append(list(ids))
        finally:
            self.block.active = False

    async def async_generate(self, context, **kwargs):
        self.generate_kwargs = kwargs
        assert context.cache_view == [self.block]
        for token in self.tokens[:kwargs["max_steps"]]:
            self.block.active = True
            try:
                await asyncio.sleep(0)
                self.decode_inputs.append(context.next_input_id)
                if self.fail:
                    raise RuntimeError("engine failed")
                context.next_input_id = token
            finally:
                self.block.active = False
            yield token


def execute(llm, budget=10, thinking=True):
    params = SimpleNamespace(temperature=1, top_p=0.95, top_k=64)
    result = asyncio.run(plain.solve(llm, "FULL PROBLEM", dict(thinking=thinking, max_new_tokens=budget),
                                     params, {0}, context_factory=SimpleNamespace))
    assert llm.generate_kwargs["sampling_params"] is params
    return result


def test_one_stream_complete_prompt_no_policy_or_token_mask():
    llm = LLM()
    row = execute(llm)
    assert llm.created == 1 and llm.block.freed
    assert llm.prefills == [[10, 11]]
    assert llm.decode_inputs == [12, 20, 21, 22, 23]
    assert "forbid_ids" not in llm.generate_kwargs
    assert row["stop_reason"] == "eos" and row["generated_tokens"] == 5
    assert row["prompt_tokens"] == 3 and row["answer_text"] == r"final \boxed{2}<eos>"
    messages, kwargs = llm.tokenizer.calls[0]
    assert messages == [dict(role="user", content=plain.INSTRUCTION + "\n\nFULL PROBLEM")]
    assert kwargs == dict(tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)


def test_length_limit_does_not_grade_thought_box():
    row = execute(LLM(), budget=2)
    assert row["stop_reason"] == "length" and row["generated_tokens"] == 2
    assert row["answer_text"] == "" and r"\boxed{1}" in row["completion_text"]


def test_non_thinking_uses_checkpoint_template_and_direct_answer():
    llm = LLM(tokens=(23, 0))
    assert execute(llm, thinking=False)["answer_text"] == r"final \boxed{2}<eos>"
    assert llm.tokenizer.calls[0][1]["enable_thinking"] is False


def test_engine_failure_releases_block():
    llm = LLM(fail=True)
    with pytest.raises(RuntimeError, match="engine failed"):
        execute(llm)
    assert llm.block.freed


def test_cancellation_waits_for_submitted_work():
    async def scenario():
        llm = LLM()
        task = asyncio.create_task(plain.solve(llm, "problem", dict(thinking=True, max_new_tokens=8),
                                              None, {0}, context_factory=SimpleNamespace))
        while not llm.block.active:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert llm.block.freed and not llm.block.active
    asyncio.run(scenario())


def test_data_is_ordinary_math_and_never_assembled_from_shards():
    row = dict(unique_id="test/algebra/1.json", problem="full", answer="2", solution="SECRET")
    assert plain.parse_data((json.dumps(row) + "\n").encode()) == [dict(row, id=row["unique_id"])]
    with pytest.raises(ValueError, match="problem"):
        plain.parse_data(json.dumps(dict(problem_shards=["a", "b"], answer="2")).encode())
    with pytest.raises(ValueError, match="Duplicate"):
        plain.parse_data(((json.dumps(row) + "\n") * 2).encode())


def test_download_opt_in_checksum_and_local_reuse(tmp_path, monkeypatch):
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return io.BytesIO(b'broken')
    monkeypatch.setattr(plain.urllib.request, "urlopen", fetch)
    path = tmp_path / "data.jsonl"
    with pytest.raises(FileNotFoundError):
        plain.load_data(path)
    assert not calls
    with pytest.raises(ValueError, match="checksum"):
        plain.load_data(path, download=True)
    assert not path.exists() and plain.REVISION in calls[0]
    path.write_text(json.dumps(dict(problem="full", answer="2")) + "\n")
    assert len(plain.load_data(path, download=True)) == 1 and len(calls) == 1


def test_seed_is_stable_across_order_and_shards():
    a = plain.question_seed(0, "a")
    b = plain.question_seed(0, "b")
    assert a == plain.question_seed(0, "a") and a != b
    assert a != plain.question_seed(1, "a") and 0 <= a < 2**63


def test_plain_sampling_and_capacity(tmp_path):
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(dict(model_path="unused", mode="plain")))
    config = plain.read_config(path)
    assert config["sampling"] == dict(temperature=1, top_p=0.95, top_k=64)
    assert config["thinking"] and config["max_new_tokens"] == 16384
    assert config["engine"]["max_running_req"] == 1


@pytest.mark.parametrize("key,value", [
    ("num_page_override", None), ("num_page_override", 0), ("page_size", -1),
    ("max_running_req", True), ("max_seq_len_override", 1.5), ("max_prefill_rows", 0),
])
def test_invalid_capacity_is_rejected_before_model_loading(tmp_path, key, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(model_path="stub", engine={key: value})))
    with pytest.raises(ValueError, match=key):
        plain.read_config(path)


def test_chat_template_does_not_mistake_metadata_for_token_ids():
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: {"input_ids": [10, 11]})
    with pytest.raises(ValueError, match="flat list"):
        plain.prompt_ids(tokenizer, "problem", True)


def test_summary_distinguishes_partial_results_and_no_answer():
    rows = {"a": dict(status="ok", correct=True, elapsed_s=2, generated_tokens=10, stop_reason="eos"),
            "b": dict(status="ok", correct=False, elapsed_s=3, generated_tokens=20, stop_reason="length")}
    partial = plain.summary(rows, 3)
    assert partial["accuracy"] is None and partial["accuracy_on_graded"] == 0.5
    assert partial["generated_tps"] == 6 and partial["length_limited"] == 1
    assert plain.summary(rows, 2)["accuracy"] == 0.5


def test_completed_resume_needs_no_torch_and_changed_thinking_is_rejected(tmp_path):
    pytest.importorskip("math_verify")
    config_path, data = tmp_path / "config.json", tmp_path / "data.jsonl"
    config_path.write_text(json.dumps(dict(model_path="unused", mode="plain")))
    data.write_text(json.dumps(dict(unique_id="a", problem="full", answer="2")) + "\n")
    args = SimpleNamespace(config=config_path, model=None, backend=None, data=data, download_data=False, start=0, limit=None,
                           shard_index=None, num_shards=None, output=tmp_path / "run", resume=False)
    config, cases = plain.read_config(config_path), plain.load_data(data)
    output = plain.prepare_run(args, config, cases)
    (output / "results.jsonl").write_text(json.dumps(dict(id="a", status="ok")) + "\n")
    args.resume = True
    plain.run(args)
    config["thinking"] = False
    with pytest.raises(ValueError, match="Resume refused"):
        plain.prepare_run(args, config, cases)
