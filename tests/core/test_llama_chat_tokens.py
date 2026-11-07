from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import EngineConfig
from minisgl.llm.llm import LLM, RequestStatus
from minisgl.message import BaseTokenizerMsg, TokenizeMsg
from minisgl.scheduler import Scheduler
from minisgl.tokenizer.detokenize import DetokenizeManager
from minisgl.tokenizer.tokenize import TokenizeManager
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import GenerationConfig, PreTrainedTokenizerFast


@pytest.fixture
def tokenizer():
    # Like Llama Instruct, the chat template includes BOS and the tokenizer also
    # inserts BOS on raw prompts.
    vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<eot>": 3, "hello": 4, "<assistant>": 5}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        chat_template="{{ bos_token }} {{ messages[0]['content'] }}{% if add_generation_prompt %} <assistant>{% endif %}",
    )


def test_chat_template_adds_bos_once_and_raw_text_keeps_tokenizer_defaults(tokenizer):
    chat = [{"role": "user", "content": "hello"}]
    manager = TokenizeManager(tokenizer)
    chat_ids, raw_ids = manager.tokenize(
        [TokenizeMsg(0, chat, SamplingParams()), TokenizeMsg(1, "hello", SamplingParams())]
    )
    assert chat_ids.tolist() == [1, 4, 5]
    assert raw_ids.tolist() == [1, 4]


@pytest.fixture
def scheduler():
    return SimpleNamespace(
        eos_token_ids={2, 3},
        finished_reqs=set(),
        cache_manager=SimpleNamespace(lazy_free_region=nullcontext),
        decode_manager=SimpleNamespace(remove_req=Mock()),
        _free_req_resources=Mock(),
        send_result=Mock(),
    )


def process_result(scheduler, req, sampled):
    batch = SimpleNamespace(reqs=[req], num_prefill=0)
    result = (None, torch.tensor([sampled]), SimpleNamespace(synchronize=lambda: None))
    Scheduler._process_last_data(scheduler, (SimpleNamespace(batch=batch), result))
    return scheduler.send_result.call_args.args[0]


@pytest.mark.parametrize(
    "generation_eos,model_eos,text_eos,fallback,expected",
    [
        ([2, 3], 2, None, 1, {2, 3}),
        (3, 2, None, 1, {3}),
        (None, 2, None, 1, {2}),
        (None, None, [2, 3], 1, {2, 3}),
        (None, None, None, 1, {1}),
        (None, None, None, None, set()),
    ],
)
def test_effective_eos_ids(monkeypatch, generation_eos, model_eos, text_eos, fallback, expected):
    config = EngineConfig(
        "unused",
        DistributedInfo(0, 1),
        torch.bfloat16,
        generation_config=GenerationConfig(eos_token_id=generation_eos),
    )
    monkeypatch.setattr(
        "minisgl.engine.config.cached_load_hf_config",
        lambda _: SimpleNamespace(
            eos_token_id=model_eos, text_config=SimpleNamespace(eos_token_id=text_eos)
        ),
    )
    assert config.get_eos_token_ids(fallback) == expected
    assert not config.get_default_sampling_params().ignore_eos


@pytest.mark.parametrize(
    "sampled,ignore_eos,budget,finished,include_token",
    [
        (2, False, 4, True, False),
        (3, False, 4, True, False),
        (3, True, 4, False, True),
        (3, True, 1, True, True),
        (4, False, 1, True, True),
    ],
)
def test_server_and_offline_eos_handling(
    tokenizer, scheduler, sampled, ignore_eos, budget, finished, include_token
):
    req = Req(torch.tensor([4]), 0, 0, budget, 0, SamplingParams(ignore_eos=ignore_eos), None)
    req.complete_one()
    replies = process_result(scheduler, req, sampled)
    [msg] = replies
    assert msg.finished == finished
    wire_msg = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    [text] = DetokenizeManager(tokenizer).detokenize([wire_msg])
    expected_ids = [sampled] if include_token else []
    assert text == tokenizer.decode(expected_ids)
    offline = SimpleNamespace(status_map={0: RequestStatus(0, [4], [])})
    LLM.offline_send_result(offline, replies)
    assert offline.status_map[0].output_ids == expected_ids


@pytest.mark.parametrize("first_token,expected", [(3, [(3, True)]), (4, [(4, False), (4, True)])])
def test_overlapped_scheduler_stops_at_returned_token(scheduler, first_token, expected):
    req = Req(torch.tensor([4]), 0, 0, 2, 0, SamplingParams(), None)
    # Both GPU forwards finish before the scheduler consumes the first result.
    req.complete_one()
    req.complete_one()
    outputs = process_result(scheduler, req, first_token) + process_result(scheduler, req, 4)
    assert [(msg.next_token, msg.finished) for msg in outputs] == expected
    scheduler._free_req_resources.assert_called_once_with(req)
