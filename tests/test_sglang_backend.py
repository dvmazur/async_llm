import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from experiment_runner.sglang_engine import Backend
from experiment_runner.logs import read_jsonl
from experiment_runner.generation import Generation
from experiment_runner.blocks import BlockHandle
from test_policy import World, pipeline


class Tokenizer:
    eos_token_id = 1
    actions = ['wait', 'forward', 'jump', 'right', 'left', 'up', 'down']

    def encode(self, text, **kwargs):
        if text in self.actions:
            return [10 + self.actions.index(text)]
        return [ord(c)+100 for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(t-100) for t in ids)


class Processor:
    tokenizer = Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=False)
        return messages[0]['content'][0]['text'] + '<image><image>'


class Native:
    def __init__(self):
        self.calls, self.closed = [], False

    async def async_generate(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        await asyncio.sleep(0)
        if kwargs['sampling_params']['max_new_tokens'] == 1:
            ids = kwargs['token_ids_logprob']
            # Native greedy output is NOT necessarily one of the action tokens.
            return dict(output_ids=[9999], meta_info=dict(completion_tokens=1,
                output_token_ids_logprobs=[[(0. if i == 10 else -2., i, None) for i in ids]]))
        return dict(output_ids=[ord('x')+100, 1],
            meta_info=dict(completion_tokens=2, cached_tokens=100))

    def shutdown(self):
        self.closed = True


def test_native_calls_keep_all_history_images_and_chosen_actions(tmp_path):
    async def run():
        native = Native()
        engine = Backend(native, Processor(), tmp_path)
        p = pipeline(tmp_path, engine, World(), max_actions=3)
        await p.run()
        await engine.close()
        assert native.closed
        # 3 roles + action on step 0, then 2 roles + action on subsequent steps.
        assert len(native.calls) == 10
        assert [len(c['image_data']) for c in native.calls] == [2]*4 + [4]*3 + [6]*3
        for previous, current in zip(native.calls, native.calls[1:]):
            assert current['input_ids'][:len(previous['input_ids'])] == previous['input_ids']
        assert 1 not in native.calls[-1]['input_ids']  # no EOS duplicate
        assert 9999 not in native.calls[-1]['input_ids']  # discarded unconstrained readout
        assert native.calls[-1]['input_ids'].count(10) == 2  # actual chosen actions
        assert all(c['return_logprob'] and c['token_ids_logprob'] for c in native.calls)
        assert all(c['token_ids_logprob'] == [0] for c in native.calls
                   if c['sampling_params']['max_new_tokens'] != 1)
        events = list(read_jsonl(tmp_path/'events.jsonl'))
        streams = [e for e in events if e['kind'] == 'stream']
        assert len(streams) == 7
        assert all((e['text'], e['sampled_tokens'], e['visible_tokens']) == ('x', 2, 1) for e in streams)
        assert len(list(read_jsonl(tmp_path/'generation.jsonl'))) == 14
        totals = json.loads((tmp_path/'engine-totals.json').read_text())
        assert totals['restricted_readouts'] == 3
        assert totals['forward_telemetry_available'] is False
    asyncio.run(run())


def test_independent_episode_histories_and_seed_streams(tmp_path):
    async def run():
        native = Native()
        engine = Backend(native, Processor(), tmp_path)
        a, b = await engine.create_block(), await engine.create_block()
        a.tokens.append(500)
        assert b.tokens == []
        r1, r2 = engine.new_generator(42), engine.new_generator(42)
        assert [r1.randrange(2**31) for _ in range(4)] == [r2.randrange(2**31) for _ in range(4)]
        await engine.free_block(a)
        await engine.close()
    asyncio.run(run())


def test_native_abort_is_not_a_successful_empty_reply(tmp_path):
    async def run():
        async def abort(**kwargs):
            return dict(meta_info=dict(finish_reason=dict(type='abort', message='context too long')))
        engine = Backend(SimpleNamespace(async_generate=abort, shutdown=lambda: None), Processor(), tmp_path)
        with pytest.raises(RuntimeError, match='context too long'):
            await engine.request([], dict(max_new_tokens=1))
        await engine.close()
        assert list(read_jsonl(tmp_path/'requests.jsonl'))[0]['status'] == 'failed'
    asyncio.run(run())


def test_normal_mode_does_not_claim_ignored_request_seeds(tmp_path):
    async def run():
        native = Native()
        engine = Backend(native, Processor(), tmp_path, request_seeds=False)
        assert engine.new_generator(123) is None
        async with await BlockHandle.create(engine, 'history') as block:
            await engine.generate('prompt', [], block, result=Generation(), generator=None,
                budget=18, temperature=.35, top_k=20, top_p=.9)
        assert 'sampling_seed' not in native.calls[0]['sampling_params']
        await engine.close()
        assert json.loads((tmp_path/'engine-totals.json').read_text())['sampling_seed_policy'] == 'engine_global'
    asyncio.run(run())
