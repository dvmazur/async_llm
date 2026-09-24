import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from experiment_runner.assessment_readout import AssessmentReadout, parse_assessment, response_schema
from pipelines.choptree import ACTION_NAMES
from test_probe import Engine, World, make


def test_schema_keeps_every_action_and_only_bounds_format():
    schema = response_schema(ACTION_NAMES)
    assert schema['properties']['action']['enum'] == list(ACTION_NAMES)
    assert schema['properties']['observation']['maxLength'] == 640
    for action in ACTION_NAMES:
        assert parse_assessment(json.dumps(dict(observation='Wood at center.', action=action)),
                                ACTION_NAMES)['action'] == action


@pytest.mark.parametrize('value', [[], {}, {'observation': 'x', 'action': 'mine'},
    {'observation': '', 'action': 'dig'}, {'observation': 'x'*641, 'action': 'dig'},
    {'observation': 'x', 'action': 'dig', 'override': 'up'}])
def test_invalid_completion_never_becomes_a_fallback_action(value):
    with pytest.raises(ValueError):
        parse_assessment(json.dumps(value), ACTION_NAMES)


def test_generated_decision_frees_kv_before_action_and_records_no_fake_probabilities(tmp_path):
    class Generated(Engine):
        async def generate_action(self, messages, target, **kwargs):
            assert len(self.live) == 1 and target.raw.num_tokens == 0
            self.samples += 1
            return 3, 1200
    e = Generated(); w = World(e)
    asyncio.run(make(tmp_path, e, w, max_actions=500, action_delay=0).run())
    assert e.samples == w.i == 500 and not e.live and w.closed


def test_generation_error_cleans_up_without_world_action(tmp_path):
    class Generated(Engine):
        async def generate_action(self, messages, target, **kwargs):
            raise RuntimeError('length limit')
    e = Generated(); w = World(e)
    with pytest.raises(RuntimeError, match='length limit'):
        asyncio.run(make(tmp_path, e, w, max_actions=500, action_delay=0).run())
    assert w.i == 0 and not e.live and w.closed


def test_generation_adapter_uses_masked_sampling_and_one_context(monkeypatch):
    class Tensor:
        device = 'cuda'
        def float(self): return self
        def reshape(self, *args): return self
        def clone(self): return Tensor()
        def to(self, device): return self
    class Matcher:
        def __init__(self, *args, **kwargs): self.n = 0
        def fill_next_token_bitmask(self, mask): return True
        def accept_token(self, token): self.n += 1; return True
        def is_terminated(self): return self.n == 2
    masks = []
    grammar = SimpleNamespace(
        TokenizerInfo=SimpleNamespace(from_huggingface=lambda *a, **kw: None),
        GrammarCompiler=lambda _: SimpleNamespace(compile_json_schema=lambda x: x),
        GrammarMatcher=Matcher, allocate_token_bitmask=lambda *a: Tensor(),
        apply_token_bitmask_inplace=lambda logits, mask: masks.append(logits))
    monkeypatch.setitem(sys.modules, 'xgrammar', grammar)
    class Backend:
        def __init__(self):
            self.n = 0; self.decodes = []
            self.llm = SimpleNamespace(engine=SimpleNamespace(config=SimpleNamespace(
                model_config=SimpleNamespace(vocab_size=100))),
                tokenizer=SimpleNamespace(decode=lambda *a, **kw:
                    '{"observation":"Ringed wood below.","action":"down"}'))
        def sample(self, output, **kwargs):
            assert output.logits is masks[-1]
            assert kwargs == dict(generator=17, temperature=.7, top_k=100, top_p=1.)
            self.n += 1
            return self.n, '', False
        async def decode(self, token, deps, target):
            self.decodes.append((token, deps, target))
            return SimpleNamespace(logits=Tensor())
    backend = Backend(); records = []
    adapter = AssessmentReadout(backend, SimpleNamespace(log=lambda *x: records.append(x)), ACTION_NAMES)
    async def prefill(messages, target): return SimpleNamespace(logits=Tensor()), 1200
    adapter.prefill_action = prefill
    target = object()
    action, rows = asyncio.run(adapter.generate_action([], target, generator=17, temperature=.7))
    assert action == 7 and rows == 1200
    assert backend.decodes == [(1, [], target)] and len(masks) == 2
    assert records[0][1]['output_tokens'] == 2
