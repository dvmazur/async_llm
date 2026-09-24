import asyncio
import sys
from types import SimpleNamespace

import pytest

from experiment_runner.json_role import JsonRole


@pytest.mark.parametrize('failure', [None, 'prefill', 'decode', 'length', 'cancel'])
def test_role_owns_fresh_kv_and_never_returns_fabricated_output(monkeypatch, failure):
    class Tensor:
        device = 'test-device'
        def float(self): return self
        def reshape(self, *args): return self
        def clone(self): return Tensor()
        def to(self, device): return self

    class Matcher:
        def __init__(self, *a, **kw): self.n = 0
        def fill_next_token_bitmask(self, mask): return True
        def accept_token(self, token): self.n += 1; return True
        def is_terminated(self): return self.n == 2

    masked = []
    monkeypatch.setitem(sys.modules, 'xgrammar', SimpleNamespace(
        TokenizerInfo=SimpleNamespace(from_huggingface=lambda *a, **kw: None),
        GrammarCompiler=lambda _: SimpleNamespace(compile_json_schema=lambda x:x),
        GrammarMatcher=Matcher, allocate_token_bitmask=lambda *a:Tensor(),
        apply_token_bitmask_inplace=lambda logits,mask:masked.append(logits)))

    class Backend:
        def __init__(self):
            self.live = set()
            self.n = self.freed = 0
            self.llm = SimpleNamespace(engine=SimpleNamespace(config=SimpleNamespace(
                model_config=SimpleNamespace(vocab_size=100))),
                tokenizer=SimpleNamespace(decode=lambda *a, **kw:'{"action":"down"}'))
        async def create_block(self):
            block = object(); self.live.add(block); return block
        async def free_block(self, block):
            self.live.remove(block); self.freed += 1
        def sample(self, output, **kwargs):
            assert output.logits is masked[-1]
            assert kwargs == dict(generator=17, temperature=.7, top_k=100, top_p=1.)
            self.n += 1
            return self.n, '', False
        async def decode(self, token, deps, target):
            assert deps == [] and target.raw in self.live and len(self.live) == 1
            if failure == 'decode': raise RuntimeError('decode failure')
            if failure == 'cancel': raise asyncio.CancelledError()
            return SimpleNamespace(logits=Tensor())

    backend = Backend(); records = []
    role = JsonRole(backend, SimpleNamespace(log=lambda *args:records.append(args)),
                   'test_role', {'type':'object'})
    async def prefill(messages, target):
        assert target.raw in backend.live
        if failure == 'prefill': raise RuntimeError('prefill failure')
        return SimpleNamespace(logits=Tensor()), 1200
    role.readout.prefill_action = prefill
    call = role([dict(role='user',content='current facts')],generator=17,
                max_tokens=1 if failure == 'length' else 8)
    if failure is None:
        assert asyncio.run(call) == ({'action':'down'},1200)
        stream = [data for kind,data in records if kind=='stream']
        assert len(stream)==1 and stream[0]['sampled_tokens']==2
    else:
        with pytest.raises(asyncio.CancelledError if failure=='cancel' else RuntimeError):
            asyncio.run(call)
        assert not [data for kind,data in records if kind=='stream']
    assert not backend.live and backend.freed == 1
