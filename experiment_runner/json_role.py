"""Sample a bounded, fresh-context on-device role; schemas constrain format only."""
import json
import hashlib
from types import SimpleNamespace
from weakref import WeakKeyDictionary

from .blocks import BlockHandle
from .probe_readout import ProbeReadout

_COMPILERS = WeakKeyDictionary()


class JsonRole:
    def __init__(self, backend, recorder, name, schema):
        import xgrammar as xgr
        self.backend, self.recorder, self.name, self.xgr = backend, recorder, name, xgr
        if backend not in _COMPILERS:
            info = xgr.TokenizerInfo.from_huggingface(backend.llm.tokenizer,
                vocab_size=backend.llm.engine.config.model_config.vocab_size)
            _COMPILERS[backend] = xgr.GrammarCompiler(info)
        self.grammar = _COMPILERS[backend].compile_json_schema(json.dumps(schema))
        self.readout = ProbeReadout(backend)

    async def __call__(self, messages, *, generator, temperature=.7, max_tokens=768):
        self.recorder.log('role_input', dict(role=self.name, messages=[dict(role=m['role'],
            text=m['content'] if isinstance(m['content'],str) else
                [p['text'] for p in m['content'] if p['type']=='text'],
            images=[] if isinstance(m['content'],str) else [
                dict(shape=list(p['image'].shape),sha256=hashlib.sha256(p['image'].tobytes()).hexdigest())
                for p in m['content'] if p['type']=='image']) for m in messages]))
        async with await BlockHandle.create(self.backend, self.name) as target:
            xgr = self.xgr
            matcher = xgr.GrammarMatcher(self.grammar, terminate_without_stop_token=True)
            vocab = self.backend.llm.engine.config.model_config.vocab_size
            bitmask = xgr.allocate_token_bitmask(1, vocab)
            output, rows = await self.readout.prefill_action(messages, target)
            tokens = []
            for _ in range(max_tokens):
                logits = output.logits.float().reshape(1, -1).clone()
                if matcher.fill_next_token_bitmask(bitmask):
                    xgr.apply_token_bitmask_inplace(logits, bitmask.to(logits.device))
                token, _, eos = self.backend.sample(SimpleNamespace(logits=logits),
                    generator=generator, temperature=temperature, top_k=vocab, top_p=1.)
                if not matcher.accept_token(token):
                    raise RuntimeError('role sample violated schema')
                tokens.append(token)
                if matcher.is_terminated():
                    text = self.backend.llm.tokenizer.decode(tokens, skip_special_tokens=True)
                    value = json.loads(text)
                    self.recorder.log('stream', dict(role=self.name, sampled_tokens=len(tokens),
                        text=text, input_rows=rows, finish_reason='complete'))
                    return value, rows
                if eos:
                    raise RuntimeError('role terminated before completing response')
                output = await self.backend.decode(token, [], target)
            self.recorder.log('role_error', dict(role=self.name, sampled_tokens=len(tokens),
                text=self.backend.llm.tokenizer.decode(tokens), finish_reason='length'))
            raise RuntimeError(f'{self.name} exceeded token budget')
