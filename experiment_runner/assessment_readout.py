"""On-device sampled JSON assessment + action, no hosted API or action heuristics."""
import json
from types import SimpleNamespace
from weakref import WeakKeyDictionary

from .probe_readout import ProbeReadout

_COMPILED = WeakKeyDictionary()


def response_schema(actions):
    return {'type': 'object', 'properties': {
        'observation': {'type': 'string', 'minLength': 1, 'maxLength': 640},
        'action': {'type': 'string', 'enum': list(actions)},
    }, 'required': ['observation', 'action'], 'additionalProperties': False}


def parse_assessment(text, actions):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'observation', 'action'}:
        raise ValueError('assessment must contain exactly observation and action')
    if not isinstance(value['observation'], str) or not 1 <= len(value['observation']) <= 640:
        raise ValueError('invalid assessment length')
    if value['action'] not in actions:
        raise ValueError('invalid action')
    return value


class AssessmentReadout(ProbeReadout):
    MAX_TOKENS = 768

    def __init__(self, backend, recorder, actions):
        super().__init__(backend)
        import xgrammar as xgr
        self.xgr, self.recorder, self.actions = xgr, recorder, tuple(actions)
        if backend not in _COMPILED:
            info = xgr.TokenizerInfo.from_huggingface(backend.llm.tokenizer,
                vocab_size=backend.llm.engine.config.model_config.vocab_size)
            compiler = xgr.GrammarCompiler(info)
            _COMPILED[backend] = (compiler, {})
        compiler, schemas = _COMPILED[backend]
        if self.actions not in schemas:
            schemas[self.actions] = compiler.compile_json_schema(
                json.dumps(response_schema(self.actions)))
        self.grammar = schemas[self.actions]
        self.policy_metadata = dict(mode='sampled-assessment-action', on_device=True,
            max_new_tokens=self.MAX_TOKENS, observation_max_characters=640,
            constrained_format='xgrammar-json-schema', legal_actions=list(self.actions),
            action_overrides=False, top_k=None, top_p=1., history=False)

    async def generate_action(self, messages, target, *, generator, temperature):
        # Each decision owns a matcher and KV block; compiled grammar is shared.
        xgr = self.xgr
        matcher = xgr.GrammarMatcher(self.grammar, terminate_without_stop_token=True)
        vocab_size = self.backend.llm.engine.config.model_config.vocab_size
        bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        tokens = []
        output, input_rows = await self.prefill_action(messages, target)
        for _ in range(self.MAX_TOKENS):
            # Never modify model/graph-owned output in place.
            logits = output.logits.float().reshape(1, -1).clone()
            if matcher.fill_next_token_bitmask(bitmask):
                xgr.apply_token_bitmask_inplace(logits, bitmask.to(logits.device))
            token, _, eos = self.backend.sample(SimpleNamespace(logits=logits),
                generator=generator, temperature=temperature, top_k=vocab_size, top_p=1.)
            if not matcher.accept_token(token):
                raise RuntimeError('sample violated the response format')
            tokens.append(token)
            if matcher.is_terminated():
                text = self.backend.llm.tokenizer.decode(tokens, skip_special_tokens=True)
                value = parse_assessment(text, self.actions)
                self.recorder.log('assessment', dict(text=text, **value,
                    output_tokens=len(tokens), input_rows=input_rows, finish_reason='complete'))
                return self.actions.index(value['action']), input_rows
            if eos:
                raise RuntimeError('EOS before complete assessment/action')
            output = await self.backend.decode(token, [], target)
        text = self.backend.llm.tokenizer.decode(tokens, skip_special_tokens=True)
        self.recorder.log('assessment_error', dict(text=text, output_tokens=len(tokens),
            input_rows=input_rows, finish_reason='length'))
        # Do not invent a fallback action, resample a nicer response or use old action.
        raise RuntimeError('assessment exceeded token budget before a complete action')
