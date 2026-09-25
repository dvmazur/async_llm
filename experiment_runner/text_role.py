"""Bounded ordinary role generation. A length limit is completion, not failure."""
import hashlib

from .blocks import BlockHandle
from .probe_readout import ProbeReadout


class TextRole:
    def __init__(self, backend, recorder, name):
        self.backend, self.recorder, self.name = backend, recorder, name
        self.readout = ProbeReadout(backend)

    async def __call__(self, messages, *, generator, temperature, max_tokens):
        async with await BlockHandle.create(self.backend, self.name) as target:
            return await self.on_block(messages, target, generator=generator,
                temperature=temperature, max_tokens=max_tokens, retain_last_token=False)

    async def on_block(self, messages, target, *, generator, temperature, max_tokens,
                       retain_last_token=True):
        """Append a role to caller-owned KV. Caller closes the assistant turn."""
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        self.recorder.log('role_input', dict(role=self.name, messages=[dict(role=m['role'],
            text=m['content'] if isinstance(m['content'],str) else
                [p['text'] for p in m['content'] if p['type']=='text'],
            images=[] if isinstance(m['content'],str) else [
                dict(shape=list(p['image'].shape),sha256=hashlib.sha256(p['image'].tobytes()).hexdigest())
                for p in m['content'] if p['type']=='image']) for m in messages]))
        output, rows = await self.readout.prefill_action(messages, target)
        tokens, sampled, finish_reason = [], 0, 'length'
        vocab = self.backend.llm.engine.config.model_config.vocab_size
        for i in range(max_tokens):
            token, piece, eos = self.backend.sample(output, generator=generator,
                temperature=temperature, top_k=vocab, top_p=1.)
            sampled += 1
            if eos or any(tag in piece for tag in ('<|im_end|>','<|endoftext|>','<|im_start|>')):
                finish_reason = 'eos' if eos else 'chat_boundary'
                break
            tokens.append(token)
            if i+1 < max_tokens or retain_last_token:
                output = await self.backend.decode(token, [], target)
        text = self.backend.llm.tokenizer.decode(tokens, skip_special_tokens=True)
        self.recorder.log('stream', dict(role=self.name, sampled_tokens=sampled,
            text=text, input_rows=rows, finish_reason=finish_reason))
        return text.strip(), rows
