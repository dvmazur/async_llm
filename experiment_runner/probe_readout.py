"""Thin Qwen prefill/readout adapter; no world state, prompts, or role policy."""


class ProbeReadout:
    def __init__(self, backend):
        self.backend = backend

    def encode(self, text):
        return self.backend.encode(text)

    def new_generator(self, seed):
        return self.backend.new_generator(seed)

    async def create_block(self):
        return await self.backend.create_block()

    async def free_block(self, block):
        await self.backend.free_block(block)

    async def prefill_action(self, messages, target):
        inputs = self.backend.llm.processor.apply_chat_template(messages,
            add_generation_prompt=True, enable_thinking=False, tokenize=True,
            return_dict=True, return_tensors='pt')
        # No system-prefix cache: the earlier split-prefix numerical check failed.
        result = await self.backend.llm(**inputs, cache_view=[target.raw])
        return result, inputs['input_ids'].numel()

    def sample_action(self, output, token_ids, *, generator, temperature):
        import torch
        import flashinfer.sampling as sampling
        logits = output.logits.reshape(-1)[token_ids].float().view(1, -1)
        probs = torch.softmax(logits / temperature, dim=-1)
        index = sampling.top_k_top_p_sampling_from_probs(
            probs, len(token_ids), 1., generator=generator).item()
        self.backend.metrics['restricted_readouts'] += 1
        return index, probs[0].tolist()
