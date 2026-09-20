"""Role-sized generation API; the mini backend retains its token-at-a-time path."""
from dataclasses import dataclass


@dataclass
class Generation:
    text: str = ''
    sampled_tokens: int = 0
    visible_tokens: int = 0


async def generate(self, prompt, deps, target, *, result, generator, budget,
                   temperature, top_k, top_p):
    output = await self.prefill(prompt, deps, target)
    for _ in range(budget):
        token, piece, eos = self.sample(output, generator=generator,
            temperature=temperature, top_k=top_k, top_p=top_p)
        result.sampled_tokens += 1
        if eos or any(tag in piece for tag in ('<|im_end|>', '<|endoftext|>', '<|im_start|>')):
            break
        output = await self.decode(token, deps, target)
        result.visible_tokens += 1
        result.text += piece
