from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from minisgl.llm import AsyncLLM


class Engine:
    """Wraps a live AsyncLLM using minisgl's real cache-block interface
    (see doom_basic/agent.py and async_thoughts/engine.py for the source
    this pattern is drawn from). This instance is constructed exactly once
    per process and then kept alive across every self-rewrite -- see
    self_edit_env.py's reload_engine_methods(), which patches new method
    bodies onto this class in place without ever calling __init__ again."""

    def __init__(self, llm: "AsyncLLM") -> None:
        self.llm = llm

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        llm = self.llm
        from generation_prompt import revision_prompt, revision_completion
        prompt = revision_prompt(llm, prompt)
        input_ids = llm.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        # llm.tokenizer.eos_token_id is often just one id, but the model's
        # real generation_config.eos_token_id can list several (this model
        # has a separate end-of-turn stop token and end-of-text stop token,
        # and both are valid places to stop) -- checking only the
        # tokenizer's single id misses a real stop signal and lets decoding
        # run away past your own turn.
        eos_ids = {llm.tokenizer.eos_token_id}
        gen_cfg = getattr(llm.config, "generation_config", None)
        gen_eos = getattr(gen_cfg, "eos_token_id", None)
        if isinstance(gen_eos, int):
            eos_ids.add(gen_eos)
        elif isinstance(gen_eos, (list, tuple, set)):
            eos_ids.update(gen_eos)

        block = await llm.create_block()
        try:
            new_token_id = await llm.sample(await llm(input_ids, cache_view=[block]))
            tokens: list[int] = []
            for _ in range(max_new_tokens):
                if on_token is not None:
                    on_token(llm.tokenizer.decode(new_token_id))
                tokens.append(int(new_token_id))
                if int(new_token_id) in eos_ids:
                    break
                new_token_id = await llm.sample(await llm(new_token_id.view(1), cache_view=[block]))
            return revision_completion(prompt, llm.tokenizer.decode(tokens, skip_special_tokens=False))
        finally:
            await llm.free_block(block)

    async def act(
        self,
        observation: Any,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Any:
        """Generic task-facing entrypoint used by tasks/runner.py. Minimal
        default: just ask generate() to respond to the observation as text.
        Specialize this (e.g. parse a numeric answer, handle an image
        observation) once a task is plugged in and its score starts showing
        up in your prompt."""
        return await self.generate(str(observation), on_token=on_token)
