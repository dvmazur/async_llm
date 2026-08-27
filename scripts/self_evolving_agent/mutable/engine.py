import numpy as np

class Engine:
    def __init__(self, llm):
        self.llm = llm
        self._step_count = 0
        self._sys_block = None
        self._history_block = None
        self._history_text = ""
        self._action_token_ids = None

    async def generate(self, prompt, max_new_tokens=512, on_token=None):
        try:
            llm = self.llm
            tokenizer = llm.tokenizer
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            block = await llm.create_block()
            try:
                output = await llm(input_ids, cache_view=[block])
                generated = []
                eos_id = tokenizer.eos_token_id
                for _ in range(max_new_tokens):
                    token = await llm.sample(output)
                    token_id = int(token.item())
                    generated.append(token_id)
                    if on_token:
                        try:
                            on_token(tokenizer.decode(token_id))
                        except Exception:
                            pass
                    if eos_id is not None and token_id == eos_id:
                        break
                    input_ids = token.reshape(1, 1)
                    output = await llm(input_ids, cache_view=[block])
                return tokenizer.decode(generated, skip_special_tokens=True)
            finally:
                await llm.free_block(block)
        except Exception as e:
            return f"[error: {type(e).__name__}: {e}]"

    def _image_to_text(self, obs):
        if not isinstance(obs, np.ndarray):
            return "unknown observation"
        h, w = obs.shape[0], obs.shape[1]
        gh, gw = 8, 12
        grid = []
        for i in range(gh):
            row = ""
            for j in range(gw):
                y0, y1 = int(i * h / gh), int((i + 1) * h / gh)
                x0, x1 = int(j * w / gw), int((j + 1) * w / gw)
                patch = obs[y0:y1, x0:x1]
                if patch.size == 0:
                    row += "."
                    continue
                brightness = patch.mean()
                if brightness > 180:
                    row += "#"
                elif brightness > 120:
                    row += "+"
                elif brightness > 60:
                    row += "."
                else:
                    row += " "
            grid.append(row)
        return "\n".join(grid)

    async def act(self, observation, on_token=None):
        try:
            self._step_count += 1
            llm = self.llm
            tokenizer = llm.tokenizer

            if not hasattr(self, '_action_token_ids') or self._action_token_ids is None:
                self._action_token_ids = {}
                for name in ["wait", "fire", "right", "left"]:
                    ids = tokenizer(name, return_tensors="pt", add_special_tokens=False).input_ids
                    self._action_token_ids[name] = ids[0].tolist()

            img_text = self._image_to_text(observation)

            if not hasattr(self, '_sys_block') or self._sys_block is None:
                sys_prompt = (
                    "You are playing Doom. You see a brightness map of the screen "
                    "(# = bright, + = medium, . = dim, space = dark). "
                    "Enemies appear as bright spots. You must respond with exactly one word: "
                    "wait, fire, right, or left. "
                    "Strategy: fire when you see bright spots (enemies) ahead. "
                    "Turn to explore. Keep firing to kill enemies.\n"
                )
                sys_ids = tokenizer(sys_prompt, return_tensors="pt", add_special_tokens=False).input_ids
                self._sys_block = await llm.create_block()
                await llm(sys_ids, cache_view=[self._sys_block])

            prompt = f"Screen (top to bottom):\n{img_text}\n\nYour action:"
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids

            frame_block = await llm.create_block()
            try:
                output = await llm(prompt_ids, cache_view=[self._sys_block, frame_block])

                logits = output.logits[0]
                scores = {}
                for name, token_ids in self._action_token_ids.items():
                    if len(token_ids) == 1:
                        scores[name] = logits[token_ids[0]].item()
                    else:
                        scores[name] = sum(logits[tid].item() for tid in token_ids) / len(token_ids)

                action = max(scores, key=scores.get)

                self._history_text += f"step {self._step_count}: {img_text[:40]}... -> {action}\n"
                if len(self._history_text) > 2000:
                    self._history_text = self._history_text[-1000:]

                return action
            finally:
                await llm.free_block(frame_block)
        except Exception as e:
            return "fire"