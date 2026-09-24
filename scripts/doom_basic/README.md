# Minimal DoomBasic demo

```python
import torch
import minisgl.llm
import transformers
import gymnasium, vizdoom.gymnasium_wrapper
import matplotlib.pyplot as plt
%matplotlib inline

llm = minisgl.llm.AsyncLLM("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, max_running_req=4, memory_ratio=0.9)
cache = await llm.create_block()  # stackable KV/GDN memory container; basic demo uses just one
env = gymnasium.make("VizdoomBasic-v1", render_mode="rgb_array", frame_skip=8)
prompt = """
This is a first-person videgame called Doom. Which action should I choose next: `wait`, `fire`, `right`, or `left`? Please use one of these actions, verbatim.
Your shot hits if the enemy is at the center of the screen, **precisely** above the gun barrel, otherwise the shot misses. You aim by moving your point of view left or right. If the enemy is to the right, move right. If it is to the left, move left.
Please analyze the image for only one paragraph, then write "Action: `your_choice`" with one of `wait`, `fire`, `right`, `left`, without quotes, e.g. Action: `wait`.
""".strip()

action_names = "wait", "fire", "right", "left"  # must be distinct tokens
action_token_ids = [llm.tokenizer.vocab[a] for a in action_names]
eos_token_ids = torch.tensor(llm.config.generation_config.eos_token_id).view(-1).tolist()
obs, *_ = env.reset()
plt.imshow(obs['screen'])
plt.show()

for i in range(100):
    inputs = llm.processor.apply_chat_template([{"role": "user", "content": [
        {"type": "image", "image": obs["screen"]}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
    cache.clear()
    for i in range(256):
        new_token_id = await llm.sample(await llm(**inputs, cache_view=[cache]))
        new_token = llm.tokenizer.decode(new_token_id)
        inputs = dict(input_ids=new_token_id.view(1))
        print(end=new_token, flush=True)
        if new_token_id in eos_token_ids or '\n' in new_token:
            break  # after the first paragraph
    inputs = llm.tokenizer('''\nAction: `''', return_tensors='pt', add_special_tokens=False)
    scores = (await llm(**inputs, cache_view=[cache])).logits.softmax(-1).flatten()
    action_index = scores[action_token_ids].argmax().item()
    obs, r, done, *_etc = env.step(action_index)
    print("Probe top:", *(f"| #{i+1} [{llm.tokenizer.decode(cand_i)}]: {scores[cand_i]:.3f}"
                          for i, cand_i in enumerate((-scores).argsort(-1)[:3])))    
    print("Action:", action_names[action_index], action_index, f"{r=}, {done=}")
    plt.imshow(obs['screen'])
    plt.show()
    if done: break
```
