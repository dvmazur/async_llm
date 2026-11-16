# Async-input distillation sanity check

This workflow distills one **mid-reasoning correction** trajectory from
[`MrDarkTesla/math_500_sharded`](https://huggingface.co/datasets/MrDarkTesla/math_500_sharded).
It is the async-input counterpart of commit `ac7f86c`: a 4B teacher generates one
recorded trajectory, a 0.6B student overfits its sampled targets, the checkpoint is
reloaded, and the student runs the same protocol independently.

Each dataset row contains two `problem_shards`. The model starts reasoning from the
first shard. After `k_steps` decode requests, the second shard is appended to the live
prompt `CacheBlock` as `ADDITIONAL INFORMATION`; subsequent decode requests keep the
same `[prompt, output]` view and see the extended prompt. This is a single reasoning
stream with an asynchronous input, not Hogwild and not the thinker/writer demo.

Open `simple_distillation.ipynb`, select an idle GPU with at least 12 GiB free, and run
the cells from top to bottom. Defaults are:

- dataset: `MrDarkTesla/math_500_sharded`, `train`, row 0;
- teacher/student: `Qwen/Qwen3-4B` and `Qwen/Qwen3-0.6B`;
- correction after 32 generated tokens;
- 512-token generation budget;
- at most 100 optimizer updates at learning rate `1e-6`.

The teacher must answer correctly before training begins. Training uses teacher forcing
over every sampled output token; prompt and correction-prefill tokens have target `-100`.
The recorded cache layout is replayed exactly with `Trace`, `Replay`, and
`supervised_loss`. Overfit success remains accuracy 100% and loss below 0.01.

The final report keeps three outcomes separate:

1. symbolic correctness of teacher and student answers using `math_verify`;
2. async-input structure: exactly one correction, injected after the requested decode
   count, followed by decode requests that see the extended prompt block;
3. exact equality of independently generated tokens, targets, and normalized cache
   history. Correct free generation does not require exact teacher reproduction.

Files:

- `async_input_rollout.py` downloads one dataset row, performs injection, and records
  the executed trace and next-token targets;
- `sanity.py` trains, reloads, grades, and compares teacher/student traces;
- `simple_distillation.ipynb` orchestrates the end-to-end experiment.

