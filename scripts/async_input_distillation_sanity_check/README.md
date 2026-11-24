# Async-input distillation sanity check

This workflow distills one **mid-reasoning correction** trajectory from
[`MrDarkTesla/math_500_sharded`](https://huggingface.co/datasets/MrDarkTesla/math_500_sharded).
It is the async-input counterpart of the thinker/writer distillation example: a 4B
teacher generates one recorded trajectory, a 0.8B student overfits its sampled targets, the checkpoint is
reloaded, and the student runs the same protocol independently.

Each dataset row contains two `problem_shards`. The model starts reasoning from the
first shard. After `k_steps` decode requests, the second shard is appended to the live
prompt `CacheBlock` and the live output stream as `ADDITIONAL INFORMATION`; subsequent
decode requests keep the same `[prompt, output]` view and see the extended prompt. This is a single reasoning
stream with an asynchronous input, not Hogwild and not the thinker/writer demo.

Install the library and examples in the notebook's environment first:

```bash
uv sync
uv pip install -e ./scripts
```

`math-verify` grades symbolic answers and `pyarrow` reads the dataset; both are runtime
dependencies. Worker processes use the notebook's Python interpreter and environment.
Open `simple_distillation.ipynb`, select an idle GPU with at least 12 GiB free, and run
the cells from top to bottom. Defaults are:

- dataset: `MrDarkTesla/math_500_sharded`, `train`, row 3;
- teacher/student: `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-0.8B`;
- correction after 16 generated tokens;
- 512-token generation budget;
- FP16 for the notebook's Turing-compatible path (`bfloat16` remains the helper default);
- sequential AdamW updates on the low-memory path (`foreach=None` remains the helper default);
- explicit Torch reference replay attention with 128-token query chunks;
- at most 100 optimizer updates at learning rate `1e-6`.

The teacher must answer correctly before training begins. Training uses teacher forcing
over every sampled output token. Prompt and correction tokens are masked with `-100`,
except the final live correction row whose logits select the next generated token.
The pending token is committed before injection without adding a supervised continuation.
`RecordedAsyncLLM` records the cache layout; `Trace.to(...)` prepares it and the examples'
`compute_supervised_loss` replays it. Overfit success remains accuracy 100% and loss below 0.01.

Set `ATTENTION_BACKEND` and `QUERY_CHUNK_SIZE` once in the notebook; training, evaluation,
and checkpoint verification receive the same settings, which are saved in their reports.
The worker CLI exposes `--attention-backend` (`torch`, `auto`, `torch_optimized`, `fa4`)
and `--query-chunk-size`. `torch` avoids hardware-dependent automatic backend selection;
it uses current training's reference math, not a promise of bit-exact legacy replay.
The 128-token default retains the original workspace budget. The FP16/FP32-master and
`foreach=False` settings retain the intended Turing path; actual hardware support and
successful convergence must be established by running the workflow.

The final report keeps three outcomes separate:

1. symbolic correctness of teacher and student answers using `math_verify`;
2. async-input structure: exactly one correction, injected after the requested decode
   count, followed by decode requests that see the extended prompt block;
3. exact equality of independently generated tokens, targets, and normalized cache
   history. Correct free generation does not require exact teacher reproduction.

Teacher validity, training overfit, and reloaded loss/accuracy are assertions. The final
student answer, async structure, and exact reproduction are separate reported outcomes;
a completed notebook alone does not mean those final checks passed. No outputs are saved
in the committed notebook.

Files:

- `async_input_rollout.py` downloads one dataset row, performs injection, and records
  the executed trace and next-token targets;
- `sanity.py` trains, reloads, grades, and compares teacher/student traces;
- `simple_distillation.ipynb` orchestrates the end-to-end experiment.
