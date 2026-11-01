# Full example → one-trajectory training

Open `train_small_model.ipynb` with the **AsyncLLM (uv)** kernel and Run All.
The notebook contains the Alice/Bob generation, trace loading, training loop and
free-generation check. `hogwild_formatting.py` retains the original Hogwild prompts
and step formatting (Apache-2.0, see `HOGWILD_LICENSE`).

The default teacher is **Qwen3-4B-Thinking-2507**, the student **Qwen3-1.7B**.
The initial prompt text and token IDs match the original QwQ-32B full example.
The periodic `insert_s1_prompt_every_tokens` intervention is disabled.
The teacher generates a new trace; training uses only that one trace.

```bash
bash scripts/parallel_thinking/setup_env.sh
.venv/bin/hf download Qwen/Qwen3-4B-Thinking-2507 --local-dir models/Qwen3-4B-Thinking-2507
.venv/bin/hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
.venv/bin/jupyter lab --no-browser --ip=127.0.0.1 --port=8890
```

On imladris, select a free RTX A4000 in the first cell (`GPU_INDEX`, default 2).
The recipe uses a 16 GiB Ampere GPU. Allow tens of GiB of host RAM for the
optimizer and FP32 master weights; BF16 forward/backward uses GPU. Use `.venv/bin/python` or `uv run --no-sync` with the prepared environment.
For remote Jupyter, open `ssh -L 8890:127.0.0.1:8890 imladris` on your computer.

Each run saves its trace, results, executable snapshots of the current notebook
functions, logs and student checkpoint under `training_runs/parallel_thinking/`.
The functions are executed in fresh processes to release GPU memory between stages;
`inspect.getsource` captures the executed cells, including unsaved edits.
Replay preserves recorded block visibility and RoPE positions. Teacher-forced
accuracy and independent free generation are reported separately.
