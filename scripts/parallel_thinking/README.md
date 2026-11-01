# Parallel thinking

- `train_small_model.ipynb`: Qwen3-4B-Thinking-2507 generates one trajectory;
  Qwen3-1.7B learns it with **BF16 master weights and AdamW moments on CPU**,
  BF16 forward/backward on GPU, FP32 RoPE buffers. The actual training targets
  and context examples are displayed. A temporary checkpoint is used only for
  independent generation and deleted automatically.
- `no_barrier.ipynb`: independent Alice/Bob loops, with synchronization only
  during cache publication. A delayed Bob demonstrates independent progress.
- `async_helpers.py`: small coroutine, publication-lock and process helpers.
  The main generation and training logic stays in the notebooks.

Both use the original Hogwild prompts without periodic collaboration reminders.
The formatter and its Apache-2.0 license are retained from Hogwild.

Run `bash scripts/parallel_thinking/setup_env.sh`, then start Jupyter from the
repository root and select **AsyncLLM (uv)**. Models are already under `models/`
on imladris. The first cell selects an RTX A4000: GPU 2 for training, GPU 1 for
the independent-loop experiment. Restart the kernel before running again.
No permanent run directories, checkpoints or JSON reports are created.
