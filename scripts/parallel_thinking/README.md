# Parallel thinking

Both notebooks use Qwen3-4B with `enable_thinking=False`. The training example
uses a reviewed 384-token-per-agent prefix; the no-barrier budget is 1024.

- `train_small_model.ipynb`: teacher generation, one-example Qwen3-1.7B training
  with BF16 CPU master weights/AdamW, and free generation. The teacher uses
  temperature 0.7, top-p 0.8, top-k 20; the student is evaluated with greedy decoding.
- `no_barrier.ipynb`: independent agent loops and a delayed-Bob check.
- `async_helpers.py`: coroutine, cache-publication and subprocess helpers.

Run `bash scripts/parallel_thinking/setup_env.sh`, select **AsyncLLM (uv)**,
and run from the repository root or this folder. GPU 2 is used for training,
GPU 1 for the no-barrier experiment. Restart the kernel before rerunning.

The chat template closes an empty thinking block in the input. No open thinking
prefix or periodic S1 reminders are added. Only a temporary checkpoint is used
for the student's free-generation check.
