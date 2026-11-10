# MATH-500 with Gemma async_thoughts

Run from the repository root, using the engine's Python environment:

```bash
uv pip install --no-deps -e ./scripts
uv pip install 'math-verify[antlr4_13_2]==0.9.0'
python -m eval.math500 run \
  --model /path/to/gemma-4-31B-it-FP8-block \
  --download-data \
  --output results/math500-gemma
```

No example config needs copying. `--model` is your local checkpoint directory.
Both original BF16 (`google/gemma-4-31B-it`) and supported pre-quantized FP8
checkpoints work: the runner reads `config.json` to select the mini-sglang
quantization setting. BF16 weights stay BF16; they are not quantized at loading.
`--download-data` downloads the pinned ordinary MATH-500 dataset (~0.45 MB)
only if `datasets/math500.jsonl` is missing. `--output` must be a new directory.
Reference answers are used only by the offline grader, never by the model.

Defaults: one question at a time, **8192 thinker + 8192 writer tokens**,
`temperature=1.0`, `top_p=0.95`, `top_k=64`, training recording enabled.
The roles run concurrently using the original
[`async_thoughts` demo](../scripts/minisgl_examples/demos/async_thoughts/demo.py).
`async_thoughts.py` adapts Gemma's chat markers and connects the demo to the evaluator;
it does not implement another thinker/writer algorithm. Gemma runs eager, without CUDA graphs.

## Select questions and resume

- `--limit 5`: first five questions.
- `--start 100 --limit 50`: questions 100–149.
- `--shard-index 0 --num-shards 4`: one partition of complete questions, for
  independent GPU instances. This is **not** the removed delayed-input
  MATH-500-Sharded experiment. Use a separate output directory per instance.
- `--resume`: continue the same output directory with the same settings,
  data, selection and code. Completed questions are not rerun.
- `--data /path/to/math500.jsonl`: use another local dataset location.

## Change settings

Optional `--config experiment.json` overrides defaults; `--model` overrides
its `model_path`. For example, a short functional run:

```json
{
  "max_think_tokens": 128,
  "max_answer_tokens": 128,
  "engine": {"num_page_override": 512}
}
```

Other settings: `sampling` (temperature/top_p/top_k), `seed`, `probe_period`,
`record_training_trace`, and `engine` parameters. The default KV pool has
1536 pages of 16 tokens, about 20.625 GiB for this Gemma. The runner checks the
prompt plus generation budget against capacity; it never silently truncates.
For a single-stream control, explicitly set `mode="plain"` and, optionally,
`max_new_tokens` and `thinking`. Normal/default mode is `async_thoughts`.

## Results

- `results.jsonl`: full thinker/writer text and token IDs, probe/state events,
  timings, stop reasons, and Math-Verify grading of the writer's final boxed answer.
- `traces/*.json`: standard `RecordedAsyncLLM` training traces with input
  batches/cache updates and row-aligned sampled targets/logprobs. No full logits
  or KV tensors. Trace paths in results are relative to the output directory.
- `summary.json`: completed/graded counts, accuracy and generated TPS; updated
  after each question. TPS includes prefill and recording during generation,
  but excludes model loading, grading and trace JSON export.
- `manifest.json`, `timing.json`: effective settings, source/data fingerprints
  and separate model-loading time.

Terminal/discarded samples count toward the generation budget; prompt/probe
inputs do not. Trace batches are session batches, not necessarily physical
forwards when prefill is chunked. Recording Gemma inference does **not** add
Gemma differentiable replay/backward to the upstream Qwen trainer.

CPU checks: `python -m pytest -o addopts= tests/eval`.

## Vanilla SGLang baseline

In a **separate SGLang environment** (`sglang==0.5.23` and the same optional
Math-Verify grader), run the same entry point:

```bash
python -m eval.math500 run --backend sglang \
  --model /path/to/gemma-4-31B-it-FP8-block \
  --download-data --limit 10 --output results/math500-sglang
```

This is ordinary single-stream generation, **not** the async thinker/writer
pipeline. Same complete problems, MATH instruction, sampling settings and 16384
total output-token budget. With thinking enabled, its initial input matches the
demo's thinker prefix exactly, including the prefilled thought channel and newline.
The same per-question seed is passed to SGLang; different engines do not promise
identical random samples. Only the final answer channel is graded.
Raw output token IDs, text and SGLang metadata (including output-token logprobs)
are saved in `results.jsonl`. There is no mini-sglang training trace for this path.

`engine` overrides in this mode are **SGLang Engine arguments**, not mini-sglang
settings. This Gemma baseline defaults to the public Triton attention backend and
a batch-1 decode graph. Checkpoint quantization is autodetected by SGLang. The repository lock
contains `sglang-kernel`, not the full SGLang runtime; do not install the baseline
over the mini-sglang environment. No SGLang library patches are applied.
For a BF16 baseline, pass the original BF16 checkpoint directory to `--model`;
`dtype=bfloat16` alone does not turn an FP8 checkpoint into an original BF16 model.
