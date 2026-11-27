# Image steering overfit

A Qwen3.5-4B teacher solves a visual question while its image is replaced during
generation. A Qwen3.5-0.8B student learns the recorded trajectory, reloads its
checkpoint, and generates its own answer under the same update schedule.
Use the [shared setup and Python API](../README.md), then install the example:

```bash
uv pip install -e './scripts[notebooks,image-steering]' 'transformers==5.14.1'
```

Open `image_steering_overfit.ipynb` with the project environment as its kernel.
Select one GPU with `CUDA_VISIBLE_DEVICES`. Outputs go to
`training_runs/image_steering_hf`; set `IMAGE_STEERING_RUN` to use another directory.
Use a fresh directory when changing code or replay settings.

The task comes from [MrDarkTesla/sharded_vqa](https://huggingface.co/datasets/MrDarkTesla/sharded_vqa),
record `0f6c4193c4c4860c` (MathVista 45). The corrected table changes the expected
answer from +3 to −3. `dataset.py` loads images, question, notification, labels,
and provenance from a pinned HF revision and checks file and image SHA256 values.
The first download is about 142 MB and stays in the Hugging Face cache.
Source attribution and terms are retained in each rollout's task metadata.

Prompt, image, and reasoning use separate cache blocks. After 96 generated tokens,
`experiment.py` replaces the image block and adds a correction notification while
retaining the reasoning cache. `RecordedAsyncLLM` captures the requests and views;
differentiable replay trains the student's language weights with the vision tower
frozen. Evaluation generates from model logits without supplied teacher tokens.
The notebook also checks the student without an update and with an A→A replacement
using the same notification.

Run individual collection and replay checks from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m minisgl_examples.training.image_steering.experiment collect \
  --model Qwen/Qwen3.5-4B --output training_runs/image_steering_hf/teacher.json
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m minisgl_examples.training.image_steering.experiment check \
  --source training_runs/image_steering_hf/teacher.json --dtype float16 \
  --output training_runs/image_steering_hf/teacher_parity.json
```

Defaults: greedy decoding, seed 7, 640 generated tokens, a 65,536-pixel image budget,
and up to 80 AdamW updates at `3e-6`. Replay uses `torch` attention with query chunks
of 64; `--replay-attention-backend` and `--replay-query-chunk-size` set these during
collection. Subsequent stages inherit them, and cached reports must match.
Use each command's `--help` for options; the notebook runs the complete workflow.

Output files cannot be overwritten. The [notebook](image_steering_overfit.ipynb)
includes executed outputs with rollouts, loss curves, and generation comparisons.
Plots are embedded in the notebook. This is an overfit check on one example and
a fixed update schedule.
