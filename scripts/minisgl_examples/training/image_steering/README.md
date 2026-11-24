# Image steering overfit

Open `image_steering_overfit.ipynb` on a GPU host using the project environment.
It records one real async image replacement, validates replay before training,
updates a student's parameters, reloads its checkpoint, and compares free-running
AsyncLLM generations before/after and without an image update.

## Source and interfaces

Developed in an isolated local worktree on `codex/image-steering-rollout-api`,
based on `origin/training-internal-refactor` at `33136eddd5d3b5f646ec88f5602baeae1a432cf0`.
`main` does not provide this training API. The older
`integrate-async-inputs-training` interface (`record`, `Replay`, `hidden`) is not
interchangeable with this checkout's public API:

```python
from minisgl.training import RecordedAsyncLLM, Trace, replay_trace

with RecordedAsyncLLM(llm) as recorded:
    # await all requests, using recorded.create_block/forward/prefill_block/free_block
    ...
trace = recorded.get_trace()
# Portable serialization: asdict(trace); restore with Trace.from_dict(data)
states = replay_trace(hf_student, trace.to("cuda"), rows=supervised_rows)
```

The reference text notebooks are under `../parallel_thinking/` and
`../distillation/`. This experiment reuses `process.isolated`,
`recording.record_sample`, and `loss.compute_supervised_loss/evaluate`.
It needs no new training framework or changes to the inference engine.

The image replacement is adapted from `origin/async_inputs` at `8f5c613`,
`scripts/image_async_inputs/image_async_thoughts_eval.py`. That evaluator has
prompt/image/thinker/writer blocks, token-count replacement barriers, optional
input reminders, and a switching probe. Here a single reasoning stream is enough:
`[prompt, image, reasoning]`. The barrier commits the last emitted token using
image A, clears A, prefills B conditioned on the prompt, then changes the view.
Historical reasoning KV is never recomputed. The notification is a new user
message using the model's chat delimiters, with no answer or supplied reasoning.
This schedule is identical for teacher and every student run.

Format-v2 traces retain request order, block IDs, write offsets, ordered visible
lengths, pixel tensors, grids, token types, three-axis MRoPE coordinates, rotary
spans, and cache mutation references. Replay rebuilds both image embeddings and
GDN state from current weights. Visibility uses token counts; rotary coordinates
use image-aware spans. Full attention performs a softmax over the actual complete
view. The loss labels the forward rows which sampled generated targets, including
the final EOS; all other rows use `-100`. Flattening the trace into ordinary
causal next-token training would change these conditional distributions.

## Task and models

The bundled normalized images are the existing MathVista correction pair
`0f6c4193c4c4860c`, problem 45, from `AI4Math/MathVista`, testmini (underlying
TabMWP). In image A September is 20, in B it is 14; August remains 17. The rate
changes from +3 to -3 minutes/month. Full upstream provenance, original revision,
SHA256 values, attribution, and source terms are in `assets/provenance.json`.
MathVista contributions are CC-BY-SA-4.0; underlying image/question rights remain
with their sources. These assets are retained for this research evaluation.

Defaults: Qwen3.5-4B teacher, Qwen3.5-0.8B student, greedy decoding, seed 7,
image replacement after exactly 96 emitted model tokens, 640 model-token cap,
65,536-pixel image budget, one GPU, no CUDA graphs. Every generated token is
selected from AsyncLLM logits. User-message delimiters and the notification are
input tokens and do not count toward the trigger or loss.

Training uses the existing differentiable forward, FP32 language weights,
AdamW (3e-6), gradient clipping, and at most 80 updates. The vision tower is
frozen but both images are re-encoded in replay. The optimizer's tracked parameter
delta must be nonzero. No teacher tokens enter evaluation generation.

## Remote setup and synchronization

SSH alias: `caradhras` (the server called caradhas in the task).
Use `~/experiments/image-steering-overfit/repo`; do not modify `~/async_llm`.
All edits originate in the local worktree. Synchronize without deletion:

```bash
rsync -az --exclude=.git --exclude=.venv --exclude=__pycache__ \
  --exclude=training_runs --exclude=uv.lock \
  /Users/warri/codex-remote/image-steering-overfit/ \
  caradhras:experiments/image-steering-overfit/repo/
```

No compatible pre-existing Python project environment was present on this server.
An isolated environment was created with uv and the project's CUDA 13 dependencies.
The training branch locks transformers 5.14.1; install that version explicitly after
syncing because the project's broad constraint permits newer versions:

```bash
uv sync --python 3.12 --extra dev
uv pip install --python .venv/bin/python -e ./scripts \
  'transformers==5.14.1' ipykernel nbclient nbformat
```

Put `.venv/bin` and `/usr/local/cuda/bin` on PATH for JIT compiler tools such as
ninja. `run_remote.sh` does this; calling `.venv/bin/python` alone is insufficient
when PATH has no ninja. Before each GPU stage inspect `nvidia-smi` and its process
list; use only a free GPU. Set `CUDA_VISIBLE_DEVICES` explicitly. Never stop other
users' processes. Separate stage CLI examples:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/minisgl_examples/training/image_steering/run_remote.sh \
  collect --model Qwen/Qwen3.5-4B --output training_runs/image_steering/teacher.json
CUDA_VISIBLE_DEVICES=0 bash scripts/minisgl_examples/training/image_steering/run_remote.sh \
  check --source training_runs/image_steering/teacher.json \
  --dtype float16 --output training_runs/image_steering/teacher_parity.json
```

The notebook executes the rest in order and is the main runnable entry point.
Select a fresh `RUN` to repeat it. It can reuse existing stage outputs only in
the explicitly selected directory, with model, task, and async setup checked.
The helper refuses to silently overwrite output files. Weights stay on the server;
copy diagnostic JSON, logs, the loss PNG and executed notebook locally. The final
report (`RUN_REPORT.md`) identifies the actual run directory and remaining limitations.

## Acceptance

1. Teacher: actual async image replacement, old visual value before the event,
   corrected value after, final -3, and EOS. Save failed attempts too.
2. Replay: compare target logprobs on unchanged weights, before/after the event;
   max absolute error ≤0.1. Validate teacher and the baseline student's own trace
   independently. Student probabilities of teacher targets need not equal teacher
   probabilities. No optimization begins if student parity fails.
3. Training: finite nonzero gradients, parameter updates, decreasing NLL, increasing
   target probabilities. Save/reload must retain teacher-forced loss.
4. Free generation: same inputs, agent settings, greedy policy, and token-count
   trigger. Report exact tokens, common prefix, positional agreement, sequence
   similarity, event sequence, visual premises, EOS, and final answer separately.
5. Counterfactual: the trained student without the update should answer +3. This
   checks image-state dependence, but also removes the notification; it is not
   a pixel-only causal ablation. The additional A→A control keeps the event and
   notification identical, but supplies the original pixels again; it must
   still answer +3.

Behavioral checks are deliberately task-specific: old 20 before, new 14 after,
correct revised answer, EOS, and the same external event. Explicit correction
language is a separate descriptive flag; a model can recompute correctly without
using a particular acknowledgement phrase. Read phase texts as well as metrics.
A low loss or final answer alone does not establish trajectory reproduction.

CPU checks: notebook schema + code parsing, helper import, lint, and
`tests/scripts/test_image_steering.py`. Existing trace/recording tests are run on
the server with CUDA hidden. No video understanding work is included.

The selected run uses `trained_low_lr/` (3e-6, early stop at 10 updates). The first
3e-5 run is preserved separately: it reproduced teacher tokens but failed both
visual controls. See `RUN_REPORT.md` for actual measured results.

## Replay settings after the training refactor

Every parity check, optimizer step, evaluation, and checkpoint reload uses the
settings stored in the collected rollout's `setup`. Defaults are
`replay_attention_backend="torch"` and `replay_query_chunk_size=64`, independent
of Hugging Face's `attn_implementation`. Reports record the effective
`replay_options`; the notebook refuses cached results with missing or different
settings. Use a fresh run directory when changing settings or code. Historical
`RUN_REPORT.md` results are not new measurements of these defaults.

The collection CLI accepts `--replay-attention-backend` and
`--replay-query-chunk-size`; subsequent stages inherit those values from the
trace. To exercise current automatic backend selection, collect teacher and
student into a separate directory with
`--replay-attention-backend auto --replay-query-chunk-size 1024`. The notebook
exposes the same fields in `SETUP`. Existing parity and quality gates still apply.
