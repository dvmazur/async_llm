# Async image thinker/writer evaluation

`image_async_thoughts_eval.py` consumes the self-contained dataset Parquet.
Four persistent blocks are used: prompt, image, thinker, writer. Thinker sees
`[prompt,image,thinker]`; writer sees `[prompt,image,thinker,writer]`.
The fifth, transient probe reads text snapshots of both streams (the same
decision task as text async thoughts), then is freed. It does not reread images.

Image 1 is prefilled exactly once. After `--k-steps` emitted thinker tokens,
both decode requests finish before any mutation; the old image block is freed,
and image 2 is prefilled into a fresh block conditioned on the prompt. Both views
are updated. Existing thinker/writer KV and emitted tokens are retained, not
recomputed. This intentionally retains reasoning based on the old image.
Thinker and writer decode concurrently within a round. Probes and mutation run
at round barriers, avoiding cache mutation while another request reads it.

Options (all routing options default on):

- `--[no-]shard-to-prompt`: append text shard 2 to the prompt before image refresh.
- `--[no-]shard-to-thinker`: append text shard 2 to the thinker after refresh.
- `--[no-]writer-reminder`: notify the writer without describing the visual edit.
- `--[no-]defer-writer-reminder`: defer notification until the next writer paragraph
  boundary; disabling it inserts the notification immediately.
- `--k-steps 0`: corrected image and both text shards from the start.
- `--k-steps -1`: original image/shard 1 only, never update.

If the writer ends before k, no replacement occurs; this is recorded, not hidden.
A deferred reminder can remain pending if the writer ends before another paragraph.
Token budgets count model-generated tokens, excluding prefills/injected fragments.
The last emitted/pending token is committed before inserting text, without
double-counting or silently adding an unlogged sampled token.

## Run

Use the repository uv environment, not the CPU dataset-construction environment:

On this machine, `bash scripts/image_async_inputs/run_qwen38_gpu1.sh` sets GPU 1,
`HF_HOME=/mnt/LLM`, CUDA 12.8 and the pinned cached model snapshot. Additional
arguments override the default run options; use a fresh output path for changed settings.

```bash
uv pip install --python .venv/bin/python pyarrow
HF_HOME=/mnt/LLM CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/image_async_inputs/image_async_thoughts_eval.py \
  --model-name Qwen/Qwen3.8-27B --k-steps 64 --budget 16384 \
  --output scripts/image_async_inputs/eval_runs/qwen38_27b_k64
```

`--model-name` also accepts a pinned local snapshot path. `--prepare-only` verifies
all image token IDs, masks and grids on CPU. `--start/--end` select a Parquet row
range. Default image size cap is 1,048,576 pixels. Matching normalized PNG dimensions
alone is not sufficient: processor parity is checked on every pair before generation.
Default KV capacity is 98,304 tokens with 256-row chunked prefills, leaving
headroom for hybrid-state workspace on an 80GB GPU. `--kv-tokens 0` instead
uses `--memory-ratio` (0.8). The ratio includes model weights, not just KV memory;
0.4 is too small to load this 27B BF16 model and allocate KV.

Each sample saves emitted token IDs/text, probe decisions, replacement/reminder
positions, EOS state, image-grid metadata, timings, and separate old/new-answer scores.
Only `row.inputs` enters generation. Labels/provenance are never put in prompts.
The output config binds dataset checksum and flags; incompatible resumes are refused.
Failures write an error file and stop rather than silently improving the denominator.
The summary is checkpointed after every sample. Results are git-ignored.

The scorer extracts the last balanced `\\boxed{...}`, normalizes exact text/choices,
compares semicolon-delimited state sets, and uses `math_verify` for numeric/formula
targets. Missing boxes count incorrect. This is conservative automated scoring, not
a universal semantic judge: retain traces and manually adjudicate prose/format variants.
Report each source/category separately and compare k=0 and k=-1 controls before
claiming a recovery effect. The k=0 baseline always includes both shards regardless
of routing ablations (which apply only to delayed updates).

CPU protocol tests:

```bash
cd scripts/image_async_inputs
../../.venv/bin/python -m unittest test_image_async_thoughts_eval.py
```

## Initial verification (2026-09-17)

### Fixed-50 sanity sweep

The initial full-dataset run was superseded by a fixed 50-sample sweep at
`k=-1,0,16,32,64,128,256,512`. `fixed_subset_50.json` binds the ordered IDs to
the dataset SHA256: MathVista 8, each other source 7. Selection uses seed 42,
source round-robin and category interleaving; it does not inspect answers or outcomes.
Per-sample sampling seeds are based on original Parquet row indices and remain
identical across k values. Budget is 16,384 tokens per stream, all routing
options enabled, writer reminder deferred. These are 400 separate evaluations.

```bash
bash scripts/image_async_inputs/run_fixed50_sweep_gpu1.sh
```

GPU 1 works through ascending k values; `run_fixed50_gpu6.sh` adds GPU 6,
working backwards from k=512. Per-condition locks and verified-completion checks
prevent duplicate evaluations. GPU 6 uses a separate rendezvous port. Individual
logs and checkpointed summaries remain under the legacy directory
`eval_runs/qwen38_27b_gpu1_fixed50_16k/k_<k>/`. Each launcher stops
on the first failed condition and holds a lock against duplicate sweep launches.
Rerunning the same code/config resumes completed samples; changing the manifest
or configuration requires fresh output paths. `--sample-manifest` cannot be
combined with `--start/--end`. At large k, a writer may finish before the update;
use the recorded image-replacement count when interpreting accuracy.

After the fixed sweep, `after_fixed50.sh` waits for its campaign lock, verifies
all eight complete 50-sample outputs (matching IDs, dataset hash, configuration
and seeds), writes `RESULTS.md` and `results.json` in the fixed sweep directory,
then runs `run_full_sweep_gpu1.sh`. Any missing/failed/mismatched condition prevents
the full run from starting. Full evaluation uses all 513 samples at the same
eight k values (4,104 evaluations), GPUs 1 and 6 and unchanged generation settings.
The full launcher starts both workers in opposite k order if GPU 6 is free;
otherwise GPU 1 handles all conditions. The GPU 6 wrapper refuses to load a model
while another compute process occupies that device. GPU availability checks are
best-effort, not a cluster-wide reservation.
Report validation permits different GPU IDs/ports, but not experiment settings.
This local handoff writes reports but does not send a chat notification.

The resumed campaign runs in detached tmux session `fixed50-16k` on the
`image-async-eval` socket, so it does not depend on a foreground tool session.
Its command runs the fixed sweep followed by `after_fixed50.sh` only on success.
Inspect it with `tmux -L image-async-eval list-sessions`. Fixed-sweep logs append
on restart, preserving earlier diagnostics alongside resumed output.

The original 2,048-token sweep was stopped at the user's request and its outputs
are preserved separately. The 16k sweep starts fresh; no old-budget results are
reused. Both thinker and writer have a 16,384 generated-token cap, excluding
injected text; writer EOS can still terminate a sample earlier. The context
limit is 65,536 tokens, and the 98,304-token shared KV pool accommodates persistent
streams plus the transient probe. The queued full sweep uses the same settings
under `eval_runs/qwen38_27b_gpu1_full_sweep_16k/`.

Seven CPU tests passed. Processor parity passed on all 513 pairs using the pinned
Qwen3.8-27B snapshot. A real GPU 1 smoke run (k=64, budget=256) completed:
sample `0055d6580f9409e9` changed from reference Yes to No; final prediction No.
Image replacement occurred at thinker token 64 / writer token 8. The deferred
reminder was inserted at thinker token 132 / writer token 40. Writer reached EOS
at 237 generated tokens; thinker reached its 256-token cap. The thinker trace
first used the edited value 10 and then recognized the corrected value 12.6.
This verifies the mechanism, not aggregate accuracy or a controlled recovery effect.
The first run took 304 seconds including first-use kernel compilation.
