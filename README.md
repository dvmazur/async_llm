# Portable Speleo runner

**Короткая инструкция установки и запуска 15×10: [QUICKSTART.md](QUICKSTART.md).**

One model per GPU, independently repeating async pipeline slots. The engine owns
batching. Pipeline code owns roles, history, its World and Recorder. No TP, image
replay, profiler, hidden warmup episodes, or runtime package installation.

The main entry point is **an ordinary Python file in `experiments/`**, not a
parameter-heavy launcher CLI. Runner/pipeline sources are read from this checkout;
they do not need to be installed in either the launching Python or the engine venv.

```text
experiments/speleo_15x5.py  # edit venv, model/config, GPUs, repeats, output path here
experiment_runner/        # execution and telemetry infrastructure
environment/              # explicit setup, dependency and Qwen/Craftium checks
pipelines/speleo.py        # role coroutines, token loop and all their communication
pipelines/prompts.py       # prompt text; no scheduling
pipelines/world.py         # async environment transport
tools/                    # optional checks, plotting and release packaging
pyproject.toml            # dependency groups and native build constraints
```

## Setup (explicit; never run by Runner)

Unzip the release. Setup installs pinned `uv` locally if it is absent, without
editing shell startup files. Choose an engine checkout or unpack the
included **exact git bundle**, without needing private GitHub credentials:

```bash
git clone engine.bundle /path/to/engine
python -m environment.setup \
  --engine /path/to/engine --venv /path/to/venv \
  --craftium /path/to/craftium --jobs 8 --system-deps \
  --download-model /path/to/Qwen3.6-35B-A3B-FP8
```

Run these commands from the extracted source directory. `--system-deps` explicitly
authorizes apt packages and sudo if needed; it **never installs GPU drivers**.
The machine needs a working CUDA toolkit compatible with its GPU/driver.
Model download uses `hf`, overlaps the native build, and pins the checkpoint
revision. Omit `--download-model` to use existing weights. Omit `--system-deps`
when system dependencies are already installed. Existing venv modification requires
`--update-existing`; existing dirty Craftium sources are not overwritten.

The bundled engine pins Torch, FlashInfer, FLA and the CUDA-13 SGLang kernels.
This checkout declares runtime/test/plot dependencies and build constraints in
`pyproject.toml`; there are no separate requirements/constraints text files.
Setup first syncs the selected engine's lock, then resolves the runtime group and
Craftium together, constrained by that lock. The exported constraints are passed
through stdin, not maintained in another configuration file. A conflict fails
instead of silently upgrading the engine. Setup does **not** install this runner.
Preflight checks actual imports, not just metadata. RTX should not fall back due
to a missing CUDA library. On GB10 SM121 the engine's intentional MoE fallback is
reported explicitly. The NVIDIA cuSPARSELt 0.8.1 ARM wheel has a known `sbsa`
internal tag despite its `aarch64` filename; setup accepts only that exact tag
defect after validating the library's AArch64 ELF header, with a warning. All
other dependency-check failures remain fatal.

These checks run during explicit Setup, not before every experiment. To rerun them
manually from the checkout with the prepared venv: `python -m environment.check`
and `python -m environment.dependencies /path/to/uv`. They are specific to the
Qwen/Craftium environment, not a generic Runner contract.

Workers prepend the selected venv's `bin` directory to PATH: JIT build tools such
as ninja must come from that environment too. Preflight checks ninja before loading
weights. Craftium uses setuptools' pinned compatible editable mode so a neighbouring
checkout directory cannot shadow the actual package via a namespace import.

Setup installs Craftium revision `8eb8707cb756df47e76131a0058ab724d2383c76`.
The default model is `Qwen/Qwen3.6-35B-A3B-FP8`, revision
`61a5771f218894aaacf97551e24a25b866750fc2`. Old results with a different weight
revision must not be treated as exactly matched controls.

## Run

Open `experiments/speleo_15x5.py` and edit its settings:

```python
VENV = Path('/path/to/engine-venv')
MODEL = Path('/path/to/Qwen3.6-35B-A3B-FP8')
RESULTS = HERE / 'results' / 'experiment-001'
GPUS = [0, 1]
PIPELINES_PER_GPU = 15
REPEATS = 3
ACTIONS = 5
# ENGINE_PARAMS is an ordinary Python dict in the same file.
```

Then run the file, from any working directory:

```bash
python /path/to/repo/experiments/speleo_15x5.py
```

The launching Python only needs the standard library. The file adds its own source
root; Runner dispatches workers through the chosen `VENV/bin/python` with the same
source tree. It never installs packages on startup. Edit the file again for another
experiment, or copy it to another name inside `experiments/`.

If you prefer neighbouring JSON files, parse them explicitly in that Python file:

```python
import json
ENGINE_PARAMS = json.loads((HERE / 'engine.json').read_text())
```

The file contains the actual `Runner(...).set_engine_params(...).set_pipeline(...)
.set_concurrency(...).set_results_directory(...).run(gpus=...)` call. There is no
separate experiment format or hidden conversion into environment variables.
Keep the call under `if __name__ == '__main__'`. Each GPU worker re-executes this same
file with the same arguments under the selected Python. Its `Runner.run()` enters
worker mode and uses the locally recreated factory directly, then exits the child.
There is no module/function-name resolver or callable pickling. Factory closures and
lambdas are supported. Keep file startup declarative: code before `run()` executes
in every process. Use one `run()` per experiment file; do not edit it during a run.

The example above runs15slots ×3episodes ×2GPUs, each with5actions. Slots start their
next repeat independently; each GPU loads the model once. No SSH orchestration or TP.

`run()` blocks and returns `None`. It prints a numeric Summary and writes all results.
It never overwrites a nonempty result directory. Engine parameters are copied on
assignment. JSON is optional: parse it yourself and pass a dict. Values must be
serializable; use strings for paths and dtypes. Prefer absolute model paths.
Unknown engine/adapter parameters fail, not silently change the run.

`experiments/speleo_15x5.py` contains the fast validated layout: KV page
size 16, 8192 pages, max sequence 32768, decode profiles 4/16/48/64, prefill profiles
256/1024/4096, prefill cap 4096, depth 16. Attention workspace is managed by the
engine's existing backend; this runner does not override it. This is **FP8 weights
with FP32 GDN state**, not the experimental BF16 GDN variant. The cap is the engine's
segment-expanded query-row cap, not a number of bytes. Larger concurrency requires
an explicitly sized config and memory check; no automatic resizing/fallback.

## Change a pipeline without changing the engine

Write a callable `make_pipeline(engine, context)`. It creates
a fresh `SpeleoPipeline(world, recorder, engine, context=context, max_actions=...)`.
Engine is shared; never close it inside a Pipeline. Context supplies episode ID,
world/model seeds, slot/repeat IDs and the episode's result directory. A custom
pipeline implements async `run()` with its own `finally` cleanup. It should check
the runner-provided `stop_requested()` between safe actions.

`RepeatedPipeline` stores a factory, not live worlds or CUDA tensors. The experiment
recreates it inside each worker; it is not serialized. Keep launch code in
the `if __name__ == '__main__'` guard. The model-facing adapter lives in
`experiment_runner/engine.py`; it has no role names, temperatures or game rules.

### The coroutine protocol is one file: `pipelines/speleo.py`

There are four persistent role tasks per episode:

- `observer` receives previous/current frames and the last action, prepares the image
  KV block, chooses its input chain and generates a live description.
- `planner` owns refresh/REPLAN decisions, freezes history, chooses its live inputs,
  and publishes completed plans at action boundaries. The executor reports observations;
  it does not construct planner prompts or decide when a plan is stale.
- `falsifier` receives the draft and evidence, decides whether review is needed
  (new plan or REPLAN), builds its own read chain and generates objections.
- `executor` drives the episode, performs its sequential draft/refine generations,
  selects an action and calls `await world.pass_action(...)`.

`generate_tokens()` is the shared model-IO loop:
prefill its instruction block, sample one token, append through decode to its KV tail,
then notify readers. A `Generation` carries the input block chain, instruction block,
growing output block, text and completion/token notifications. The executor passes
these live KV blocks to other roles rather than copying partially generated text.
Observer, falsifier and executor await this loop directly. Planner explicitly owns
one `planner/tokens` child task: a long generation must not prevent its controller
from acknowledging later observations. It starts no second generation until the
first is finished/published, and drains its child before shutdown. This is local
policy concurrency, not a second scheduler or a new engine abstraction.

All queues, waits, read/write chains, planner publication, REPLAN decisions and
environment steps are visible in that file. The observer/falsifier four-token waits
do not wait for full generation; planner can continue across environment actions.
The executor's draft and refine use separate RNG streams, as before, but never ran
concurrently and now execute in the same executor coroutine.

Role budgets, temperatures, top-k/top-p and seed offsets are `ROLE_PARAMS` in the
policy. An experiment can pass `role_params=...` to override them. Engine sampling
receives explicit parameters and an RNG, with no role lookup. Token scoring is a
generic engine operation; action names and argmax selection belong to the executor.
`BlockHandle` in `experiment_runner/blocks.py` owns a native engine block.
Each consumer acquires its own handle with `share()` and releases it with
`await handle.aclose()` (or `async with`). Closing the last handle calls the engine's
normal `free_block`; there is no separate pool, allocator or block registry.
Closing a handle twice is harmless; reading/sharing a closed handle is an error.
Snapshot uses the native merge operation. Cached system prefixes have one engine
owner plus an independent handle per episode. All handles must be closed before
engine shutdown; Python assignment is only an alias, not an ownership acquisition.
We do not call asynchronous GPU cleanup from a Python destructor.
The old orchestration classes `Client`/`Stream` and `pipelines/core.py` are removed.

World returns an Observation from `await reset()` / `await pass_action(action)`.
The unmodified Craftium bridge can hold the GIL, so World uses a spawned helper
process and async IPC. Images are owned snapshots. Native rendering uses software
Mesa with two rendering threads and an Xvfb display when no DISPLAY is provided.
Position is logged for evaluation, **not added to model prompts**.

The original observer/planner/draft/falsifier/refine prompts, default token budgets,
live-tail/snapshot dependencies, REPLAN condition and incoming-feedback order are
preserved. Async environment/recording and independent repeats intentionally change
scheduling. Identical seeds do not promise bit-identical generated trajectories.

## Telemetry and optional media

```python
Recorder(context.results_directory, dump_images=False, gif_on=False)
```

Both flags default to false. No PNG/GIF/base64 images are stored in that mode;
model image inputs are unchanged. GIF-only mode streams frames without permanent
PNG files or retaining the entire episode in RAM. Image queues are bounded. Writer
errors are propagated. The agent's initial height and height after every action,
role text, rewards, actions and event times remain in numeric/text logs.

Summary runs automatically after workers/log writers finish. It is also reusable:

```bash
python -m experiment_runner summary /path/to/results/run
python -m tools.plot_height --run 'Minimal=/path/to/results/run' --output height.png
python -m experiment_runner.artifacts /path/to/results/run results.zip
```

Plotting is local-only and requires matplotlib (`uv pip install --project /path/to/repo
--python /path/to/analysis-venv/bin/python --group plot`). Plot data includes the number of available episodes per step;
missing heights/steps are not replaced by zero. Use `--descent` for height loss
relative to reset. Packing never removes source results. Partial results require
`--allow-partial`. ZIP checksums are stored and CRC is verified.

Metrics explicitly distinguish:

* Generated TPS: sampled **role** tokens, including terminal tokens, divided by
  full workload wall time, including the first action and background role drain.
* Tokens/action: the same role-token count / completed actions, an amortized cost.
* Mean decode batch: active workers per actual decode-forward, excluding padding.
* Mean prefill batch: both active requests and new token rows per prefill-forward.
* `output_tokens_including_readouts_tps`: additionally counts action readouts,
  matching the old launcher's numerator; it does not drop its first-action interval.
* `decode_rows_per_second`: input decode rows, **not** all generated output tokens.
* Sampled GPU utilization: nvidia-smi samples, **not** a trace-derived idle fraction.

No CUDA synchronization is added per forward. Per-client overlapping windows are
not summed as global token counts. Means use sums/counts, not means of means.
Partial/failed runs are marked, not silently treated as successful repeats.
Model initialization and episode/reset/flush boundaries remain separately recorded.
`experiment_runner/fi_warmup.py` owns the per-process sampler warmup. It makes one
synthetic call with its own RNG after Engine initialization, before any episodes.
This is not a Setup check or a discarded agent action.

There is no runner workspace option or custom engine workspace parameter.
The existing Attention backend allocates/binds its scratch; prefill graph setup
queries FlashInfer's required sizes and grows buffers when necessary. Runner does
not inspect or patch these wrappers.
Construction is timed as a whole; loader/capture methods are no longer patched just
to split their internal timings.

## Validation status

CPU tests cover independent repeats, seed
assignment, event-loop-safe recording, all media switches, feedback order and block
cleanup, fail-fast, arithmetic of overlapping episodes, plotting data, result packing,
and Setup ordering/download error handling.
Run them with `python -m pytest tests -q` in a test environment with NumPy/Pillow.
`tools.check_world` and `tools.small_check` retain the previous integration checks.
Before the experiment-file layout refactor they passed native reset/step and a real Qwen3.5-0.8B FP8 run on GB10:
2 slots × 2 repeats × 2 actions, one engine load, all 8 actions completed, role
and engine token counts reconciled, PNG/GIF switches exercised. Setup completed
in an isolated venv with the pinned native Craftium build.

Historical reference measurement (including the old 2 GiB workspace override), with
real Qwen3.6-35B-A3B FP8 on RTX PRO6000,15pipelines ×5actions:

| Metric | New runner | Original portable control |
| --- | ---: | ---: |
| Actions completed |75|75|
| Full workload seconds |42.507|48.263|
| Role tokens/sec |222.03|198.70|
| Tokens/sec including action readouts (old numerator) |223.80|200.26|
| Mean decode batch |28.16|29.07|
| Mean prefill requests |6.75|9.86|
| Mean new prefill rows |422.08|511.10|

Same engine commit70942c4, checkpoint revision, layout and assigned role seeds;
no profiler and no first-action exclusion. This is a single paired experiment,
not a statistical speed guarantee. The old control writes PNGs and steps worlds
synchronously; new runner uses async World/Recorder with media disabled. Scheduling
and initial-world readiness differ. Prompts and role/core logic are preserved,
not bit-identical trajectories. Native imports passed; no OOM; all229result files
were downloaded and SHA256-verified before releasing the GPU. Real multi-GPU
execution was not available; GPU assignment/repeat scheduling are covered by CPU tests.

The current stock-workspace/coroutine path was tested on GB10 with the small FP8 model;
RTX/A3B throughput has not been remeasured after these refactors. The
historical table is not a performance claim for the new workspace policy.

Coroutine tests control producer progress explicitly: the executor reads unfinished
KV tails after four tokens, planner spans actions with an unchanged history snapshot,
failed producers unblock consumers and release ownership, and two episodes share an
engine without sharing role RNG state. A deterministic comparison with the frozen
old policy matched52protocol operations (prompts, ordered read chains, writes,
snapshots, images and sampling parameters), ignoring interleaving order. Actual
scheduling can change, so bit-identical trajectories are still not promised.

## Release tools

`tools/build_release.py` builds a distributable ZIP including an exact engine git
bundle. It is for packaging, not installation or experiments:

```bash
python tools/build_release.py --engine /path/to/engine --output /path/to/release.zip
```

The generated `tools/release.json` contains the engine revision and file checksums.
Neither Runner nor an experiment reads it. Results packaging is a separate helper
(`python -m experiment_runner.artifacts`), not the source-release builder.
