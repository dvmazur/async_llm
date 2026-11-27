# MATH-500 Hogwild budget sweeps

This workflow generates traced reasoning for ordinary Qwen and Hogwild with
1, 2, 4, 6, 8 and 16 workers, measures short answers from truncated traces, and
exports accuracy tables on the same 500 MATH-500 problems.

The final score uses the original answer when the saved reasoning finishes at or
before the requested cutoff. It uses the forced eight-token answer only when
reasoning continues beyond that cutoff. Selection depends on the recorded
reasoning length, including an exact tie, and preserves the selected answer's
original grade.

## Environment

Use Linux x86-64, Python 3.12.3, an NVIDIA driver supporting CUDA 13, and a free
NVIDIA H200 141 GB GPU for each evaluator. One GPU can execute the full workflow
serially. The longest measured readout contained 256,446 input tokens, so the
16-worker and 16k configurations require the documented context and memory
settings. Multiple GPUs divide independent problem indices across evaluators.
Install a CUDA-13-compatible C++20 host compiler and Linux build tools, such as
GCC and Make, for the JIT-compiled kernels. The historical host compiler version
was not retained.

The repository's `uv.lock` pins the inference dependencies. The supplemental
[requirements.txt](requirements.txt) pins the grader and CUDA compiler packages
present in the captured experiment environment. Principal versions are:

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| Torchvision | 0.28.0+cu130 |
| Transformers | 5.14.1 |
| FlashInfer | 0.6.17 |
| Flash Linear Attention / fla-core | 0.5.2 |
| sglang-kernel | 0.4.6.post1+cu130 |
| safetensors | 0.8.0 |
| CUDA compiler | 13.0.88 |
| Math-Verify | 0.9.0 |
| latex2sympy2_extended | 1.11.0 |
| ANTLR Python runtime | 4.13.2 |
| SymPy | 1.14.0 |
| NumPy | 2.4.4 |

The retained Python 3.12.3 environment record comes from the initial B200 stage.
A separate complete Python-patch and driver-version record for the H200 stage
was not retained. The package lock and supplemental pins below preserve the
available software versions.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create
the environment from a clean checkout:

```bash
git clone --branch tmp/hogwild_multiworkers https://github.com/dvmazur/async_llm.git
cd async_llm

uv python install 3.12.3
uv sync --frozen --python 3.12.3
uv pip install --no-deps -r docs/hogwild/requirements.txt
uv pip install --no-deps -e ./scripts
```

The following commands use `uv run --no-sync` so the supplemental packages stay
installed. The launchers locate the environment's CUDA compiler, expose a single
GPU UUID per evaluator, and load checkpoint tensors through CPU-mapped
safetensors followed by blocking copies to CUDA. The copies preserve dtype and
weight transformations.

## Download the pinned inputs

Set a writable Hugging Face cache and download both immutable revisions before
enabling offline execution:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export PYTHONPATH="$PWD/scripts:$PWD/python"
unset HF_HUB_OFFLINE

uv run --no-sync hf download Qwen/Qwen3.8-27B \
  --revision 8c0b3d30f3ae96f5c7f774de704bf0ada83e422a

uv run --no-sync hf download HuggingFaceH4/MATH-500 test.jsonl \
  --repo-type dataset \
  --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be

export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8
export MAX_JOBS=8
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
```

The dataset file must have SHA256
`35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`.
The loader checks all 500 unique problem IDs and the pinned revision.

## Generate the full reasoning traces

Choose currently free physical GPU indices. The examples use GPU 0. Check
ownership immediately before launch:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

./run_hogwild.sh --full-math500 runs/math500 \
  --save-traces --gpus 0 --repetition-policy record --no-wait-gpus
```

A device must have no compute processes, zero utilization and at most 8 MiB of
idle driver memory. Busy devices are skipped. A GPU that becomes occupied is
dropped from that attempt. The coordinators leave availability stops for an
explicit operator decision. A successful full sweep contains 500 tasks in each
of seven conditions and 3,500 verified trace files. Use a fresh output directory
for a new experiment. Reusing its directory resumes only its verified committed
cases with the same experiment configuration.

All arms use BF16 Qwen3.8-27B, the native thinking prompt, `reasoning_effort=xhigh`,
seed 42, temperature 1, top-k 20 and top-p 0.95. The budget is 16,384 sampled
reasoning rounds per worker. Ordinary Qwen ends at its own EOS or budget.
Hogwild publishes complete encoded paragraphs including their separator with
`encoded_separator_v1`. Its answer selector uses the terminating EOS worker's
last complete box across that worker's entire sampled stream. Missing EOS or a
missing box invokes a greedy cached finisher of at most 128 tokens outside the
reasoning budget. Existing KV is reused during this full-trace stage.

Repetition findings remain in the scores and efficiency totals and are recorded
under `suspicious-samples/`. Trace, source, token, selector and cache integrity
failures stop the run with a quality hold. Inspect that evidence before taking
any further action.

Regenerate the CPU source report after the run completes:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python \
  -m minisgl_examples.training.parallel_thinking.report runs/math500
```

This writes `runs/math500/report/summary.json`, `summary.csv` and `tasks.csv`.

## Measure the truncated readouts

Run the three groups sequentially on a free GPU. Each group includes all seven
conditions and all 500 problems. Change `--gpus 0` to an explicit list such as
`--gpus 0 1` to divide work across currently free devices.

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python \
  -m minisgl_examples.training.parallel_thinking.budget_short_sweep \
  --source-root runs/math500 --output runs/budget-short --gpus 0

./run_hogwild_budget.sh \
  --source-root runs/math500 --output runs/budget-1k-8k --gpus 0

CUDA_VISIBLE_DEVICES='' uv run --no-sync python \
  -m minisgl_examples.training.parallel_thinking.budget_long_sweep \
  --source-root runs/math500 --output runs/budget-9k-16k --gpus 0
```

| Group | Cutoffs | Expected rows |
| --- | --- | --- |
| Short | 256, 512, 1024 | 10,500 |
| 1k–8k | 1000, 2000, …, 8000 | 28,000 |
| 9k–16k | 9000, 10000, …, 16000 | 28,000 |

Cutoffs count sampled reasoning rounds per worker. The 1000 and 1024 cutoffs are
distinct. Each input reconstructs the first worker's view from the original
prompt, chronological shared publications, peer residuals in stable order, and
the first worker's residual last. Sampled pending non-EOS tokens through the
cutoff are included. Future reasoning and the original finisher are excluded.

The readout performs a fresh flat causal prefill of those preserved token IDs,
closes an open thinking span and appends `Given the current thoughts, the answer
is \boxed{`. It greedily samples at most eight new tokens, including closure and
EOS, and stops at a complete balanced box or EOS. Incomplete or unparsed answers
count as incorrect. Within one case, identical contexts can reuse the exact
readout with provenance. Every cutoff still receives its own result row.

Readout context caps are 136,192 tokens through 8k and 262,144 tokens for 9k–16k,
with memory ratio 0.75 and prefill chunks of at most 256 rows. The cutoff axis
excludes this new prefill and answer computation. The original generation and
this flat-prefill experiment use different attention computations.

The GPU stages retain forced readouts even where the source already finished.
The final numeric report below applies the requested original-answer rule.

## Export the conditional accuracies

After all three readout stages complete, run:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python \
  -m minisgl_examples.training.parallel_thinking.budget_conditional_report \
  --source runs/math500 \
  --short runs/budget-short \
  --previous runs/budget-1k-8k \
  --extended runs/budget-9k-16k \
  --output reports/conditional
```

The report verifies the shared 500 problem IDs, source trace hashes, all 66,500
unique result keys and the eight-token cap. It exports per-problem choices in
`reports/conditional/tasks.csv`, aggregate cells in `summary.csv` and provenance
in `summary.json`. Three wide CSV tables under `three-ranges/` cover:

- `accuracy-256-512-1024.csv`
- `accuracy-1k-to-10k.csv`
- `accuracy-10k-to-16k.csv`

The 10,000-round endpoint appears in both adjacent tables. The original answer
and grade are retained whenever `source_reasoning_rounds <= cutoff`. All other
rows retain the forced answer and grade. No grade is recomputed by this report.

## Interpreting a reproduction

The commands above generate a fresh set of 3,500 source trajectories. The
reported study reused 900 unchanged baseline executions collected on B200/H200
and generated 2,600 fresh executions on H200. Hardware and numerical execution
can affect sampled trajectories even with the same seed. Exact reproduction of
the previously reported per-problem table requires its original saved source
and readout artifacts as the four report inputs.

Math-Verify 0.9.0 grades remain frozen, including its known matrix-spacing,
interval, ordinal and tuple-representation limitations. An eight-token answer
cap can truncate an otherwise valid answer on cases still reasoning at the
cutoff. Repetitive samples remain in the evaluated cohort. A cutoff of 16,000
rounds remains distinct from the full generation budget of 16,384.

## Code entry points

All Python modules live under
`scripts/minisgl_examples/training/parallel_thinking/`.

| File | Purpose |
| --- | --- |
| `eval.py`, `native.py`, `single_worker.py`, `rollout.py` | Full reasoning and cached finisher |
| `sweep.py`, `report.py` | Full-trace scheduling and integrity/accuracy report |
| `budget_views.py` | Reconstruct the truncated first-worker context |
| `budget_curve_eval.py` | Flat prefill and greedy eight-token readout |
| `budget_curve_sweep.py`, `budget_short_sweep.py`, `budget_long_sweep.py` | Readout scheduling and completion verification |
| `budget_conditional_report.py` | Original/forced answer selection and numeric tables |
| `math500.py`, `trace_io.py` | Pinned dataset, grading and trace storage |
