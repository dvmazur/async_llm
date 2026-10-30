# Source-based construction pipeline

The MathVista adapter and a first real-source run are now implemented. See
[SOURCE_RUN.md](SOURCE_RUN.md) for commands, results and review artifacts. The generic
local-manifest workflow below remains available for other datasets.

The existing `pilot_v1` is synthetic and stays separate. `pipeline.py` implements
the local construction workflow for benchmark-derived examples. `mathvista_sources.py`
downloads and selects MathVista candidates. The workflow does not edit images, solve
arbitrary problems, or run model inference. Those actions must not be confused with
the bookkeeping and review checks implemented here.

## Environment and execution

Use this directory as the working directory. Its independent uv project has no
dependencies, so it does not install the inference engine or GPU packages.

```bash
uv sync --python /usr/bin/python3
uv run --no-sync python -m unittest -v test_pipeline
uv run --no-sync python pipeline.py --workspace work/source_v1 init
```

Only the optional synthetic renderer needs Pillow (`uv sync --extra render`).
Do not add GPU dependencies to this construction environment.

## Data flow

| State | Action | Output / gate |
| --- | --- | --- |
| imported | Import a normalized local source manifest | Preserved corrected image and source provenance |
| edit_proposed | Submit a proposed corruption and both derivations | Ready to discuss editing method; no editing happens |
| rendered | Attach a separately produced erroneous image | Records agreed method and hashes; identical files rejected |
| normalized | Normalize both images on CPU | Matched dimensions, aspect ratio, RGB PNG, stripped metadata |
| accepted / rejected | Submit named review and reasons | Acceptance requires every review criterion |
| export | Package accepted records into a new directory | Model inputs separated from evaluator annotations |

States are deliberately sequential. A rejected sample remains recorded. For revised
construction rounds use a new workspace/version; in-place proposal revision is not
implemented. Each command is resumable between records; reimporting unchanged sources
does nothing. Use one writer per workspace. This is a local pipeline, not a concurrent
job service. File writes use temporary replacement for records; batch export is not
transactional and a failed partial export should be inspected before choosing a new destination.

## 1. Import existing problems

Prepare JSONL, one source problem per line. Images must already exist locally; relative
paths resolve against the manifest directory. Every listed field is required and a string:

```json
{"dataset":"DATASET_NAME","source_id":"ORIGINAL_ID","source_split":"validation","category":"geometry","question":"Complete original question, including choices and required units","answer":"68","image":"images/example.png","source_url":"ORIGINAL_SOURCE_URL","license":"SOURCE_TERMS_OR_REFERENCE"}
```

This is a format example, not an actual benchmark record. The import preserves any
additional fields, such as original metadata or answer precision. Dataset-specific
adapters should map into this format; MathVista is implemented in `mathvista_sources.py`.
MathVision/CharXiv adapters remain future work. Preserve question choices, units, source IDs and split names.
Use labeled source splits. Never invent missing answers or infer source terms.

```bash
uv run --no-sync python pipeline.py --workspace work/source_v1 import /path/to/sources.jsonl
uv run --no-sync python pipeline.py --workspace work/source_v1 report
```

Open `work/source_v1/review.html` to inspect candidates and their generated IDs.
This page is evaluator-only and must never be provided to the model.

## 2. Prepare an edit proposal

For the first source-based subsample, aim for six candidates: two charts, two geometry
problems, and two document/table calculations. Review the original image before selecting
the corruption. Save a proposal JSON file using this structure:

```json
{
  "edit_type": "length_label",
  "changed_fact_before": "Rectangle side labeled 6",
  "changed_fact_after": "Rectangle side labeled 15",
  "expected_answer_before": "40",
  "solution_before": "sqrt(8^2 + 6^2) = 10; square perimeter = 40",
  "solution_after": "sqrt(8^2 + 15^2) = 17; square perimeter = 68",
  "editing_method": "PROPOSED: local label replacement; discuss with user",
  "reasoning_change": "Recompute diagonal, then propagate to square perimeter"
}
```

The example describes our synthetic geometry case only. Actual source proposals must
be derived from the imported image and problem.

```bash
uv run --no-sync python pipeline.py --workspace work/source_v1 propose SAMPLE_ID proposal.json
uv run --no-sync python pipeline.py --workspace work/source_v1 report
```

**Discuss manual editing or image generation with the user here.** The CLI does not
perform or approve those actions. The `editing_method` field is a proposal, not consent.
After the user selects an approach, produce the image separately and attach it:

```bash
uv run --no-sync python pipeline.py --workspace work/source_v1 attach SAMPLE_ID /path/to/before.png --approval-note "Actual agreed method and reference to the discussion"
```

## 3. Normalize, then review the completed pair

After `attach`, run:

```bash
uv run --no-sync python pipeline.py --workspace work/source_v1 normalize SAMPLE_ID
```

Requires Pillow (`uv sync --extra sources`, or `--extra render`). The original corrected
image's EXIF-oriented dimensions define the target canvas. The edited image is resized
with Lanczos, without cropping or padding. Relative aspect-ratio differences above 1%
are rejected for alignment review; differences below that threshold are recorded and
corrected by the resize. This policy handles the sub-1% drift in our three test images.

Both images receive EXIF orientation correction, ICC-to-sRGB conversion when a profile
exists (otherwise sRGB is assumed), alpha compositing onto white, and fresh RGB PNG
encoding without embedded metadata. Originals remain untouched. `normalization.json`
records raw/output hashes, dimensions, transformation policy and pending checks.

Acceptance now requires `normalized` status. Review and any blind answer verification
must use these normalized files; earlier raw-image solves do not automatically validate
resampled inputs. Reports display normalized pairs when available. Exports point model
inputs at normalized PNGs and retain raw assets for provenance. Pixel-perfect preservation
outside the edit region is not implied by normalization.

An optional Qwen-style **processor-only** parity check is available:

```bash
uv run --no-sync python pipeline.py --workspace work/source_v1 processor-check SAMPLE_ID --model /path/to/cached/model-processor
```

This checks `image_grid_thw`, image-token count and pixel-tensor shape under the same
processor settings, records its configuration and image hashes, and fails on mismatch.
It requires an installed compatible Transformers environment and a cached processor;
it loads no model weights, uses no GPU and does not download files. It supports processors
exposing `image_token` and `image_grid_thw`; other families need an adapter. This path has
not been run against an actual processor yet. Processor selection is still pending.
Dataset export is allowed with an explicit pending marker, but the evaluation harness
must check parity using its exact processor/configuration before controlled async runs.

A reviewer must inspect both normalized images and independently check both solutions. Save:

```json
{
  "reviewer": "REVIEWER_NAME",
  "decision": "accept",
  "notes": "Describe checks and any remaining limitations",
  "answer_before": "40",
  "answer_after": "68",
  "isolated_edit": true,
  "readable": true,
  "both_solvable": true,
  "visual_required": true,
  "answers_distinct": true,
  "source_terms_checked": true,
  "same_dimensions": true
}
```

```bash
uv run --no-sync python pipeline.py --workspace work/source_v1 review SAMPLE_ID review.json
uv run --no-sync python pipeline.py --workspace work/source_v1 report
```

These booleans are explicit reviewer attestations, not automated mathematical or image
verification. In particular, different strings may encode equivalent answers, and
different image hashes may encode visually identical pixels. Review must catch both.
Use `decision: "reject"` and explain why for unsuitable pairs. Rejection still requires
reviewer, notes, and both answer fields; use `unresolved` when no reliable answer exists.

## 4. Export an evaluation version

```bash
uv run --no-sync python pipeline.py --workspace work/source_v1 export exports/source_v1
```

Exports never overwrite an existing destination. Image hashes are rechecked. Output:

- `inputs.jsonl`: IDs, relative before/after image paths, original question and fixed notice.
- `annotations.jsonl`: accepted records, answers, provenance, proposals and reviews.
- `assets/`: paired images.
- `manifest.json`: schema version, sample count, fixed notice and JSONL hashes.

Only decoded images and text shards belong in model prompts. Filenames, review data,
and answer annotations must not be exposed. Output-format instructions and model chat
templates belong in the later evaluation harness. No GPU or model runs occur here.

## Scaling to 100 per category

Use several problem families per category and import larger candidate batches. Review
acceptance rates before estimating how many sources are needed. Record rejection reasons
even when a problem seems difficult. Do not filter on eventual async success.

Stable IDs use dataset, original split, and source ID. `group_id` is the source image
SHA-256, grouping exact shared-image questions even across sources. This does not detect
resized copies or semantically equivalent problems. Add near-duplicate review and
base-problem grouping before producing train/dev/evaluation splits; the current pipeline
does not split data automatically.

Next extensions, after reviewing the six real candidates: dataset-specific adapters,
candidate eligibility annotations, edit-family tooling agreed with the user, independent
numeric validators for suitable families, category/rejection summaries, and reproducible
grouped splits. Static model qualification and async evaluation remain separate work.

## Optional model assistance (not enabled or run yet)

`model_assist.py` prepares proposal jobs offline by default. It can later send images
to an existing local OpenAI-compatible vision-model server (which may use a GPU),
or to OpenRouter. It does not start a server or install a model. The provider paths
are untested against live services; validate compatibility with the chosen VLM first.
Do not execute either backend until the user says that GPU/OpenRouter is ready.

```bash
uv run --no-sync python model_assist.py --workspace work/source_v1 --output work/plan_v1 --limit 6
```

Future execution examples, **not commands to run now**:

```bash
uv run --no-sync python model_assist.py --workspace work/source_v1 --output work/local_suggestions_v1 --limit 6 --execute --backend local --model MODEL_ID
uv run --no-sync python model_assist.py --workspace work/source_v1 --output work/router_suggestions_v1 --limit 6 --execute --backend openrouter --model PROVIDER_MODEL_ID
```

OpenRouter reads `OPENROUTER_API_KEY` from the environment; local serving optionally
reads `LOCAL_MODEL_API_KEY`. Keys are never written to job files. Requests are serial,
bounded by `--limit` and `--max-output-tokens`, and are not automatically retried.
The latter limits output tokens, not cost. Raw responses retain provider usage metadata.
Review completed responses before retrying a failed run using a new output directory.

Model output remains an untrusted suggestion: inspect its eligibility decision,
verify both solutions independently, and submit the reviewed proposal via `propose`.
It cannot approve editing, accept samples, or modify dataset records. Images sent to
OpenRouter leave the machine; source eligibility/terms should be checked beforehand.

## Eliza gateway connection

The `eliza` backend uses the user-specified endpoint
`https://api.eliza.yandex.net/openrouter/v1/chat/completions`, `Authorization: OAuth`
with `API_KEY`, pool `YR_all`, and `X-Request-Timeout: 60m`. Client timeout defaults
to 7200 seconds. Load the repo `.env` in the shell without printing it:

```bash
# From repository root:
set -a
source .env
set +a
cd scripts/image_async_inputs/dataset_construction
uv run --no-sync python gateway.py --list-models --filter gemini
```

Catalog access was verified with TLS verification enabled. `--insecure` is available
for an explicit troubleshooting choice; it is not the default. Eliza's response
envelope is unwrapped and its `key` field is never persisted by the transport.

Proposed first models, both listed as image-capable by the gateway:
`google/gemini-3.8-flash` for screening/proposals, and
`google/gemini-3.1-pro-preview` for a separate solution check. The latter review stage
is not yet automated: the current model-assistance prompt generates proposals.
No chat-generation request has been made as part of connection setup.

Example for later proposal generation:

```bash
uv run --no-sync python model_assist.py --workspace work/source_v1 --output work/eliza_suggestions_v1 --limit 6 --execute --backend eliza --model google/gemini-3.8-flash --service-tier flex --reasoning-effort medium --max-output-tokens 4096 --seed 42
```

`--provider` can be repeated to supply `provider.only` with fallbacks disabled. Do not
guess an internal provider name. `--reasoning-effort` and `--reasoning-tokens` are
mutually exclusive; a token budget must leave room for final output. We do not infer
reasoning settings by matching the substring `gpt` in a model name.

Flex handling follows [OpenRouter service-tier documentation](https://openrouter.ai/docs/guides/features/service-tiers).
Requesting `service_tier=flex` does not prove discounted billing: models without flex
endpoints may route at standard rates. Inspect the returned tier and usage before
scaling, and confirm the gateway's behavior. `:batch` catalog entries are not assumed
to mean flex. Raw upstream responses preserve tier/usage metadata.
# Executed diverse expansion

See [DATASET_V2.md](DATASET_V2.md) for the completed 41-pair, three-source run,
concurrent budget accounting, validation/review gates, and exported evidence.
The source-based workflow below remains the underlying record state machine.
