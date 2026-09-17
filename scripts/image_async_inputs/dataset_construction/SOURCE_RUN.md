# First real-source run

Follow-up: the three approved image-model edits have now been generated and blind
solved. See [IMAGE_EDIT_RESULT.md](IMAGE_EDIT_RESULT.md). The proposal-stage results
below are preserved as the record of the earlier run.

This run starts from **MathVista testmini**, not the synthetic pilot. Cached 1,000
labeled examples and selected six candidates. Gemini 3.8 Flash screened all six on
flex through Google AI Studio. Reported total cost: **$0.015250875**.

Source revision: `2b6ad69445fbb5695c9b165475e8decdbeb97747`.
Parquet SHA-256: `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`.
Selection: seed 42, offset 0, two candidates each from chart, geometry and table
contexts; prefer math-targeted examples, then rank by seeded hash of source ID.
Source image bytes, original question/choices/units and metadata are preserved.

| ID | Task | Proposed initial error | Before → correct answer | Codex review |
| --- | --- | --- | --- | --- |
| 88 | Square/straight-line angle equation | Right label 2x° becomes x° | 45 → 30 | Shortlist |
| 926 | Shaded area under semicircle | Shade whole semicircle | 14.14 → 7.07 | Shortlist |
| 45 | Monthly waiting-time change | September 14 becomes 20 | +3 → -3 | Easy calibration only |
| 441 | Triangle perimeter | Base label 3 becomes 4 | 10 → 9 | Exclude: simple summation |
| 542 | Count models above threshold | Move a point above threshold | 3 → 2 | Exclude: counting and imprecise visual claim |
| 188 | Count table entries above threshold | None | — | Agree with model rejection |

Model screening returned five positive proposals and one rejection. All six source
images and proposed calculations were inspected by Codex. These are proposals,
not accepted pairs: **no images have been edited**. The image-editing method must be
discussed with the user. Source AFTER is fixed; erroneous BEFORE is constructed later.

[Open source-image proposal review packet](work/mathvista_batch01_proposals/review.html).
The same directory contains `summary.json`, `review_notes.json`, exact `jobs.json`,
and raw `*.response.json` files. Work artifacts are ignored by git.

## Reproduce and scale

From this directory:

```bash
uv sync --extra sources
uv run --no-sync python mathvista_sources.py --cache work/mathvista_cache --workspace work/mathvista_batch02 --per-category 2 --offset 2 --seed 42
```

First download pins the source revision; subsequent runs reuse and hash-check the
cache. Offset advances within each category's deterministic ordering. Increasing
`--per-category` scales candidate counts, not accepted pair counts. The initial
batch is `work/mathvista_batch01`; the commands above prepare a disjoint next batch.

After sourcing `.env` from the repository root and returning to this directory:

```bash
uv run --no-sync python model_assist.py --workspace work/mathvista_batch02 --output work/mathvista_batch02_proposals --limit 6 --execute --backend eliza --model google/gemini-3.8-flash --service-tier flex --reasoning-effort medium --max-output-tokens 4096 --seed 42
uv run --no-sync python collect_proposals.py --workspace work/mathvista_batch02 --responses work/mathvista_batch02_proposals
```

`--resume` skips saved responses when the entire job manifest/settings match. Invalid
saved responses remain visible to the collector; retries are deliberate, not automatic.
The collector parses JSON, checks proposal fields and obvious unchanged answers,
and builds the review packet. Optional `review_notes.json`, keyed by original source
ID, adds review decisions without overwriting model suggestions.

After agreeing on edits, submit a reviewed `*.proposal.json` through `pipeline.py
propose`, produce the image, attach it, review the completed pair, then export. See
PIPELINE.md. Suggestions never automatically accept data or authorize image editing.

## Lessons and remaining work

Metadata alone does not guarantee multi-step reasoning. Model screening accepted one
threshold-counting task while rejecting another; the main set needs a stricter rubric.
Keep simple cases as a separate calibration set, if useful. The source answer was
supplied in proposal generation, so this is not blind independent verification.
Before accepting edited pairs, solve both images without expected answers and compare
with reference derivations. No GPU or async inference experiment ran.

For ID 88, discuss localized removal of the digit 2. For ID 926, discuss a controlled
fill mask preserving the curve and labels. These may be programmable edits once
approved; AI image editing remains an alternative to discuss, not an assumed step.
