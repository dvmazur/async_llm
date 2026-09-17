# Image correction dataset construction

This directory constructs datasets for asynchronous image correction experiments.
It does not run engine inference or change the engine.

## Git checkout

The versioned Parquet datasets use Git LFS. After cloning, run `git lfs pull`.
Use the Parquet loader below: v2/v3 unpacked image directories, raw API evidence,
and construction caches are local-only and are not committed. The JSONL manifests
and Markdown galleries retain their original unpacked paths for audit purposes;
those paths are not a standalone dataset in a fresh checkout. Complete evaluation
inputs, images, labels and provenance are embedded in each Parquet file.

## Latest: 513 pairs across seven sources

Ready-to-load one-file artifact: [diverse_corrections_v3.parquet](datasets/diverse_corrections_v3.parquet).
Both images, text shards, separate labels and provenance are embedded.
See [loader/protocol](PACKAGE.md), [completed run and cost report](RUN_V3.md), and
[release counts](datasets/diverse_corrections_v3/summary.json).

MathVista 100, MathVision 72, CharXiv 64, ChartQA 72, TabMWP 113, MapQA-U 32,
CLEVR 60. Includes the earlier 41 pairs. New reported API cost is $74.47;
conservative accounting with unknown-call reserves is $75.27, below $150.
All release pairs passed model checks and explicit Codex visual review, not
independent human annotation. No GPU/AsyncLLM evaluation was run.

## Earlier: 41 diverse source-derived pairs under $10

Ready-to-load single-file artifact: [diverse_corrections_v2.parquet](datasets/diverse_corrections_v2.parquet)
(12.8 MB, all 82 PNGs embedded). [Loader, schema and evaluation protocol](PACKAGE.md).
Build/validate offline with `package_dataset.py`; no API calls or GPU required.

The completed MathVista / MathVision / CharXiv expansion is described in
[DATASET_V2.md](DATASET_V2.md), with [41 image/text pairs](datasets/diverse_corrections_v2/examples.md)
and [dataset metadata](datasets/diverse_corrections_v2/README.md).
New reported spend: $4.955348; including unknown-call reservations: $5.055348.
No GPU or async inference was run.

## Earlier: source-derived dataset under $2

See [DATASET_V1.md](DATASET_V1.md) for the executed construction workflow, budget,
review gates and limitations. The exported six-pair MathVista pilot is in
[datasets/mathvista_corrections_v1](datasets/mathvista_corrections_v1/README.md),
with [image/text shards](datasets/mathvista_corrections_v1/examples.md).

For the implemented **source-based import → proposal → review → export pipeline**,
see [PIPELINE.md](PIPELINE.md). The synthetic pilot below is separate from that workflow.

## Earlier synthetic pilot (historical)

Five base problems, ten PNGs, five kinds of visual correction. This is a diversity
preview, not a statistically useful evaluation set or a claim of model readiness.
All images are programmatically rendered. No external dataset, AI image generator,
or manually edited image is used.

| ID | Error corrected | Required reasoning | Before → after answer |
| --- | --- | --- | --- |
| 01_geometry | Rectangle length label: 6 → 15 | Pythagorean theorem, then square perimeter | 40 → 68 |
| 02_bar | A's bar height: 12 → 18 | Read outputs, divide by workforce, compare | B → A |
| 03_legend | North/South legend mapping swapped | Identify series, subtract, divide by baseline | 100/3% → 25% |
| 04_receipt | Notebook quantity: 2 → 5 | Multiply quantities/prices, sum, apply discount | $12.60 → $23.40 |
| 05_graph | A–T edge weight: 2 → 9 | Sum competing routes and minimize | 6 → 9 |

The corrected state is the canonical problem. The initial image contains a plausible
error, but still defines a solvable problem. Both versions have verified, different
answers. The geometry drawing is explicitly not to scale. The legend swap changes
two text labels but one semantic mapping. Bar heights must be read from the axis;
the bar sample deliberately does not duplicate their values in text.

## Reproduce and inspect

Requires Python 3.10+, Pillow, and the DejaVu Sans font discoverable by Pillow.
Pillow is already a project dependency. Run from the repository root:

```bash
python3 scripts/image_async_inputs/dataset_construction/generate_pilot.py
```

Run the command above from the repository root, or use the script's absolute path.
It regenerates the files in `pilot_v1/`; do not manually edit generated files.

- `pilot_v1/contact_sheet.png`: before/after visual review, with evaluator-only headings.
- `pilot_v1/inputs.jsonl`: only IDs, relative image paths, and the two text shards.
- `pilot_v1/annotations.jsonl`: evaluator-only answers, intermediate facts, derivations,
  source specifications, edit categories, pixel-difference bounding boxes, and image hashes.
- `pilot_v1/*_before.png`, `*_after.png`: individual 960 × 640 RGB model images.

Resolve image paths relative to `inputs.jsonl`. Never send annotations, filenames,
or contact-sheet headings to the model. Both stages use the identical question.
The correction notice is fixed and reveals no values, locations, or edit types:

> The earlier picture contained an error. It has now been corrected. Recheck your reasoning using the current picture.

## Construction pipeline

1. **Specify the correct problem.** Define visual facts, question, exact answer,
   and a derivation. Require essential visual evidence and at least two reasoning steps.
2. **Derive one erroneous state.** Change one answer-relevant semantic fact. Preserve
   a plausible, uniquely solvable initial problem. Keep unrelated visual content fixed.
3. **Solve both states.** Derive answers and intermediate facts from the rendering
   specification. Compare with independently hand-calculated expected answers.
   Reject identical answers, ties, inconsistent geometry, and ambiguous questions.
4. **Render paired images.** Use identical image sizes and styling. Record the changed
   pixel region and hashes. Confirm the visible edit matches the intended semantic edit.
5. **Package the input shards.** Keep essential changed facts out of the question and
   use the generic correction notice. Separate evaluator annotations from model inputs.
6. **Review every pilot pair.** Inspect readability, unintended cues, mathematical
   validity, and whether the requested calculation really depends on the image.
   Current records remain `pending_user_review`; automated checks do not approve them.
7. **Later, qualify with static inference.** Run before/after baselines only once
   experiments are requested. Preserve failures and report baseline-solvable subsets
   separately. Do not select examples using async success.
8. **Later, evaluate async updates.** At a recorded decoding boundary, replace the
   image and append the notice while retaining generated output. Record whether the
   update was reached, the changed fact was used, dependent calculations were revised,
   and the final answer was correct. Mutation details belong in the future eval harness.

Generator checks cover visible differences, expected derived answers, and changed
answers. They do not establish visual readability for a particular processor/model,
cache correctness, or the model's ability to reason about these examples. Match and
record processed image grids/token counts when integrating a model processor.

## After review: expand carefully

First expand accepted families to 8–10 base problems, including less obvious errors
and some changes to the solution route. Eventually target 100 base problems per
category across several templates and edit types. Keep variants of the same base
problem grouped in splits and uncertainty estimates. Version specifications, prompts,
renderers, and scoring together. Add explicit answer-preserving/no-op control pairs;
the current five pairs are all answer-changing.

Planned eval controls: no update, corrected input from the start, corrected image
only, correction notice only, both updates, and a neutral-notice condition. Use exact
or mathematical-equivalence scoring for these short answers; inspect intermediate
facts separately, without requiring a particular wording of the generated explanation.

## Original synthetic-pilot decisions (historical)

The following describes the initial pilot only. Source downloads and OpenRouter
editing were subsequently approved and executed; see DATASET_V1.md.

No image-generation or manual-editing decision is needed for this pilot. Before
adding either of the following, discuss the proposed sources and approach with the user:

- **Benchmark-derived images:** choose actual MathVista/MathVision/CharXiv or document
  samples and inspect them first. Decide whether each edit should use a clean redraw,
  a local manual edit, or an image-editing model. Preserve provenance and applicable
  source terms; revalidate both answers. No such images have been downloaded here.
- **Natural scenes or complex image edits:** agree on generation/editing tooling,
  effort, and acceptance criteria before producing assets. Check for unintended
  differences and factual inconsistencies; these require additional visual review.

Do not treat programmatically rendered substitutes as original benchmark samples.
