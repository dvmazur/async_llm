# Source-derived pilot under $2

Six MathVista pairs: three reasoning examples (angle algebra, shaded area, table
mean), one missing-angle reveal control, two easy calibration examples (signed
rate and scale factor). Pic 2 is the source image; pic 1 is an AI-edited variant.
These are not wholly synthetic problems. Three edits were reused from the earlier
pilot and three were newly generated. All six pairs were normalized and rechecked.

## Executed construction path

1. `mathvista_sources.py`: pinned MathVista testmini download, checksum, category
   sampling, original question/answer/choices and image provenance.
2. `model_assist.py` and `collect_proposals.py`: Gemini 3.8 Flash screening and
   localized edit proposals. Twelve new candidates were screened. Selection remains
   supervised: reject textual answer leakage, trivial reasoning-only candidates,
   and edits inconsistent with other visual facts.
3. `build_dataset.py build`: import a reviewed plan, attach reused edits or call
   Gemini 3.1 Flash Image, preserve raw assets, then normalize both image states.
4. `normalize_pair.py`: oriented sRGB/RGB, same dimensions and aspect ratio, PNG,
   no incidental metadata. Raw aspect drift over 1% stops processing. Sample 331
   needed an explicitly reviewed 1.5% allowance (actual drift 1.29%); no crop or
   missing table row was found. Normalized images were inspected afterwards.
5. Two blind normalized-image solves per pair, followed by a paired semantic
   audit. All 12 solves matched their reference values; all six audits passed.
   These checks use the same model family as proposals, not an independent oracle.
6. Explicit Codex visual/arithmetic review; `build_dataset.py finalize` exports
   accepted pairs, separate answers, provenance, checks, reviews and budget ledger.

Text shard 1 preserves the source question. Text shard 2 only announces correction;
it does not describe the changed label, number or shading. The reveal control is
identified separately because its first answer is insufficient information.

## Budget and execution

New reported spend: **$0.26043025 / $2** across 33 requests: 12 screening, three
image edits, 12 blind solves and six audits. Earlier pilot calls are excluded;
three existing edited assets were reused. Flex was requested: text requests reported
flex, image edits reported default. No GPU or async inference was run.

Use the uv-managed `.venv` from this directory (see pyproject.toml extras).
`DATASET_BUDGET_LEDGER` names a JSON file with `limit_usd` and `requests` fields.
`budget.py` records each request before sending it and reconciles reported costs;
unknown costs retain conservative reservations. This is serial accounting, not a
provider-enforced hard spending cap. Do not share the ledger between concurrent runs.
Credentials come from the repository `.env`; never copy it into artifacts.

```bash
export DATASET_BUDGET_LEDGER="$PWD/work/budget_dataset_v1.json"
.venv/bin/python build_dataset.py build --workspace work/dataset_v1_build --plan work/dataset_v1_plan.json
.venv/bin/python build_dataset.py finalize --workspace work/dataset_v1_build --reviews work/dataset_v1_reviews.json --output datasets/mathvista_corrections_v1
.venv/bin/python -m unittest discover -p 'test_*.py'
```

Build resumes completed calls without paying for them again. Keep the plan and
assets immutable while resuming; changed plans require a new version. Export to a
new directory. Final artifacts carry the construction plan and review decisions.

## Scaling and remaining work

Increase category quotas/source offsets, screen candidates, review proposals, then
run the same build stages with a fresh budget ledger. Track rejection rates and
group source variants together in splits. This pilot is selected, not representative;
do not extrapolate six successes into a benchmark-wide acceptance rate or budget.

The construction path is end-to-end but **not fully unattended**: proposal selection,
alignment exceptions and final acceptance need review. Add no-op and answer-preserving
controls before a larger evaluation. Schematic diagrams use labels, not pixel-scale
measurements (especially sample 206). Exact Qwen processor grid/token parity remains
pending target model configuration; metadata parity alone does not establish it.
Async update timing, cache replacement and final-answer recovery still need the
separate inference experiment. Retain MathVista attribution and original-source rights;
this local research export does not grant new redistribution rights.
