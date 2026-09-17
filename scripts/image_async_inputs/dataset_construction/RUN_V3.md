# Expansion toward 600 accepted pairs

## Completed release

513 pairs selected: 472 new and 41 retained from v2. This meets the requested
500–600 range; provisional quotas were not forced at the expense of quality.

| Source | Pairs |
|---|---:|
| MathVista | 100 |
| MathVision | 72 |
| CharXiv | 64 |
| ChartQA | 72 |
| TabMWP | 113 |
| MapQA-U | 32 |
| CLEVR | 60 |

New construction: 5,723 requests; $74.47368075 reported, $75.27368075
conservatively accounted including $0.80 reserved for five unknown-cost calls.
This is local accounting, not a verified provider invoice. Prior v2 costs are
excluded; adding its $5.055348 accounted cost still totals about $80.33.
No additional generation is needed and no GPU/AsyncLLM evaluation was run.

All release pairs passed blind before/after solves, a model pair audit, and
explicit Codex visual/semantic review. This is not independent human annotation.
Of 520 initially retained pairs, final review excluded one duplicate source image,
four ambiguous within-bin map comparisons, and two conflicting color-name legends.
The nine cross-source duplicate candidates were viewed: one exact duplicate,
eight template-similar but distinct tables. Earlier review also removed a
MathVista/ChartQA duplicate. Heuristic near-duplicate detection is not exhaustive.
Exclusion reasons remain in work/v3_final_exclusions.json; old review history and
source artifacts are preserved rather than overwritten.

Groups: 484 reasoning, 28 calibration, one reasoning-reveal control. These are
construction tags, not empirical difficulty estimates. Some simple one-step
tasks remain in reasoning; report by source/category as well as these tags.
MapQA extrema use displayed bin ranks and include ties; exact within-bin values
cannot be recovered. All normalized pairs match size, aspect ratio, RGB/PNG mode,
and empty metadata. Raw image-generation output can change global rasterization;
this is not a pixel-local-edit guarantee. Qwen processor parity is still pending.

Release: datasets/diverse_corrections_v3.parquet, with embedded image bytes,
input shards, separate labels and source provenance. Expanded evidence remains
in datasets/diverse_corrections_v3 and work/. Mixed source licenses require review
before public redistribution; this release is for local evaluation.

Offline reproduction of the selection/package (no new model calls):

```bash
uv run --offline --extra sources python finalize_selection.py
DATASET_BUDGET_LEDGER=work/budget_diverse_v3.json uv run --offline --extra sources python assemble_dataset.py --config work/v3_assembly.json --staging work/v3_assembled --output datasets/diverse_corrections_v3
uv run --offline --extra sources python package_dataset.py --source datasets/diverse_corrections_v3 --output datasets/diverse_corrections_v3.parquet
uv run --offline --extra sources python package_dataset.py --validate datasets/diverse_corrections_v3.parquet
```

Assembly and packaging deliberately refuse to overwrite existing releases.
Use fresh staging/output paths for reproduction. The uv offline test suite passed
20 tests, including budget reservations, request-cache/unknown-outcome protection,
normalization, review gates and standalone package/input-label separation.

## Original execution plan

Authorized 2026-09-16: at most $150 in new API construction charges, retaining
the existing 41 accepted pairs. Target 600 total; do not weaken validation or
exceed the spending cap to reach the target. No GPU inference is authorized.

Provisional source quotas: MathVista 100, MathVision 100, CharXiv 100,
TabMWP 100, ChartQA 80, MapQA 60, CLEVR 60. New sources require acquisition,
license/provenance checks, adapters and a small pilot before scale-out.
Controlled re-rendering is awaiting the user's choice; existing source-image
editing can proceed independently.

Ledger: `work/budget_diverse_v3.json`. All paid calls must use
`DATASET_BUDGET_LEDGER` pointing to this file. Existing v1/v2 ledgers are not
modified. Failed/unknown requests retain reservations; no blind retries.
The ledger is local conservative accounting, not a provider-side spending cap.
Checkpoint at each 100 accepted pairs and stop before budget exhaustion.

Models: google/gemini-3.8-flash for proposal/validation, with flex requested;
google/gemini-3.1-flash-image for edits. Availability checked via the configured
gateway's model catalog. Credentials stay in the repository .env and must not
appear in output artifacts.

Outputs will be versioned separately from the existing 41-pair Parquet. Model
checks alone do not constitute visual acceptance. Retain rejected evidence,
split/group provenance, normalized image hashes, and explicit review notes.
