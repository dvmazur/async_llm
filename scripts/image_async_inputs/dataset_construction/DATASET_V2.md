# Diverse source-derived dataset: 41 pairs under $10

For evaluation, use the self-contained [Parquet file](datasets/diverse_corrections_v2.parquet)
(41 rows, 82 embedded PNGs, 12,842,767 bytes). See [loading and protocol](PACKAGE.md).
SHA256: `b4baddb5f543b3c4df672b73fb77944e52b0ca1c3a6a1628cde2ce2c666da991`.
This compact package excludes raw construction logs; answer keys and provenance
remain separate from model inputs. No extra API spend or GPU use was required.

The completed export is [datasets/diverse_corrections_v2](datasets/diverse_corrections_v2/README.md).
See [plain image/text examples](datasets/diverse_corrections_v2/examples.md).

| Source | Accepted pairs |
|---|---:|
| MathVista | 15 |
| MathVision | 14 |
| CharXiv | 12 |

There are 35 reasoning-tagged pairs, five easier calibration pairs and one
missing-information reveal control. Difficulty tags are construction labels, not
measured model difficulty. Tasks include geometric constraints, probability,
linked arithmetic, sorting/aggregation, chart comparisons and visual selection.
Six MathVista pairs reuse the previous pilot; 35 accepted edits are new.

## Cost and acceptance

New reported API spend is **$4.955348**. Two interrupted screening requests have
unknown upstream cost, so their $0.05 reservations remain charged to the budget:
**$5.055348 accounted against the $10 cap**. Earlier pilot calls are excluded.
All API work has stopped. Flex was requested; text calls reported flex and image
generation reported default. No GPU or engine inference was run.

Prepared 220 candidates; 209 screening responses were saved: 74 proposals, 126
model rejections and nine invalid/incomplete responses. Eleven candidates have no
saved response (including the two interrupted requests; the others were not reached).
Generated 56 edited images; 35 were accepted after all checks and review. The other
21 are excluded. This is selected evaluation data, not a representative random sample.

The full cost ledger, screening responses, prompts, rejected builds and review notes
are included in the export. Source downloads are pinned and checksum-verified:
[MathVista](https://huggingface.co/datasets/AI4Math/MathVista),
[MathVision](https://huggingface.co/datasets/MathLLMs/MathVision),
[CharXiv](https://huggingface.co/datasets/princeton-nlp/CharXiv).
CharXiv uses validation reasoning questions and excludes the documented incorrect
sample 0. Source metadata and original question/answer fields are retained.

## Executed pipeline

1. `mathvista_sources.py` / `diverse_sources.py`: acquire pinned sources and sample
   candidates on CPU. MathVision prefers levels 1–3; CharXiv prefers at most four panels.
2. `model_assist.py` / `diverse_batch.py screen`: Gemini 3.8 Flash proposes a localized
   visual error, before answer and solutions, leaving the source question unchanged.
3. Review source images/proposals, then create immutable per-source plans.
4. `diverse_batch.py build`: Gemini 3.1 Flash Image makes pic 1 from the original
   pic 2. Save raw output, source hashes, exact prompt and sanitized response metadata.
5. Normalize both states to identical dimensions/aspect ratio, RGB PNG and empty
   embedded metadata. Raw aspect drift over 1% requires explicit per-pair review.
6. Solve each normalized state separately without answer keys; audit both images
   for intended changes and preservation of unrelated content. Non-exact answer
   strings may have a separate semantic equivalence check. One source double-hyphen
   spelling was explicitly adjudicated by Codex against the original legend.
7. Visually inspect normalized edits. A model-passing scatter edit was rejected
   because it moved a point horizontally as well as vertically. Acceptance is not
   merely copying the model audit decision.
8. `assemble_dataset.py`: combine explicit accepted reviews with the prior pilot,
   export model inputs separately from answer keys, and include provenance/evidence.

`retry_incomplete.py` permits one explicit higher-token retry of a truncated text
response, preserving both attempts. Two retries remained truncated and were excluded.
There are no automatic billable retries. `budget.py` locks reservations/reconciliation
across concurrent requests; pending/unknown calls are not assumed free. Conservative
local accounting is not a provider-enforced hard spending cap.

## Running or scaling

Use the uv-managed `.venv` and the extras in `pyproject.toml`; no model weights are
needed. Run from this directory. Keep `.env` credentials outside dataset artifacts.

```bash
# Offline checks, no API calls:
.venv/bin/python -m unittest discover -p 'test_*.py'
.venv/bin/python construction_report.py

# Paid steps require exported API_KEY and a fresh explicitly capped budget ledger:
export DATASET_BUDGET_LEDGER="$PWD/work/budget_diverse_v2.json"
.venv/bin/python diverse_batch.py screen --workspace work/v2_charxiv --output work/v2_charxiv_proposals --workers 3
.venv/bin/python diverse_batch.py build --plan work/v2_charxiv_plan.json --output work/v2_builds --workers 3
```

The commands above show the workflow, **not permission to spend again**. Existing
responses are reused only with matching requests/hashes; interrupted missing responses
require deliberate investigation before another call. New source offsets and quotas
should use new workspaces and reviewed plans. Do not run two builders on the same
sample workspace concurrently. Export to a fresh versioned directory.

## Limitations and next stage

The pipeline is end-to-end but supervised: source/proposal selection, alignment
exceptions and acceptance require review. Model checks use the same model family
as proposal generation; Codex review is not independent human annotation. Raw edits
may have global rasterization/style differences. Some geometric pictures are
schematic; printed labels, not pixel measurements, determine answers. Sample-specific
deviations are documented, including a widened/moved target bar in CharXiv 291.

Text shard 2 announces a correction without describing the visual change. The export
has no decoding-step schedule: k, cache manipulation and async generation remain
separate evaluation work. Exact target-Qwen processor grid/token parity is pending
model configuration; metadata parity alone is not sufficient. Add no-op/answer-
preserving controls and group shared sources across splits before larger experiments.

Retain source attribution and original image rights. CharXiv prohibits training use;
the current export is for local evaluation, not a grant of redistribution rights.
