# One-file async-image evaluation dataset

Each versioned `datasets/diverse_corrections_*.parquet` contains source-derived pairs.
Consult the embedded manifest and actual rows for version-specific counts and sources.
Both normalized RGB PNG images are
embedded losslessly: no downloads, asset folders, model weights or API keys are
needed. This document is also embedded in the Parquet schema metadata under
`image_async.readme`; the source-export manifest is under `image_async.manifest`.
The package schema version is `image_async.schema_version = 1`.

## Load and evaluate

From this directory, use the uv-managed environment:

```bash
uv sync --extra sources
uv run --extra sources python package_dataset.py --validate datasets/diverse_corrections_v3.parquet
```

Any PyArrow reader can load the file without this repository:

```python
import io
import pyarrow.parquet as pq
from PIL import Image

table = pq.read_table("diverse_corrections_v3.parquet")
row = table.slice(0, 1).to_pylist()[0]
inputs = row["inputs"]
pic1 = Image.open(io.BytesIO(inputs["image_before"]["bytes"]))
pic2 = Image.open(io.BytesIO(inputs["image_after"]["bytes"]))
shard1, shard2 = inputs["text_shard_1"], inputs["text_shard_2"]
target = row["labels"]["answer_after"]  # evaluator only
```

With the repository helper, `model_context(row, 1)` and `model_context(row, 2)`
return only a PIL image and text. Do not serialize the full row into a prompt.

Protocol: start generation with pic1 + shard1. At a harness-selected decoding
step k, **replace** pic1 with pic2 and append shard2. Do not append pic2 alongside
pic1. Keep the generated prefix according to the async engine's update semantics.
The stage-2 helper describes the resulting input context, not an engine mutation
or a restart implementation. The file intentionally does not choose k.

Report final corrected-answer accuracy and persistence of the old answer, grouped
by source and difficulty. Compare async updates with static pic1 and static pic2
baselines. Trajectory changes require evaluator-generated traces; none are bundled.

## Schema and labels

Each row has `id`, `dataset`, `source_id`, `source_split`, `category`,
`difficulty_group`, `group_id`, `width`, `height`, and:

- `inputs`: two image structs (`bytes`, `sha256`) and both text shards.
- `labels`: original reviewed `answer_before`, canonical source `answer_after`,
  choices and JSON evaluation metadata (units/precision and prior answer checks).
- `provenance_json`: source revision, attribution/terms, original source record,
  edit proposal and review notes. This is **evaluator-only**, including solutions.

Answer keys are preserved, not silently rewritten. Exact string comparison alone
is insufficient: option letters, option values, LaTeX fractions and equivalent
spellings occur. Use a source-aware scorer or explicit adjudication. In particular,
CharXiv 1212 preserves source `B, N--C`; `B, N-C` was adjudicated equivalent during
construction. Observed blind-solve answers are evidence, not a universal alias list.
Group by `group_id` when splitting to prevent shared-source leakage.

MapQA highest/lowest answers refer to the highest/lowest **displayed value bin**,
not unknowable exact rankings within a bin. Include all states in the requested
extreme bin; state-list order is irrelevant. Exact same-value and within-bin
higher/lower comparisons found in final review were excluded. Template-similar
tables with different data remain; this is not a template-held-out benchmark.

## Scope, quality and rights

Pic2 is the source image; pic1 is an AI-edited variant. Report the per-row
reasoning, calibration and reasoning-reveal difficulty groups separately.
Both images have identical dimensions/aspect ratio, RGB PNG encoding and empty
embedded metadata. Bytes and SHA256s are preserved from the accepted export.
Target-Qwen processor grid/token parity remains pending model selection.
No GPU/AsyncLLM evaluation has been run. There are no no-op controls yet.

Proposal/validation used the same model family; Codex visual review is not
independent human annotation. Some edits change global rasterization/style;
pixel-local isolation is not guaranteed. Geometry may be schematic. Individual
review caveats remain in provenance (including the widened bar in CharXiv 291).
The large construction/API logs and rejected samples are omitted from this file
but remain in the original export. Packaging makes no new paid calls.

Retain per-row attribution and source terms. MathVista contributions are
CC-BY-SA-4.0 with original image/question rights retained; MathVision's dataset
card lists MIT with original competition/image rights retained. CharXiv questions
are CC-BY-SA-4.0, charts belong to original authors, and training is prohibited.
This is an **evaluation-only artifact**, not a blanket redistribution license.
Check original content rights before public redistribution. Pinned source URLs
and original chart identifiers are included in every applicable row.

Expanded versions may also include ChartQA (dataset card GPL-3.0; retain original
chart rights), TabMWP (CC-BY-NC-SA-4.0), MapQA-U (CC-BY-SA-4.0; retain KFF map
rights), and CLEVR (CC-BY-4.0). CLEVR original images are synthetic; the other
sources include charts, diagrams and rendered tables. No new source-scene
rerendering is implied by packaging. Treat the combined artifact as local
evaluation data until source-specific redistribution permissions are established.

## Rebuild

```bash
uv run --extra sources python package_dataset.py --source datasets/diverse_corrections_v3 --output datasets/diverse_corrections_v3.parquet
```

The packager verifies source manifest hashes, IDs, acceptance, image hashes,
dimensions and metadata; validates the written file; and refuses to overwrite an
existing output. Use `--output` with a fresh filename to build another copy.
