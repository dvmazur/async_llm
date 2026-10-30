# Flash Image: three real-source edits

Date: 2026-09-16. User approved one model edit per shortlisted MathVista source.
Editing model: `google/gemini-3.1-flash-image`. Verification model:
`google/gemini-3.8-flash`. Both used Eliza's `/openrouter/v1/chat/completions` endpoint.
The editor used `modalities: [image, text]` and returned inline images in
`choices[0].message.images`. The gateway wrapper's key field is not retained.

[Side-by-side images, text shards, and blind answers](work/flash_image_test_v1/review.html).
[Machine-readable summary](work/flash_image_test_v1/summary.json).

| MathVista source | Model-produced initial image | Original corrected image | Blind answers before → after |
| --- | --- | --- | --- |
| 88 | Right angle label x°, left label x° | Right label 2x°, left label x° | 45 → 30 |
| 926 | Entire semicircle shaded | Only right quadrant shaded | 14.14 → 7.07 |
| 45 | September value 20 | September value 14 | 3 → -3 |

All three requested semantic edits are visible on inspection. Gemini solved each of
the six images in a separate request, with only its image and original question.
Neither the edit instruction nor expected answer was supplied during solving.
All six answers matched manually checked reference calculations. This is one blind
solve per state using another model in the same family, not statistical validation
or a guarantee of pixel-level preservation.

## Preservation limitation

All three outputs are larger JPEGs despite requesting the original canvas dimensions:

| ID | Source PNG dimensions | Generated JPEG dimensions |
| --- | --- | --- |
| 88 | 442 × 277 | 1312 × 816 |
| 926 | 433 × 312 | 1216 × 880 |
| 45 | 273 × 218 | 1152 × 928 |

The raw originals and raw generated images are preserved, with no resizing or
compositing applied. HTML displays them at comparable widths only. Regeneration and
JPEG encoding mean unrelated pixels are not preserved exactly. These pairs are **not
accepted for the controlled async dataset**: their resolution and possibly processed
image grids differ. A future normalization/masked-compositing stage would need its
own verification, and resizing alone would not guarantee that unrelated content is
unchanged. No GPU or async cache experiment ran.

## Cost and service tiers

Flex was requested throughout, but the three image calls reported `default` tier:

- Image edits: $0.0682165 + $0.0682725 + $0.0681425 = **$0.2046315**.
- Six blind solves: **$0.008019375**, all reported `flex`.
- Total reported cost: **$0.212650875**.

These are API-reported costs, not a billing audit. Catalog or requested flex settings
must not be interpreted as evidence of flex image-generation pricing.

## Artifacts and reproduction

`image_edit_test.py` implements `generate`, `verify`, and `report` stages. It uses the
existing uv-managed environment (`uv sync --extra sources`). Source `.env` at the
repository root before running live stages. For example, from this directory:

```bash
uv run --no-sync python image_edit_test.py report --workspace work/mathvista_batch01 --output work/flash_image_test_v1
uv run --no-sync python -m unittest -v test_image_edit test_gateway test_pipeline
```

The seven CPU tests passed. One solve returned unescaped LaTeX in otherwise JSON
output. The report parser repaired invalid backslash escapes, flagged that repair,
and retained the raw response unchanged; the answer itself was unchanged.

Each `work/flash_image_test_v1/{88,926,45}/` directory contains:

- `before.jpg`: actual model-generated erroneous image.
- `pair.json`: source location, hashes, sizes, question, and pending-review status.
- `generation.request.json`: exact final editing prompt and model settings.
- `generation.response.json`: response metadata, with image bytes stored separately.
- `solve_before.*.json`, `solve_after.*.json`: exact blind prompts and raw answers.

Generation refuses to overwrite a saved response. Verification skips completed
responses. No retries or model fallback were used. Work artifacts are gitignored.

The imagegen skill informed preservation-focused prompts and visual inspection.
Execution used the explicitly requested OpenRouter/Eliza API, not the built-in
image generator. Prompts specify one edit, exact affected text/region, and preservation
of all other labels, borders, colors, layout and framing. Their complete text is saved
per sample in `generation.request.json`.
