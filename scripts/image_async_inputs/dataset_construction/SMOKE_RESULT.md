# Gemini 3.8 Flash gateway smoke test

Date: 2026-09-16. One synthetic geometry image, two sequential requests. No real
benchmark samples, GPU execution, image editing, or async-inference experiment.

Model: `google/gemini-3.8-flash`; returned provider: `Google AI Studio`.
Both responses reported `service_tier: flex`, `finish_reason: stop`.
Requested seed 42, medium reasoning effort, maximum 4096 output tokens.

The supplied image has a rectangle with sides 8 and 15. The question asks for the
perimeter of a square whose side equals the rectangle's diagonal. Source answer: 68.

| Attempt | Prompt tokens | Completion tokens (including reasoning) | Reported cost | Result |
| --- | --- | --- | --- | --- |
| 1 | 1282 | 1230 | $0.002787 | Correct calculations, reversed before/after fields |
| 2 | 1425 | 812 | $0.002056875 | Correct facts, answers, and solution-field orientation |

Total reported cost: **$0.004843875**. These are returned API costs, not a separate
billing audit or a forecast for harder real-source examples.

The first attempt proposed replacing label 15 with 6, but called the source image
"before". The prompt now explicitly defines AFTER as the supplied source and BEFORE
as the proposed erroneous version. The second attempt returned parseable JSON:

- Before: sides 8 and 6; diagonal 10; answer 40.
- After: sides 8 and 15; diagonal 17; answer 68.
- Proposed editing method: replace the source label 15 with 6, matching typography.

Both calculations were checked manually. The second proposal's `reasoning_change`
sentence still describes the image-construction direction (15 → 6), rather than the
experimental correction (6 → 15). That sentence needs normalization during review.
This is not automatically accepted dataset content. No proposed image edit was made.

Local raw responses and exact job prompts:

- `work/eliza_smoke_result_v1/d1f9abd549f94d90.response.json`
- `work/eliza_smoke_result_v2/d1f9abd549f94d90.response.json`
- `work/eliza_smoke_result_v1/jobs.json`
- `work/eliza_smoke_result_v2/jobs.json`

The work directory is ignored by git. The original synthetic image remains under
`pilot_v1/01_geometry_after.png`. Tests used the interpreter in the uv-managed `.venv`.

Next: real-source candidate import and review, with explicit timeline checks on model
suggestions. This smoke test establishes transport and basic image-proposal behavior,
not dataset-generation quality at scale.
