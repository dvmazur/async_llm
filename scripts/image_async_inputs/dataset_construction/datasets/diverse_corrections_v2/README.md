# Diverse async image corrections v2

41 reviewed pairs. New reported API spend: $4.955348 / $10.

| Dataset | Pairs |
|---|---|
| AI4Math/MathVista | 15 |
| MathLLMs/MathVision | 14 |
| princeton-nlp/CharXiv | 12 |

[Image/text shards](examples.md). Model inputs are `inputs.jsonl`; answers, edit descriptions, source metadata and reviewer notes are separate in `annotations.jsonl`.

Conservative accounted cost, including unknown-cost request reservations: $5.055348. Two screening processes ended before their final response was saved; their reservations are retained rather than assumed free.

These are source-derived questions, not wholly synthetic problems. Pic 1 is AI-edited; pic 2 preserves source content. Prior pilot assets may be reused (see assembly review). New costs exclude those prior calls.

Each pair passed separate blind solves of both normalized image states and a semantic edit audit, followed by Codex visual review. Proposals, solves and audits use Gemini 3.8 Flash; image edits use Gemini 3.1 Flash Image. This is not independent human ground truth. Non-exact equivalent answers have explicit semantic grading evidence.

All pairs match dimensions, aspect ratio, RGB/PNG mode and empty embedded metadata. Original raw files are retained. Minor global rasterization changes are possible. Any relaxed raw-aspect tolerance is explicitly recorded; no blanket alignment override is used. Exact Qwen processor grid/token parity is deferred. No GPU or async inference was run.

Selection is supervised and intentionally biased toward unambiguous editable samples. Report calibration/reveal controls separately. Keep common-source images grouped across splits; this is a local evaluation pilot, not a representative benchmark score.

Sources and terms: [MathVista](https://huggingface.co/datasets/AI4Math/MathVista), [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision), [CharXiv](https://huggingface.co/datasets/princeton-nlp/CharXiv). Revisions and original metadata are retained per sample. MathVista contributions and CharXiv questions use CC-BY-SA-4.0; MathVision card lists MIT. Original image rights remain with their sources. CharXiv is evaluation-only, not training data. Preserve attribution and check rights before redistribution.
