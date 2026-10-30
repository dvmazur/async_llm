# MathVista async image corrections v1
Small reviewed construction pilot, not a model benchmark result.
| Source ID | Group | Before answer | After answer |
|---|---|---|---|
| 45 | calibration | 3 | -3 |
| 331 | reasoning | 65 | 64 |
| 234 | reasoning_reveal | It cannot be determined from the information given | 6 |
| 206 | calibration | 2 | 3 |
| 926 | reasoning | 14.14 | 7.07 |
| 88 | reasoning | 45 | 30 |

New reported API spend: $0.26043025; budget: $2. Earlier pilot edits were reused.

Inputs: `inputs.jsonl` and normalized PNGs. Answers/provenance: `annotations.jsonl`. `evidence/` holds normalized-image blind solves and semantic pair audits. `examples.md` displays both image and text shards.

Every accepted pair was visually reviewed and its normalized before/after states solved separately without reference answers. The same Gemini 3.8 Flash family was used for proposals, solves and audits; these checks are not independent human labels. Codex additionally checked the image evidence and calculations.

Report calibration and reveal subsets separately. The pilot is intentionally selected for valid edits and is not a random representative sample. Original images may contain printed question text. Questions and choices are preserved.

Image dimensions, aspect ratio, RGB PNG mode and metadata match within each pair. Generated edits can change unrelated pixels; only semantic preservation was reviewed. Exact Qwen processor grid/token parity is pending model selection. No GPU or async inference experiment was run.

Provenance: AI4Math/MathVista testmini at the pinned revision in annotations. MathVista contributions are CC-BY-SA-4.0; original image/question rights are retained by their sources. This is a local derived evaluation set, not training data.
