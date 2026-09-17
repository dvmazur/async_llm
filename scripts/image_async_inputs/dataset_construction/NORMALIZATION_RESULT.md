# Pair normalization

Normalization is now a required step between image attachment and acceptance.
Implemented in `normalize_pair.py` and integrated into `pipeline.py normalize`.

Applied on CPU to all three Flash Image test pairs, preserving original files:

| MathVista ID | Both normalized images | Format / mode | Embedded metadata |
| --- | --- | --- | --- |
| 88 | 442 × 277 | PNG / RGB | Empty |
| 926 | 433 × 312 | PNG / RGB | Empty |
| 45 | 273 × 218 | PNG / RGB | Empty |

Files: `work/flash_image_test_v1/{88,926,45}/normalized/{before,after}.png`.
Each directory also contains `normalization.json` with source/output hashes and policy.

Policy: orient using EXIF, convert ICC profiles to sRGB when present, composite alpha
on white, resize the generated image to oriented source dimensions with Lanczos,
then save fresh RGB PNGs without metadata. Source color is assumed sRGB when no ICC
profile exists. Aspect drift above 1% is rejected for review instead of silently
stretching substantially. No cropping or padding occurs.

These outputs match file-level properties but are not automatically accepted dataset
pairs. Normalization does not remove unrelated generative changes. Earlier blind solves
were on raw images, and normalized-image answer validation remains pending. Exact model
processor grids/token counts also remain pending. No API calls or GPU use occurred
during normalization. Nine CPU tests passed, including alpha handling, metadata stripping,
raw-file preservation, rejection of excessive aspect changes, and the export gate.

Reproduce the batch with a fresh test-output directory, or normalize a fresh pair:

```bash
uv run --no-sync python normalize_pair.py --before /path/to/raw-before.jpg --after /path/to/raw-after.png --output /path/to/new-normalized-directory
```

Existing normalized directories are not overwritten. Optional processor checking is
documented in PIPELINE.md and does not download or load model weights.
