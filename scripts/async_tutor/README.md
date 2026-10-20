# async-tutor

Camera/photo tutoring with concurrent observation, hint probes, and replies.
Readers share completed image/observation cache blocks and append private suffixes;
snapshots live until their last reader finishes.

Install from the repository root into the same environment as `minisgl`:

```bash
uv sync --locked --python 3.12
uv pip install -e scripts/async_tutor
uv run --no-sync async-tutor
```

Choose an available GPU. The default Qwen3.8-27B needs one 80 GB GPU, CUDA 13
(driver 580+), and FFmpeg with `libx264`/`libx265`. Open **http://localhost:7860**;
for a remote GPU, forward port 7860 over SSH. Camera access needs localhost or HTTPS.
The demo serves one session and does not save live images or dialogue.

Record three lessons and three simulated student scenarios, or regenerate saved MP4s:

```bash
uv run --no-sync async-tutor-record --download
uv run --no-sync async-tutor-record --output artifacts/tutor-eval/run
uv run --no-sync async-tutor-record --render artifacts/tutor-eval/run/*/events.json
uv run --no-sync async-tutor-record --report artifacts/tutor-eval/run --output artifacts/tutor-eval/index.html
```

Use a new output directory; `--help` lists case, duration, and student-turn options.
The commands also work as `python -m async_tutor` and `python -m async_tutor.record`.
Keep `--no-sync` with `uv run` so it retains the separately installed demo package.

To reproduce an MP4, retain `events.json` and every referenced video/JPEG at its recorded
path, with the same FFmpeg, Pillow, and DejaVu Sans font. Rendering uses CPU, preserves
1x event timing and the final frame hold, and adds four reading seconds.
