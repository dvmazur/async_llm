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
The demo serves up to four concurrent sessions using one shared model and does not
save live images or dialogue. Each open browser tab uses a session slot; additional
connections receive an explicit resource-limit message. `MAX_USERS` and
`BRANCHES_PER_USER` at the top of `async_tutor/demo.py` determine the session limit
and decode batch capacity. Run one server worker per model instance.

Uploaded videos are sampled during playback. Pausing or reaching the end sends
the stopped frame once for analysis; the connection stays open without repeatedly
encoding and sending the same image.

The **Prompts** tab edits System and Observe for the current browser tab. **Apply &
restart session** clears that session's conversation, media, and KV cache; upload
the video again afterward. **Reset to defaults** only fills the editor until Apply
is clicked. Applied prompts are stored in the tab's `sessionStorage` and survive
a page reload. Each session owns its system KV while sharing the model weights.

The **Trace** button beside the chat opens a compact live log on the right:
Observe assessments, Speak decisions, tutor replies, errors, and session starts.
Repeated WAIT results and equivalent Speak decisions collapse into counters, with
the frame/time range and latest measurements. Recent history stays available
across session restarts; **Clear** clears the displayed log. Technical tracing
remains in the code, while the browser receives only these selected events.
Server durations include waiting for the engine as well as execution.

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
