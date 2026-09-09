"""FP8 correctness, with pytest as the public entry point.

Fresh run (launches mini, SGLang and Transformers, sequentially on the GPU)::

    python -m pytest tests/e2e/fp8 --fp8-model /path/to/Qwen-FP8 --fp8-results /tmp/fp8-run -v

Serving only (mixed + sequential + both external references)::

    python -m pytest tests/e2e/fp8 --fp8-model /path/to/Qwen-FP8 -k serving -v

Without --fp8-model E2E is skipped. Once explicitly enabled, it always runs
the current mini and external references afresh. Session-scoped pytest fixtures
share only the results produced during this pytest invocation. There is no
disk-cache/replay mode. Normal unit tests remain in tests/core and tests/kernel.

Shared-cache uses 16 text/paired-image cases x 32 teacher tokens. Serving uses
8 text cases x 12 tokens, with ordinary per-request KV/GDN state, not shared-cache
mixed GDN. The two profiles collect references for exactly the same cases and
histories as their mini runs. Dependencies/interpreters must already be installed;
--fp8-sglang-python and --fp8-transformers-python allow separate reference venvs.

Known numerical distinction: the strict serving mixed-vs-sequential gate can fail
on GB10 (FlashInfer attention paths), while SG-relative FP8 quality gates pass.
The refactor does not relax either set of thresholds or mark that failure xfail.
"""
