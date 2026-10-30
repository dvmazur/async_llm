"""FP8 correctness, with pytest as the public entry point.

Fresh run (launches mini, SGLang and Transformers, sequentially on the GPU)::

    python -m pytest tests/e2e/fp8 --fp8-model /path/to/Qwen-FP8 --fp8-results /tmp/fp8-run -v

Serving only (mixed + sequential + both external references)::

    python -m pytest tests/e2e/fp8 --fp8-model /path/to/Qwen-FP8 -k serving -v

Short causal-chain parity (three prompts, eight teacher positions)::

    python -m pytest tests/e2e/fp8 -k test_chain_external --fp8-model /path/to/Qwen-FP8 -v
    python -m pytest tests/e2e/fp8 -k bf16_chain --chain-bf16-model /path/to/Qwen-BF16 -v
    python -m pytest tests/e2e/fp8 -k moe_chain --fp8-moe-model /path/to/Qwen-MoE-FP8 -v
    python -m pytest tests/e2e/fp8 -k moe_serving --fp8-moe-model /path/to/Qwen-MoE-FP8 -v

The chain profile runs flat, three-block, and shared-prefix chains of depths
2/3/4 in ONE mini model load. It tests the actual eager shared-cache API of
qwen-fp8: no graph runner, graph replay counters, or fusion-branch dependencies.
It checks exact causal histories, physical prefix sharing, empty write tails,
unchanged prefixes, growing write blocks and changing worker order. Each layout
has 21 decode positions and nine cold-prefill comparisons. BF16 uses the same
topologies with a separately selected unquantized checkpoint; it never weakens
the FP8 storage checks. Separate MoE is opt-in, never substituted with dense;
its native FP8 expert calls must be observed in both prefill and decode.
The optional MoE serving profile also requires actual FP8 expert calls inside
mixed batches. It uses the inherited eight text prompts and twelve teacher
positions, not a behavioral task. Select -k moe_chain or -k moe_serving to run
one profile; supplying --fp8-moe-model with no filter opts into both profiles.
SGLang's memory fraction can be selected with --fp8-reference-memory-fraction;
the default is .15 for the primary/small control, .6 for the separate MoE model.
No behavior benchmarks, free generation, answer scoring, or dataset downloads
are part of these tests. Ordinary mixed/sequential serving and the inherited
image tests remain as separate, explicit profiles.

Without --fp8-model E2E is skipped. Once explicitly enabled, it always runs
the current mini and external references afresh. Session-scoped pytest fixtures
share only the results produced during this pytest invocation. There is no
disk-cache/replay mode. Normal unit tests remain in tests/core and tests/kernel.

Shared-cache uses 16 text/paired-image cases x 32 teacher tokens. Serving uses
8 text cases x 12 tokens, with ordinary per-request KV/GDN state, not shared-cache
mixed GDN. The two profiles collect references for exactly the same cases and
histories as their mini runs. Dependencies/interpreters must already be installed;
--fp8-sglang-python and --fp8-transformers-python allow separate reference venvs.

All E2E profiles use the
same quality criterion: mini's distance to SGLang must not exceed Transformers'
distance to SGLang plus the established tolerance. Different schedules need not
produce identical logits. Mixed-batch coverage is still checked explicitly.

TEMPORARY FRAGILE POLICY (user-approved 2026-09-15): each mean/p95 TV/L2 error
must be at most 2 times the corresponding Transformers-to-SGLang error.
This relaxes the former 1.05x gate; it is NOT a numerical bugfix or a calibrated
task-quality guarantee. The threshold is reference/batch/checkpoint-sensitive
and may be revised or removed. Max TV/top1 remain diagnostic; zero reference
error still permits only zero error. Reports explicitly mark this policy fragile.
User-approved follow-up: unquantized BF16 chain controls alone allow 2.2x for
prefill p95 TV (observed ratio 2.1603); all other metrics and FP8 remain at 2x.
This post-hoc exception is also TEMPORARY AND FRAGILE, not a numerical fix.
Reports expose per-metric error_factors; relative_tolerance_percent is the default.
Only this single gate controls acceptance. Direct Mini-to-Transformers distances
are report-only. Optional ablations/layer dumps live in the separate
python -m tests.e2e.fp8.diagnostics entry point, never in acceptance execution.
"""
