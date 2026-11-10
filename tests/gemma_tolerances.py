"""Temporary, user-approved Gemma numerical thresholds (2026-11-08)."""

# TODO: investigate the remaining differences and restore the target factor 1.0:
# error(Mini, Transformers) <= error(SGLang, Transformers), for every metric.
# 1.5 is a temporary allowance, NOT an accuracy improvement. Reports keep the
# strict 1.0 result too. The current learned-model panel reached factor 1.430306.
GEMMA_REFERENCE_ERROR_FACTOR = 1.5

# TODO: investigate batch/chunk-dependent API differences and tighten these.
# Learned 31B: initial max logit gap 2.75; checking the remaining batch rows
# exposed 3.125 (old limits 0/.01/.02/.2). The max
# per-token probability gap .0281 (old .0025). These are absolute errors, not TV.
GEMMA_API_LOGIT_ATOL = 3.5
GEMMA_API_PROBABILITY_ATOL = .03

# Long image/window layout tests previously required bitwise equality. Checking
# ALL saved tensors exposed max absolute logit gaps 9.96875 (split) and 5.5
# (append/chunked), beyond the short API panel above. This is a loose regression
# ceiling, NOT a claim of equivalent layouts. TODO: investigate these large gaps.
# The external 1.5x TV/L2 gate still applies independently to every layout.
GEMMA_LAYOUT_LOGIT_ATOL = 10.0

# Tiny random BF16 models: initial failures reached .0403; continuing past them
# exposed .1358 on a second-image append. Old limits were .01/.015.
# TODO: investigate these differences, rather than assume rounding only.
# Do not apply these allowances to KV/state/lifetime or direct kernel tests.
GEMMA_SMALL_LOGIT_ATOL = .15
