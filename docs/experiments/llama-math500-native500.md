# Tülu native-chat adaptation vs ordinary generation on MATH-500

The Tülu native-chat writer solved **209/500 (41.8%)** problems on MATH-500. Ordinary Tülu solved **196/500 (39.2%)** with the same prompts and grader. The observed advantage was 2.6 percentage points (exact paired two-sided p = 0.1048). On the 436 problems added after the pilot, the writer solved 176 and ordinary solved 165.

We evaluated `allenai/Llama-3.1-Tulu-3-8B` (revision `666943798adbde0b1aff34626007e26986a3c107`) with greedy BF16 decoding. Ordinary Tülu used one output stream capped at 16,384 sampled tokens. The native-chat adaptation used separate 16,384-token thinker and writer streams with a 40,960-token context and a probe period of 20. Both methods received identical user prompt tokens on each evaluated problem. We scored the trimmed contents of the last literal balanced `\boxed{...}` in the answer against the trimmed gold string, without mathematical normalization.

Native-chat preserves the concurrent thinker and writer of [AsyncReasoning (Appendix B)](https://arxiv.org/html/2512.10931v3#A2). It changes the first admission decision to a separate chat over the committed notes, then gives the writer a new user turn asking it to check a partial assistant draft. Later switches still use the live probe. The pure score measures these changes together. An experiment varying the probe alone would be needed to attribute the difference to admission timing.

| Cohort | Cases | Ordinary | Pure concurrent writer | With optional retry | Pure gains/losses | Δ (pp) | Paired p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reused pilot | 64 | 31/64 (48.4%) | 33/64 (51.6%) | 33/64 (51.6%) | 4/2 | +3.1 | 0.6875 |
| Complement | 436 | 165/436 (37.8%) | 176/436 (40.4%) | 176/436 (40.4%) | 30/19 | +2.5 | 0.1524 |
| Full MATH-500 | 500 | 196/500 (39.2%) | 209/500 (41.8%) | 209/500 (41.8%) | 34/21 | +2.6 | 0.1048 |

The pure writer and ordinary generation both solved 175 problems, only the pure writer solved 34, only ordinary solved 21, and neither solved 270. The table reports the two-sided exact sign test over discordant pairs. All 64 ordinary pilot replicas reproduced the archived ordinary output and stop token exactly.

The optional retry would use ordinary generation when the writer reached EOS within 16 content tokens. No draft triggered it. All 209 correct answers came from the concurrent writer.

Across the 500 matched problems, native-chat emitted 478,115 thinker and 424,481 pure writer content tokens. Ordinary emitted 461,046. The pure writer reached EOS on 491/500 cases and its length cap on 9. Ordinary reached its length cap on 6. The traces show joint decoding and growth in writer-visible thinker state on 500/500 cases.

Median per-problem inference time was 9.23 s for ordinary, 12.22 s for the pure writer, and 12.22 s including the optional retry policy. The median time to the first writer content token was 1.46 s across 500/500 nonempty writer outputs. Summed per-problem inference time was 6,943.3 s for ordinary and 12,599.0 s for the pure writer. These times exclude model loading, smoke tests, and trace I/O. Native-chat predictions came from 7 GPU sessions. The archived ordinary evaluation took 6,952.4 s wall time (6,965.3 s for the process).

[Per-problem strict grades, timings, token counts, and trace digests](llama-math500-native500.jsonl).

## Comparison with the earlier Tülu port

The earlier working-notes port scored 128/500 (25.6%) with a 16,384-token limit per role. Native-chat gained 81 net correct answers on the same problems. Of the 106 problems ordinary solved and the earlier port missed, native-chat recovered 93. Of the 90 both earlier methods solved, native-chat retained 82 and missed eight. Among the 304 that ordinary missed, native-chat retained 13 of the earlier port’s 38 exclusive solves and added 21 that both earlier methods missed. These comparisons use the same 500 ordinary trace hashes. The earlier traces informed the adaptation, so the 93/106 recovery rate describes a development cohort.

## Timing and compute

The seven native-chat sessions contributed 12, 52, 141, 144, 88, 44 and 19 new results, respectively. The first two supplied the retained pilot. Summing inference time once per problem gives 3 h 30 min for native-chat and 1 h 56 min for ordinary. Native-chat took 1.81 times as long by this measure. The writer emitted 33.7 answer tokens/s versus 66.4 for ordinary. Including private thinker tokens gives 71.6 tokens/s across both native-chat streams. A per-stream token limit allows native-chat to generate more tokens in total. The runs used separate A100-SXM4-80GB devices. These figures describe serial problems and exclude rental setup and handoffs.

## Acquisition and billing

The November 9 extension ran from 08:37:14 UTC until the final pod’s absence was confirmed at 13:52:27 UTC, or 5 h 15 min 13 s. The six sequential rental windows totaled 4 h 01 min 01 s. A 70-minute gap during SSH recovery incurred no pod charge. The evaluator reused all 64 pilot problems and the archived ordinary answers. The table covers the costs of acquiring the remaining 436 answers, including temporary disk.

| Rental | New native answers | Posted by 13:54 UTC | Estimated total where billing was incomplete |
|---|---:|---:|---:|
| 1 | 141 | $1.587929 | — |
| 2 | 144 | $1.600801 | — |
| 3 | 88 | $1.597668 | — |
| 4 (setup only) | 0 | $0.138521 | — |
| 5 | 44 | $0.537887 | $1.554341 |
| 6 | 19 | $0.000000 | $0.705333 |

Posted charges totaled $5.462806 at the stated billing check. Using creation through confirmed deletion for the two incomplete bills gives an estimated **$7.18459 total**. All individual rentals were below $2 including disk, and Runpod confirmed all six pods deleted.

All 500 matched prompt hashes and source traces passed offline verification. The final mirrored catalog had 1,284 files and SHA-256 `3ec4fc2abd4b8728e7d103ed665914bd2127d55eb23f76a50a46e6cc8d4a9594`. The public ledger has SHA-256 `24411345e713b72088c7a4930258b66ec2144b811ff4781b9ca2bffd85aece83`. [Protocol design decisions and earlier experiments](llama-asyncreasoning.md) · [Per-problem ledger](llama-math500-native500.jsonl).
