# Tülu 3 8B writer gate and reasoning baselines on MATH-500

The previous native-chat writer solved 209/500 problems, compared with 196/500 for ordinary Tülu. It generated 902,596 content tokens across its two streams versus 461,046 for ordinary generation. When the saved native-chat answers were cut to nearly the same aggregate token spend, they scored 169/500. This rerun asks whether changing when the writer runs can improve that accuracy–work tradeoff. [Previous measurement](llama-math500-efficiency.md).

## Design fixed before the pilot

All new arms use the same instruction-tuned checkpoint, `allenai/Llama-3.1-Tulu-3-8B` at revision `666943798adbde0b1aff34626007e26986a3c107`, greedy BF16 decoding, the pinned MATH-500 file and the existing literal last-box grader. Only the visible final answer is scored. A sampled EOS counts toward its generation allowance. Generated content counts reported in tables exclude EOS. Direct and serial thinking are prompted protocols on the same model.

| Arm | Procedure | Token budget |
|---|---|---|
| Direct answer | One ordinary chat response requesting just the final answer, with no separate reasoning pass. | 16,384 samples total. |
| Serial thinking | Generate private scratch work, then generate a visible answer with the scratch work available. | Up to 8,192 private samples. All remaining samples of a shared 16,384 are available to the answer. |
| Native-chat with the revised gate | Use the same two live cache views, task and writer bridge as the previous native-chat arm. Preserve the first readiness chat. Each subsequent grant lasts one paragraph or at most 20 writer tokens. Check again after new committed thought, and give the live probe the pending writer token in its private input. When thinking ends, the writer may finish. | 16,384 samples per role to match the previous native-chat setup. Also score at shared sample and content budgets, specifying which was counted. |
| Archived ordinary and original native-chat | Join the verified scores on the same problem IDs. The archived ordinary prompt requests working notes. | Original budgets. |

The continuation question asks whether the notes justify a concrete next step that the answer has not covered. The new schedule avoids polling an active writer. The pending-token option lets the probe see the writer's most recent emitted token without adding it a second time to the writer's cache. The primary patched arm tests these changes together. The mechanisms can also be enabled separately for an ablation.

The first 64 problems are the previous deterministic pilot, in SHA-256 rank order. Run all new arms on each rank. Report the 64 separately from any extension, retain completed rows and label partial coverage. Previously measured results supply matched controls. Newly prompted baselines are generated fresh. Record reports and compressed traces outside each rental, with checksums, before deleting it.

## Evaluation

Report the number correct, mean/median writer tokens, mean thinker tokens, mean decode batches and mean total forward batches. A joint thinker/writer batch counts as one step. Report paired gains and losses, wall time, probe counts and joint decoding. Compare the revised and original writer at matched total token spend as well as their full token allowances. For prefix scores, grade the writer text available before the cap; a joint step that does not fit is excluded in full. A prefix comparison measures an interruption of an existing trajectory. Compare the representative direct and serial baselines on the same IDs, including cases where the writer uses few tokens but does not produce a correct boxed answer.

A reduction in probe calls alone would not demonstrate that the accuracy–work gap has closed. The comparison should show how much work bought each accuracy result, and whether gains came from solving more problems or from spending less after the thinker stopped.

## Reproduce the measurement

With the pinned checkpoint in the Hugging Face snapshot cache and the pinned MATH-500 `test.jsonl` on disk, run from the repository root on one A100:

```sh
export PYTHONPATH=scripts:python
module=minisgl_examples.demos.async_thoughts.math500_tulu_paired
python -m "$module" --model "$MODEL" --dataset "$DATASET" --output "$OUT/smoke" --smoke-only
python -m "$module" --model "$MODEL" --dataset "$DATASET" --output "$OUT/eval" --limit 500 --budget 16384 --context 40960 --arms direct serial-thinking evidence-turns
```

The fixture run is separate from the scored data. The second command rotates arm order by problem rank; it resumes complete arm records from the same run directory after checking their identities, trace hashes, prompts and grades. It never substitutes an ordinary answer for the concurrent answer. Keep the same limit when resuming. Use `math500_tulu_compare` to recheck the native token IDs, physical cache writes, sample limits, gate events, grades and paired outcomes; use `math500_tulu_content_budget` for comparisons against the archived controls at common content budgets. Both accept `--expected 64` for the pilot and `--expected 500` for the full set. The tables should show writer/thinker tokens separately from decode and forward steps; an earlier end to visible writing can still move work into the private stream.

The archived 64-case pilot provides a check on this analysis: ordinary Tülu scored 31/64 with 78,907 content tokens, and the previous native-chat writer scored 33/64 with 113,282. Truncating the saved native-chat traces at a uniform 2,063 combined tokens per problem yielded 78,908 content tokens and 31/64: two paired gains and two losses relative to ordinary. These figures come from the original protocol. They do not measure the revised gate or either newly prompted baseline.
