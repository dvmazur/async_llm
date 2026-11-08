# Llama AsyncReasoning on MATH-500

## Question and comparison

Can a concurrent thinker and writer produce accurate answers on an ordinary instruction-tuned Llama checkpoint? The primary score grades the writer's own answer. A second, separately reported experiment allows an ordinary generation to replace a short draft. This distinction matters when reporting support for models beyond Qwen.

The original AsyncReasoning work studies concurrent thought and answer generation, including the delay to the first output token. Here the benchmark records both elapsed time to the first generated writer content and end-to-end time. A retry policy that buffers its draft exposes its selected answer later. [Asynchronous Reasoning](https://arxiv.org/abs/2512.10931)

The model is allenai/Llama-3.1-Tulu-3-8B at revision 666943798adbde0b1aff34626007e26986a3c107. MATH-500 is HuggingFaceH4/MATH-500 at revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be. The test.jsonl SHA-256 is 35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132. The ordinary and async models receive the same user message, including the request for a boxed answer and the working-notes instruction. The grader extracts the final literal balanced box, strips outer whitespace, and compares it to the gold string. Mathematically equivalent expressions can receive different grades.

## Decisions before the new GPU measurements

| Design choice | Evidence and decision |
| --- | --- |
| Independent thinker | The thinker uses the original prompt and its own KV. The writer sees both actors' committed KV. Commit 836426d corrected a reciprocal read in the initial Llama port. |
| Ordered readiness votes | Commit 743d3b6 ordered writer steps and votes so a queued step cannot race a newer WAIT decision. |
| Probe labels | Commit 23eba4d included single-token capitalization and leading-space variants for Llama. In archived GPU observations the old lowercase-only vote disagreed with the unconstrained greedy label on 30/126 first checks. |
| First admission | Commits c54def5 and 605fc6e added a separate native readiness chat over committed notes. After first admission, subsequent checks use live KV. In eight held-out diagnostic problems this delayed five unfinished openings versus one with the earlier live probe. |
| Writer heading | A controlled change from Final answer to Solution reduced immediate box openings, but scored 3/6 versus 4/6 for the original heading in fresh CPU reference runs. The original heading remains in the working-notes control. The new first-admission probe with that heading also scored 4/6. |
| Native thinker start | The old thinker injected a role label and a forced newline after the checkpoint's native assistant prompt. Only 221/500 historical thinker outputs matched the first ordinary generated token. The native-chat candidate feeds the final real prompt token exactly once, making the first think step comparable to ordinary generation. |
| Native writer turn | The old writer heading was encoded before the thinker had produced any notes. The candidate builds a native user follow-up and assistant-generation suffix from the checkpoint chat template, then encodes that suffix when the writer is first admitted. It receives the notes committed by that point. The writer continues to read evolving thinker KV after admission. |
| Short-answer retry | The opt-in hybrid retries only an EOS-terminated async writer draft with at most 16 content tokens. The threshold came from the earlier 106-loss analysis. It uses no label at inference time. The retry starts from the original ordinary prompt on an independent KV block. Draft, EOS and retry all consume the same writer token allowance. The selected output and both phase costs are recorded. |
| Scope | The working-notes protocol remains available for reproduction. The native-chat variant bundles native thinker seeding, a different writer turn, and deferred writer prefill. A score difference between these variants cannot identify an individual component's causal effect. |

The historical original async run scored 128/500, ordinary scored 196/500. Of the 106 ordinary-only losses, 63 had writers of at most 16 tokens and 105 reached EOS. All 38 async-only wins also began writing after at most 20 visible thinker tokens, so first-writer timing alone did not distinguish wins from losses. Four ordinary-only loss traces contained the correct thinker box eventually. In three the literal correct box was already visible to the writer by the wrong box.

Replaying the short-draft rule over the archived original async and ordinary generations yields 177/500. It routes 188 cases, gains 63 and loses 14 relative to the historical async outputs. Five archived ordinary outputs are clipped to the remaining writer allowance with no strict grade changes. This is an exploratory retrospective hybrid estimate, derived on a dataset already examined during development.

## GPU protocol fixed before inspecting new scores

- Use one A100, BF16, greedy decoding, probe period 20, up to 16,384 sampled tokens per role and a 40,960-token engine context. Ordinary generation uses a single 16,384-token cap. Async has separate thinker and writer caps. The optional retry shares the latter with the draft.
- Smoke-test two arithmetic prompts before evaluation. Verify the executed native prompt, compare initial thinker samples with ordinary generation, require a correct boxed answer on 2 + 3, and verify a real joint decode in which the writer subsequently sees more thinker KV.
- Rank the 500 canonical IDs by SHA-256 of the UTF-8 bytes of the literal tulu-native-concurrent-v1, a zero byte and the unique ID. The first 64 ID list has digest b89ce2ed9f90f6ebb65dc3c5c39baeb076d35a2743f260c309756506fade02a9. The digest for all 500 ordered IDs is 3af38ebff7179f04eeb979b1d8037562cf694e6f5962dfc99cc10e5b69648095. These are SHA-256 hashes of compact JSON string arrays.
- On the first 64, generate fresh ordinary, working-notes and native-chat answers. Rotate arm order by rank. Record pure writer scores, retry-selected scores, paired gains and losses, output source, wall time, token use and concurrent decode witnesses. Use only complete same-ID pairs for comparisons. Continue native-chat in the same fixed order toward 500 as the rental permits.
- MATH-500 had previously been inspected. The ordering and model decisions were frozen before the new GPU outputs. The cohort provides prospective rollouts on a previously analyzed benchmark; it is not an untouched test set.

Run the evaluator with a pinned local snapshot and the byte-identical dataset from the repository root.

~~~bash
python -m minisgl_examples.demos.async_thoughts.math500 \
  --model "$TULU_SNAPSHOT" --model-revision 666943798adbde0b1aff34626007e26986a3c107 \
  --dataset "$MATH500_TEST_JSONL" --output "$RESULTS" \
  --paired 64 --limit 500 --budget 16384 --context 40960
~~~

The evaluator writes per-arm answers, selected sources, exact token sequences and compressed trace hashes. It resumes complete, checksummed rows only. Smoke-only output can be generated in a separate directory with --smoke-only.

## Validation log

The measured A100 source is commit `7526585`. An initial startup attempt queried the CUDA device name before the engine could initialize CUDA. It produced zero answers. The measured run queries the device after engine initialization and records ordinary forwards through the same trace recorder used for verification.

On the A100, ordinary and native-chat both boxed 5 on the addition smoke and 9 on the divisor smoke. The native thinker matched 61 and 309 initial ordinary tokens, respectively. The writer saw the thinker block grow, with 34 and 211 joint decode batches. This checks execution and concurrency on Tülu independently of the benchmark score.

To regrade a mirrored run offline, use the pinned tokenizer, dataset and measured source directory. The report checks the actual token inputs and compressed traces before calculating paired counts, timing summaries and output-source frequencies.

~~~bash
python -m minisgl_examples.demos.async_thoughts.math500_report \
  --eval-dir "$RESULTS" --dataset "$MATH500_TEST_JSONL" --tokenizer "$TULU_TOKENIZER" \
  --source-dir "$MEASURED_MODULE_DIR" --output "$REPORT"
~~~

