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
| Higher confidence threshold | On the 126 archived first-probe logits, requiring a write-minus-wait logit margin above one would delay 61/94 ordinary-only losses and 24/32 async-only wins. A higher cutoff had poor selectivity and was dropped. |
| Live question wording | Three stricter phrasings over the live role caches each released the writer on all eight unfinished development openings. They did not justify changing the live question for subsequent paragraph checks. |
| Readiness-chat examples | On 24 development evidence states, the independent task-general zero-shot classifier made 21 correct decisions. The two example orderings made 16 and 18. The fixed zero-shot version went to the eight held-out problems. These probes diagnose readiness independently of answer accuracy. |
| First admission | Commits c54def5 and 605fc6e added a separate native readiness chat over committed notes. After first admission, subsequent checks use live KV. In eight held-out diagnostic problems this delayed five unfinished openings versus one with the earlier live probe. |
| Writer heading | On 16 frozen note prefixes, changing Final answer to Solution reduced immediate greedy box openings from four to zero. The heading change scored 3/6 versus 4/6 for the original heading in fresh CPU reference runs. The original heading remains in the working-notes control. The new first-admission probe with that heading also scored 4/6. |
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

Run the paired cohort with a pinned local snapshot and the byte-identical dataset from the repository root.

~~~bash
python -m minisgl_examples.demos.async_thoughts.math500 \
  --model "$TULU_SNAPSHOT" --model-revision 666943798adbde0b1aff34626007e26986a3c107 \
  --dataset "$MATH500_TEST_JSONL" --output "$RESULTS" \
  --paired 64 --limit 64 --budget 16384 --context 40960
~~~

The evaluator writes per-arm answers, selected sources, exact token sequences and compressed trace hashes. It resumes complete, checksummed rows only. Smoke-only output can be generated in a separate directory with --smoke-only. Setting --limit 500 requests the native-chat extension on the remaining 436 items, as in the first rental's original plan.

To reproduce the retrospective 500-case calculation, supply the original full and selective-retry phase directories and the parent directory of the matched ordinary comparison.

~~~bash
python -m minisgl_examples.demos.async_thoughts.short_retry_replay \
  --async-full "$ORIGINAL_RUN/full" --async-retry "$WRITER_RETRY_RUN/retry" \
  --comparison-run "$MATCHED_RUN" --tokenizer "$TULU_TOKENIZER" --output "$ARCHIVE_REPORT"
~~~

## Validation log

The measured A100 source is commit `7526585`. An initial startup attempt queried the CUDA device name before the engine could initialize CUDA. It produced zero answers. The measured run queries the device after engine initialization and records ordinary forwards through the same trace recorder used for verification.

On the A100, ordinary and native-chat both boxed 5 on the addition smoke and 9 on the divisor smoke. The native thinker matched 61 and 309 initial ordinary tokens, respectively. The writer saw the thinker block grow, with 34 and 211 joint decode batches. This checks execution and concurrency on Tülu independently of the benchmark score.

To regrade a mirrored run offline, use the pinned tokenizer, dataset and measured source directory. The report checks the actual token inputs and compressed traces before calculating paired counts, timing summaries and output-source frequencies.

~~~bash
python -m minisgl_examples.demos.async_thoughts.math500_report \
  --eval-dir "$RESULTS" --dataset "$MATH500_TEST_JSONL" --tokenizer "$TULU_TOKENIZER" \
  --source-dir "$MEASURED_MODULE_DIR" --output "$REPORT"
~~~


## Recovered A100 results

The local artifact mirror stopped synchronizing when the controller Mac slept. Runpod terminated the pod before the missing records were imported. Of 628 scheduled arm records, 34 were recoverable. The first 11 ranks have all three arms. The next rank has only native-chat and is excluded from the paired table. The remaining ranks have no verified A100 results. The posted rental total was $5.026661, including disk.

| Output on the same 11 problems | Strict correct | Median time to first draft content | Median pure generation wall time |
| --- | ---: | ---: | ---: |
| Ordinary | 4/11 | 0.047 s | 11.05 s |
| Working-notes writer | 5/11 | 1.271 s | 4.22 s |
| Native-chat writer | 5/11 | 1.158 s | 16.74 s |

Both async protocols produced joint decode batches and the writer saw the thinker block grow on all 11 problems. Native-chat gained one correct answer and lost none against ordinary. Against working-notes it gained one and lost one. Ten of the 11 native thinkers shared at least eight initial tokens with ordinary generation. In one hard algebra question, native-chat exhausted both 16,384-token caps in 663 seconds. Ordinary also exhausted its output cap on that question. The ordinary first token is visible response text; the async measurement starts with the writer after admission.

The short retry ran on three of the 11 working-notes drafts, raising that score from 5/11 to 6/11. It did not trigger on any of the 11 native-chat drafts. Across the full historical archive it raised the selected score from 128/500 to 177/500, compared with ordinary's 196/500. That archive result closes 49 of the original 68 net misses, at the cost of 196,850 extra sampled tokens. The threshold was developed on those earlier traces, and the archived compositions do not supply fresh latency measurements.

On the 64 prespecified study IDs, the historical archive has ordinary at 31/64, original async at 23/64 and the short retry at 28/64. The retry routed 23 cases, gaining eight and losing three. These are grades of earlier archived outputs. The fresh A100 arm records and their trace hashes are in the [pilot results](llama-math500-pilot.json).

**What the paper can report.** The A100 traces establish that an instruction-tuned Llama checkpoint performed concurrent reasoning and writing. The surviving paired prefix is small, and the native-chat pure score tied the current working-notes score there. A new 64-problem or 500-problem accuracy result requires another measured run. No accuracy improvement for pure AR over the full benchmark has been established.

**Artifact handling for a repeat.** Put the benchmark output on storage that survives pod deletion, or have a separate always-on host pull and verify completed row/trace pairs while the pod is running. Verify the sink on an actual smoke result before starting the cohort. Retain the prompt, source and trace digests and report verified coverage independently of the pod exit code. The first trial used the Mac as the only persistent mirror; that choice lost the unaudited portion of the run.

## Paired-cohort relaunch

The next rental has a separate $2 ceiling. Its target is the original first 64 ranked IDs and the same three arms. The 34 verified arm records from the first rental are retained, including the native-chat output at rank 11. Finishing the paired cohort requires 158 additional arm records. Each imported row keeps its original generation time, trace digest and acquisition-session provenance.

The relaunch uses the same four measured source modules from commit `7526585`, checkpoint, dataset bytes, token limits and grader. Smoke verification occurs before the new benchmark records. An independently running host reads and verifies row/trace pairs directly from the pod, including a smoke artifact before the main phase. No model, prompt, or routing decision is selected using the grades observed in the first rental.

Two paired outputs illustrate why the heading and turn structure were tested. On `test/algebra/1937.json`, the working-notes writer used five tokens to box 883. The native-chat writer evaluated the middle term as 798 and boxed the correct sum, 1600. The opposite happened on `test/intermediate_algebra/1300.json`. The working-notes writer correctly boxed 16 for the minimum of the product of a sum and the sum of reciprocals. The native-chat writer generated a longer derivation and boxed 64. These are individual cases. The paired totals determine the reported accuracy.

## Completed paired results

All **64 fixed problems** have three verified A100 runs, for **192 arm records**. The second rental reused 34 records and generated the remaining 158. Eleven complete triples came from the first A100 and 52 from the second. At rank 11, the native-chat row came from the first rental and the other two arms from the second. Both devices were A100-SXM4-80GB and used the same four measured source modules, PyTorch 2.13.0+cu130, BF16 and greedy decoding. The public ledger contains the boxed outputs, source labels, session labels and trace hashes. [64 paired cases](llama-math500-paired64.jsonl)

| Protocol | Pure strict matches | With short retry | Native retries | Median first draft content | Median pure wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary Tülu | 31/64 (48.4%) | — | — | 0.047 s | 9.11 s |
| Working-notes AR | 24/64 (37.5%) | 30/64 (46.9%) | 18 | 1.416 s | 6.86 s |
| Native-chat AR | **33/64 (51.6%)** | **33/64 (51.6%)** | **0** | 1.761 s | 12.74 s |

The native-chat writer gained 14 correct answers and lost five against working-notes, a net increase of nine answers or 14.1 percentage points. The unadjusted two-sided exact paired sign test gives p = 0.0636. Against ordinary generation it gained four and lost two, a net increase of two answers or 3.1 points (p = 0.6875). All 33 correct native-chat answers came from the concurrent writer. These estimates describe Tülu 3 8B on the 64 ranked problems. The patched native-chat protocol has no fresh 500-problem accuracy measurement.

Both async protocols had joint decode batches and growing writer-visible thinker state on **64/64** problems. The native thinker matched at least eight initial ordinary tokens on 63/64 and at least 32 on 61/64. Its median shared prefix was 213.5 tokens. Across the cohort, native-chat generated 61,535 thinker content tokens and 51,747 writer content tokens. Ordinary generation produced 78,907 content tokens. The median wall times above exclude model load, smoke tests and artifact I/O. The buffered working-notes hybrid's median time to selected output was 9.04 seconds.

On the magic-square item, test/prealgebra/1930.json, ordinary generation used its entire 16,384-token allowance without a box in 273.8 seconds. The native-chat writer boxed 7 in 21.8 seconds using 964 writer and 1,230 thinker content tokens. In the hard polynomial item, test/intermediate_algebra/1411.json, native-chat exhausted both stream budgets and missed in 663.4 seconds. The longer conversations can help or cost substantial computation depending on the question.

The independent relay accepted the final 409-file SHA-256 catalog, combining the two sessions without resampling completed rows. Catalog SHA-256 is 42c31eb2c9ed012a589b067718bed0d83ab21f706cbb0325ca8103f5683cff97. The canonical verified report SHA-256 is 326478405dce1cfde7a4ca14290c77a399144d45c3985d3e65b190c62105e38e. The native-only 436-problem extension was outside the relaunched cohort.

Both A100 pods were deleted and the durable lab copy contains all 192 records. The first rental posted $5.026661 against its $5.50 ceiling. The second rental's charge was estimated at $1.30 through the verified lab acknowledgment, or $1.33 through confirmed provider absence, against its separate $2 ceiling. No billing lines for the second rental had posted at the 21:22 UTC check on November 8.
