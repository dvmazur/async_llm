# Tülu 3 8B on MATH-500: accuracy, generated tokens and inference steps

We compared the 500 matched problems using the same greedy checkpoint, user prompts, strict boxed-answer grader and 16,384-token limit per stream. Writer tokens are emitted answer content tokens, excluding EOS. For ordinary generation the sole stream is the writer. A decode step is one executed backend decode batch, including EOS when sampled. A joint thinker/writer batch counts once. Forward steps count all backend batches, including prompt, bridge and probe prefills. Different types of forward step can take different amounts of time.

| Method | Strict accuracy | Mean writer tokens | Median writer tokens | Mean decode steps | Mean forward steps |
|---|---:|---:|---:|---:|---:|
| Ordinary Tülu | 196/500 (39.2%) | 922.1 | 628.5 | 923.1 | 924.1 |
| Earlier working-notes port | 128/500 (25.6%) | 459.2 | 178 | 556.6 | 595.3 |
| Native-chat adaptation | 209/500 (41.8%) | 849.0 | 473.5 | 1,374.6 | 1,445.4 |

Native-chat generated 956.2 thinker tokens per problem in addition to its 849.0 writer tokens. It used more total content tokens on 478/500 problems and more inference time on 430/500. Its writer used fewer tokens on 402/500. The working-notes row uses the verified 16k result, including the writer and thinker budget repairs. [Full benchmark and case ledger](llama-math500-native500.md).

## Accuracy at a fixed token spend

We replayed the saved greedy outputs and scored only the writer text available before each shared content-token cap. Both native streams count toward the cap. A joint batch producing two content tokens is kept only if both fit. EOS, probe prefills and prompt processing consume no content-token allowance. All 500 unrestricted answers and grades matched the archived outputs. Truncation can cut an answer before it contains a box. A model given a budget-aware prompt or an instruction to finalize early would be a different experiment.

| Uniform cap per problem | Ordinary matches | Native-chat matches | Ordinary total content tokens | Native-chat total content tokens |
|---:|---:|---:|---:|---:|
| 512 | 150 | 46 | 226,923 | 249,636 |
| 1,024 | 192 | 156 | 326,181 | 433,514 |
| 1,134 | 194 | 169 | 337,255 | 461,102 |
| 2,048 | 196 | 201 | 373,153 | 597,058 |
| 4,096 | 196 | 208 | 387,318 | 658,769 |
| 8,192 | 196 | 208 | 411,894 | 704,019 |
| 16,384 | 196 | 208 | 461,046 | 794,131 |
| Complete saved outputs | 196 | 209 | 461,046 | 902,596 |

For **equal aggregate generated content**, we selected the native cap closest to ordinary’s full 461,046-token spend using lengths alone. At 1,134 combined tokens per problem, native-chat used 461,102 tokens and scored **169/500 (33.8%)**, versus ordinary’s **196/500 (39.2%)**. For **equal per-problem caps**, ordinary led at 1,024, and native-chat led at 2,048 while consuming substantially more tokens in aggregate. These comparisons measure output available at a cutoff. Generated content, forward steps, GPU time and FLOPs are different quantities.

## When the writer ran

Only **91/500** problems admitted the writer on the first standalone readiness check. The 500 admissions required 4,029 checks. At its first decode the writer could see a median of **81 generated thinker tokens**. All 500 first writer steps shared a batch with the thinker.

The traces record requests and cache views. For the 29,905 later checks, a writer decoded while the thinker was still active after 19,475 checks, no writer decoded before the next check in 10,383 intervals, and 47 ended with thinker completion confounding the release. This is an observed scheduler activity rate of about 65%. Decisions can reaffirm an already open gate, and triggers can coalesce. The saved traces do not log the yes/no logits. Of 424,972 writer decode requests, 208,668 (49.1%) ran after the thinker's final decode. The writer then runs independently to EOS or its cap.

The notes sometimes contribute directly. On `test/intermediate_algebra/1837.json`, the writer saw 299 generated thought tokens. It picked up two polynomial expansions from the notes and began with step 3. It gave the gold answer 34, while ordinary Tülu gave −8. On `test/intermediate_algebra/1300.json`, the writer saw 140 thought tokens, restarted the derivation and repeated a reciprocal-sum error. It gave 64, while ordinary gave the gold answer 16. These examples show how either behavior can occur. Varying the writer’s access to the notes with other settings fixed would test their causal contribution.

The 4,029 readiness checks processed 1,604,935 question input tokens and the 29,905 later checks processed 986,865. These are prefill input rows, additional to the generated content above. A useful next controlled comparison is a stricter writer policy evaluated at the same total content-token allowance, with per-probe events recorded.

## Reproduce the replay

Run the following command from the repository with the examples installed. Use the final native evaluation directory, the archived ordinary result journal, their verified comparison directory and the tokenizer from the pinned checkpoint. The script checks the supplied journals and native trace hashes, reconstructs the joint decode order, and regrades prefixes. It needs no GPU.

```sh
python -m minisgl_examples.demos.async_thoughts.math500_token_budget \
  --eval-dir "$native_run/s06/eval" \
  --ordinary-results "$ordinary_run/matched/results.jsonl" \
  --comparison "$native_run/comparison" \
  --tokenizer "$pinned_tokenizer"
```
