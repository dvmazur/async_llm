# Query-Rotation Decode for Shared Cache — Design & Plan

> **STATUS (2026-06-09): Phases A–D implemented and verified.**
> `shared_cache/attention.py` (FlashInfer decode + `merge_states`, torch
> reference behind `MINISGL_SC_SDPA=1`, per-layer cross-check behind
> `MINISGL_SC_COMPARE=1`); `decode_step` rewritten block-relative; the
> per-step KV copy is gone.  All 28 oracle tests + 27 shared-cache tests
> pass (incl. a new interleaved-growth test).  Decode semantics now match
> the reference's same-step cross-worker visibility.  Per-step latency:
> Qwen3-8B, 2 workers, ~4.6k shared tokens: **23.9ms vs 37.8ms** (old),
> flat vs linear in shared-context size.
> Caveat found during verification: AsyncReasoning's *slow* reference
> mishandles ragged (unequal view length) multi-worker batches under
> transformers>=4.56 — it disagrees with its own single-worker forward.
> Oracle tests therefore use equal-length views; minisgl's ragged path is
> self-consistent (bit-exact vs solo chains) and matches the reference's
> single-worker forward.

Replace the per-step KV re-rotation in shared-cache decode with the **query-rotation**
scheme from [*Asynchronous Reasoning: Training-Free Interactive Thinking LLMs*
(arXiv:2512.10931)](https://arxiv.org/abs/2512.10931), Yakushev et al.

Reference implementation is cloned locally at `/home/dvmazur/AsyncReasoning/` — both the
HF driver (`async_reasoning_inference/attention.py`) and the fused CUDA kernel
(`inference_lib/src/kernels/kernel_v21.cuh`, lineage: Hogwild! arXiv:2504.06261).

---

## 1. What attention this project uses

Three interchangeable backends (`python/minisgl/attention/`), auto-selected in
`engine/engine.py` (`_adjust_config`, ~line 222):

| Backend | Kernel | Used when |
|---|---|---|
| `fa`  | `sgl_kernel.flash_attn.flash_attn_with_kvcache` | SM90 (prefill half of `fa,fi`) |
| `fi`  | FlashInfer `BatchDecode/PrefillWithPagedKVCacheWrapper` (fa2) | default decode |
| `trtllm` | FlashInfer `trtllm_batch_{decode,context}` | SM100 |

They share one contract (`attention/base.py`): a **single paged page-table per request**,
one query position per token, standard causal attention. RoPE is applied to **both q and k**
in `AttentionLayer.forward` (`layers/attention.py:54`) from `batch.positions`, *before* the
backend runs. KV lives in a paged, **token-major** pool `[slot, n_kv_heads, head_dim]` with
`page_size=1` for `fi` (`kvcache/mha_pool.py`).

### The shared-cache decode path (the thing we're changing)

`SharedCacheSession.decode_step` (`shared_cache/session.py`, driven from
`scheduler/scheduler.py:327`) builds one `Req` per worker with a **concatenated** page table
over all of that worker's blocks, then calls the normal backend. To keep RoPE correct when a
block sits at a different position than where it was written, `_apply_corrections` →
`correct_kv_pages` (`session.py:175`, `shared_cache/rope_correction.py:66`) **copies every
block's K/V to fresh temp pages across all layers and re-rotates the keys — on every decode
step.**

**This is the inefficiency:** `O(Σ block_len × n_layers)` HBM read+write+rotate per step,
repeated and growing as blocks grow.

---

## 2. The target scheme

**RoPE depends only on relative position:** `⟨R(q, a), R(k, b)⟩ = g(q, k, a − b)`. To place a
key at logical position `n` without touching it, rotate the **query** instead.

The reference makes this clean by **storing every block's keys at block-relative positions
`0…L−1`** (each block is position-independent). At decode, for worker `w` and segment `s`
starting at view-offset `O_s`:

```
loc(s, w) = P_w − O_s          # P_w = query's logical position in worker w's view
```

rotate the query to `loc`, and compute **one softmax over the concatenation of per-segment
scores**, each segment using its own rotated query. The fused kernel does this flash-style
(online softmax over fragments + split-K reduction); keys are read in place, only the tiny
query tensor `[F, W, Hq, D]` is rotated.

**Masking:** `key l valid iff l ≤ loc`. In **decode this is automatically satisfied** — non-self
blocks are entirely in the past (`O_s + L_s ≤ P_w ⇒ loc ≥ L_s`), and the self (write) block's
newest token *is* the query (`loc = L_self − 1`). So **no intra-block mask is needed** for the
decode (S=1) case.

### Why block-relative storage is required (not "rotate q by P−Δ on the current store")

A tempting shortcut is to keep mini-sglang's current store (keys RoPE'd at their view-global
write position) and rotate the query by `P − Δ`, where `Δ` is the existing correction delta.
This works **only if `Δ` is constant within a block**, which is **false in general**:

- The writer block is written while the thinker block keeps growing. Writer token `t` is stored
  at view position `P_prompt + len(thinker@t) + len(close) + t`. Because `len(thinker)` grows
  between writer steps, the writer block's stored view-positions are **non-contiguous**, so the
  per-token `Δ` is not constant within the block.

The current per-token KV-copy handles this; a single per-block query rotation against the
current store would not. **Block-relative storage removes the dependency entirely** — each
segment's keys are self-contained, so one rotation per `(segment, worker)` is always exact,
regardless of interleaving. This is exactly why the reference stores block-relative.

---

## 3. Constraints verified in this environment

- **FlashInfer 0.6.9** here exposes `merge_state` / `merge_states` / `merge_state_in_place`
  and `return_lse` / `out` / `lse` on the decode wrapper `.run()` → a **head-dim-agnostic
  correctness path with no new CUDA is feasible.** This is the primary path.
- The reference CUDA kernel is compiled only for **head_dim=128 and GQA ∈ {2,4,5,8,16}**.
  Qwen3-8B/32B fit (hd128, GQA 4/8), but the **oracle/test model Qwen2.5-0.5B is hd64, GQA7 →
  unsupported.** So the CUDA kernel can only ever be an opt-in fast path; the general path must
  not depend on it.
- mini-sglang KV is paged + **token-major** `[slot, n_kv_heads, head_dim]`; the reference kernel
  wants contiguous **head-major** `[Hkv, L, D]` fragments — a layout gap to bridge in the perf
  phase (adapt kernel addressing for zero-copy, or gather per block).
- `_cos_sin_cache` convention (first-half cos, second-half sin) and rotate-half match the
  reference exactly (the oracle kernel-parity test already proves this), so the query rotation
  is a few lines of PyTorch.

---

## 4. Plan (phased)

### Phase A — block-relative storage (decode only; prefill already is)
`prefill_block` already writes positions `0…L−1` (the oracle's layer-0 KV-parity test confirms
the stored K matches the reference). Only decode writes change:
- `_record_writes`: record `stored_pos = write_block.num_tokens` (block-relative), not
  `req.cached_len`.
- The new token's K must be RoPE'd block-relative → set decode `positions[w] =
  write_to[w].num_tokens` (before grow).

### Phase B — shared-cache flag in `AttentionLayer.forward`
Mirror the reference attention module: when `ctx.batch` carries shared-cache metadata, rotate
**only K** by `batch.positions` (block-relative) and pass the **raw q** to the shared-cache
attention op (which does the per-segment query rotation). ~3 guarded lines; the normal path is
untouched.

### Phase C — shared-cache attention op (`shared_cache/attention.py`)
Metadata is precomputed **once per step** (layer-independent: segments, per-`(worker, segment)`
page indices, `loc = P_w − O_s`, fragment lengths) and attached to the batch; each layer reads
its own KV via those page indices. Two implementations:
- **`_sdpa_torch`** — pure-PyTorch port of the reference `async_reasoning_sdpa_pt`: gather each
  segment's K/V, rotate q to `loc`, masked-softmax over the concatenation. CPU/eager ground
  truth for tests.
- **`_flashinfer`** (primary GPU path, any head_dim/GQA): build a flat `(worker × segment)`
  batch, rotate queries via `_cos_sin_cache`, run the paged wrapper with `return_lse=True`, then
  `merge_states` per worker (pad short workers with `lse = −inf`). The new-token page goes in
  the write block's segment so the query attends to itself.

### Phase D — delete the copy
Remove `_apply_corrections` / `_cleanup_corrections` / `correct_kv_pages` and temp-page
allocation from the decode hot path. **Keep `apply_rope_correction`** (the oracle's
kernel-parity test imports it directly). `_fill_page_tables` becomes per-segment page lists +
`loc` instead of one concatenated table.

### Phase E — optional fast kernel (head_dim=128 only)
Either build the reference `inference_lib` extension, or port `kernel_v21` adapted to read
mini-sglang's **paged token-major** KV through a per-fragment page-index table (zero-copy — the
real win). Gate on head_dim/GQA; fall back to `_flashinfer` otherwise.

---

## 5. Validation
- Keep the existing oracle suite green: `tests/core/test_shared_cache_async_reasoning_oracle.py`
  already exercises decode against the reference, including **block reorder at nonzero offset**.
- Add: (1) an **interleaved-growth** test (writer grows while thinker grows — the case that
  proves block-relative is required); (2) new-path-vs-current-impl logit parity within bf16 tol;
  (3) a microbench of per-step latency / HBM traffic, KV-copy vs query-rotate, swept over
  shared-token count and `n_layers`.

## 6. Out of scope (first cut)
- **CUDA-graph capture:** the shared-cache path already bypasses graphs and `F` (segment count)
  is dynamic; leave graph capture out initially (consistent with the deferred items in
  `docs/shared_cache_api_v2.md`).
- ~~Prefilling a block *in the context of other blocks*~~ — **DONE** (2026-06-09):
  `prefill_block(block, ids, context=[...])` runs the causal self-segment through a second
  FlashInfer prefill wrapper and the context segments with `causal=False`, merged per token
  with `merge_states`; exposed as `context` on `POST /v1/shared-cache/blocks`.  The demo
  prefills the `</think>` close block in context, which fixes the writer's answer style.

---

## 7. Open decisions
1. **Storage:** adopt block-relative (recommended — single exact rotation, handles interleaving,
   deletes the copy, matches the reference) vs. keep the current store.
2. **Phase-1 engine:** FlashInfer `merge_states` first (recommended — works for the hd64 test
   model and hd128 prod), CUDA port deferred to Phase E.
3. **Start point:** full A–D, or prototype `_sdpa_torch` first as an executable spec to lock the
   math before touching the engine.
