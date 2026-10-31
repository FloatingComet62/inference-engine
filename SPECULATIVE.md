# SPECULATIVE.md — Two-model speculative decoding with MobileLLM-125M as the draft

Plan for extending the engine's existing greedy prompt-lookup speculation (`--spec ngram`)
to a real two-model draft: **MobileLLM-125M drafts, TinyLlama-1.1B verifies**. Both models
share the Llama 32k vocabulary, so drafts map to target logits 1:1 with no projection.

Read this together with `PLAN.md` (Priority 5) and the current implementation in
`src/main.cpp` (`ngram_draft` + the greedy verify loop) and `src/model.cpp`
(`forward_batch(..., all_logits=true)`).

---

## 1. What exists today

- `Model::forward_all(tokens, T, pos)` runs T tokens in one batched forward and returns
  `[T][VOCAB]` logits; row `i` predicts the token after `tokens[i]` (built for prefill).
- `main.cpp` `--spec ngram / --draft-k K`: drafts by n-gram lookup, verifies
  `[next, d1..dk]` in one batch, accepts `d_{i+1}` while it equals row `i`'s argmax,
  takes row `m`'s argmax for free on mismatch. Greedy only (`TEMP=0`); `--repeat-penalty`
  disables spec (commit 0c69691). KV rollback is free: rejected positions are simply
  overwritten later (position-indexed cache).
- Verified correctness invariant (commit 130cd5c, experiment 20): spec output is
  byte-identical to plain greedy on the toy model and on real prompts.
- Measured anchors (laptop, i5-13420H, q4a + kv q8, EXPERIMENTS.md):
  target decode ≈ 60.5 tok/s, batched prefill ≈ 198 tok/s, ngram spec ≈ 2.0x decode on
  repetitive/summarisation prompts (83 % acceptance), neutral on normal prompts.

The ngram loop is already a complete verify/accept/emit/rollback harness. Two-model
spec swaps `ngram_draft` for a draft-model pass; everything else (acceptance, EOS,
stop strings, room clamps, stats counters `verifies/drafted/accepted`) transfers.

## 2. The draft model: MobileLLM-125M (verified facts)

From the official `facebookresearch/MobileLLM` repo, `configs/125M/config.json`
(mirrored unchanged by `vonjack/MobileLLM-125M-HF`):

| Property | Value | Notes |
|---|---|---|
| architecture | `LlamaForCausalLM` | plain Llama 2 — matches TinyLlama's forward pass |
| hidden / intermediate | 576 / 1536 | deep-and-thin; HIDDEN divisible by 64 |
| layers / heads / KV heads | 30 / 9 / 3 | GQA, `KV_MUL=3`, `HEAD_DIM=64` |
| vocab | 32000 | **same as TinyLlama** (`bos=1`, `eos=2`, `unk=0`) |
| max position | 2048 | same as target `MAX_SEQ` |
| rope theta / rms eps | 1e4 / 1e-5 | identical to target |
| dtype / weights | fp16, `pytorch_model.bin` 249,359,178 B | 124.6M params ≈ 124.6M × 2 B ⇒ embeddings shared input/output (`lm_head` absent or aliased) despite `tie_word_embeddings: false` |
| tokenizer | `LlamaTokenizer`, `tokenizer.model` 499,723 B | the Llama 2 SentencePiece model, same as TinyLlama's |
| license | **gated**, FAIR noncommercial research (manual approval) | see §7.1 |

Mismatch risk to close **before** any engine work: TinyLlama's `tokenizer.bin` in this
repo and MobileLLM's vocabulary must map identical strings to identical ids (same spm
model ⇒ expected, not guaranteed). Cheap check: `convert_tokenizer` both → compare
token→id maps and merge-score order (stdlib script, §4 step 1).

## 3. Why it should pay (roofline)

Decode is memory-bound: the target re-reads ~805 MB of q4a weights per token
(877 MB file minus embed). The draft at q4a is ~63 MB per token
(124.6M params × 0.5 B + scales). At the engine's ~50 GB/s effective read rate:

| Quantity | Value | Source |
|---|---|---|
| target decode step | ~16.7 ms | measured 60.5 tok/s |
| target batched verify (5 tokens, one weight pass) | ~25 ms | measured prefill speed, 5 rows |
| draft decode step (q4a) | ~1.1–1.3 ms | [ESTIMATE] 54 MB/50 GB/s; greedy only thanks to hard-coded sizes |
| draft proposal + catch-up per cycle (k=4) | ~5–7 ms | [ESTIMATE] k + α^k ≈ 4–5 draft decodes |
| one full spec cycle (k=4, 5 tokens, all accepted) | ~32 ms | = verify + draft + overhead ⇒ ~156 tok/s theoretical |

Speedup per cycle = `E[tokens] / (t_verify + (k+1)·t_draft + overhead)` against plain
decode's 4 steps for 4 tokens:

| acceptance α | E[tokens/cycle] | est. decode tok/s | vs 60.5 |
|---|---|---|---|
| 0.50 | 1.94 | 61 | ~1.0x |
| 0.60 | 2.31 | 72 | 1.2x |
| 0.70 | 2.77 | 87 | 1.4x |
| 0.80 | 3.36 | 105 | 1.7x |
| 0.90 | 4.10 | 128 | 2.1x |

Break-even is α ≈ 0.5 — weak drafts still pay, unlike ngram (which is near-zero only on
repetitive text). These are [ESTIMATE]s pinning the measured anchors; the first real
benchmark (§5) is what decides whether draft-k/α targets hold. Two caveats:

- Batched verification is compute-bound on this CPU (prefill ≈ 3.3x decode), so the
  verify batch must stay small (k ≤ 8); a faster matmul kernel raises every number above.
- The draft must be quantised (q4a). An fp32 draft reads ~500 MB/token ⇒ ~10 ms/step
  and roughly halves the modelled gains.

## 4. Implementation

### Step 1 — vocab gate (before engine work)

`tools/` stdlib script: load `models/tokenizer.json` (TinyLlama) and the draft
`tokenizer.json`, compare vocab id→token and merge-score ordering. Equal ⇒ reuse
`models/tokenizer.bin` for both models (no second tokenizer). Not equal ⇒ spec still
works but needs an id remap table at verification time — remeasure before proceeding.

### Step 2 — second compiled config

The repo convention is hard-coded sizes ("a second compiled config", PLAN.md). The
engine core (`model.cpp`, `kv_cache.cpp`, header-only kernels/sampler) references the
global `tl` constants directly, and the KV cache classes hard-code `N_LAYERS`, `KV_DIM`,
`MAX_SEQ` (e.g. `RadixKVCache::PAGE_BYTES`, `StreamingKVCache::ROW_BYTES`). So a second
model instance means recompiling the core with draft constants in a separate namespace:

- `src/config.hpp`: add `#elif defined(TL_DRAFT)` branch with the §2 table
  (576/1536/30/9/3/32000/2048) and, under `TL_TINY_TEST`, toy-draft sizes
  (e.g. DIM=48, HIDDEN=128, L=2, NH=3, NKV=1, V=300, S=128 — must stay ≦ VOCAB of the
  toy target so the toy draft shares its vocabulary; `tests/make_tiny_model.py` grows
  a matching toy draft checkpoint).
- `CMakeLists.txt`: compile `model.cpp` + `kv_cache.cpp` twice — once as today
  (`tl::Model`), once as `model_draft.o`/`kv_cache_draft.o` with `-DTL_DRAFT -Dtl=tld`
  (token-substitution renames `namespace tl` → `tld` in the draft TU only). Kernels are
  header-only and instantiate per TU for free. Main links both.
  Alternative if the macro grates: template `Model`/kernels on a `constexpr Cfg` struct —
  cleaner, bigger diff, same compile-time unrolling. Start with the dual-TU; it touches
  zero existing code paths.
- Draft weights: `models/mobilellm-f32.bin` from the same `convert_weights` (already
  config-driven — reads shapes from `config.json`, already falls back to
  `model.embed_tokens.weight` when `lm_head.weight` is absent, i.e. handles the tied
  case). Header check in the draft TU validates against `tld` constants. Draft q4a:
  reuse the existing `save_quantized` calibration flow (`make model WTYPE=q4a`,
  parameterised by model name).

### Step 3 — draft runtime (`src/main.cpp`)

- CLI: `--spec off|ngram|draft`, `--draft-model PATH` (default
  `models/mobilellm-q4a.bin`), existing `--draft-k` (clamped to `PREFILL_CHUNK - 1`).
- Construct a second `tld::Model` against its own cache instance. Draft KV is tiny
  (30 × 2048 × 192; ~47 MB fp32, halved with kv q8) — a `tld::BaselineKVCache` is
  enough; radix/streaming for the draft is future work (§6).
- Per cycle, replacing `ngram_draft`:

```mermaid
flowchart TD
    A["committed token `next` at pos"] --> B["draft.forward(next, pos)\nstores next, → d1 = argmax"]
    B --> C["for j = 2..k:\ndraft.forward(d[j-1], pos+j-1)\nstores d[j-1], → d[j]"]
    C --> D["target.forward_all([next, d1..dk], k+1, pos)\nrows 0..k-1 predict d1..dk, row k = next target token next2"]
    D --> E{accept m consecutive drafts\nwhile row[i] == d[i+1]}
    E -->|m < k| F["emit d1..dm + next2 = row[m] argmax\npos ← pos + m + 1"]
    E -->|m = k| G["emit d1..dk + next2 = row[k] argmax\npos ← pos + k + 1\nlockstep: draft.forward(dk, pos+k)\nstores dk (its prediction is discarded)"]
    F --> H["next cycle proposal starts at\ndraft.forward(next2, pos)"]
    G --> H
    H --> A
```

- Lockstep rule: the draft's KV cache must cover the **same committed prefix** as the
  target's at every cycle. Proposal inputs `next, d1..dk-1` store positions
  `pos..pos+k-1`; the draft's *predictions* `d1..dk` are not stored. So after a partial
  accept (`m < k`) the draft KV is already contiguous up to the new `pos - 1`; after a
  full accept (`m = k`) `dk` is missing and must be stored once
  (`draft.forward(dk, pos+k)`, prediction discarded) before the next proposal runs.
  Rejected rows are overwritten next cycle, exactly like the target's existing
  rollback. Never verify across a streaming re-base (`g_cache_cap` guard already
  exists). Draft cost per cycle is therefore `k + α^k` decodes (α^k expected
  catch-ups). Catch-up was needed per cycle in the k=4 estimates in §3; the stake is
  ~1 ms.
- Stats: existing `verifies/drafted/accepted` counters extend to the draft path;
  report acceptance rate and tokens/verify.

### Step 4 — sampling (exactness at TEMP > 0)

- Greedy (`temp=0`): the existing argmax-equality rule is exact — keep it.
- Sampling: Leviathan et al. rejection sampling (arXiv:2302.01318): draft samples
  `d_j ~ q_j`, verify with target rows; accept with `min(1, p_j(d_j)/q_j(d_j))`; on
  rejection at position `m`, resample from `max(0, p_m(x) − q_m(x))` normalised.
  Needs per-step draft probabilities `q_j(d_j)` (capture during proposal) and the target
 's full row (forward_all already returns them). This is what makes the mode usable from
  the chat API (temp 0.3) — keep spec off when `--repeat-penalty` is set (unchanged).
- Sequence: implement greedy first (existing invariant: byte-identical output), then
  rejection sampling with a distribution-preservation check on the toy model.

### Step 5 — tuning and benchmarks

- Adaptive k from the running acceptance rate (PLAN.md item): raise k while α stays
  above a threshold, stop early when the expected-gain model (§3) turns negative.
- Extend `bench/compare.py` rows: acceptance rate, tokens/verify, decode tok/s —
  on normal prompts **and** the repetitive/summarisation suite where ngram wins.
  Log to `docs/EXPERIMENTS.md` per repo convention.
- `make model-draft WTYPE=q4a`, `make bench-draft`, smoke coverage:
  `tests/make_tiny_model.py` emits a toy draft so `make smoke --spec draft` asserts
  output == plain greedy on the toy pair.

## 5. Acceptance criteria (done when)

1. Greedy: `--spec draft` output **byte-identical** to `--spec off` (toy smoke + real
   model, fixed prompt).
2. Sampling: rejection-sampling path passes a distribution-preservation check on the
   toy model (draft distribution known), and real-text outputs are statistically
   consistent with plain sampling (spot checks).
3. Bench: acceptance rate, tokens/verify and decode speedup reported for k ∈ {2,4,8}
   on normal and repetitive prompts; spec ≥ ~1.2x decode on at least one normal-prompt
   suite, or a documented reason why not (e.g. α < 0.5).
4. `docs/EXPERIMENTS.md` rows added; `PLAN.md` Priority 5 item 2 marked done.

## 6. Future work (only once 1–3 land)

- Draft KV via radix prefix reuse across chat turns (draft prefill is cheap, but free is
  free); hybrid draft (draft model, ngram as fallback when the model stalls).
- Self-speculative layer-skip draft (PLAN.md item 3) as a zero-download comparison point.
- Larger draft (MobileLLM-350M, +? vocab check) only if 125M acceptance is weak.

## 7. Risks and open questions

| Risk | Mitigation |
|---|---|
| 7.1 **Gated download** — official repo needs HF login + FAIR license acceptance; checkpoint is a torch pickle (`pytorch_model.bin`), and the repo rule is stdlib-only tooling | Use the ungated `vonjack/MobileLLM-125M-HF` mirror (same weights, `model.safetensors` + `tokenizer.json`, works with `convert.py` as-is). Document the license; non-commercial research only |
| 7.2 **Vocab mismatch** | Closed by the §4-step-1 gate before any engine work — the single highest-risk unknown |
| 7.3 Low acceptance on real prompts (α < 0.5) | Auto-disable spec at runtime below an α floor; revert to ngram where it wins (repetitive text). Measure first |
| 7.4 Verify batch compute-bound ⇒ k only modestly helpful | Keep k ≤ 8, adaptive-k (§4 step 5); gains grow with matmul speedups |
| 7.5 Tied embeddings (`lm_head` absent/aliased) | `convert.py` already falls back to `model.embed_tokens.weight`; verify at conversion, keep the fallback |
| 7.6 Draft KV lockstep bug (hole at `dk`) | The catch-up forward is mandatory (§4 step 3); toy-smoke mismatch tests catch it — the existing byte-identical invariant is the harness |
| 7.7 fp32 draft kills the roofline | Draft is converted to q4a before benchmarking; fp32 only for calibration |

## 8. Work order

1. Vocab gate script + mirror download + `mobilellm-f32.bin` conversion (`scripts/download.sh`/`Makefile`).
2. `TL_DRAFT` config branch + dual-TU build; draft standalone sanity (`tokenize`-driven single-token generations).
3. Greedy draft loop in `main.cpp`; toy draft in `tests/`; byte-identical smoke.
4. Rejection sampling + distribution check.
5. Adaptive k, benchmark rows, EXPERIMENTS.md, PLAN.md update.
