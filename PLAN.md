# Layout

```
src/config.hpp      TinyLlama sizes as compile-time constants
src/kernels.hpp     matmul, rmsnorm, softmax, silu  <- optimise here first
src/kv_cache.hpp    KV cache interface (RadixKVCache default, radix-tree

                    prefix reuse + LRU eviction; StreamingKVCache = bounded
                    rolling window + attention sinks; paged vLLM-style +
                    baseline kept for A/B; free-on-finish remaining)
src/model.cpp       weight loading (mmap) and the forward pass
src/tokenizer.cpp   SentencePiece-style BPE + chat template
src/sampler.hpp     greedy, temperature, top-p
src/main.cpp        CLI: run, chat, bench, tokenize
scripts/convert.py  safetensors -> engine format (Python standard library only)
bench/compare.py    benchmark vs Ollama
tools/bandwidth.cpp memory bandwidth probe
tests/              toy model for `make smoke`
```

# Suggested next steps

1. `make bench` and `make bandwidth` to record the fp32 baseline.
2. Quantise weights to 4-bit blocks in `scripts/convert.py`, with a new `matmul_q4` in `kernels.hpp`.
3. Quantise activations to int8 and use AVX2 integer dot products.
4. Fuse Q/K/V and gate/up matrices, fold RMSNorm weights into the next matrix.
5. Re-run `make compare` after each step and watch `history.csv`.
# Priority 5: Speculative decoding

Decode is memory-bound: every token re-reads all ~615 MB of Q4 weights. A cheap draft proposes
k tokens, and the real model verifies all k in one batched forward pass (`forward_batch`, built
for prefill), reading the weights once. Keep the longest accepted prefix plus the model's own
next token. The KV cache is position-indexed, so rollback is just resetting `pos`.

Catch on CPUs: verification is cheap only while it stays memory-bound. With the batched kernel
compute-bound (prefill ~64 tok/s vs decode ~34 tok/s on the i5-13420H), verifying 4 tokens costs
about 2 decode steps, so it pays only if about 2+ of 4 are accepted. Faster prefill kernels directly
raise the gain.

Draft sources, in build order:
1. Prompt lookup (n-gram): continue n-grams already in the context. No model, near-zero cost.
2. Small draft model sharing the Llama 32k vocabulary (e.g. JackFram/llama-68m or llama-160m;
   check the vocabulary matches first), run by this engine with a second compiled config.
3. Self-speculative layer skip: first N of 22 layers + lm_head as the draft (research option).

Novel parts: adaptive draft length from the running acceptance rate, and k chosen per CPU by
the auto-tuner (it depends on this machine's prefill/decode ratio).

Steps:
1. `forward_batch` returns logits for every position, not only the last.
2. `--spec off|ngram|draft` and `--draft-k K` in src/main.cpp; greedy acceptance first.
3. Rejection sampling for temperature > 0 so the output distribution is unchanged.
4. bench/compare.py: acceptance rate, tokens per verification, decode speedup.

Done when: greedy output with speculation is identical to plain greedy (strict correctness
test), and decode tok/s improves on chat and summarisation prompts, reported with acceptance rates.
