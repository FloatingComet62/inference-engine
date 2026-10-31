# Layout

```
src/config.hpp      TinyLlama sizes as compile-time constants
src/kernels.hpp     matmul, rmsnorm, softmax, silu  <- optimise here first
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