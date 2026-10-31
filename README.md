# Irregs

A from-scratch CPU inference engine for **TinyLlama-1.1B-Chat-v1.0**, written in C++ with no ML libraries.
Every model size is hard-coded in `src/config.hpp`, so the code is specific to this one model on purpose.

## Requirements

- Linux, macOS, or Windows via **WSL2** (the engine uses `mmap`)
- `cmake` 3.16+, a C++17 compiler (g++ or clang++), `python3`, `curl`
- about 7 GB of disk (2.2 GB download + 4.4 GB fp32 weights) and 8 GB of RAM
- optional: [Ollama](https://ollama.com) for the comparison benchmark
- macOS only: `brew install libomp` for multithreading

## Quick start

```bash
make setup      # checks tools, builds, downloads and converts the model (once)
make run        # generates an answer
make chat       # interactive chat
```

To compare against Ollama, start Ollama in a second terminal and leave it running:

```bash
make ollama-serve   # = OLLAMA_LLM_LIBRARY=cpu_avx2 ollama serve  (GPU hidden)
make ollama-pull    # first time only: ollama pull tinyllama:1.1b
```

Then, back in the first terminal:

```bash
make compare
```

## Commands

| Command | What it does |
| --- | --- |
| `make setup` | Check tools, build, download and convert the model |
| `make run PROMPT="..." N=200 TEMP=0.7` | Generate one answer, streamed |
| `make chat` | Multi-turn chat (type `/exit` to quit) |
| `make bench` | Benchmark this engine only |
| `make compare` | Benchmark this engine and Ollama side by side |
| `make bench-ollama` | Benchmark Ollama only |
| `make bandwidth` | Measure memory read bandwidth (the decode speed ceiling) |
| `make smoke` | 30-second pipeline test with a toy random model, no download |
| `make debug` | Debug build with address and UB sanitizers, then `make run BUILD=build-debug` |
| `make ollama-serve` / `ollama-pull` / `ollama-run` | Ollama helpers |
| `make clean` / `make distclean` | Remove builds / also remove downloaded models |

Every variable can be overridden: `make compare THREADS=4 N=256 RUNS=5`.
`THREADS` defaults to the number of physical cores.

You can also call the binary directly: `build/tinyllama run|chat|bench|tokenize --help`.
`tokenize` prints the chat-template token IDs, which is useful for checking prompt
token counts against Ollama's.

## How the benchmark stays fair

`make compare` gives both engines the same prompts, the same TinyLlama (Zephyr) chat
template (Ollama receives it with `raw: true`), greedy decoding, the same max new
tokens and the same thread count. It does one warm-up run, then `RUNS` measured runs,
each with a different prompt so Ollama's prompt cache cannot skip prefill. It reports
medians, decode and prefill speed, model size, and the percentage of the memory
bandwidth limit each engine achieves.

Results are saved to `bench/results/<timestamp>.json` and appended to
`bench/results/history.csv` with the git commit, so you can chart progress over time.

Expect the starting point to be far slower than Ollama: this baseline streams 4.4 GB
of fp32 weights per token, while Ollama's tinyllama is about 0.6 GB of 4-bit weights.
Closing that gap is the project.

## Layout

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

## Correctness

The baseline forward pass has been checked token-for-token against an independent
reference implementation on the toy model. After every optimisation, re-run
`make smoke` and check that `make run TEMP=0` still produces the same text as before
the change (fp32 kernels) or still reads sensibly with a small perplexity change
(quantised kernels).

## Suggested next steps

1. `make bench` and `make bandwidth` to record the fp32 baseline.
2. Quantise weights to 4-bit blocks in `scripts/convert.py`, with a new `matmul_q4` in `kernels.hpp`.
3. Quantise activations to int8 and use AVX2 integer dot products.
4. Fuse Q/K/V and gate/up matrices, fold RMSNorm weights into the next matrix.
5. Re-run `make compare` after each step and watch `history.csv`.
