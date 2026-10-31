# PROJECT.md — Irregs (TinyLlama CPU engine)

Reference for maintenance and optimisation tasks. Facts below are from reading
the repo at commit `7369e53` (2026-10-31).

## What this is

A from-scratch C++17 CPU inference engine for **exactly one model**:
TinyLlama-1.1B-Chat-v1.0 (Llama 2 architecture, Zephyr chat template). No ML
libraries anywhere — not even in Python tooling (stdlib only). The performance
target: fp32 baseline streams ~4.4 GB of weights per decoded token; Ollama's
tinyllama (Q4 0.6 GB) is the benchmark opponent. Closing that gap is the point
of the project; optimising `matmul` is ~95% of runtime.

State: baseline works end-to-end (real weights downloaded in `models/`,
`build/tinyllama` built). Per `PLAN.md`, no optimisation has landed yet.

## Layout

| Path | Role |
| --- | --- |
| `src/config.hpp` | All model sizes as compile-time constants. `TL_TINY_TEST` switches to toy sizes. |
| `src/kernels.hpp` | `matmul`, `rmsnorm`, `softmax`, `silu`. **Optimise here first.** |
| `src/kv_cache.hpp/.cpp` | KV cache abstraction. `KVCache` interface; `PagedKVCache` (engine default) + `BaselineKVCache` (A/B). Further experiments plug in here. |
| `src/model.cpp` | mmap weight loading, forward pass (uses a `KVCache&`), RoPE cache. |
| `src/tokenizer.hpp/.cpp` | SentencePiece-style BPE + Zephyr template. |
| `src/sampler.hpp` | greedy / temperature / top-p. |
| `src/main.cpp` | CLI: `run|chat|bench|tokenize`, `generate()` loop, timings. |
| `scripts/convert.py` | safetensors + tokenizer.json → engine binary formats. No numpy/torch. |
| `scripts/download.sh` | Resumable HF download (2.2 GB). |
| `bench/compare.py` | Engine vs Ollama benchmark; writes `bench/results/`. |
| `tools/bandwidth.cpp` | Memory read bandwidth probe (decode speed ceiling). |
| `tests/make_tiny_model.py` | Random toy model in HF layout for `make smoke`. |
| `Makefile` | All user-facing commands (see below). |

## Model config (real model, `src/config.hpp`)

```
DIM=2048  HIDDEN=5632  N_LAYERS=22  N_HEADS=32  N_KV_HEADS=4
VOCAB=32000  MAX_SEQ=2048  HEAD_DIM=64  KV_DIM=256  KV_MUL=8  ROPE_THETA=1e4
BOS=1  EOS=2
```
Toy config (used by smoke): DIM=64, HIDDEN=172, L=2, NH=4, NKV=2, V=300, S=128.

## Data formats (change these on both sides!)

### `models/tinyllama-f32.bin` ("TLM1")
- 4-byte magic `TLM1`, then 8×int32 LE: `{version=1, DIM, HIDDEN, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB, MAX_SEQ}`; total 256-byte header.
- Tensors in fixed order, **each aligned to 64 bytes**: embed `[V][D]`, then per
  layer `i`: `input_layernorm[D]`, `q[D][D]`, `k[KV][D]`, `v[KV][D]`, `o[D][D]`,
  `post_attention_layernorm[D]`, `gate[H][D]`, `up[H][D]`, `down[D][H]`; then
  `model.norm[D]`, `lm_head[V][D]`.
- All fp32, row-major (as Hugging Face stores). When weights are bf16 in the
  safetensors, `convert.py` widens to fp32.
- Header is validated against compiled `config.hpp` at load; mismatch → error
  "re-run convert" (`make distclean && make setup`).

### `models/tokenizer.bin` ("TLT1")
- Magic `TLT1`, uint32 count `n` (=VOCAB), then per token: float32 merge score,
  uint32 byte length, UTF-8 bytes.
- Score: `-merge_rank` (earlier merge = higher priority); absent pieces get
  `-1e9`. Merging in `encode()` picks the adjacent pair with the **highest**
  score; low score = never merged.
- Byte-fallback tokens `<0x00>`… are ids 3..258 (`BYTE_TOKEN_OFFSET=3`).

## Forward pass internals (`model.cpp`)

- One `forward(token, pos)` call per token; weights come from mmap (`MAP_PRIVATE`,
  `MADV_WILLNEED`); all preallocated buffers in the `Model` object:
  `x,xb,xb2,q(2048)`, `hb,hb2(5632)`, `att_[N_HEADS][MAX_SEQ]`, `logits_[V]`,
  `k_tmp_/v_tmp_(KV_DIM)` scratch rows, RoPE cos/sin tables `[MAX_SEQ][HEAD_DIM/2]`
  precomputed in the constructor.
- KV rows live in a `KVCache&` (one sequence per Model, created in the ctor,
  freed in the dtor). Per layer, forward computes K/V into scratch, calls
  `store(seq, l, pos, token, …)`, then fetches per-position row pointers
  (`key_row`/`value_row` — one virtual call per (layer, pos), done before the
  parallel head loop) and runs attention exactly as the old flat arrays did.
  Swap algorithms at the single construction site in `src/main.cpp`.
- Per layer: rmsnorm → q/k/v matmuls → RoPE (HF `rotate_half`: dim `i` pairs
  with `i+HEAD_DIM/2`) → per-head attention over `t <= pos` (Grouped-Query:
  head `h` uses kv head `h/KV_MUL`) → `wo` → residual → SwiGLU FFN
  (`silu(gate∘up)`, `down`) → residual. Final: rmsnorm → `lm_head` logits.
- Threading is `#pragma omp parallel for` (static schedule) on matmul rows and
  attention heads; OpenMP optional at build time.
- `bytes_per_token()` = file size − embed table + one embed row (embed only
  contributes one row per token).

## Build & test workflow

```bash
make setup      # deps check → cmake build → download + convert (once)
make run PROMPT="..." N=200 TEMP=0.7
make chat       # multi-turn interactive
make smoke      # ~30 s: toy model end-to-end; primary regression test
make debug      # builds build-debug with -fsanitize=address,undefined; then make run BUILD=build-debug
make bench      # engine-only benchmark
make compare    # engine vs Ollama (needs ollama running: make ollama-serve)
make bandwidth  # memory read bandwidth
make run TEMP=0 # deterministic greedy; must reproduce previous output after fp32 changes
make clean / make distclean   # rm builds / also rm models
```

CMake options: `TL_NATIVE` (-march=native, default ON), `TL_OPENMP`, `TL_SANITIZE`,
`TL_TINY_TEST`. Builds: `build/` (Release), `build-debug/`, `build-tiny/` (smoke).
`compile_commands.json` is exported (clangd). Make vars: `THREADS` (physical
cores via `scripts/physical_cores.sh`), `N`, `RUNS`, `TEMP`, `PROMPT`, `BUILD`.

## Verification protocol (critical for optimisation tasks)

1. After any change, run `make smoke` (pipeline still works).
2. fp32 kernel change → `make run TEMP=0` must produce **identical text** to
   before the change (baseline checked token-for-token against an independent
   reference on the toy model).
3. Quantised kernels → same prompt may change; require sensible text + only a
   small perplexity change (per README).
4. Benchmark: `make compare` then check history in `bench/results/`.
   Greedy + same Zephyr template (Ollama gets `raw:true`), same prompts/threads/
   max tokens, warm-up run first, `RUNS` measured runs with **different prompts**
   so Ollama's prompt cache can't skip prefill. Medians reported.
   `bench/results/<timestamp>.json` + appended `history.csv` (with git commit).
   `--ignore-eos` is used by smoke so generation always runs N tokens.

## Conventions & pitfalls

- Changing a tensor/dtype/order anywhere means touching **both**
  `scripts/convert.py` **and** `src/model.cpp` (and the header check), plus the
  toy model in `tests/make_tiny_model.py` when it affects sizes.
- Keep the stdlib-only rule in all new Python tooling.
- Keep sizes as compile-time constants; the compiler unrolls/vectorises only if
  nothing makes them runtime.
- `bench` mode forces `temp=0` in `main.cpp` (greedy, deterministic).
- First streamed token has its leading space stripped on output, not on decode.
- `peak_rss_mb()`: macOS `ru_maxrss` is bytes, Linux is kilobytes (handled).
- OpenMP not found → builds single-threaded with a warning (macOS needs
  `brew install libomp`).
- Do not commit `models/`, `build*`, `bench/results/` (gitignored).

## KV cache experiments (in progress)

`src/kv_cache.hpp` defines the interface (per-token rows, sequence handles,
content-keyed prefix API, capacity + stats). Two backends live in
`src/kv_cache.cpp`:

- **`PagedKVCache`** — engine default. vLLM-style (PagedAttention, SOSP'23):
  fixed pool of physical blocks (`BLOCK_SIZE=16` tokens, all layers per block),
  per-sequence logical block tables, lazy allocation, blocks released to the
  pool on `end_sequence`. `capacity_tokens() = BLOCK_SIZE * pool_blocks`
  (default `MAX_SEQ` = 2048). Pool exhaustion makes `store()` throw — no
  eviction yet, and none is safe until the engine can re-prefill gaps.
  `key_row()`/`value_row()` return `nullptr` for positions ≥ `stored_tokens`.
- **`BaselineKVCache`** — flat per-sequence `[N_LAYERS][MAX_SEQ][KV_DIM]`
  arrays, kept for A/B; swap back at the single construction site in
  `src/main.cpp`.

Remaining experiments: `FreeOnFinish` (drop rows at end), `PrefixCaching`
(content-keyed re-use, skip prefill), `BoundedGrowth` (eviction under a cap).
Note: real cross-request prefix caching needs the engine to consult
`longest_cached_prefix` in `generate()` before prefilling — not wired yet.
Memory benefit of paging: peak allocation tracks stored tokens (a full real
context caps at `MAX_SEQ/BLOCK_SIZE = 128` blocks × 5.8 MB ≈ 738 MB, same
ceiling as baseline but only reached when actually used).

## Planned next steps (from PLAN.md, in priority order)

1. Record fp32 baseline (`make bench`, `make bandwidth`).
2. 4-bit block quantisation of weights in `scripts/convert.py` + new
   `matmul_q4` in `kernels.hpp` (new weight format → update loader + header).
3. int8 quantised activations + AVX2 integer dot products.
4. Fuse Q/K/V and gate/up matmuls; fold RMSNorm weights into the next matrix.
5. `make compare` after each step; watch `history.csv` for decode tok/s and
   % of bandwidth ceiling.