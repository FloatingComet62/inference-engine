# TinyLlama CPU Engine: Technical Deep-Dive and Q&A

Nov 1, 2026 · for presenting and answering questions on how it works. Companion: [PRESENTATION_BRIEF.md](PRESENTATION_BRIEF.md).

## Overview and architecture

The engine is a single C++17 binary (`build/tinyllama`) specialised for TinyLlama-1.1B, plus standard-library Python for conversion, benchmarks and serving; there are no ML libraries anywhere.

**Data flow for one request**

1. **Tokenize:** the chat template (system / user / assistant markers and the end-of-sequence token) is applied and the text is split into BPE tokens (`src/tokenizer.cpp`).
2. **Prefill:** the prompt runs through the 22 transformer layers in batches of 32 tokens, writing each token's keys and values into the KV cache (`Model::prefill` -> `forward_batch`).
3. **Decode loop:** one token per step: sample from the logits, print it, feed it back as the next input (`generate` in `src/main.cpp`).
4. **Stop** at end-of-sequence, a stop string, or the token limit; print speed stats to stderr.

**Repository map** (repo: FloatingComet62/inference-engine, branch `master`)

| Path | What it does |
| --- | --- |
| `src/config.hpp` | TinyLlama sizes as compile-time constants |
| `src/model.cpp`, `model.hpp` | weight loading (mmap), batched forward pass, quantize/calibrate |
| `src/kernels_q4.hpp` | 4-bit and 8-bit weight formats, AVX2/AVX-VNNI dot products, activation quantisation |
| `src/kernels_attn.hpp` | vectorised attention (fp32 and int8 KV) |
| `src/kernels.hpp` | fp32 dot, RMSNorm, softmax, SiLU |
| `src/threadpool.hpp` | persistent pinned thread pool, dynamic chunking, P-core detection |
| `src/kv_cache.cpp`, `.hpp` | teammate's KV cache backends (baseline, paged, radix, streaming) |
| `src/sampler.hpp` | greedy, temperature, top-p, repetition penalty |
| `src/tokenizer.cpp` | SentencePiece-style BPE + chat template |
| `src/main.cpp` | CLI: run, chat, bench, ppl, quantize, tokenize |
| `scripts/convert.py` | safetensors -> our fp32 format (stdlib only) |
| `bench/compare.py`, `compare-longer.py`, `eval.py` | benchmarks vs Ollama, WikiText-2 quality |
| `tools/openai_server.py`, `dashboard.html` | OpenAI-compatible API, results dashboard |
| `docs/EXPERIMENTS.md` | experiment log |

## TinyLlama and why speed is limited by memory

TinyLlama-1.1B-Chat is a standard Llama-2-style decoder; generating a token reads every weight once, so decode speed is capped by memory bandwidth divided by bytes per token.

| Property | Value |
| --- | --- |
| Parameters | 1.1 billion |
| Layers | 22 |
| Hidden size / FFN size | 2048 / 5632 |
| Attention | 32 query heads, 4 key/value heads (grouped-query attention), head size 64 |
| Position encoding | RoPE (rotary), theta 10,000 |
| Normalisation / activation | RMSNorm (eps 1e-5) / SwiGLU (SiLU gate) |
| Vocabulary / context | 32,000 tokens / 2,048 tokens |

**One layer:** RMSNorm -> Q, K, V projections -> RoPE on Q and K -> attention over all previous tokens (each group of 8 query heads shares one K/V head) -> output projection Wo -> add to residual -> RMSNorm -> gate and up projections -> SwiGLU -> down projection -> add to residual. After 22 layers: final RMSNorm and the output head (2048 -> 32,000 logits).

**Decode is memory-bound.** One token does about 2.2 GFLOP but must stream all weights from RAM: 4.4 GB in fp32, 615 MB in our 4-bit format. Speed is roughly bandwidth / bytes per token: the engine reads at ~37 GB/s in performance mode, and 37 / 0.615 = ~60 tok/s, which is what we measure (23 GB/s and ~38 tok/s in power-saving mode). Fewer bytes = faster, which is why quantisation is the first lever.

**Prefill is compute-bound.** A batch of 32 prompt tokens shares each weight read, so memory traffic drops 32x and arithmetic becomes the limit. That is why kernel tricks (tiling, VNNI) help prefill much more than decode.

**Grouped-query attention** makes the KV cache 8x smaller than full multi-head attention: per token it stores 22 layers x 256 keys + 256 values = 11,264 numbers (45 KB in fp32).

## Weights and quantisation

Weights are stored as 4-bit integers in blocks of 32 with one 16-bit scale each (4.5 bits per weight), and multiplied by 8-bit activations with integer SIMD instructions.

**Pipeline:** `scripts/convert.py` turns Hugging Face safetensors (bf16) into our fp32 file (`TLM1` header + 64-byte-aligned tensors, stdlib Python). `tinyllama quantize` turns that into a quantised file in about 6 seconds (100 s for `q4a`). The engine `mmap`s the file, so loading is near-instant and the OS shares pages.

**Formats**

| Format | Layout | Bits per weight | Used for |
| --- | --- | --- | --- |
| Q4 (Q4_32) | fp16 scale + 16 bytes; weight i in the low nibble of byte i, weight i+16 in the high nibble; stored as q+8 (0..15) | 4.5 | all 154 layer matrices |
| Q8 (Q8_32) | fp16 scale + 32 int8 | 8.5 | output head (sensitive) |
| fp32 | plain floats | 32 | embedding table (only one row read per token), norms |

The same scheme as llama.cpp's Q4_0. Scale = max abs(w) / 7, value = round(w / scale) clamped to -8..7.

**Activation quantisation.** Before each matrix multiply the input vector is quantised to int8 in blocks of 32 (scale = max abs(x) / 127), vectorised with AVX2. Each 32-element block dot product is then pure integer arithmetic, multiplied once by weight scale x activation scale.

**The kernel (AVX2 / AVX-VNNI).** Per block: unpack 32 nibbles with a shift and a mask; multiply-add unsigned bytes by signed bytes (`_mm256_dpbusd_avx_epi32` with VNNI, `maddubs` + `madd` without); convert to float; fused multiply-add with the combined scale.

**Unsigned-nibble trick.** The instruction needs one unsigned operand. Instead of re-signing weights every time, we use the stored 0..15 nibbles directly and subtract 8 x (sum of the activation block), precomputed once during activation quantisation. Mathematically exact, saves instructions, and keeps results bit-identical.

**Adaptive `q4a` (Priority 1).** Run the fp32 model over ~500 tokens of text and record the mean squared value of every input channel of every matrix (how much each weight column matters). For every 32-weight block, try 16 scales between 0.80x and 1.10x of max abs(w)/7 and keep the one with the lowest error weighted by those activation statistics (the idea behind AWQ, applied offline). Same file format, same kernels, zero runtime cost. Result: perplexity 9.409 -> 9.104 (fp32 = 9.058); calibrating on unrelated text gives 9.135, so it is not overfitting to the test text.

## Forward pass and kernels

One function, `Model::forward_batch(tokens, T, pos)`, handles both prefill (T up to 32) and decode (T = 1), so the two paths share arithmetic and give bit-identical results.

**Batched prefill.** The prompt is processed 32 tokens at a time. Each matrix multiply becomes matrix x matrix: a weight row is loaded once and used for all 32 tokens while it sits in the L1 cache. Measured: prefill went from 23 tok/s (one token at a time) to 56, then higher with the kernel work below.

**Register tiling.** The prefill kernel (`dot_tile<2,4>`) computes 2 weight rows x 4 tokens at once with 8 independent accumulators in AVX registers. Each weight block is unpacked once and each activation block loaded once per tile, and independent accumulators hide the 4-cycle latency of fused multiply-add. 2x4, 3x4 and 4x4 measured the same end to end; 2x4 was best in a single-core microbenchmark (28.8 vs 15.8 GMAC/s for 1x1).

**Fusion (Priority 3).** Before: 9 parallel steps per layer, each ending in a barrier, 199 per token. After: 5 per layer, 111 per token.

- Q, K, V: one shared activation quantisation and one parallel loop over all 2,560 output rows (row index -> wq, wk or wv; no weight copying).
- Gate and up: one loop where each thread computes matching gate and up rows and writes silu(g) x u directly; the separate SwiGLU pass is gone.
- Wo and down: each thread adds its rows into the residual stream directly (`x += W.h`) instead of a separate serial loop.

**Vectorised attention** (`src/kernels_attn.hpp`). Query pre-scaled and held in 8 AVX registers; each key row is 8 FMAs; softmax uses a degree-6 polynomial exp (max relative error 1.2e-7); the 64-value output stays in 8 registers across all positions. +17-27% prefill and +18% decode at long context.

**Exactness guarantee.** `make smoke` checks on a toy model that batched prefill + thread pool give byte-identical output to sequential prefill + OpenMP, and that speculative decoding equals plain greedy. Changes that alter floating-point order (vectorised attention) are checked with perplexity instead (9.4262 -> 9.4093, within noise).

## Threading on hybrid CPUs

The biggest single speed lever after quantisation was where threads run: pinning to performance cores doubled decode, and too many threads costs 25-40% on both engines.

**Persistent thread pool** (`src/threadpool.hpp`). Workers are created once. Each parallel step bumps a generation counter; workers spin briefly (with `pause`), then yield, then sleep on a condition variable so an idle chat does not burn CPU. The caller is thread 0 and waits for a completion counter. Counters sit on separate cache lines. We found and fixed a startup race (workers read the generation before the first dispatch; 23% of runs hung) and verified 0 hangs in 500 runs.

**Pinning.** Threads are pinned to the P-cores, one per physical core first, then their hyperthread siblings (CPU order 0 2 4 6 1 3 5 7 on the i5-13420H). P-cores are read from the kernel's `/sys/devices/cpu_core/cpus` list, not from max frequency, because favored-core turbo makes 1-2 cores report a higher maximum. Unpinned, Linux mixed in E-cores and every barrier waited for the slowest: 13.5 vs 26.6 tok/s.

**Dynamic chunking** (the ggml approach). A parallel loop is split into ~4 chunks per thread that threads take from a shared atomic counter, so faster cores simply do more chunks. At 12 threads this lifted prefill 120 -> 190 tok/s; at 8 pinned threads nothing changed.

**Default thread count** = the P-cores' hardware threads on hybrid Intel (8 on the laptop, 16 on the teammate's i9), otherwise physical cores; the same rule as llama.cpp. Override with `--threads` or `THREADS=`.

| Threads (laptop, performance mode) | Our decode tok/s | Ollama decode tok/s |
| --- | --- | --- |
| 4 | 56.7 | 52.7 |
| 8 (default) | 60.5 | 59.8 |
| 12 (all CPUs) | 44.8 | 48.1 |

**Energy finding:** fewer threads did not save energy here: 0.72 J/token at 8 threads vs 0.97 at 4 (finish fast, then idle).

## KV cache

The KV cache stores each past token's keys and values so they are computed once; it sits behind a teammate-built interface with four swappable backends, and the engine reads it through a compact working copy that can be int8.

**Backends** (`src/kv_cache.*`, choose with `--cache` or `CACHE=`):

| Backend | Idea | Based on |
| --- | --- | --- |
| baseline | one flat array per sequence, no eviction | the original engine |
| paged | fixed 16-token blocks + a block table, allocated on demand | vLLM PagedAttention |
| radix (default) | radix tree of token prefixes, shared pages, LRU eviction; a repeated prompt prefix is skipped entirely | SGLang RadixAttention |
| streaming | fixed window + first 4 "attention sink" tokens kept forever; chat can run past the 2,048 limit | StreamingLLM |

**Working copy.** Keys are stored unrotated (StreamingLLM re-bases positions when tokens are evicted), so the engine keeps a contiguous per-layer copy with keys rotated at their current slot. Only new slots are copied each step; everything is recopied only after a streaming re-base or when positions are rewound. The teammate's original version re-rotated every cached key on every token, which would have erased our long-context gains. Streaming output matched the teammate's own build byte for byte through 60 evictions.

**int8 KV (`--kv q8`, Priority 4).** The working copy stores keys and values as int8 with one scale per 64-value head row; q.k uses the same integer dot-product instruction as the weights. At ~1.7k tokens: prefill +13%, decode +13%, 65 MB less RAM, perplexity 8.26 -> 8.24 (no loss). Per-token KV size: 45 KB fp32 -> ~11.6 KB int8.

**Rollback** for speculative decoding is free: the cache is indexed by position, so rejected tokens are simply overwritten by the next write.

## Speculative decoding and sampling

Speculative decoding guesses several tokens cheaply and checks them all in one batched pass; with greedy decoding the output is provably identical to normal decoding.

**How `--spec ngram` works (Priority 5).**

1. **Draft:** find the most recent earlier place in the prompt + output where the last 3 tokens (then 2) appeared, and propose the up-to-4 tokens that followed it. No draft model needed ("prompt lookup").
2. **Verify:** run [current token, d1..d4] through `forward_all` in one batch, which reads the weights once and returns logits for every position.
3. **Accept** drafts while each matches the model's own greedy choice at that position, plus one more token from the first mismatch for free.
4. **Roll back** rejected positions by overwriting them later.

Result: 2.0x decode on "repeat / summarise this paragraph" (57 -> 116 tok/s, 83% of drafts accepted, ~3.3 tokens per verify); neutral on normal questions because matches are rare. 24/24 test outputs byte-identical. It only runs at temperature 0 without a repetition penalty, since verification compares raw greedy choices. On this CPU verification is not free (prefill ~200 tok/s vs decode ~60), so it pays only when about 2+ of 4 drafts are accepted; a faster prefill kernel would raise the gain.

**Sampler** (`src/sampler.hpp`):

| Setting | What it does | Default |
| --- | --- | --- |
| `-t` temperature | 0 = greedy (argmax); higher = more random | 0.7 CLI, 0 benchmarks, 0.3 chat API |
| `--top-p` | sample only from the smallest set of tokens covering p of the probability | 0.9 |
| `--repeat-penalty` | divide positive logits (multiply negative) of the last 64 sampled tokens, as llama.cpp/Ollama do | 1 (off) CLI, 1.1 chat API |
| `--stop` | end the reply when the text appears | chat API: the four chat-template markers (user, assistant, system, end of sequence) |

The chat API defaults match Ollama's TinyLlama configuration; without them TinyLlama at temperature 0.8 rambled and wrote both sides of a dialogue, and greedy decoding looped.

## Benchmarking and evaluation

Every claim is measured against Ollama on the same machine under the same rules, and three benchmark pitfalls we discovered are now built into the scripts.

**Fairness rules** (`bench/compare.py`, `make compare`): same prompts and chat template (Ollama gets `raw: true`), greedy decoding, same max tokens and thread count, Ollama CPU-only (GPU hidden), one warm-up then N measured runs, medians reported, a unique tag per run so no prompt cache helps either engine. `compare-longer.py` (teammate) adds very long prompts.

**Metrics**

| Metric | How it is measured |
| --- | --- |
| Decode tok/s | generated tokens / decode time |
| Prefill tok/s, time to first token | prompt tokens / prefill time; prompt-in to first-token-out (Ollama: plus its load time) |
| Peak RAM | `getrusage` (ours, includes mmap'd weights); `/proc/<pid>/status` VmHWM (Ollama runner) |
| CPU % | CPU time / wall time (100% = one busy core) |
| Energy (J/token) | Intel RAPL package counters before/after each run (`make rapl-perms` once per boot) |
| Read GB/s | decode tok/s x weight bytes per token |
| Quality | perplexity, KL divergence and top-1 agreement vs the fp32 model (`tinyllama ppl --ref-model`, `make eval` on WikiText-2) |

**Pitfalls we found**

- **Ollama's prompt cache** inflated its prefill (150 -> 114 tok/s real on the laptop; repeat runs on the i9 showed an impossible 18,000-27,000 tok/s, which would need ~60 TFLOP/s).
- **Power mode flips the result:** in power-saving mode we decoded 1.65x faster than Ollama, in performance mode it is a tie. Record the mode with every result.
- **The bandwidth probe under-reads:** a pure read loop measured 16-19 GB/s while our engine sustains 23+, so a "% of bandwidth limit" column was replaced by effective read GB/s.
- **Running the script directly** used fp32 and 12 threads; defaults now match the Makefile (Q4, P-core threads).
- **Parallel agents sharing the CPU** produced noisy numbers; final numbers were re-measured on an idle machine, alternating baseline and candidate runs.

## Serving: API, Open WebUI and dashboard

`make up` starts the whole demo (Ollama, our OpenAI-compatible API, Open WebUI); `make down` stops it.

**OpenAI-compatible API** (`tools/openai_server.py`, `make serve`, port 8000, stdlib Python). `GET /v1/models` and `POST /v1/chat/completions` with streaming. Each request writes the conversation to a temp file and runs `tinyllama run --messages FILE`, which applies the Zephyr chat template to every turn; loading is near-instant thanks to mmap. Output streams back as server-sent events, holding back partial UTF-8 characters and partial stop strings. The final chunk carries Ollama-style speed stats (`response_token/s`, `prompt_token/s`, time to first token) so Open WebUI's (i) button shows comparable numbers for both engines. One request at a time, because the engine already uses every core.

**Open WebUI** (`make webui`, port 3000) runs via `uvx` with Python 3.11 and CPU-only PyTorch (~1 GB download instead of ~5 GB of CUDA libraries; the Docker pull stalled inside the Docker daemon). It is connected to Ollama natively and to our API as an OpenAI endpoint. Both TinyLlama models are set to temperature 0.3 and "legacy" function calling, because TinyLlama cannot call tools and Ollama otherwise rejects every chat.

**Dashboard** (`http://localhost:8000/dashboard`, linked from a banner in Open WebUI): latest-run cards with "Nx better/worse than Ollama", history charts (decode, prefill, time to first token, energy), all runs filterable by suite and model, and this project's experiment log. It reads `bench/results/*.json`; results now record suite, model file and engine flags.

| Command | Starts |
| --- | --- |
| `make up` / `make down` | everything below / stop it |
| `make serve WTYPE=q4a ENGINE_FLAGS="--kv q8"` | our API + dashboard on :8000 |
| `make ollama-serve` | Ollama, CPU only, on :11434 |
| `make webui` | Open WebUI on :3000 |

## Full experiment log

We ran 24 experiments; 19 were kept, 3 were dropped, and 2 were benchmark corrections that changed how we read every other result. Newest first.

| # | Experiment | Result | Kept? |
| --- | --- | --- | --- |
| 24 | Dashboard + per-reply speed stats in Open WebUI | results record suite, model, flags | yes |
| 23 | Chat temperature 0.3 for both models | focused answers instead of rambling | yes |
| 22 | Repetition penalty 1.1 + Ollama stop strings in chat | fixes loops and self-dialogue; benchmarks unchanged | yes |
| 21 | OpenAI-compatible API + Open WebUI | chat UI for both engines | yes |
| 20 | P5 speculative decoding (prompt lookup) | 2.0x on repeat/summarise, neutral otherwise, identical output | yes, opt-in |
| 19 | P4 int8 KV cache | +13% at 1.7k tokens, -65 MB, no quality loss | yes, opt-in |
| 18 | P3 fusion | 199 -> 111 barriers, +5% prefill, decode unchanged on laptop | yes |
| 17 | P2 auto-tuner | noisy samples chose a 24% slower prefill setting | no |
| 16 | P1 activation-aware q4a | ppl gap +3.9% -> +0.5% | yes |
| 15 | Default threads = P-core threads | avoids the 24/32-thread collapse on hybrid i9 | yes |
| 14 | Dynamic chunking in the thread pool | 12 threads: prefill 120 -> 190 | yes |
| 13 | Report effective read GB/s instead of % of a probe | the probe under-read | correction |
| 12 | Unique tag per benchmark prompt | Ollama's prefill had been inflated by its cache | correction |
| 11 | Merge teammate's KV cache backends, incremental working copy | speed unchanged, 4 backends available | yes |
| 10 | Use E-cores for prefill | no prefill gain, decode -9% | no |
| 9 | Vectorised attention | +18-27% at long context | yes |
| 8 | Unsigned nibbles + 2x4 register tiles | prefill +19%, decode +8%, bit-identical | yes |
| 7 | AVX-VNNI | prefill 56 -> 65 | yes |
| 6 | Batched prefill + tiled kernel | prefill 23 -> 56 | yes |
| 5 | Vectorised activation quantisation | removed ~6 ms/token | yes |
| 4 | Pin threads to P-cores | 13.5 -> 26.6 tok/s | yes |
| 3 | Persistent thread pool, unpinned | 35% slower than OpenMP | fixed by #4 |
| 2 | Q4 weights + Q8 activations, AVX2 | 3.3 -> 22 tok/s | yes |
| 1 | fp32 baseline | 3.3 tok/s, 4.1 GB per token | baseline |

Per-step numbers before #10 were taken in power-saving mode; `docs/EXPERIMENTS.md` has the same log and is updated with every new experiment.

## Questions we expect, with answers

Short, honest answers to the questions mentors and judges are most likely to ask.

**Why not just use llama.cpp?** To show that knowing the exact model and the exact hardware can match a general engine, and to measure by how much. llama.cpp supports hundreds of models through a generic compute graph; we hard-code one model's sizes and tune for the machine.

**Are you actually faster than Ollama?** In power-saving mode yes, 1.65x on decode. In performance mode decode is a tie (~60 tok/s each). Ollama is ~2.3x faster at prompt processing, and on a desktop i9 with fast memory it wins decode too (~107 vs 66 tok/s).

**Why does power mode change the winner?** Our pinned, spinning threads keep the P-cores at high clocks; Ollama's threads go idle between steps and the power-saving governor slows the CPU. In performance mode the CPU stays fast for both.

**Does 4-bit hurt quality?** Plain Q4 costs +3.9% perplexity. Our activation-aware `q4a` brings that to +0.5% at the same size and speed. We have not yet scored Ollama's Q4_0 on the same text; that is our next measurement.

**What does "no ML libraries" mean exactly?** The C++ engine uses only the standard library, OpenMP (as an optional fallback) and CPU intrinsics. Python tooling uses only the Python standard library (no torch, no numpy). Open WebUI is a separate demo UI, not part of the engine.

**Why is decode memory-bound?** Each token reads all 615 MB of weights once and does only ~2.2 GFLOP with them, so the CPU waits on RAM; speed is roughly bandwidth / bytes per token (37 GB/s / 0.615 GB = ~60 tok/s).

**Why is your prefill slower than Ollama's?** About 90% of prefill time is the batched matrix kernel. Ollama has AVX-512 paths and a more aggressive accumulation order; ours is AVX2/AVX-VNNI and keeps batched output bit-identical to sequential, which costs some speed.

**How do you know the optimisations are correct?** `make smoke` checks bit-identical output between batched and sequential prefill, both thread runtimes, and speculative vs plain greedy. Every quantisation or reordering change is checked with perplexity, KL divergence and top-1 agreement against fp32.

**What is speculative decoding here, without a draft model?** Prompt lookup: when the last few tokens appeared earlier, propose what followed and verify up to 4 tokens in one batched pass. With greedy decoding the output is identical; it doubled speed on summarise/repeat tasks.

**What did not work?** Using E-cores (power budget is shared), the auto-tuner on a noisy machine (picked a worse setting), and fusion on the laptop (memory-bound, so no decode gain).

**Does it generalise to other models?** Only Llama-family models with the same structure, by changing `src/config.hpp` and re-converting; that is a deliberate trade-off for speed and simplicity.

**What would you do with more time?** An AVX-512 kernel path, quality comparison against Q4_0, a reliable auto-tuner, and folding RMSNorm into the weights.

## Limitations and next steps

The engine is specialised and laptop-tuned; the main open gaps are prefill speed, fast-memory desktops, and an unproven quality comparison against Ollama's own quantisation.

**Known limitations**

- One model (TinyLlama-1.1B), sizes compiled in; AVX2/AVX-VNNI only (no AVX-512 or ARM NEON path yet).
- Prefill ~2.3x slower than Ollama on the laptop, ~3.2x on the i9.
- Decode reaches only ~60% of bandwidth on the 74 GB/s i9 (per-token overhead).
- `q4a` and `--kv q8` are opt-in; `--kv q8` quality was checked on only 469 tokens.
- Speculative decoding is greedy-only; the chat API serves one request at a time.
- Open WebUI settings (temperature, legacy function calling, banner) live in the local `~/.open-webui`, not the repo.

**Next steps, in order**

1. Score `q4a` against Ollama's Q4_0 on WikiText-2 with the same engine and text; if it wins, make `q4a` + `--kv q8` the defaults.
2. Re-run on the teammate's i9 to see whether fusion closed the decode gap there.
3. AVX-512 / AVX512-VNNI kernel path chosen at runtime, and a faster prefill accumulation order (trading the bit-identical guarantee for speed, as a flag).
4. Make the auto-tuner robust: require a >3% win in two sweeps, run on an idle machine.
5. Offline folding of RMSNorm gains into the weights (the rest of Priority 3).
