# Experiments log

Everything we tried on the engine, what it did, and whether it was kept. Numbers are medians
of 5 alternating runs unless noted. Main machine: i5-13420H laptop (4 P-cores with HT + 4 E-cores,
AVX2 + AVX-VNNI), on AC, **performance** power profile, 8 threads, TinyLlama-1.1B-Chat.
Add a row whenever you try something; keep the losers, they are results too.

## Recommended commands

| Goal | Command |
|---|---|
| Best quality + speed (recommended) | `make run WTYPE=q4a ENGINE_FLAGS="--kv q8"` |
| Chat in the terminal | `make chat WTYPE=q4a ENGINE_FLAGS="--kv q8"` |
| Summaries / quoting text (2x decode) | add `--spec ngram` to ENGINE_FLAGS (greedy only: `TEMP=0`) |
| Compare against Ollama | `make compare N=64 RUNS=5 WTYPE=q4a ENGINE_FLAGS="--kv q8"` |
| Quality vs fp32 (WikiText-2) | `make eval WTYPE=q4a` |
| Chat UI with both engines | `make serve WTYPE=q4a ENGINE_FLAGS="--kv q8"` + `make ollama-serve` + `make webui` (via uv, first run downloads ~1 GB) -> http://localhost:3000 |
| Sanity check after any change | `make smoke` |
| Benchmark dashboard (all runs, charts, this log) | `make serve`, then http://localhost:8000/dashboard |
| Both models side by side in Open WebUI | click **+** next to the model name, add `tinyllama-cpp` and `tinyllama:1.1b`; the (i) under each reply shows tok/s |

`WTYPE=q4a` is built once from fp32 (~100 s, `make model WTYPE=q4a`). Before benchmarking:
plug in, set the performance profile, close other heavy apps, keep Ollama idle.

## Combinations (performance mode, 58-token prompts, 64 new tokens, `make compare`)

| Config | Prefill tok/s | Decode tok/s | TTFT ms | Peak RAM MB | J/token | ppl vs fp32 |
|---|---|---|---|---|---|---|
| A: `q4` (previous default) | 220.6 | 57.9 | 258 | 703 | 1.31 | +3.88% |
| B: `q4a` | 189.5 | 58.5 | 294 | 704 | 1.30 | **+0.50%** |
| C: `q4a --kv q8` | 198.4 | **60.5** | 290 | **639** | **0.97** | +0.50% |
| D: `q4a --kv q8 --spec ngram` | 197.9 | 59.1 | 281 | 639 | 0.98 | +0.50% |
| Ollama tinyllama:1.1b (Q4_0), same runs | 454-499 | 59.2-60.4 | 168-177 | 697-738 | 0.88-1.08 | not measured |

Prefill differences between A-D are run-to-run noise (same kernels). The ppl column is weights
only (`ppl` mode does not take `--kv`). Speculative decoding is neutral here because these prompts
do not repeat text; see P5 below for where it pays.

## Log

| # | What | Result | Kept? |
|---|---|---|---|
| 1 | fp32 baseline (scalar matmul, OpenMP) | 3.3 tok/s decode, 4.1 GB read per token | baseline |
| 2 | Q4_32 weights + Q8 activations, AVX2 `maddubs` kernel | decode 3.3 -> 22 tok/s (7x); ppl +3.7% | yes |
| 3 | Persistent spin thread pool (unpinned) | 35% *slower* than OpenMP: threads landed on E-cores | fixed by 4 |
| 4 | Pin threads to P-cores, physical cores first | 13.5 -> 26.6 tok/s at 8 threads | yes |
| 5 | Vectorised activation quantisation (was scalar `lround`, ~6 ms/token) | decode -> 29.8 tok/s | yes |
| 6 | Batched prefill (32-token chunks), 4- then 8-token tiled kernel | prefill 23 -> 56 tok/s | yes |
| 7 | AVX-VNNI `dpbusd` instead of `maddubs+madd` | prefill 56 -> 65, decode -> 32 | yes |
| 8 | Unsigned nibbles + exact -8*sum(x) correction, 2x4 register tiles | prefill +19%, decode +8%, bit-identical | yes |
| 9 | AVX2 attention (vector q.k, polynomial exp, V in registers) | long ctx: prefill +27%, decode +18% | yes |
| 10 | Use E-cores for prefill (dynamic chunks over P+E) | prefill no gain, decode -9% (power/thermal budget) | no |
| 11 | Merge teammate's KV cache backends (baseline/paged/radix/streaming) | engine keeps a working copy, rotates only new rows; speed unchanged | yes |
| 12 | Benchmark fix: per-run prompt tag | Ollama's prefill was inflated by its prompt cache (150 -> 114 real; repeat runs showed 18k-27k) | yes |
| 13 | Benchmark fix: "% BW limit" -> effective read GB/s | the read probe under-reads on laptops (powersave) | yes |
| 14 | Dynamic chunking in `parallel_for` (ggml-style) | 12 threads: prefill 120 -> 190, decode 38.9 -> 44.0; 8 threads unchanged | yes |
| 15 | Default threads = P-core hardware threads (kernel `cpu_core` list) | avoids the 24/32-thread collapse on hybrid i9 (39 vs 66 tok/s) | yes |
| 16 | **P1** activation-aware `q4a`: weighted scale search per Q4 block | ppl gap to fp32 +3.9% -> **+0.5%**, KL 0.069 -> 0.047, same size/speed | yes (`WTYPE=q4a`) |
| 17 | **P2** auto-tuner (`tinyllama tune`, per-phase threads) | noisy timings picked 4 prefill threads: prefill -24%. Finding: 8 threads best for decode and *cheapest* in energy (0.72 vs 0.97 J/token at 4) | no (branch `p2-autotune`) |
| 18 | **P3** fusion: QKV one dispatch, gate/up + SwiGLU epilogue, residual in epilogues | 199 -> 111 barriers/token, bit-identical; laptop: prefill +5%, decode unchanged (memory-bound). Expected to help fast-memory machines | yes |
| 19 | **P4** `--kv q8` int8 KV working copy, integer q.k | ~1.7k ctx: prefill +13%, decode +13%, -65 MB RAM; no ppl loss (469-token test) | yes (opt-in) |
| 20 | **P5** `--spec ngram` prompt-lookup speculative decoding, greedy | repeat/summarise prompt: 2.0x decode (57 -> 116 tok/s, 83% acceptance); normal prompts neutral; output byte-identical | yes (opt-in) |
| 21 | OpenAI-compatible API (`make serve`) + Open WebUI (`make webui`) | chat UI against both engines | yes |
| 22 | Repetition penalty (`--repeat-penalty`, Ollama's 1.1 in the chat API) + stop strings (`--stop`, Ollama's TinyLlama stops) | fixes greedy loops and the model writing both sides of a dialogue; benchmarks unchanged (penalty off there) | yes |
| 23 | Chat API temperature default 0.3 (both models set to 0.3 in Open WebUI) | at 0.8 TinyLlama rambles/role-plays; 0.3 gives focused answers | yes |
| 24 | Dashboard (`/dashboard`) + per-reply speed stats in Open WebUI | results now record suite, model and flags | yes |

## Things we learned

- **Power mode changes the winner.** In powersave/balanced we decoded 1.65x faster than Ollama
  (spinning pinned threads keep P-cores clocked up); in performance mode decode is a tie (~60 tok/s).
- **Thread count is the biggest single knob** on hybrid CPUs: too many threads (E-cores, all
  logical CPUs) costs 25-40% decode on both engines.
- **Ollama's prompt cache** makes repeat benchmark runs meaningless unless every prompt is unique.
- **Prefill is our largest gap** (Ollama ~2.3x on the laptop, ~3.2x on the teammate's i9). ~90%
  of prefill time is the batched matmul kernel; next steps are an AVX-512 path and a faster
  accumulation order.

## Open items

- Compare `q4a` against Ollama's Q4_0 on WikiText-2 (same engine, Q4_0 rounding) - the headline claim.
- Validate `--kv q8` on 2,048-token WikiText chunks, then make `q4a` + `--kv q8` the defaults.
- Re-run on the teammate's i9 after `git pull` (fusion + thread defaults).
- P2 tuner: require a >3% win in two sweeps before leaving the default; run on an idle machine.

## Open WebUI notes

- If an Ollama model fails with "does not support tools" (TinyLlama has no tool calling): in Open
  WebUI open Workspace -> Models -> edit the model -> Advanced Params -> Function Calling = **Legacy**,
  and untick the **Builtin Tools** capability. Do the same for `tinyllama-cpp`. Then start a new chat.
  The setting is stored in `~/.open-webui`, so it survives restarts.
- Both TinyLlama models are set to temperature 0.3 in Open WebUI (Workspace -> Models) so the
  side-by-side comparison is fair. Two models answering at once share the CPU, so compare tok/s
  from single-model chats or `make compare`, not from a side-by-side chat.
- Open WebUI writes `.webui_secret_key` into the directory it starts from; it is git-ignored.
