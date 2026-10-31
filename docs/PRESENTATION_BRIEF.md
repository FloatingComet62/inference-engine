# TinyLlama CPU Engine: Presentation Brief

Nov 1, 2026 · context for building the slide deck. Companion: [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md).

## The pitch

We built a from-scratch C++ inference engine that runs TinyLlama-1.1B on an ordinary laptop CPU with no ML libraries, and it matches Ollama on generation speed while giving measurably better answers at the same model size.

**The problem.** Running a useful language model locally usually means a GPU, or a large general-purpose stack like llama.cpp/Ollama. We wanted to show that an engine written for one model and tuned for the actual hardware it runs on can compete with that stack on a mid-range laptop (Intel i5-13420H, no GPU used).

**The constraint.** No PyTorch, no GGML, no BLAS, no numpy. Everything is ours: the weight format, the tokenizer, the 4-bit kernels (hand-written AVX2 and AVX-VNNI), the thread pool, attention, the KV cache, the sampler, and the OpenAI-compatible server. Python tooling uses only the standard library.

**The one-line story.** From a 3.3 tok/s fp32 baseline to ~60 tok/s: level with Ollama on decode in performance mode, 1.65x faster in power-saving mode, and within 0.5% of full-precision quality where 4-bit normally costs about 4%.

## Headline results

On the laptop in performance mode we tie Ollama on generation, use less memory, and have far better quality per bit; Ollama still leads prompt processing by about 2.3x.

Setup: i5-13420H (4 P-cores with HT + 4 E-cores), on AC, 8 threads, TinyLlama-1.1B-Chat, 64 new tokens, medians of 5 runs, every prompt tagged so Ollama's prompt cache cannot help it. Ours = `q4a` weights + int8 KV cache (`--kv q8`).

| Metric | tinyllama-cpp | Ollama (Q4_0) | Verdict |
| --- | --- | --- | --- |
| Decode speed (tok/s) | 60.5 | 59.3 | tie (we lead 1.65x in power-saving mode: 38 vs 23) |
| Prefill speed (tok/s) | 198 | 454-499 | Ollama ~2.3x |
| Time to first token (ms) | 281-290 | 168-177 | Ollama ~1.6x |
| Peak RAM (MB) | 639 | 697-738 | ours ~10% less |
| Model bytes read per token (MB) | 615 | 638 | ours 4% smaller |
| Energy (J/token) | 0.97 | 0.88-1.08 | about equal |
| Quality: perplexity vs fp32 | +0.5% | not yet measured (Q4_0 is typically +2-4%) | open item |

**On a teammate's desktop i9 (74 GB/s memory) Ollama wins clearly:** decode ~107 vs our 66 tok/s at best thread count, prefill ~620 vs ~190. That machine exposes our per-token overhead and our AVX2-only kernels; it is the main open gap.

**Honest caveats to say out loud:**

- "Faster than Ollama" is only true in power-saving mode; in performance mode decode is a tie.
- The quality claim versus Ollama's own Q4_0 still needs the same-text perplexity comparison.
- All numbers are TinyLlama-1.1B only; the engine is deliberately specialised to this one model.

## Slide-by-slide outline

Twelve slides, about 8-10 minutes, ending on a live demo; each slide carries one claim and one exhibit.

1. **Title.** "An LLM engine from scratch that keeps up with Ollama on a laptop CPU." Team names, TinyLlama-1.1B, i5-13420H.
2. **The problem.** Local AI needs a GPU or a heavy generic stack. Show: laptop photo or spec line; "no GPU, no ML libraries".
3. **What we built.** Architecture diagram: tokenizer -> weights (mmap) -> 4-bit kernels -> thread pool -> attention + KV cache -> sampler -> OpenAI API -> Open WebUI. Everything in C++17, Python only for tooling (stdlib).
4. **Why decode is memory-bound.** Every new token reads all weights once: 4.4 GB in fp32, 615 MB in 4-bit. Speed ceiling = memory bandwidth / bytes per token. This motivates quantisation.
5. **The progression chart.** Decode tok/s step by step from 3.3 to ~60 (see the progression section). The single most important visual.
6. **Quantisation that keeps quality.** Activation-aware 4-bit (`q4a`): perplexity gap to fp32 from +3.9% to +0.5% at the same size and speed. Bar chart of ppl: fp32 / Q4 / q4a.
7. **Hardware-aware threading.** Hybrid CPU problem: E-cores stall every step. Pinning to P-cores: 13.5 -> 26.6 tok/s. Thread-count chart (4/8/12 threads).
8. **Prefill and attention.** Batched prefill, tiled VNNI kernel, vectorised attention, fusion (199 -> 111 sync points per token).
9. **Long context and speculation.** int8 KV cache: +13% at 1.7k tokens, -65 MB. Speculative decoding: 2x on summarise/repeat prompts, byte-identical output.
10. **Head to head with Ollama.** The headline table, both power modes, plus the teammate's i9 result stated honestly.
11. **What did not work and what we learned.** E-cores for prefill, the auto-tuner, benchmark pitfalls (prompt cache, power modes).
12. **Live demo + next steps.** Open WebUI side by side, the dashboard; next: AVX-512 path, quality vs Q4_0, auto-tuner.

## Progression: how we got from 3 to 60 tok/s

Decode went from 3.3 to ~60 tok/s in eight measured steps; the two biggest jumps were 4-bit weights (7x) and pinning threads to performance cores (2x).

Use these rows for the progression chart (x = step, y = decode tok/s; a second series for prefill). Steps 1-7 were measured in balanced/power-saving mode, so absolute numbers are lower than the final performance-mode figures; say this on the slide.

| Step | Change | Decode tok/s | Prefill tok/s | Note |
| --- | --- | --- | --- | --- |
| 1 | fp32 baseline (scalar matmul, OpenMP) | 3.3 | 3.3 | reads 4.1 GB per token |
| 2 | 4-bit weights + int8 activations, AVX2 | 22 | 23 | 7x; quality +3.7% ppl |
| 3 | Persistent thread pool, unpinned | 14 | 20 | slower: threads landed on E-cores |
| 4 | Pin threads to P-cores | 26.6 | 30 | the hybrid-CPU fix |
| 5 | Vectorised activation quantisation | 29.8 | 35 | removed ~6 ms/token of scalar work |
| 6 | Batched prefill + tiled kernel | 31 | 56 | prefill is now batched |
| 7 | AVX-VNNI instructions | 33.8 | 64 | first time ahead of Ollama (power-saving mode) |
| 8 | Unsigned-weight trick + 2x4 register tiles | 36.5 | 81 | bit-identical output |
| 9 | Vectorised attention | 38 | 80 | +18-27% at long context |
| 10 | Performance power mode (same code) | 58-60 | 210-220 | the machine, not the code |
| 11 | Fusion, int8 KV, q4a (final) | 60.5 | 198 | +13% at long context; quality +0.5% |

## The five priority features

Four of the five shipped with a measured win; the auto-tuner was dropped because noisy timings made it pick a worse setting.

| Priority | Feature | Result | Status |
| --- | --- | --- | --- |
| P1 | Adaptive INT4 (`q4a`): calibrate activations, pick each 4-bit block's scale to minimise activation-weighted error | ppl gap to fp32 +3.9% -> **+0.5%**; KL 0.069 -> 0.047; same size and speed; one-time build ~100 s | shipped (`WTYPE=q4a`) |
| P2 | CPU auto-tuning (`tinyllama tune`): measure thread counts and chunk sizes per machine | picked 4 prefill threads from noisy samples: prefill -24%. Useful finding: 8 threads is fastest and cheapest (0.72 vs 0.97 J/token at 4) | dropped; branch kept |
| P3 | Fused computation: Q/K/V in one pass, SwiGLU in the matmul, residual adds folded in | sync points 199 -> 111 per token, output bit-identical; laptop: prefill +5%, decode unchanged (memory-bound) | shipped (always on) |
| P4 | int8 KV cache (`--kv q8`) | at ~1.7k tokens: prefill +13%, decode +13%, -65 MB RAM, no ppl loss | shipped (opt-in) |
| P5 | Speculative decoding (`--spec ngram`), prompt-lookup drafts, greedy | 2.0x decode on summarise/repeat prompts (57 -> 116 tok/s, 83% acceptance); neutral on normal questions; output byte-identical in 24/24 tests | shipped (opt-in) |

The best slide material is P1 (quality) and P5 (2x with identical output); P3 is the best "engineering depth" story; P2 is a good "what we learned" story.

## What did not work, and what we learned

Our most useful lessons came from failures and from benchmark mistakes we caught, and each one is a good answer to "what surprised you?".

| What we tried or saw | What happened | Lesson |
| --- | --- | --- |
| Unpinned thread pool | 35% slower than OpenMP | On hybrid CPUs the OS puts threads on slow E-cores and every step waits for them |
| E-cores for prefill (dynamic work split) | no prefill gain, decode -9% | A laptop's power and heat budget is shared; more cores does not mean more work done |
| Auto-tuner on a busy machine | chose a setting 24% slower | Tuning needs quiet machines and repeated measurements; a fixed good default beats a noisy tuner |
| Fusion on the laptop | decode unchanged | When memory is the limit, removing compute overhead does not help; it should help on fast-memory machines |
| Ollama's "18,000-27,000 tok/s" prefill | Ollama reused cached prompts between runs | Every benchmark prompt must be unique; we tag each run |
| Our "1.65x faster than Ollama" | only true in power-saving mode | Power mode can flip the winner; always record it |
| Memory bandwidth probe | reported less than our engine actually reads | A pure read loop is down-clocked on laptops; we report effective read GB/s instead |
| 12 or 24 threads | 25-40% slower for both engines | Default to the P-core threads; this is what llama.cpp does too |
| Docker for Open WebUI | image pull stalled | Ran it with uv instead, with CPU-only PyTorch (1 GB instead of 5 GB) |
| TinyLlama in chat at temperature 0.8 | talked to itself, looped | Ollama-style stop strings, repetition penalty 1.1, temperature 0.3 |

## Demo plan

Run `make up` ten minutes before presenting, then show three things: a side-by-side chat, a speculative-decoding moment, and the dashboard.

**Before the talk (checklist)**

- [ ] Laptop on AC, power profile set to performance; close browsers and other heavy apps.
- [ ] `cd ~/Projects/BIOS/inference-engine && git pull && make up` (starts Ollama, our API, Open WebUI).
- [ ] Open http://localhost:3000 and http://localhost:8000/dashboard in two tabs.
- [ ] Send one warm-up message to each model so nothing loads during the demo.

**Script**

1. **Side by side.** In Open WebUI click **+** next to the model name, pick `tinyllama-cpp` and `tinyllama:1.1b`, ask "Explain how a CPU cache works, in simple terms." Both stream at once. Click (i) under each reply to show tok/s.
2. **Speculative decoding (terminal).** `build/tinyllama run --model models/tinyllama-q4a.bin --tokenizer models/tokenizer.bin -t 0 -n 120 -p "Repeat this paragraph, then summarise it: ..." --spec ngram` vs the same without `--spec`. Point at tok/s and at identical text.
3. **Dashboard.** Latest-run cards ("Nx better/worse than Ollama"), history charts, the experiment log.

**What could go wrong**

- Side-by-side halves both speeds (they share the CPU): say so, and quote numbers from single-model runs.
- Ollama error "does not support tools": set Function Calling = Legacy for the model in Workspace -> Models (already done on this laptop).
- Wi-Fi is not needed once `make up` has run before; everything is local.

## Numbers cheat sheet and visuals

Memorise the first six numbers; build the four charts from the listed sources.

| Number | Value | Context |
| --- | --- | --- |
| Speed-up over our fp32 baseline | ~18x decode (3.3 -> 60.5 tok/s) | same laptop, final config, performance mode |
| Decode vs Ollama | tie (60.5 vs 59.3); 1.65x in power-saving mode (38 vs 23) | i5-13420H, 8 threads |
| Quality vs fp32 | +0.5% perplexity (`q4a`), down from +3.9% | same 615 MB of weights read per token |
| Speculative decoding | 2.0x decode on repeat/summarise prompts | output byte-identical |
| int8 KV cache | +13% speed at 1.7k tokens, -65 MB | no measurable quality loss |
| Sync points per token | 199 -> 111 | fusion, bit-identical output |
| Hybrid-CPU pinning | 13.5 -> 26.6 tok/s | one fix, 2x |
| Ollama's real prefill lead | ~2.3x laptop, ~3.2x desktop i9 | our biggest gap |
| Lines of code | ~3k C++ + stdlib Python | no ML libraries |

**Charts to build**

1. **Progression line** (decode and prefill per step): the progression table above.
2. **Head-to-head bars** (ours vs Ollama per metric, both power modes): the headline table, or the dashboard's latest-run cards.
3. **Quality bars** (ppl: fp32 9.058, Q4 9.409, q4a 9.104): from `make compare WTYPE=q4` and `WTYPE=q4a` quality lines.
4. **Thread-count curve** (4 / 8 / 12 threads, both engines): from the thread sweep (ours 56.7 / 60.5 / 44.8; Ollama 52.7 / 59.8 / 48.1 decode tok/s).

Raw data for every run is in `bench/results/*.json` and on the dashboard at http://localhost:8000/dashboard; the experiment log is `docs/EXPERIMENTS.md` in the repo.
