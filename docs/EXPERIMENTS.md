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
| Chat UI with both engines (+ dashboard) | `make up` -> http://localhost:3000 and http://localhost:8000/dashboard; `make down` stops all three |
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
| 25 | OpenMP tuned vs spin pool (`--pool omp` + `OMP_PLACES` P-cores, `OMP_PROC_BIND=close`, `OMP_WAIT_POLICY=active`) | ties: decode 60.5 vs 60.6, prefill 223 vs 226; untuned OpenMP 56 / 198 | no change (pool stays default) |
| 26 | Multi-machine pipeline over Open MPI (`make mpi-run`, `tinyllama pipe`) with micro-batching | 1 laptop: 4 requests in one micro-batch 183 tok/s total vs 61 for one (3.0x); 8 in one 197. 2 ranks on the SAME laptop: 187 (they share its memory bandwidth, so no gain expected); stream 0 output byte-identical to `run -t 0` | yes (opt-in build, `-DTL_MPI=ON`) |
| 27 | Two laptops over Wi-Fi: enigma (i5-13420H, 41 GB/s) + pegasus (i7-14650HX, 79 GB/s), layers split 8/14 | 40 requests, batches of 4: 133 tok/s total vs 317 pegasus alone, 177 enigma alone; machines busy 19-35%. Link measured 2.7 MB/s, ping 8-207 ms: ~30 MB of activations is ~11 s on that link. Output identical | works; needs wired LAN to pay off |

## Multi-machine (Open MPI)

`make mpi-run NP=2 MPI_HOSTS=192.168.1.10:1,192.168.1.11:1` (LAN IPs, one rank each). MPI traffic is pinned to this machine's LAN subnet (`scripts/lan_subnet.sh`, e.g. `192.168.1.0/24`), which works whatever each machine calls its interface; override with `MPI_IFACE=<subnet or interface>`, or `MPI_IFACE=lo` with no hosts for one machine.
Every machine needs the repo at the same path, `make mpi-build`, `models/tinyllama-q4a.bin` + `tokenizer.bin`,
Open MPI and passwordless SSH from the first host. `make mpi-bench` sweeps streams x micro-batch into
`bench/results/mpi.jsonl` (aggregate tok/s, per-stream tok/s, layer split, measured bandwidth/RAM, link RTT, busy %).

- Layers are split by each rank's measured read bandwidth; the output head counts as ~2.6 layers on the last rank.
  `--split 12,10` overrides. Ranks sharing a machine take separate core slices.
- One request alone gains nothing from more machines (they take turns, plus a network hop per stage). Gains need
  `--streams` >= ranks x `--micro`, so every stage always has a micro-batch to work on.
- `--ctx` caps tokens per request (K/V buffers are per request and per local layer).

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
