// Multi-machine inference: pipeline parallel over Open MPI (`tinyllama pipe`).
//
// Each MPI rank owns a contiguous block of layers; rank 0 also does the embedding and
// drives the requests, the last rank also runs the output head and picks the next token
// (greedy). A message carries a micro-batch: R rows, each one token of one request
// (slot) at its position, plus the R x DIM hidden state (8 KB per row).
//
//   rank 0 --x--> rank 1 --x--> ... --x--> last rank --tokens--> rank 0
//
// Several micro-batches are in flight at once, so while the last rank works on one group
// of requests, rank 0 is already working on the next: every machine stays busy. Within a
// micro-batch, the rows share every weight read (decode is memory-bandwidth bound).
//
// Layers are split by each rank's measured memory bandwidth (the decode speed limit),
// with the output head counted as extra layers on the last rank; --split overrides it.
#include "pipeline.hpp"
#include "kv_cache.hpp"
#include "model.hpp"
#include "threadpool.hpp"
#include "tokenizer.hpp"

#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tl {
namespace {

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

constexpr int TAG_WORK = 1, TAG_RESULT = 2, TAG_STOP = 3;

// Work message: [group, R, need_out, slot[R], pos[R], token[R]] as int32, then x[R][DIM].
struct Work {
    int group = 0, R = 0, need_out = 0;
    std::vector<int> slot, pos, token;
    std::vector<float> x;
};

std::vector<char> pack(const Work& w) {
    const size_t ni = 3 + 3 * (size_t)w.R;
    std::vector<char> b(ni * 4 + w.x.size() * 4);
    int* h = reinterpret_cast<int*>(b.data());
    h[0] = w.group; h[1] = w.R; h[2] = w.need_out;
    std::memcpy(h + 3, w.slot.data(), w.R * 4);
    std::memcpy(h + 3 + w.R, w.pos.data(), w.R * 4);
    std::memcpy(h + 3 + 2 * w.R, w.token.data(), w.R * 4);
    std::memcpy(b.data() + ni * 4, w.x.data(), w.x.size() * 4);
    return b;
}

Work unpack(const std::vector<char>& b) {
    const int* h = reinterpret_cast<const int*>(b.data());
    Work w;
    w.group = h[0]; w.R = h[1]; w.need_out = h[2];
    w.slot.assign(h + 3, h + 3 + w.R);
    w.pos.assign(h + 3 + w.R, h + 3 + 2 * w.R);
    w.token.assign(h + 3 + 2 * w.R, h + 3 + 3 * w.R);
    w.x.resize((size_t)w.R * DIM);
    std::memcpy(w.x.data(), b.data() + (3 + 3 * (size_t)w.R) * 4, w.x.size() * 4);
    return w;
}

// Blocking receive of a message of unknown size.
std::vector<char> recv_bytes(int src, int& tag, int& from) {
    MPI_Status st;
    MPI_Probe(src, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
    int n = 0;
    MPI_Get_count(&st, MPI_BYTE, &n);
    std::vector<char> b(n);
    MPI_Recv(b.data(), n, MPI_BYTE, st.MPI_SOURCE, st.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    tag = st.MPI_TAG; from = st.MPI_SOURCE;
    return b;
}

struct Spec {
    double bw_gbs;     // measured read bandwidth with this rank's threads
    double avail_gb;   // available RAM
    int threads;
    char host[64];
};

// Read bandwidth: sum a 256 MB buffer with the thread pool, best of 3.
double measure_bandwidth() {
    const size_t n = (size_t)64 << 20;   // floats
    std::vector<float> buf(n, 1.0f);
    std::atomic<int> sink{0};
    double best = 0;
    for (int it = 0; it < 3; it++) {
        const auto t0 = Clock::now();
        parallel_for((int)(n / 4096), [&](int b, int e) {
            float s = 0;
            for (size_t i = (size_t)b * 4096; i < (size_t)e * 4096; i++) s += buf[i];
            if (s == -1.0f) sink++;   // keeps the loop from being optimised away
        });
        best = std::max(best, n * 4.0 / (ms(t0, Clock::now()) / 1000.0) / 1e9);
    }
    return best;
}

// MemAvailable (free + reclaimable page cache); Linux only, else free pages.
double available_gb() {
    std::ifstream f("/proc/meminfo");
    std::string k, unit;
    long kb = 0;
    while (f >> k >> kb >> unit) if (k == "MemAvailable:") return kb / 1e6;
    return (double)sysconf(_SC_AVPHYS_PAGES) * (double)sysconf(_SC_PAGESIZE) / 1e9;
}

// Layers per rank in proportion to bandwidth; the head counts as `head_layers` extra
// layers of work on the last rank. Every rank gets at least one layer.
std::vector<int> split_layers(const std::vector<Spec>& sp, double head_layers) {
    const int P = (int)sp.size();
    if (P > N_LAYERS) throw std::runtime_error("more ranks than layers");
    double total_bw = 0;
    for (auto& s : sp) total_bw += s.bw_gbs;
    std::vector<double> want(P);
    for (int r = 0; r < P; r++)
        want[r] = (N_LAYERS + head_layers) * sp[r].bw_gbs / total_bw - (r == P - 1 ? head_layers : 0);
    std::vector<int> n(P, 1);
    for (int left = N_LAYERS - P; left > 0; left--) {   // next layer to the rank furthest below its share
        int best = 0;
        for (int r = 1; r < P; r++) if (want[r] - n[r] > want[best] - n[best]) best = r;
        n[best]++;
    }
    return n;
}

}  // namespace

int run_pipeline(int argc, char** argv, const PipeArgs& a) {
    MPI_Init(&argc, &argv);
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    int rc = 0;
    try {
        // Ranks sharing a machine split its fast cores instead of pinning onto the same ones.
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
        int lrank = 0, lsize = 1;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_size(local, &lsize);
        MPI_Comm_free(&local);
        int threads = a.threads > 0 ? a.threads : std::max(1, default_threads() / lsize);
        g_cpu_offset = lrank * threads;
        Parallel::get().init(threads, true);

        // ---- probe every machine, then split the layers ----
        Spec me{};
        MPI_Barrier(MPI_COMM_WORLD);   // ranks on one machine measure together: shared bandwidth
        me.bw_gbs = measure_bandwidth();
        me.avail_gb = available_gb();
        me.threads = threads;
        gethostname(me.host, sizeof(me.host) - 1);
        std::vector<Spec> sp(P);
        MPI_Allgather(&me, sizeof(Spec), MPI_BYTE, sp.data(), sizeof(Spec), MPI_BYTE, MPI_COMM_WORLD);

        BaselineKVCache unused_cache;
        Model model(a.model, unused_cache, false, /*stage_only=*/true);
        const double layer_gb = model.layer_bytes() / 1e9;
        std::vector<int> n(P);
        if (!a.split.empty()) {
            n.clear();
            size_t i = 0;
            while (i < a.split.size()) {
                size_t j = a.split.find(',', i);
                if (j == std::string::npos) j = a.split.size();
                n.push_back(std::stoi(a.split.substr(i, j - i)));
                i = j + 1;
            }
            int sum = 0;
            for (int v : n) sum += v;
            if ((int)n.size() != P || sum != N_LAYERS)
                throw std::runtime_error("--split needs one count per rank, summing to " + std::to_string(N_LAYERS));
        } else {
            n = split_layers(sp, (double)model.head_bytes() / model.layer_bytes());
        }
        std::vector<int> l0(P + 1, 0);
        for (int r = 0; r < P; r++) l0[r + 1] = l0[r] + n[r];
        const int my_l0 = l0[rank], my_l1 = l0[rank + 1];
        const double need_gb = n[rank] * layer_gb + (rank == P - 1 ? model.head_bytes() / 1e9 : 0.0);
        if (need_gb > me.avail_gb * 0.9)
            std::fprintf(stderr, "warning: rank %d (%s) needs %.2f GB of weights, %.2f GB available\n",
                         rank, me.host, need_gb, me.avail_gb);

        const int slots = a.streams;
        model.stage_init(my_l0, my_l1, slots, a.ctx);
        const bool first = rank == 0, last = rank == P - 1;

        // ---- link latency: 8 KB ping-pong between rank 0 and each rank ----
        std::vector<double> rtt(P, 0.0);
        {
            std::vector<char> ping(DIM * 4);
            for (int r = 1; r < P; r++) {
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank != 0 && rank != r) continue;
                const int iters = 20;
                const auto t0 = Clock::now();
                for (int i = 0; i < iters; i++) {
                    if (rank == 0) {
                        MPI_Send(ping.data(), (int)ping.size(), MPI_BYTE, r, 9, MPI_COMM_WORLD);
                        MPI_Recv(ping.data(), (int)ping.size(), MPI_BYTE, r, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    } else {
                        MPI_Recv(ping.data(), (int)ping.size(), MPI_BYTE, 0, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Send(ping.data(), (int)ping.size(), MPI_BYTE, 0, 9, MPI_COMM_WORLD);
                    }
                }
                if (rank == 0) rtt[r] = ms(t0, Clock::now()) / iters;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double busy_ms = 0;   // time this rank spent computing
        auto compute = [&](Work& w) {
            const auto t0 = Clock::now();
            if (first) { w.x.resize((size_t)w.R * DIM); model.embed(w.token.data(), w.R, w.x.data()); }
            model.stage_forward(w.x.data(), w.R, w.slot.data(), w.pos.data());
            if (last) model.head_argmax(w.x.data(), w.R, w.token.data());
            busy_ms += ms(t0, Clock::now());
        };

        if (!first) {
            // ---- middle / last stage: receive, compute, pass on ----
            while (true) {
                int tag, from;
                auto b = recv_bytes(rank - 1, tag, from);
                if (tag == TAG_STOP) {
                    if (!last) MPI_Send(nullptr, 0, MPI_BYTE, rank + 1, TAG_STOP, MPI_COMM_WORLD);
                    break;
                }
                Work w = unpack(b);
                compute(w);
                if (!last) {
                    auto out = pack(w);
                    MPI_Send(out.data(), (int)out.size(), MPI_BYTE, rank + 1, TAG_WORK, MPI_COMM_WORLD);
                } else {   // [group, R, need_out, token[R]]
                    std::vector<int> res{w.group, w.R, w.need_out};
                    res.insert(res.end(), w.token.begin(), w.token.end());
                    MPI_Send(res.data(), (int)res.size(), MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
                }
            }
        } else {
            // ---- rank 0: drive the requests ----
            Tokenizer tok(a.tokenizer);
            const int S = a.streams, M = std::max(1, std::min(a.micro, PREFILL_CHUNK));
            std::vector<std::vector<int>> prompt(S), out(S);
            std::vector<int> cur_pos(S);
            for (int s = 0; s < S; s++) {
                // stream 0 gets the prompt exactly (to compare with `run -t 0`), the rest a variant
                prompt[s] = tok.encode_first_turn(a.system, s == 0 ? a.prompt : a.prompt + " (request " + std::to_string(s) + ")");
                if ((int)prompt[s].size() + a.n > a.ctx) throw std::runtime_error("prompt + -n exceeds --ctx");
            }
            std::deque<std::vector<int>> local;   // results when P == 1
            auto submit = [&](Work& w) {
                compute(w);
                if (P == 1) {
                    std::vector<int> res{w.group, w.R, w.need_out};
                    res.insert(res.end(), w.token.begin(), w.token.end());
                    local.push_back(res);
                } else {
                    auto b = pack(w);
                    MPI_Send(b.data(), (int)b.size(), MPI_BYTE, 1, TAG_WORK, MPI_COMM_WORLD);
                }
            };
            auto result = [&]() {
                if (P == 1) { auto r = local.front(); local.pop_front(); return r; }
                MPI_Status st;
                MPI_Probe(P - 1, TAG_RESULT, MPI_COMM_WORLD, &st);
                int cnt = 0;
                MPI_Get_count(&st, MPI_INT, &cnt);
                std::vector<int> r(cnt);
                MPI_Recv(r.data(), cnt, MPI_INT, P - 1, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                return r;
            };

            // Prefill: every stream's prompt in chunks of <= PREFILL_CHUNK rows, all sent
            // back to back (the pipeline overlaps them); the last chunk returns the first token.
            const auto t_pre = Clock::now();
            int pending = 0;
            for (int s = 0; s < S; s++) {
                const int L = (int)prompt[s].size();
                for (int i = 0; i < L; i += PREFILL_CHUNK) {
                    Work w;
                    w.group = s; w.R = std::min(PREFILL_CHUNK, L - i); w.need_out = i + w.R == L;
                    for (int r = 0; r < w.R; r++) { w.slot.push_back(s); w.pos.push_back(i + r); w.token.push_back(prompt[s][i + r]); }
                    submit(w);
                    pending++;
                }
            }
            for (; pending > 0; pending--) {
                auto r = result();
                if (r[2]) { out[r[0]].push_back(r[3 + r[1] - 1]); cur_pos[r[0]] = (int)prompt[r[0]].size(); }
            }
            const double prefill_ms = ms(t_pre, Clock::now());

            // Decode: streams in groups of M (one micro-batch each); every group is in
            // flight at once, and a group's next step is sent as soon as its tokens return.
            const int G = (S + M - 1) / M;
            auto step = [&](int g) {
                Work w;
                w.group = g; w.need_out = 1;
                for (int s = g * M; s < std::min(S, (g + 1) * M); s++) {
                    w.slot.push_back(s); w.pos.push_back(cur_pos[s]); w.token.push_back(out[s].back());
                }
                w.R = (int)w.slot.size();
                submit(w);
            };
            const auto t_dec = Clock::now();
            const double busy_before = busy_ms;
            std::vector<int> steps(G, 1);   // tokens produced so far per stream (first came from prefill)
            int inflight = 0;
            for (int g = 0; g < G; g++) if (a.n > 1) { step(g); inflight++; }
            while (inflight > 0) {
                auto r = result();
                inflight--;
                const int g = r[0];
                for (int i = 0; i < r[1]; i++) { const int s = g * M + i; out[s].push_back(r[3 + i]); cur_pos[s]++; }
                if (++steps[g] < a.n) { step(g); inflight++; }
            }
            const double decode_ms = ms(t_dec, Clock::now());
            const double busy0_dec = busy_ms - busy_before;
            if (P > 1) MPI_Send(nullptr, 0, MPI_BYTE, 1, TAG_STOP, MPI_COMM_WORLD);

            std::string text;
            for (int t : out[0]) text += tok.decode(t);
            std::fprintf(stderr, "[stream 0] %s\n", text.c_str());

            // per-rank busy time (decode + prefill) from the other ranks
            std::vector<double> busy(P, 0.0);
            busy[0] = busy_ms;
            for (int r = 1; r < P; r++)
                MPI_Recv(&busy[r], 1, MPI_DOUBLE, r, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            const long gen = (long)S * a.n;
            std::printf("{\"engine\": \"tinyllama-mpi\", \"ranks\": %d, \"streams\": %d, \"micro\": %d, "
                        "\"groups\": %d, \"gen_tokens_per_stream\": %d, \"prompt_tokens\": %zu, "
                        "\"prefill_ms\": %.1f, \"decode_ms\": %.1f, \"aggregate_tps\": %.3f, "
                        "\"per_stream_tps\": %.3f, \"rank0_decode_busy_pct\": %.1f",
                        P, S, M, G, a.n, prompt[0].size(), prefill_ms, decode_ms,
                        (gen - S) / (decode_ms / 1000.0), (a.n - 1) / (decode_ms / 1000.0),
                        100.0 * busy0_dec / decode_ms);
            auto list = [&](const char* key, auto get) {
                std::printf(", \"%s\": [", key);
                for (int r = 0; r < P; r++) { if (r) std::printf(", "); get(r); }
                std::printf("]");
            };
            list("hosts", [&](int r) { std::printf("\"%s\"", sp[r].host); });
            list("layers", [&](int r) { std::printf("%d", n[r]); });
            list("threads", [&](int r) { std::printf("%d", sp[r].threads); });
            list("bw_gbs", [&](int r) { std::printf("%.1f", sp[r].bw_gbs); });
            list("avail_ram_gb", [&](int r) { std::printf("%.1f", sp[r].avail_gb); });
            list("rtt_ms", [&](int r) { std::printf("%.3f", rtt[r]); });
            list("busy_pct", [&](int r) { std::printf("%.1f", 100.0 * busy[r] / (prefill_ms + decode_ms)); });
            std::printf("}\n");
            std::fflush(stdout);
        }
        if (!first) MPI_Send(&busy_ms, 1, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rank %d error: %s\n", rank, e.what());
        rc = 1;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return rc;
}

}  // namespace tl
