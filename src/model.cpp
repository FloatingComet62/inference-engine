#include <tuple>
#include "model.hpp"
#include "kernels.hpp"
#include "kernels_attn.hpp"
#include "threadpool.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tl {

namespace {
constexpr size_t HEADER_BYTES = 256;
constexpr size_t ALIGN = 64;

// Hugging Face Llama RoPE layout ("rotate_half"): dimension i pairs with i + HEAD_DIM/2.
inline void rope(float* v, const float* cs, const float* sn) {
    constexpr int half = HEAD_DIM / 2;
    for (int i = 0; i < half; i++) {
        const float a = v[i], b = v[i + half];
        v[i]        = a * cs[i] - b * sn[i];
        v[i + half] = a * sn[i] + b * cs[i];
    }
}
}  // namespace

Model::Model(const std::string& path, KVCache& cache, bool kv_q8) : kv_q8_(kv_q8), cache_(&cache) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open weights: " + path + " (run `make setup`)");
    struct stat st{};
    fstat(fd, &st);
    file_size_ = (size_t)st.st_size;
    map_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map_ == MAP_FAILED) { map_ = nullptr; throw std::runtime_error("mmap failed"); }
    madvise(map_, file_size_, MADV_WILLNEED);

    const char* base = static_cast<const char*>(map_);
    if (file_size_ < HEADER_BYTES || std::memcmp(base, "TLM1", 4) != 0)
        throw std::runtime_error("not a TLM1 weight file: " + path);
    int32_t h[8];
    std::memcpy(h, base + 4, sizeof(h));
    const int version = h[0];
    if (version != 1 && version != 2) throw std::runtime_error("unknown weight file version");
    h[0] = 1;
    const int expect[8] = {1, DIM, HIDDEN, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB, MAX_SEQ};
    for (int i = 0; i < 8; i++)
        if (h[i] != expect[i])
            throw std::runtime_error("weight file does not match compiled config (re-run convert)");

    size_t off = HEADER_BYTES;
    // Version 2 stores one int8 type code per matrix after the sizes (layer order, then lm_head).
    const int8_t* types = reinterpret_cast<const int8_t*>(base + 4 + sizeof(h));
    int next_type = 0;
    auto take_bytes = [&](size_t n) -> const void* {
        off = (off + ALIGN - 1) & ~(ALIGN - 1);
        const void* p = base + off;
        off += n;
        if (off > file_size_) throw std::runtime_error("weight file truncated");
        return p;
    };
    auto take = [&](size_t n) { return static_cast<const float*>(take_bytes(n * sizeof(float))); };
    auto mat = [&](int n_out, int n_in) {
        Mat m;
        m.t = version == 1 ? WType::F32 : (WType)types[next_type++];
        m.w = take_bytes(row_bytes(m.t, n_in) * n_out);
        return m;
    };
    w_.embed = take((size_t)VOCAB * DIM);
    for (auto& L : w_.layers) {
        L.attn_norm = take(DIM);
        L.wq        = mat(DIM, DIM);
        L.wk        = mat(KV_DIM, DIM);
        L.wv        = mat(KV_DIM, DIM);
        L.wo        = mat(DIM, DIM);
        L.ffn_norm  = take(DIM);
        L.w_gate    = mat(HIDDEN, DIM);
        L.w_up      = mat(HIDDEN, DIM);
        L.w_down    = mat(DIM, HIDDEN);
    }
    w_.final_norm = take(DIM);
    w_.lm_head    = mat(VOCAB, DIM);
    constexpr size_t T = PREFILL_CHUNK;
    xq_.resize(T * HIDDEN);
    xd_.resize(xq_.size() / QK);
    xs_.resize(xd_.size() * 8);

    x_.resize(T * DIM); xb_.resize(T * DIM); xb2_.resize(T * DIM); q_.resize(T * DIM);
    att_.resize(T * N_HEADS * MAX_SEQ);
    hb_.resize(T * HIDDEN); hb2_.resize(T * HIDDEN); logits_.resize(VOCAB);
    k_tmp_.resize(T * KV_DIM); v_tmp_.resize(T * KV_DIM);
    if (kv_q8_) {
        k8_.resize((size_t)N_LAYERS * MAX_SEQ * KV_DIM);
        v8_.resize((size_t)N_LAYERS * MAX_SEQ * KV_DIM);
        ks_.resize((size_t)N_LAYERS * MAX_SEQ * N_KV_HEADS);
        vs_.resize((size_t)N_LAYERS * MAX_SEQ * N_KV_HEADS);
    } else {
        k_rot_.resize((size_t)N_LAYERS * MAX_SEQ * KV_DIM);
        v_work_.resize((size_t)N_LAYERS * MAX_SEQ * KV_DIM);
    }
    seq_ = cache_->begin_sequence();

    constexpr int half = HEAD_DIM / 2;
    rope_cos_.resize((size_t)MAX_SEQ * half);
    rope_sin_.resize((size_t)MAX_SEQ * half);
    for (int p = 0; p < MAX_SEQ; p++)
        for (int i = 0; i < half; i++) {
            const double freq = std::pow((double)ROPE_THETA, -2.0 * i / HEAD_DIM);
            rope_cos_[(size_t)p * half + i] = (float)std::cos(p * freq);
            rope_sin_[(size_t)p * half + i] = (float)std::sin(p * freq);
        }
}

Model::~Model() {
    if (cache_) cache_->end_sequence(seq_);
    if (map_) munmap(map_, file_size_);
}

int Model::use_cached_prefix(const int* tokens, int n) {
    return cache_->longest_cached_prefix(seq_, tokens, n);
}

size_t Model::bytes_per_token() const {
    return file_size_ - HEADER_BYTES - (size_t)VOCAB * DIM * sizeof(float) + DIM * sizeof(float);
}

// prefill register tile: 2 rows x 4 tokens measured best (2x4, 3x4, 4x4 within noise; 4x2 slower)
static constexpr int TR = 2, TC = 4;

// Calibration (quantize --wtype q4a): accumulate per-input-channel sum of x^2 for
// matrix m, plus the token count in the last slot.
void Model::record_calib(const float* x, const Mat& m, int n_in, int T) {
    if (!calib_) return;
    auto& acc = (*calib_)[m.w];
    acc.resize(n_in + 1);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n_in; i++) acc[i] += (double)x[(size_t)t * n_in + i] * x[(size_t)t * n_in + i];
    acc[n_in] += T;
}

// Quantise T activation rows of length n_in into xq_/xd_/xs_ (no-op for fp32 weights).
// Every matmul input passes through here, so it also feeds calibration.
void Model::quant_act(const float* x, const Mat& m, int n_in, int T) {
    record_calib(x, m, n_in, T);
    if (m.t == WType::F32) return;
    const int nb = n_in / QK;
    auto quant = [&](int b, int e) {
        for (int t = b; t < e; t++)
            quantize_act(x + (size_t)t * n_in, xq_.data() + (size_t)t * n_in, xd_.data() + t * nb, n_in,
                         xs_.data() + (size_t)t * nb * 8);
    };
    if (T == 1) quant(0, 1); else parallel_for(T, quant);
}

// Rows [b, e) of out = W.x for all T tokens; out row stride (per token) is ldo.
// Expects quant_act() already run for x. Called from inside parallel_for workers, so
// several matrices / epilogues can share one dispatch (see forward_batch).
void Model::mm_rows(float* out, int ldo, const float* x, const Mat& m, int n_in, int b, int e, int T) {
    if (m.t == WType::F32) {
        const float* W = static_cast<const float*>(m.w);
        for (int i = b; i < e; i++)
            for (int t = 0; t < T; t++)
                out[(size_t)t * ldo + i] = dot_f32(W + (size_t)i * n_in, x + (size_t)t * n_in, n_in);
        return;
    }
    const int nb = n_in / QK;
    const int8_t* xq = xq_.data();
    const float* xd = xd_.data();
    const int32_t* xs = xs_.data();
    if (T == 1) {   // decode: one dot product per row
        for (int i = b; i < e; i++)
            out[i] = m.t == WType::Q4 ? dot_q4(static_cast<const BlockQ4*>(m.w) + (size_t)i * nb, xq, xd, xs, nb)
                                      : dot_q8(static_cast<const BlockQ8*>(m.w) + (size_t)i * nb, xq, xd, nb);
        return;
    }
    // prefill: TR rows x TC tokens register tiles; remainders use narrower tiles
    auto rows = [&](auto* W) {
        auto col = [&](auto R, int i) {   // R rows starting at i, all tokens
            constexpr int r = decltype(R)::value;
            const auto* row = W + (size_t)i * nb;
            int t = 0;
            auto at = [&](int t) { return std::make_tuple(xq + (size_t)t * n_in, xd + t * nb, xs + (size_t)t * nb * 8,
                                                         out + (size_t)t * ldo + i); };
            for (; t + TC <= T; t += TC) {
                auto [a, d, c, o] = at(t);
                dot_tile<r, TC>(row, a, d, c, nb, o, ldo);
            }
            for (; t < T; t++) {
                auto [a, d, c, o] = at(t);
                dot_tile<r, 1>(row, a, d, c, nb, o, ldo);
            }
        };
        int i = b;
        for (; i + TR <= e; i += TR) col(std::integral_constant<int, TR>{}, i);
        for (; i < e; i++) col(std::integral_constant<int, 1>{}, i);
    };
    if (m.t == WType::Q4) rows(static_cast<const BlockQ4*>(m.w));
    else rows(static_cast<const BlockQ8*>(m.w));
}

void Model::mm(float* out, const float* x, const Mat& m, int n_in, int n_out, int T) {
    quant_act(x, m, n_in, T);
    parallel_for(n_out, [&](int b, int e) { mm_rows(out, n_out, x, m, n_in, b, e, T); });
}

// x += W.in, with the residual add folded into the matmul epilogue (each worker adds
// its own rows; same x + dot as a separate pass, so results are unchanged).
void Model::mm_residual(float* x, const float* in, const Mat& m, int n_in, int T) {
    float* tmp = xb2_.data();
    quant_act(in, m, n_in, T);
    parallel_for(DIM, [&](int b, int e) {
        mm_rows(tmp, DIM, in, m, n_in, b, e, T);
        for (int t = 0; t < T; t++)
            for (int i = b; i < e; i++) x[(size_t)t * DIM + i] += tmp[(size_t)t * DIM + i];
    });
}

const float* Model::prefill(const int* tokens, int n, int pos) {
    // Prompt tokens already cached for this sequence are skipped entirely; the last
    // one is always run because its logits are needed.
    int i = pos == 0 ? use_cached_prefix(tokens, n) : 0;
    if (i > n - 1) i = n - 1;
    const int cap = (int)cache_->capacity_tokens();
    const float* logits = nullptr;
    while (i < n) {
        int T = n - i < PREFILL_CHUNK ? n - i : PREFILL_CHUNK;
        if (pos + i + T > cap) T = pos + i < cap ? cap - pos - i : 1;   // never straddle a re-base
        logits = forward_batch(tokens + i, T, pos + i);
        i += T;
    }
    return logits;
}

void Model::save_quantized(const std::string& path, WType mats, WType head,
                           const std::map<const void*, std::vector<double>>* act_ms) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    char hdr[HEADER_BYTES] = {};
    const int32_t sizes[8] = {2, DIM, HIDDEN, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB, MAX_SEQ};
    std::memcpy(hdr, "TLM1", 4);
    std::memcpy(hdr + 4, sizes, sizeof(sizes));
    int8_t* types = reinterpret_cast<int8_t*>(hdr + 4 + sizeof(sizes));
    for (int i = 0; i < N_LAYERS * 7; i++) types[i] = (int8_t)mats;
    types[N_LAYERS * 7] = (int8_t)head;
    f.write(hdr, sizeof(hdr));

    auto put = [&](const void* p, size_t n) {
        const size_t pad = (ALIGN - (size_t)f.tellp() % ALIGN) % ALIGN;
        static const char zeros[ALIGN] = {};
        f.write(zeros, pad);
        f.write(static_cast<const char*>(p), n);
    };
    auto put_mat = [&](const Mat& m, int n_out, int n_in, WType t) {
        if (m.t != WType::F32) throw std::runtime_error("quantize needs an fp32 source model");
        const float* w = static_cast<const float*>(m.w);
        std::vector<char> buf(row_bytes(t, n_in) * n_out);
        std::vector<float> imp;   // mean x^2 per input channel, if calibrated
        if (act_ms && t == WType::Q4) {
            auto it = act_ms->find(m.w);
            if (it == act_ms->end()) throw std::runtime_error("matrix missing from calibration");
            for (int i = 0; i < n_in; i++) imp.push_back((float)(it->second[i] / it->second[n_in]));
        }
#pragma omp parallel for schedule(static)
        for (int r = 0; r < n_out; r++) {
            char* dst = buf.data() + row_bytes(t, n_in) * r;
            const float* src = w + (size_t)r * n_in;
            if (t == WType::Q4 && !imp.empty()) quantize_row_q4_weighted(src, reinterpret_cast<BlockQ4*>(dst), n_in, imp.data());
            else if (t == WType::Q4) quantize_row_q4(src, reinterpret_cast<BlockQ4*>(dst), n_in);
            else if (t == WType::Q8) quantize_row_q8(src, reinterpret_cast<BlockQ8*>(dst), n_in);
            else std::memcpy(dst, src, n_in * sizeof(float));
        }
        put(buf.data(), buf.size());
    };
    put(w_.embed, (size_t)VOCAB * DIM * sizeof(float));
    for (const auto& L : w_.layers) {
        put(L.attn_norm, DIM * sizeof(float));
        put_mat(L.wq, DIM, DIM, mats);
        put_mat(L.wk, KV_DIM, DIM, mats);
        put_mat(L.wv, KV_DIM, DIM, mats);
        put_mat(L.wo, DIM, DIM, mats);
        put(L.ffn_norm, DIM * sizeof(float));
        put_mat(L.w_gate, HIDDEN, DIM, mats);
        put_mat(L.w_up, HIDDEN, DIM, mats);
        put_mat(L.w_down, DIM, HIDDEN, mats);
    }
    put(w_.final_norm, DIM * sizeof(float));
    put_mat(w_.lm_head, VOCAB, DIM, head);
    if (!f) throw std::runtime_error("write failed: " + path);
}

const float* Model::forward_batch(const int* tokens, int T, int pos0) {
    float* x = x_.data();
    float* xb = xb_.data();
    float* q = q_.data();
    float* hb = hb_.data();
    float* hb2 = hb2_.data();
    constexpr int half = HEAD_DIM / 2;
    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    // Cache-relative position (RoPE re-basing, StreamingLLM). Below the cache
    // capacity a token sits at its own position; a streaming backend past its
    // window puts the newest token at the last retained slot. prefill() feeds
    // tokens one at a time in that region, so a batch never straddles it.
    const int cap = (int)cache_->capacity_tokens();
    const bool rebased = pos0 >= cap;
    // re-based slots or re-stored positions (ppl rewinds) invalidate the working copy
    if (rebased) work_valid_ = 0;
    else if (pos0 < work_valid_) work_valid_ = pos0;

    for (int t = 0; t < T; t++)
        std::memcpy(x + (size_t)t * DIM, w_.embed + (size_t)tokens[t] * DIM, DIM * sizeof(float));

    int n_ctx = 0;   // slots the newest token attends to
    for (int l = 0; l < N_LAYERS; l++) {
        const LayerWeights& L = w_.layers[l];
        float* kw = kv_q8_ ? nullptr : k_rot_.data() + (size_t)l * MAX_SEQ * KV_DIM;
        float* vw = kv_q8_ ? nullptr : v_work_.data() + (size_t)l * MAX_SEQ * KV_DIM;
        int8_t* k8 = kv_q8_ ? k8_.data() + (size_t)l * MAX_SEQ * KV_DIM : nullptr;
        int8_t* v8 = kv_q8_ ? v8_.data() + (size_t)l * MAX_SEQ * KV_DIM : nullptr;
        float* ks = kv_q8_ ? ks_.data() + (size_t)l * MAX_SEQ * N_KV_HEADS : nullptr;
        float* vs = kv_q8_ ? vs_.data() + (size_t)l * MAX_SEQ * N_KV_HEADS : nullptr;

        // ---- attention ----
        for (int t = 0; t < T; t++) rmsnorm(xb + (size_t)t * DIM, x + (size_t)t * DIM, L.attn_norm, DIM);
        // Q, K and V in one dispatch over their concatenated rows, sharing one activation
        // quantisation (all three take xb). Each row's dot product is unchanged.
        quant_act(xb, L.wq, DIM, T);
        record_calib(xb, L.wk, DIM, T);   // fused: wk/wv share wq's input
        record_calib(xb, L.wv, DIM, T);
        parallel_for(DIM + 2 * KV_DIM, [&](int b, int e) {
            auto seg = [&](float* out, int ld, const Mat& m, int lo) {   // rows [lo, lo+ld) of the concat
                const int sb = b > lo ? b - lo : 0, se = e - lo < ld ? e - lo : ld;
                if (sb < se) mm_rows(out, ld, xb, m, DIM, sb, se, T);
            };
            seg(q, DIM, L.wq, 0);
            seg(k_tmp_.data(), KV_DIM, L.wk, DIM);
            seg(v_tmp_.data(), KV_DIM, L.wv, DIM + KV_DIM);
        });
        for (int t = 0; t < T; t++) {
            const int qpos = rebased ? cap - 1 : pos0 + t;
            for (int h = 0; h < N_HEADS; h++)
                rope(q + (size_t)t * DIM + h * HEAD_DIM, &rope_cos_[(size_t)qpos * half],
                     &rope_sin_[(size_t)qpos * half]);
            // K is stored UNROTATED (the cache may re-base it); rotation is applied below
            cache_->store(seq_, l, pos0 + t, tokens[t], k_tmp_.data() + (size_t)t * KV_DIM,
                          v_tmp_.data() + (size_t)t * KV_DIM);
        }

        // Bring the working copy up to date: only slots not yet copied.
        const int stored = cache_->stored_tokens(seq_);
        n_ctx = stored < pos0 + T ? stored : pos0 + T;
        for (int sl = work_valid_; sl < n_ctx && kv_q8_; sl++) {
            // rotate in a scratch row, then quantise each head row to int8
            float kr[KV_DIM];
            std::memcpy(kr, cache_->key_row(seq_, l, sl), KV_DIM * sizeof(float));
            const float* vr = cache_->value_row(seq_, l, sl);
            const size_t o = (size_t)sl * KV_DIM, so = (size_t)sl * N_KV_HEADS;
            for (int h = 0; h < N_KV_HEADS; h++) {
                rope(kr + h * HEAD_DIM, &rope_cos_[(size_t)sl * half], &rope_sin_[(size_t)sl * half]);
                quantize_head_q8(kr + h * HEAD_DIM, k8 + o + h * HEAD_DIM, ks + so + h);
                quantize_head_q8(vr + h * HEAD_DIM, v8 + o + h * HEAD_DIM, vs + so + h);
            }
        }
        for (int sl = work_valid_; sl < n_ctx && !kv_q8_; sl++) {
            float* kr = kw + (size_t)sl * KV_DIM;
            std::memcpy(kr, cache_->key_row(seq_, l, sl), KV_DIM * sizeof(float));
            std::memcpy(vw + (size_t)sl * KV_DIM, cache_->value_row(seq_, l, sl), KV_DIM * sizeof(float));
            for (int h = 0; h < N_KV_HEADS; h++)
                rope(kr + h * HEAD_DIM, &rope_cos_[(size_t)sl * half], &rope_sin_[(size_t)sl * half]);
        }

        // causal: token t sees slots 0..pos0+t (all retained slots once re-based)
        parallel_for(T * N_HEADS, [&](int b, int e) {
            for (int idx = b; idx < e; idx++) {
                const int t = idx / N_HEADS, h = idx % N_HEADS;
                const int n = rebased ? n_ctx : pos0 + t + 1;
                const int kvh = h / KV_MUL;
                if (kv_q8_) {
                    attend_head_q8(xb + (size_t)t * DIM + h * HEAD_DIM, q + (size_t)t * DIM + h * HEAD_DIM,
                                   k8 + kvh * HEAD_DIM, ks + kvh, v8 + kvh * HEAD_DIM, vs + kvh,
                                   KV_DIM, N_KV_HEADS, n, scale, att_.data() + (size_t)idx * MAX_SEQ);
                    continue;
                }
                attend_head(xb + (size_t)t * DIM + h * HEAD_DIM, q + (size_t)t * DIM + h * HEAD_DIM,
                            kw + kvh * HEAD_DIM, vw + kvh * HEAD_DIM, KV_DIM, n, scale,
                            att_.data() + (size_t)idx * MAX_SEQ);
            }
        });
        mm_residual(x, xb, L.wo, DIM, T);

        // ---- SwiGLU feed-forward ----
        for (int t = 0; t < T; t++) rmsnorm(xb + (size_t)t * DIM, x + (size_t)t * DIM, L.ffn_norm, DIM);
        // gate and up in one dispatch: each worker computes matching gate/up rows and
        // applies SwiGLU in the epilogue (no separate silu pass, one barrier fewer).
        quant_act(xb, L.w_gate, DIM, T);
        record_calib(xb, L.w_up, DIM, T);   // fused: w_up shares w_gate's input
        parallel_for(HIDDEN, [&](int b, int e) {
            mm_rows(hb, HIDDEN, xb, L.w_gate, DIM, b, e, T);
            mm_rows(hb2, HIDDEN, xb, L.w_up, DIM, b, e, T);
            for (int t = 0; t < T; t++)
                for (int i = b; i < e; i++) {
                    const size_t k = (size_t)t * HIDDEN + i;
                    hb[k] = silu(hb[k]) * hb2[k];
                }
        });
        mm_residual(x, hb, L.w_down, HIDDEN, T);
    }

    work_valid_ = rebased ? 0 : n_ctx;
    float* last = x + (size_t)(T - 1) * DIM;
    rmsnorm(last, last, w_.final_norm, DIM);
    mm(logits_.data(), last, w_.lm_head, DIM, VOCAB, 1);
    return logits_.data();
}

}  // namespace tl
