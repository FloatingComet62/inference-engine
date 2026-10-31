#include "model.hpp"
#include "kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
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

Model::Model(const std::string& path, KVCache& cache) : cache_(&cache) {
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
    const int expect[8] = {1, DIM, HIDDEN, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB, MAX_SEQ};
    for (int i = 0; i < 8; i++)
        if (h[i] != expect[i])
            throw std::runtime_error("weight file does not match compiled config (re-run convert)");

    size_t off = HEADER_BYTES;
    auto take = [&](size_t n) -> const float* {
        off = (off + ALIGN - 1) & ~(ALIGN - 1);
        const float* p = reinterpret_cast<const float*>(base + off);
        off += n * sizeof(float);
        if (off > file_size_) throw std::runtime_error("weight file truncated");
        return p;
    };
    w_.embed = take((size_t)VOCAB * DIM);
    for (auto& L : w_.layers) {
        L.attn_norm = take(DIM);
        L.wq        = take((size_t)DIM * DIM);
        L.wk        = take((size_t)KV_DIM * DIM);
        L.wv        = take((size_t)KV_DIM * DIM);
        L.wo        = take((size_t)DIM * DIM);
        L.ffn_norm  = take(DIM);
        L.w_gate    = take((size_t)HIDDEN * DIM);
        L.w_up      = take((size_t)HIDDEN * DIM);
        L.w_down    = take((size_t)DIM * HIDDEN);
    }
    w_.final_norm = take(DIM);
    w_.lm_head    = take((size_t)VOCAB * DIM);

    x_.resize(DIM); xb_.resize(DIM); xb2_.resize(DIM); q_.resize(DIM);
    att_.resize((size_t)N_HEADS * MAX_SEQ);
    hb_.resize(HIDDEN); hb2_.resize(HIDDEN); logits_.resize(VOCAB);
    k_tmp_.resize(KV_DIM); v_tmp_.resize(KV_DIM);
    k_rows_.resize(MAX_SEQ); v_rows_.resize(MAX_SEQ);
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
    matched_ = cache_->longest_cached_prefix(seq_, tokens, n);
    return matched_;
}

size_t Model::bytes_per_token() const {
    return file_size_ - HEADER_BYTES - (size_t)VOCAB * DIM * sizeof(float) + DIM * sizeof(float);
}

const float* Model::forward(int token, int pos) {
    float* x = x_.data();
    float* xb = xb_.data();
    float* xb2 = xb2_.data();
    float* q = q_.data();
    float* hb = hb_.data();
    float* hb2 = hb2_.data();
    const float* cs = &rope_cos_[(size_t)pos * (HEAD_DIM / 2)];
    const float* sn = &rope_sin_[(size_t)pos * (HEAD_DIM / 2)];
    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    std::memcpy(x, w_.embed + (size_t)token * DIM, DIM * sizeof(float));

    for (int l = 0; l < N_LAYERS; l++) {
        const LayerWeights& L = w_.layers[l];

        // ---- attention ----
        rmsnorm(xb, x, L.attn_norm, DIM);
        matmul(q, xb, L.wq, DIM, DIM);
        for (int h = 0; h < N_HEADS; h++)    rope(q + h * HEAD_DIM, cs, sn);
        if (pos >= matched_) {   // cached prefix: rows read back below, not recomputed
            matmul(k_tmp_.data(), xb, L.wk, DIM, KV_DIM);
            matmul(v_tmp_.data(), xb, L.wv, DIM, KV_DIM);
            for (int h = 0; h < N_KV_HEADS; h++) rope(k_tmp_.data() + h * HEAD_DIM, cs, sn);
            cache_->store(seq_, l, pos, token, k_tmp_.data(), v_tmp_.data());
        }

        // Per-position row pointers (single-threaded; the parallel loop only reads).
        for (int t = 0; t <= pos; t++) {
            k_rows_[t] = cache_->key_row(seq_, l, t);
            v_rows_[t] = cache_->value_row(seq_, l, t);
        }

#pragma omp parallel for schedule(static)
        for (int h = 0; h < N_HEADS; h++) {
            const float* qh = q + h * HEAD_DIM;
            float* att = att_.data() + (size_t)h * MAX_SEQ;
            const int kvh = h / KV_MUL;
            for (int t = 0; t <= pos; t++) {
                const float* kt = k_rows_[t] + kvh * HEAD_DIM;
                float s = 0.0f;
                for (int i = 0; i < HEAD_DIM; i++) s += qh[i] * kt[i];
                att[t] = s * scale;
            }
            softmax(att, pos + 1);
            float* out = xb + h * HEAD_DIM;
            std::memset(out, 0, HEAD_DIM * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float* vt = v_rows_[t] + kvh * HEAD_DIM;
                const float a = att[t];
                for (int i = 0; i < HEAD_DIM; i++) out[i] += a * vt[i];
            }
        }
        matmul(xb2, xb, L.wo, DIM, DIM);
        for (int i = 0; i < DIM; i++) x[i] += xb2[i];

        // ---- SwiGLU feed-forward ----
        rmsnorm(xb, x, L.ffn_norm, DIM);
        matmul(hb, xb, L.w_gate, DIM, HIDDEN);
        matmul(hb2, xb, L.w_up, DIM, HIDDEN);
        for (int i = 0; i < HIDDEN; i++) hb[i] = silu(hb[i]) * hb2[i];
        matmul(xb2, hb, L.w_down, HIDDEN, DIM);
        for (int i = 0; i < DIM; i++) x[i] += xb2[i];
    }

    rmsnorm(x, x, w_.final_norm, DIM);
    matmul(logits_.data(), x, w_.lm_head, DIM, VOCAB);
    return logits_.data();
}

}  // namespace tl
