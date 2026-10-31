#include "model.hpp"
#include "kernels.hpp"

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

Model::Model(const std::string& path) {
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
    xq_.resize(HIDDEN > DIM ? HIDDEN : DIM);
    xd_.resize(xq_.size() / QK);

    x_.resize(DIM); xb_.resize(DIM); xb2_.resize(DIM); q_.resize(DIM);
    att_.resize((size_t)N_HEADS * MAX_SEQ);
    hb_.resize(HIDDEN); hb2_.resize(HIDDEN); logits_.resize(VOCAB);
    key_cache_.resize((size_t)N_LAYERS * MAX_SEQ * KV_DIM);
    value_cache_.resize((size_t)N_LAYERS * MAX_SEQ * KV_DIM);

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
    if (map_) munmap(map_, file_size_);
}

size_t Model::bytes_per_token() const {
    return file_size_ - HEADER_BYTES - (size_t)VOCAB * DIM * sizeof(float) + DIM * sizeof(float);
}

void Model::mm(float* out, const float* x, const Mat& m, int n_in, int n_out) {
    if (m.t == WType::F32) return matmul(out, x, static_cast<const float*>(m.w), n_in, n_out);
    quantize_act(x, xq_.data(), xd_.data(), n_in);
    matmul_q(out, xq_.data(), xd_.data(), m.w, m.t, n_in, n_out);
}

void Model::save_quantized(const std::string& path, WType mats, WType head) const {
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
#pragma omp parallel for schedule(static)
        for (int r = 0; r < n_out; r++) {
            char* dst = buf.data() + row_bytes(t, n_in) * r;
            const float* src = w + (size_t)r * n_in;
            if (t == WType::Q4) quantize_row_q4(src, reinterpret_cast<BlockQ4*>(dst), n_in);
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
        float* kcache = key_cache_.data() + (size_t)l * MAX_SEQ * KV_DIM;
        float* vcache = value_cache_.data() + (size_t)l * MAX_SEQ * KV_DIM;
        float* k = kcache + (size_t)pos * KV_DIM;
        float* v = vcache + (size_t)pos * KV_DIM;

        // ---- attention ----
        rmsnorm(xb, x, L.attn_norm, DIM);
        mm(q, xb, L.wq, DIM, DIM);
        mm(k, xb, L.wk, DIM, KV_DIM);
        mm(v, xb, L.wv, DIM, KV_DIM);
        for (int h = 0; h < N_HEADS; h++)    rope(q + h * HEAD_DIM, cs, sn);
        for (int h = 0; h < N_KV_HEADS; h++) rope(k + h * HEAD_DIM, cs, sn);

#pragma omp parallel for schedule(static)
        for (int h = 0; h < N_HEADS; h++) {
            const float* qh = q + h * HEAD_DIM;
            float* att = att_.data() + (size_t)h * MAX_SEQ;
            const int kvh = h / KV_MUL;
            for (int t = 0; t <= pos; t++) {
                const float* kt = kcache + (size_t)t * KV_DIM + kvh * HEAD_DIM;
                float s = 0.0f;
                for (int i = 0; i < HEAD_DIM; i++) s += qh[i] * kt[i];
                att[t] = s * scale;
            }
            softmax(att, pos + 1);
            float* out = xb + h * HEAD_DIM;
            std::memset(out, 0, HEAD_DIM * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float* vt = vcache + (size_t)t * KV_DIM + kvh * HEAD_DIM;
                const float a = att[t];
                for (int i = 0; i < HEAD_DIM; i++) out[i] += a * vt[i];
            }
        }
        mm(xb2, xb, L.wo, DIM, DIM);
        for (int i = 0; i < DIM; i++) x[i] += xb2[i];

        // ---- SwiGLU feed-forward ----
        rmsnorm(xb, x, L.ffn_norm, DIM);
        mm(hb, xb, L.w_gate, DIM, HIDDEN);
        mm(hb2, xb, L.w_up, DIM, HIDDEN);
        for (int i = 0; i < HIDDEN; i++) hb[i] = silu(hb[i]) * hb2[i];
        mm(xb2, hb, L.w_down, HIDDEN, DIM);
        for (int i = 0; i < DIM; i++) x[i] += xb2[i];
    }

    rmsnorm(x, x, w_.final_norm, DIM);
    mm(logits_.data(), x, w_.lm_head, DIM, VOCAB);
    return logits_.data();
}

}  // namespace tl
