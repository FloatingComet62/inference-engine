// Quantised kernels: Q4_32 / Q8_32 weights x Q8_32 activations (Q4_0-style).
//   Q4 block: fp16 scale + 16 bytes; element i in low nibble of byte i, i+16 in the high nibble,
//             stored as q+8 with q in [-8, 7]                          -> 4.5 bits per weight
//   Q8 block: fp16 scale + 32 int8                                       -> 8.5 bits per weight
// Activations are quantised to int8 once per matmul, one fp32 scale per 32 values.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace tl {

constexpr int QK = 32;   // weights per block
struct BlockQ4 { _Float16 d; uint8_t qs[QK / 2]; };
struct BlockQ8 { _Float16 d; int8_t qs[QK]; };
static_assert(sizeof(BlockQ4) == 18 && sizeof(BlockQ8) == 34, "block layout");

enum class WType : int32_t { F32 = 0, Q4 = 1, Q8 = 2 };

inline size_t row_bytes(WType t, int n_in) {
    switch (t) {
        case WType::Q4: return (size_t)(n_in / QK) * sizeof(BlockQ4);
        case WType::Q8: return (size_t)(n_in / QK) * sizeof(BlockQ8);
        default:        return (size_t)n_in * sizeof(float);
    }
}

// ---------------------------------------------------------------- quantisers
inline void quantize_row_q4(const float* x, BlockQ4* out, int n) {
    for (int b = 0; b < n / QK; b++, x += QK) {
        float amax = 0.0f;
        for (int i = 0; i < QK; i++) amax = std::fmax(amax, std::fabs(x[i]));
        const float d = amax / 7.0f, id = d ? 1.0f / d : 0.0f;
        out[b].d = (_Float16)d;
        for (int i = 0; i < QK / 2; i++) {
            auto q = [&](float v) { int r = (int)std::lround(v * id); return (uint8_t)((r < -8 ? -8 : r > 7 ? 7 : r) + 8); };
            out[b].qs[i] = q(x[i]) | (uint8_t)(q(x[i + QK / 2]) << 4);
        }
    }
}

inline void quantize_row_q8(const float* x, BlockQ8* out, int n) {
    for (int b = 0; b < n / QK; b++, x += QK) {
        float amax = 0.0f;
        for (int i = 0; i < QK; i++) amax = std::fmax(amax, std::fabs(x[i]));
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        out[b].d = (_Float16)d;
        for (int i = 0; i < QK; i++) out[b].qs[i] = (int8_t)std::lround(x[i] * id);
    }
}

// Activations: int8 values + one fp32 scale per block.
inline void quantize_act(const float* x, int8_t* xq, float* xd, int n) {
    for (int b = 0; b < n / QK; b++) {
        const float* v = x + b * QK;
        float amax = 0.0f;
        for (int i = 0; i < QK; i++) amax = std::fmax(amax, std::fabs(v[i]));
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        xd[b] = d;
        for (int i = 0; i < QK; i++) xq[b * QK + i] = (int8_t)std::lround(v[i] * id);
    }
}

// ---------------------------------------------------------------- dot products
#ifdef __AVX2__
inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
// signed int8 w . signed int8 x -> 8 x int32 (|w| * sign-transferred x, no int16 overflow for |w|,|x| <= 127)
inline __m256 dot_i8(__m256i w, __m256i x) {
    const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, _mm256_set1_epi16(1)));
}
#endif

inline float dot_q4(const BlockQ4* w, const int8_t* xq, const float* xd, int nb) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    const __m256i lo = _mm256_set1_epi8(0x0F), eight = _mm256_set1_epi8(8);
    for (int b = 0; b < nb; b++) {
        const __m128i raw = _mm_loadu_si128((const __m128i*)w[b].qs);
        __m256i wv = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw), lo);
        wv = _mm256_sub_epi8(wv, eight);
        const __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + b * QK));
        acc = _mm256_fmadd_ps(_mm256_set1_ps((float)w[b].d * xd[b]), dot_i8(wv, xv), acc);
    }
    return hsum(acc);
#else
    float s = 0.0f;
    for (int b = 0; b < nb; b++) {
        int isum = 0;
        for (int i = 0; i < QK / 2; i++) {
            isum += ((w[b].qs[i] & 0x0F) - 8) * xq[b * QK + i];
            isum += ((w[b].qs[i] >> 4) - 8) * xq[b * QK + i + QK / 2];
        }
        s += (float)w[b].d * xd[b] * isum;
    }
    return s;
#endif
}

inline float dot_q8(const BlockQ8* w, const int8_t* xq, const float* xd, int nb) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const __m256i wv = _mm256_loadu_si256((const __m256i*)w[b].qs);
        const __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + b * QK));
        acc = _mm256_fmadd_ps(_mm256_set1_ps((float)w[b].d * xd[b]), dot_i8(wv, xv), acc);
    }
    return hsum(acc);
#else
    float s = 0.0f;
    for (int b = 0; b < nb; b++) {
        int isum = 0;
        for (int i = 0; i < QK; i++) isum += w[b].qs[i] * xq[b * QK + i];
        s += (float)w[b].d * xd[b] * isum;
    }
    return s;
#endif
}

// out[n_out] = W . x for quantised W; xq/xd hold x already quantised by quantize_act.
inline void matmul_q(float* __restrict out, const int8_t* xq, const float* xd,
                     const void* W, WType t, int n_in, int n_out) {
    const int nb = n_in / QK;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_out; i++) {
        out[i] = t == WType::Q4 ? dot_q4((const BlockQ4*)W + (size_t)i * nb, xq, xd, nb)
                                : dot_q8((const BlockQ8*)W + (size_t)i * nb, xq, xd, nb);
    }
}

}  // namespace tl
