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

// Activations: int8 values + one fp32 scale per block. Runs before every matmul, so it is
// vectorised (round-to-nearest-even; the scalar path matches it with nearbyint).
inline void quantize_act(const float* x, int8_t* xq, float* xd, int n) {
#ifdef __AVX2__
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int b = 0; b < n / QK; b++) {
        const float* v = x + b * QK;
        __m256 v0 = _mm256_loadu_ps(v), v1 = _mm256_loadu_ps(v + 8);
        __m256 v2 = _mm256_loadu_ps(v + 16), v3 = _mm256_loadu_ps(v + 24);
        __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                                 _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float d = _mm_cvtss_f32(m4) / 127.0f;
        xd[b] = d;
        const __m256 id = _mm256_set1_ps(d ? 1.0f / d : 0.0f);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, id)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, id));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, id)), i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, id));
        // packs work per 128-bit lane; the final permute restores element order
        __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i*)(xq + b * QK), p);
    }
#else
    for (int b = 0; b < n / QK; b++) {
        const float* v = x + b * QK;
        float amax = 0.0f;
        for (int i = 0; i < QK; i++) amax = std::fmax(amax, std::fabs(v[i]));
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        xd[b] = d;
        for (int i = 0; i < QK; i++) xq[b * QK + i] = (int8_t)std::nearbyint(v[i] * id);
    }
#endif
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
// unsigned |w| x signed x -> 8 x int32: one instruction with AVX-VNNI, two without
inline __m256i dot_u8s8(__m256i aw, __m256i sx) {
#ifdef __AVXVNNI__
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sx);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), _mm256_set1_epi16(1));
#endif
}
inline __m256 dot_i8(__m256i w, __m256i x) {
    return _mm256_cvtepi32_ps(dot_u8s8(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)));
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

// N tokens x one weight row: each weight block is unpacked once and used N times, and the
// N independent accumulators hide the FMA latency. Used by batched prefill.
// out[k] = row . x_k, where x_k = xq + k * n_in (scales xd + k * nb).
template <int N, class Blk>
inline void dot_rows_n(const Blk* w, const int8_t* xq, const float* xd, int nb, float* out, size_t out_stride) {
#ifdef __AVX2__
    const int n_in = nb * QK;
    __m256 acc[N];
    for (int k = 0; k < N; k++) acc[k] = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i wv;
        if constexpr (sizeof(Blk) == sizeof(BlockQ4)) {
            const __m128i raw = _mm_loadu_si128((const __m128i*)w[b].qs);
            wv = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw), _mm256_set1_epi8(0x0F));
            wv = _mm256_sub_epi8(wv, _mm256_set1_epi8(8));
        } else {
            wv = _mm256_loadu_si256((const __m256i*)w[b].qs);
        }
        const __m256i aw = _mm256_sign_epi8(wv, wv);
        const float wd = (float)w[b].d;
        for (int k = 0; k < N; k++) {
            const __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + (size_t)k * n_in + b * QK));
            const __m256 p = _mm256_cvtepi32_ps(dot_u8s8(aw, _mm256_sign_epi8(xv, wv)));
            acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(wd * xd[(size_t)k * nb + b]), p, acc[k]);
        }
    }
    for (int k = 0; k < N; k++) out[k * out_stride] = hsum(acc[k]);
#else
    for (int k = 0; k < N; k++) {
        const int8_t* xk = xq + (size_t)k * nb * QK;
        const float* dk = xd + (size_t)k * nb;
        if constexpr (sizeof(Blk) == sizeof(BlockQ4)) out[k * out_stride] = dot_q4((const BlockQ4*)w, xk, dk, nb);
        else out[k * out_stride] = dot_q8((const BlockQ8*)w, xk, dk, nb);
    }
#endif
}

}  // namespace tl
