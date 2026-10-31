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

#ifdef __AVX2__
// unsigned u8 x signed s8 -> 8 x int32 added to acc: one instruction with AVX-VNNI, two without
inline __m256i dot_u8s8(__m256i aw, __m256i sx, __m256i acc = _mm256_setzero_si256()) {
#ifdef __AVXVNNI__
    return _mm256_dpbusd_avx_epi32(acc, aw, sx);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), _mm256_set1_epi16(1)));
#endif
}
#endif

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
// xs (optional): per block 8 x int32, lane j = -8 * (x[4j]+..+x[4j+3]). It is the correction
// that turns an unsigned-nibble dot (0..15) into the signed (q-8) dot, lane for lane, exactly.
inline void quantize_act(const float* x, int8_t* xq, float* xd, int n, int32_t* xs = nullptr) {
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
        if (xs) _mm256_storeu_si256((__m256i*)(xs + b * 8),
                                    _mm256_sub_epi32(_mm256_setzero_si256(), dot_u8s8(_mm256_set1_epi8(8), p)));
    }
#else
    for (int b = 0; b < n / QK; b++) {
        const float* v = x + b * QK;
        float amax = 0.0f;
        for (int i = 0; i < QK; i++) amax = std::fmax(amax, std::fabs(v[i]));
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        xd[b] = d;
        for (int i = 0; i < QK; i++) xq[b * QK + i] = (int8_t)std::nearbyint(v[i] * id);
        if (xs)
            for (int j = 0; j < 8; j++) {
                const int8_t* g = xq + b * QK + 4 * j;
                xs[b * 8 + j] = -8 * (g[0] + g[1] + g[2] + g[3]);
            }
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
inline __m256 dot_i8(__m256i w, __m256i x) {
    return _mm256_cvtepi32_ps(dot_u8s8(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)));
}
#endif

#ifdef __AVX2__
// Q4 nibbles as unsigned 0..15 (no -8, no sign trick)
inline __m256i unpack_q4u(const BlockQ4& blk) {
    const __m128i raw = _mm_loadu_si128((const __m128i*)blk.qs);
    return _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw), _mm256_set1_epi8(0x0F));
}
#endif

// xs: per-block correction from quantize_act. The integer lane sums equal the signed (q-8).x
// dot exactly, so results are bit-identical to the tiled prefill kernel below.
inline float dot_q4(const BlockQ4* w, const int8_t* xq, const float* xd, const int32_t* xs, int nb) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + b * QK));
        const __m256i c = _mm256_loadu_si256((const __m256i*)(xs + b * 8));
        const __m256 p = _mm256_cvtepi32_ps(dot_u8s8(unpack_q4u(w[b]), xv, c));
        acc = _mm256_fmadd_ps(_mm256_set1_ps((float)w[b].d * xd[b]), p, acc);
    }
    return hsum(acc);
#else
    (void)xs;
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

// Register tile: R weight rows x C tokens. Each weight block is unpacked once and used C times,
// each activation block is loaded once and used R times. Per (row, token) the arithmetic is the
// same as dot_q4 / dot_q8 (same int lanes, same fmadd order, same hsum), so batched and
// sequential paths give bit-identical results.
// out[c * out_stride + r] = row_r . x_c, rows at w + r * nb, tokens at xq + c * nb * QK.
template <int R, int C, class Blk>
inline void dot_tile(const Blk* w, const int8_t* xq, const float* xd, const int32_t* xs, int nb,
                     float* out, size_t out_stride) {
    constexpr bool q4 = sizeof(Blk) == sizeof(BlockQ4);
#ifdef __AVX2__
    const size_t n_in = (size_t)nb * QK;
    __m256 acc[R][C];
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) acc[r][c] = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i wv[R];
        __m256 wd[R];
        for (int r = 0; r < R; r++) {
            const Blk& blk = w[(size_t)r * nb + b];
            if constexpr (q4) wv[r] = unpack_q4u(blk);
            else wv[r] = _mm256_loadu_si256((const __m256i*)blk.qs);
            wd[r] = _mm256_set1_ps((float)blk.d);
        }
        for (int c = 0; c < C; c++) {
            const __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + c * n_in + b * QK));
            const __m256 d = _mm256_broadcast_ss(xd + (size_t)c * nb + b);
            if constexpr (q4) {
                const __m256i corr = _mm256_loadu_si256((const __m256i*)(xs + ((size_t)c * nb + b) * 8));
                for (int r = 0; r < R; r++)
                    acc[r][c] = _mm256_fmadd_ps(_mm256_mul_ps(wd[r], d),
                                                _mm256_cvtepi32_ps(dot_u8s8(wv[r], xv, corr)), acc[r][c]);
            } else {
                for (int r = 0; r < R; r++)
                    acc[r][c] = _mm256_fmadd_ps(_mm256_mul_ps(wd[r], d), dot_i8(wv[r], xv), acc[r][c]);
            }
        }
    }
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) out[c * out_stride + r] = hsum(acc[r][c]);
#else
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) {
            const int8_t* xk = xq + (size_t)c * nb * QK;
            const float* dk = xd + (size_t)c * nb;
            if constexpr (q4) out[c * out_stride + r] = dot_q4((const BlockQ4*)w + (size_t)r * nb, xk, dk, xs + (size_t)c * nb * 8, nb);
            else out[c * out_stride + r] = dot_q8((const BlockQ8*)w + (size_t)r * nb, xk, dk, nb);
        }
#endif
}

}  // namespace tl
