// Attention kernels: one (token, head) of causal attention over the fp32 KV cache.
#pragma once
#include "config.hpp"
#include "kernels_q4.hpp"   // dot_u8s8
#include <cmath>
#include <cstdint>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace tl {

#ifdef __AVX2__
// exp(x) for 8 floats: Cephes-style range reduction x = n*ln2 + r (two-part ln2),
// degree-6 minimax polynomial on |r| <= ln2/2. Max relative error ~2e-7 (about
// 1-2 ulp) over [-87, 88]; inputs below -87 are clamped (result ~1e-38, i.e. 0
// for softmax purposes).
inline __m256 exp256(__m256 x) {
    x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.0f)), _mm256_set1_ps(-87.0f));
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
#endif

// out[0..HEAD_DIM) = softmax(scale * q.K[0..n)) . V[0..n)
// k, v point at this KV head's slice of row 0; rows are `stride` floats apart.
// att is scratch of >= n floats.
inline void attend_head(float* __restrict out, const float* __restrict q,
                        const float* k, const float* v, size_t stride, int n,
                        float scale, float* __restrict att) {
    static_assert(HEAD_DIM % 16 == 0, "kernel assumes HEAD_DIM is a multiple of 16");
#ifdef __AVX2__
    constexpr int NV = HEAD_DIM / 8;  // 8 ymm registers for HEAD_DIM = 64
    __m256 qv[NV];
    for (int j = 0; j < NV; j++) qv[j] = _mm256_mul_ps(_mm256_loadu_ps(q + 8 * j), _mm256_set1_ps(scale));
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    for (int p = 0; p < n; p++) {
        const float* kt = k + p * stride;
        __m256 a0 = _mm256_mul_ps(qv[0], _mm256_loadu_ps(kt));
        __m256 a1 = _mm256_mul_ps(qv[1], _mm256_loadu_ps(kt + 8));
        for (int j = 2; j < NV; j += 2) {
            a0 = _mm256_fmadd_ps(qv[j], _mm256_loadu_ps(kt + 8 * j), a0);
            a1 = _mm256_fmadd_ps(qv[j + 1], _mm256_loadu_ps(kt + 8 * j + 8), a1);
        }
        const float s = hsum256(_mm256_add_ps(a0, a1));
        att[p] = s;
        vmax = _mm256_max_ps(vmax, _mm256_set1_ps(s));
    }
    // vmax is a broadcast of the max
    __m256 vsum = _mm256_setzero_ps();
    int p = 0;
    for (; p + 8 <= n; p += 8) {
        __m256 e = exp256(_mm256_sub_ps(_mm256_loadu_ps(att + p), vmax));
        _mm256_storeu_ps(att + p, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    float sum = hsum256(vsum);
    const float mx = _mm256_cvtss_f32(vmax);
    for (; p < n; p++) { att[p] = std::exp(att[p] - mx); sum += att[p]; }
    __m256 o[NV];
    for (int j = 0; j < NV; j++) o[j] = _mm256_setzero_ps();
    for (p = 0; p < n; p++) {
        const float* vt = v + p * stride;
        const __m256 a = _mm256_broadcast_ss(att + p);
        for (int j = 0; j < NV; j++) o[j] = _mm256_fmadd_ps(a, _mm256_loadu_ps(vt + 8 * j), o[j]);
    }
    const __m256 inv = _mm256_set1_ps(1.0f / sum);
    for (int j = 0; j < NV; j++) _mm256_storeu_ps(out + 8 * j, _mm256_mul_ps(o[j], inv));
#else
    float mx = -INFINITY;
    for (int p = 0; p < n; p++) {
        const float* kt = k + p * stride;
        float s = 0.0f;
        for (int i = 0; i < HEAD_DIM; i++) s += q[i] * kt[i];
        att[p] = s * scale;
        mx = att[p] > mx ? att[p] : mx;
    }
    float sum = 0.0f;
    for (int p = 0; p < n; p++) { att[p] = std::exp(att[p] - mx); sum += att[p]; }
    const float inv = 1.0f / sum;
    std::memset(out, 0, HEAD_DIM * sizeof(float));
    for (int p = 0; p < n; p++) {
        const float* vt = v + p * stride;
        const float a = att[p] * inv;
        for (int i = 0; i < HEAD_DIM; i++) out[i] += a * vt[i];
    }
#endif
}

// ---------------------------------------------------------------- int8 KV (--kv q8)
// One KV head row (HEAD_DIM values) is stored as int8 with one fp32 scale: x ~= d * q.
inline void quantize_head_q8(const float* x, int8_t* q, float* d) {
    float amax = 0.0f;
    for (int i = 0; i < HEAD_DIM; i++) amax = std::fmax(amax, std::fabs(x[i]));
    const float s = amax / 127.0f, is = s ? 1.0f / s : 0.0f;
    for (int i = 0; i < HEAD_DIM; i++) q[i] = (int8_t)std::lrintf(x[i] * is);
    *d = s;
}

// Same contract as attend_head, with int8 K/V: k, v point at this head's row 0
// (rows `stride` bytes apart), ks, vs at its scale for row 0 (`sstride` floats apart).
// q is quantised to int8 once, q.k is an integer dot (u8 x s8 via the sign trick),
// and values are accumulated as (p * v_scale) * int8 in fp32.
inline void attend_head_q8(float* __restrict out, const float* __restrict q,
                           const int8_t* k, const float* ks, const int8_t* v, const float* vs,
                           size_t stride, size_t sstride, int n, float scale,
                           float* __restrict att) {
    int8_t qi[HEAD_DIM];
    float qd;
    quantize_head_q8(q, qi, &qd);
    qd *= scale;
    float mx = -INFINITY;
#ifdef __AVX2__
    if constexpr (HEAD_DIM == 64) {   // SIMD path: two 32-byte halves per row
    const __m256i q0 = _mm256_loadu_si256((const __m256i*)qi);
    const __m256i q1 = _mm256_loadu_si256((const __m256i*)(qi + 32));
    const __m256i aq0 = _mm256_sign_epi8(q0, q0), aq1 = _mm256_sign_epi8(q1, q1);   // |q| as u8
    for (int p = 0; p < n; p++) {
        const int8_t* kt = k + p * stride;
        const __m256i k0 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)kt), q0);
        const __m256i k1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(kt + 32)), q1);
        const __m256i acc = dot_u8s8(aq1, k1, dot_u8s8(aq0, k0));
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        s4 = _mm_add_epi32(s4, _mm_unpackhi_epi64(s4, s4));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 1));
        const float s = (float)_mm_cvtsi128_si32(s4) * qd * ks[p * sstride];
        att[p] = s;
        mx = s > mx ? s : mx;
    }
    const __m256 vmax = _mm256_set1_ps(mx);
    __m256 vsum = _mm256_setzero_ps();
    int p = 0;
    for (; p + 8 <= n; p += 8) {
        __m256 e = exp256(_mm256_sub_ps(_mm256_loadu_ps(att + p), vmax));
        _mm256_storeu_ps(att + p, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    float sum = hsum256(vsum);
    for (; p < n; p++) { att[p] = std::exp(att[p] - mx); sum += att[p]; }
    constexpr int NV = HEAD_DIM / 8;
    __m256 o[NV];
    for (int j = 0; j < NV; j++) o[j] = _mm256_setzero_ps();
    for (p = 0; p < n; p++) {
        const int8_t* vt = v + p * stride;
        const __m256 a = _mm256_set1_ps(att[p] * vs[p * sstride]);
        for (int j = 0; j < NV; j++) {
            const __m256 x = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(vt + 8 * j))));
            o[j] = _mm256_fmadd_ps(a, x, o[j]);
        }
    }
    const __m256 inv = _mm256_set1_ps(1.0f / sum);
    for (int j = 0; j < NV; j++) _mm256_storeu_ps(out + 8 * j, _mm256_mul_ps(o[j], inv));
    return;
    }
#endif
    for (int p = 0; p < n; p++) {
        const int8_t* kt = k + p * stride;
        int32_t s = 0;
        for (int i = 0; i < HEAD_DIM; i++) s += qi[i] * kt[i];
        att[p] = (float)s * qd * ks[p * sstride];
        mx = att[p] > mx ? att[p] : mx;
    }
    float sum = 0.0f;
    for (int p = 0; p < n; p++) { att[p] = std::exp(att[p] - mx); sum += att[p]; }
    const float inv = 1.0f / sum;
    std::memset(out, 0, HEAD_DIM * sizeof(float));
    for (int p = 0; p < n; p++) {
        const int8_t* vt = v + p * stride;
        const float a = att[p] * vs[p * sstride] * inv;
        for (int i = 0; i < HEAD_DIM; i++) out[i] += a * vt[i];
    }
}

}  // namespace tl
