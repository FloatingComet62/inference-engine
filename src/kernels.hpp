// Maths primitives. matmul() is ~95% of runtime: this file is where most of
// the hackathon optimisation happens (4-bit weights, int8 dot products, AVX2).
#pragma once
#include "config.hpp"
#include <cmath>
#include <cstddef>

namespace tl {

// out[n_out] = W[n_out x n_in] . x[n_in]   (W row-major, as stored by Hugging Face)
inline void matmul(float* __restrict out, const float* __restrict x,
                   const float* __restrict W, int n_in, int n_out) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_out; i++) {
        const float* row = W + (size_t)i * n_in;
        float s = 0.0f;
        for (int j = 0; j < n_in; j++) s += row[j] * x[j];
        out[i] = s;
    }
}

inline void rmsnorm(float* out, const float* x, const float* w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / std::sqrt(ss / n + RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = w[i] * (ss * x[i]);
}

inline void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) mx = x[i] > mx ? x[i] : mx;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

}  // namespace tl
