// Maths primitives. The matmuls are ~95% of runtime: this file is where most of
// the hackathon optimisation happens (4-bit weights, int8 dot products, AVX2).
#pragma once
#include "config.hpp"
#include <cmath>
#include <cstddef>

namespace tl {

// dot product of one fp32 weight row with x
inline float dot_f32(const float* __restrict row, const float* __restrict x, int n) {
    float s = 0.0f;
    for (int j = 0; j < n; j++) s += row[j] * x[j];
    return s;
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
