#pragma once
#include "config.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace tl {

class Sampler {
public:
    Sampler(float temperature, float top_p, uint64_t seed)
        : temp_(temperature), top_p_(top_p), rng_(seed) {}

    int sample(const float* logits) {
        if (temp_ <= 0.0f)   // greedy: used by the benchmark so runs are deterministic
            return (int)(std::max_element(logits, logits + VOCAB) - logits);

        probs_.resize(VOCAB);
        const float mx = *std::max_element(logits, logits + VOCAB);
        double sum = 0.0;
        for (int i = 0; i < VOCAB; i++) { probs_[i] = std::exp((logits[i] - mx) / temp_); sum += probs_[i]; }
        for (auto& p : probs_) p = (float)(p / sum);
        const float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_);

        if (top_p_ <= 0.0f || top_p_ >= 1.0f) {
            float acc = 0.0f;
            for (int i = 0; i < VOCAB; i++) { acc += probs_[i]; if (r < acc) return i; }
            return VOCAB - 1;
        }

        // top-p (nucleus): drop tiny probabilities first so the sort stays cheap
        const float cutoff = (1.0f - top_p_) / (VOCAB - 1);
        cand_.clear();
        for (int i = 0; i < VOCAB; i++)
            if (probs_[i] >= cutoff) cand_.push_back({probs_[i], i});
        std::sort(cand_.begin(), cand_.end(), [](const Cand& a, const Cand& b) { return a.p > b.p; });
        float cum = 0.0f;
        size_t last = cand_.size() - 1;
        for (size_t k = 0; k < cand_.size(); k++) {
            cum += cand_[k].p;
            if (cum > top_p_) { last = k; break; }
        }
        const float target = r * cum;
        float acc = 0.0f;
        for (size_t k = 0; k <= last; k++) { acc += cand_[k].p; if (target < acc) return cand_[k].id; }
        return cand_[last].id;
    }

private:
    struct Cand { float p; int id; };
    float temp_, top_p_;
    std::mt19937_64 rng_;
    std::vector<float> probs_;
    std::vector<Cand> cand_;
};

}  // namespace tl
