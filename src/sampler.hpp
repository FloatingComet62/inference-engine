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
    // repeat_penalty > 1 discourages tokens among the last `repeat_last_n` sampled ones
    // (llama.cpp/Ollama rule: positive logits / penalty, negative logits * penalty).
    Sampler(float temperature, float top_p, uint64_t seed, float repeat_penalty = 1.0f, int repeat_last_n = 64)
        : temp_(temperature), top_p_(top_p), rng_(seed), penalty_(repeat_penalty), last_n_(repeat_last_n) {}

    int sample(const float* logits) {
        if (penalty_ != 1.0f && !recent_.empty()) {
            pen_.assign(logits, logits + VOCAB);
            for (int t : recent_) pen_[t] = pen_[t] > 0.0f ? pen_[t] / penalty_ : pen_[t] * penalty_;
            logits = pen_.data();
        }
        const int tok = pick(logits);
        if (penalty_ != 1.0f) {
            recent_.push_back(tok);
            if ((int)recent_.size() > last_n_) recent_.erase(recent_.begin());
        }
        return tok;
    }

private:
    int pick(const float* logits) {
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

    struct Cand { float p; int id; };
    float temp_, top_p_;
    std::mt19937_64 rng_;
    float penalty_;
    int last_n_;
    std::vector<int> recent_;      // last sampled tokens (duplicates penalised once each time)
    std::vector<float> pen_;
    std::vector<float> probs_;
    std::vector<Cand> cand_;
};

}  // namespace tl
