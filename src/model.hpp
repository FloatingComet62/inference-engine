#pragma once
#include "config.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace tl {

struct LayerWeights {
    const float *attn_norm, *wq, *wk, *wv, *wo;
    const float *ffn_norm, *w_gate, *w_up, *w_down;
};

struct Weights {
    const float* embed;            // [VOCAB][DIM]
    LayerWeights layers[N_LAYERS];
    const float* final_norm;       // [DIM]
    const float* lm_head;          // [VOCAB][DIM]
};

class Model {
public:
    explicit Model(const std::string& path);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Runs one token at position `pos`; returns logits[VOCAB].
    const float* forward(int token, int pos);

    // Bytes of weights streamed from RAM per generated token
    // (everything except the embedding table, of which only one row is read).
    size_t bytes_per_token() const;

private:
    void* map_ = nullptr;
    size_t file_size_ = 0;
    Weights w_{};

    std::vector<float> x_, xb_, xb2_, q_, att_, hb_, hb2_, logits_;
    std::vector<float> key_cache_, value_cache_;   // [N_LAYERS][MAX_SEQ][KV_DIM]
    std::vector<float> rope_cos_, rope_sin_;       // [MAX_SEQ][HEAD_DIM/2]
};

}  // namespace tl
