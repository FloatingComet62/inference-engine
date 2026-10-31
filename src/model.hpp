#pragma once
#include "config.hpp"
#include "kv_cache.hpp"
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
    // `cache` must outlive the Model; the model occupies one sequence in it.
    // A new Model always starts a fresh sequence (the cache may keep others).
    Model(const std::string& path, KVCache& cache);
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
    std::vector<float> k_tmp_, v_tmp_;             // [KV_DIM] freshly computed rows
    std::vector<const float*> k_rows_, v_rows_;    // row pointers for pos 0..pos
    std::vector<float> rope_cos_, rope_sin_;       // [MAX_SEQ][HEAD_DIM/2]

    KVCache* cache_ = nullptr;
    SeqId seq_ = -1;
};

}  // namespace tl
