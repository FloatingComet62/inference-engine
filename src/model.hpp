#pragma once
#include "config.hpp"
#include "kernels_q4.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace tl {

// A weight matrix [n_out][n_in], stored as fp32 rows or quantised blocks.
struct Mat { const void* w = nullptr; WType t = WType::F32; };

struct LayerWeights {
    const float *attn_norm, *ffn_norm;
    Mat wq, wk, wv, wo, w_gate, w_up, w_down;
};

struct Weights {
    const float* embed;            // [VOCAB][DIM]
    LayerWeights layers[N_LAYERS];
    const float* final_norm;       // [DIM]
    Mat lm_head;                   // [VOCAB][DIM]
};

class Model {
public:
    explicit Model(const std::string& path);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Runs one token at position `pos`; returns logits[VOCAB].
    const float* forward(int token, int pos) { return forward_batch(&token, 1, pos); }

    // Runs n prompt tokens starting at `pos` in batches of PREFILL_CHUNK
    // (matrix x matrix); returns the logits of the last token.
    const float* prefill(const int* tokens, int n, int pos);

    // Bytes of weights streamed from RAM per generated token
    // (everything except the embedding table, of which only one row is read).
    size_t bytes_per_token() const;

    // Writes this (fp32) model as a version-2 file: matrices as `mats`, output head as `head`.
    void save_quantized(const std::string& path, WType mats, WType head) const;

private:
    void* map_ = nullptr;
    size_t file_size_ = 0;
    Weights w_{};

    // T tokens at positions pos..pos+T-1; logits for the last one only.
    const float* forward_batch(const int* tokens, int T, int pos);
    // out[T][n_out] = x[T][n_in] . W^T
    void mm(float* out, const float* x, const Mat& m, int n_in, int n_out, int T);
    std::vector<int8_t> xq_;   // quantised activations for mm()
    std::vector<float> xd_;
    std::vector<int32_t> xs_;  // per-block Q4 zero-point correction (-8 * group sums)

    std::vector<float> x_, xb_, xb2_, q_, att_, hb_, hb2_, logits_;
    std::vector<float> key_cache_, value_cache_;   // [N_LAYERS][MAX_SEQ][KV_DIM]
    std::vector<float> rope_cos_, rope_sin_;       // [MAX_SEQ][HEAD_DIM/2]
};

}  // namespace tl
