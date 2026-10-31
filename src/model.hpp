#pragma once
#include "config.hpp"
#include "kernels_q4.hpp"
#include "kv_cache.hpp"
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
    // `cache` must outlive the Model; the model occupies one sequence in it.
    // A new Model always starts a fresh sequence (the cache may keep others).
    Model(const std::string& path, KVCache& cache);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Runs one token at position `pos`; returns logits[VOCAB].
    const float* forward(int token, int pos) { return forward_batch(&token, 1, pos); }

    // Runs n prompt tokens starting at `pos` in batches of PREFILL_CHUNK
    // (matrix x matrix); returns the logits of the last token.
    const float* prefill(const int* tokens, int n, int pos);

    // Speculative verification: runs T tokens at positions pos..pos+T-1 in one batch
    // (T <= PREFILL_CHUNK) and returns logits for ALL of them, [T][VOCAB]; row t
    // predicts the token after tokens[t]. Rewinding pos later simply overwrites rows.
    const float* forward_all(const int* tokens, int T, int pos) { return forward_batch(tokens, T, pos, true); }

    // Consults the cache for tokens[0..n) already stored for this sequence.
    // prefill() calls it at position 0 and skips the matched prefix entirely
    // (its KV rows are read back from the cache). Returns the matched length.
    int use_cached_prefix(const int* tokens, int n);

    // Bytes of weights streamed from RAM per generated token
    // (everything except the embedding table, of which only one row is read).
    size_t bytes_per_token() const;

    // Writes this (fp32) model as a version-2 file: matrices as `mats`, output head as `head`.
    void save_quantized(const std::string& path, WType mats, WType head) const;

private:
    void* map_ = nullptr;
    size_t file_size_ = 0;
    Weights w_{};

    // T tokens at positions pos..pos+T-1; logits for the last one only (or all T).
    const float* forward_batch(const int* tokens, int T, int pos, bool all_logits = false);
    // out[T][n_out] = x[T][n_in] . W^T
    void mm(float* out, const float* x, const Mat& m, int n_in, int n_out, int T);
    std::vector<int8_t> xq_;   // quantised activations for mm()
    std::vector<float> xd_;
    std::vector<int32_t> xs_;  // per-block Q4 zero-point correction (-8 * group sums)

    std::vector<float> x_, xb_, xb2_, q_, att_, hb_, hb2_, logits_;
    std::vector<float> k_tmp_, v_tmp_;             // [T][KV_DIM] freshly computed rows
    // Contiguous working copy of the cache for the attention kernel, per layer:
    // keys rotated at their LOGICAL slot, values as stored. A slot's content only
    // changes when the cache re-bases (StreamingLLM past its window) or a position
    // is re-stored (ppl rewinds), so only new slots are copied each step and
    // everything is recopied after a re-base.
    std::vector<float> k_rot_, v_work_;            // [N_LAYERS][MAX_SEQ][KV_DIM]
    int work_valid_ = 0;                           // slots [0, work_valid_) are current
    std::vector<float> rope_cos_, rope_sin_;       // [MAX_SEQ][HEAD_DIM/2]

    KVCache* cache_ = nullptr;
    SeqId seq_ = -1;
};

}  // namespace tl
