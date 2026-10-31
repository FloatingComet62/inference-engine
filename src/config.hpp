// TinyLlama-1.1B-Chat-v1.0 hyperparameters, hard-coded on purpose.
// The engine supports exactly this model, so every size is a compile-time
// constant and the compiler can unroll and vectorise freely.
#pragma once

namespace tl {

#ifndef TL_TINY_TEST
constexpr int DIM        = 2048;   // hidden size
constexpr int HIDDEN     = 5632;   // feed-forward (SwiGLU) size
constexpr int N_LAYERS   = 22;
constexpr int N_HEADS    = 32;     // query heads
constexpr int N_KV_HEADS = 4;      // key/value heads (grouped-query attention)
constexpr int VOCAB      = 32000;
constexpr int MAX_SEQ    = 2048;   // context length
#else
// Toy sizes used only by `make smoke` to test the pipeline without the real model.
constexpr int DIM        = 64;
constexpr int HIDDEN     = 192;   // multiple of 32 for Q4 blocks
constexpr int N_LAYERS   = 2;
constexpr int N_HEADS    = 4;
constexpr int N_KV_HEADS = 2;
constexpr int VOCAB      = 300;
constexpr int MAX_SEQ    = 128;
#endif

constexpr int HEAD_DIM = DIM / N_HEADS;            // 64
constexpr int KV_DIM   = N_KV_HEADS * HEAD_DIM;    // 256
constexpr int KV_MUL   = N_HEADS / N_KV_HEADS;     // 8 query heads share one KV head

constexpr float ROPE_THETA = 10000.0f;
constexpr float RMS_EPS    = 1e-5f;

constexpr int BOS = 1;   // <s>
constexpr int EOS = 2;   // </s>

}  // namespace tl
