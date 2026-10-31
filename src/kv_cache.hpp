// KV cache interface for the TinyLlama engine.
//
// The engine stores one K row and one V row per (layer, position) — KV_DIM
// floats each — during prefill/decode and reads them back for attention.
// KVCache abstracts *where and how* those rows live so caching algorithms can
// be swapped and compared without touching the forward pass.
//
// Intended implementations (only BaselineKVCache exists today):
//   BaselineKVCache  flat per-layer arrays, no eviction        [kv_cache.cpp]
//   PagedBlocks      fixed-size token blocks + block tables (vLLM style)
//   FreeOnFinish     drop a sequence's rows when it ends
//   PrefixCaching    content-keyed reuse of stored prefixes (skip prefill)
//   BoundedGrowth    capacity cap with an eviction policy
//
// Contracts (implementations must honour; the engine relies on them):
//   * A row is KV_DIM contiguous floats. The returned pointer stays valid
//     until the next store() on the same sequence, end_sequence(), or clear().
//   * store() is called per position in order (0, 1, 2, ...); overwriting an
//     already-stored position is allowed (ppl mode restarts at pos 0).
//   * Reads (key_row, value_row, stored_tokens, longest_cached_prefix) are
//     thread-safe and may run concurrently (OpenMP attention loop); writes
//     (store, begin_sequence, end_sequence, clear) are single-threaded.
//   * Positions are in [0, MAX_SEQ); the engine enforces that limit.
//   * Evicting implementations must keep every position < stored_tokens(seq)
//     readable; shrinking that range or dropping an engine-visible position
//     requires teaching the engine to re-prefill gaps first.
//   * longest_cached_prefix may return 0 at any time (safe: no skip); it must
//     never return k while positions [0, k) hold KV for different tokens.
//
// The engine pays one virtual call per (layer, position) row fetch (see
// Model::forward). If that ever matters, a batch row-pointer API can be added;
// row-at-a-time keeps the surface small while layering is opaque.
#pragma once

#include "config.hpp"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace tl {

// Opaque identifier for one logical sequence (a chat session / request).
using SeqId = int;

// Per-cache counters so benchmark code can compare the algorithms.
struct KVCacheStats {
    size_t sequences = 0;      // currently active (begun, not ended)
    size_t tokens = 0;         // KV positions stored across all sequences
    size_t bytes = 0;          // KV payload bytes (rows only, excl. bookkeeping)
    size_t row_fetches = 0;    // key_row + value_row calls
    size_t prefix_checks = 0;  // longest_cached_prefix calls
    size_t prefix_hits = 0;    // calls that returned > 0
    size_t evictions = 0;      // positions dropped by a policy (0 for baseline)
};

class KVCache {
public:
    virtual ~KVCache() = default;

    // ---- lifecycle ----
    virtual SeqId begin_sequence() = 0;
    virtual void end_sequence(SeqId seq) = 0;
    virtual void clear() = 0;  // drop every sequence and its rows

    // ---- storage ----
    // Store one token's K/V rows for `layer` at `pos`. `token` is recorded so
    // longest_cached_prefix can verify content; layered calls for the same
    // position pass the same token.
    virtual void store(SeqId seq, int layer, int pos, int token,
                       const float* k_row, const float* v_row) = 0;
    virtual const float* key_row(SeqId seq, int layer, int pos) const = 0;
    virtual const float* value_row(SeqId seq, int layer, int pos) const = 0;
    // Number of valid leading positions for `seq` ([0, stored_tokens) readable).
    virtual int stored_tokens(SeqId seq) const = 0;

    // ---- prefix caching ----
    // Number of leading tokens of `tokens[0..n)` whose KV is already stored for
    // `seq` and matches by content. The engine can prefill from that offset.
    virtual int longest_cached_prefix(SeqId seq, const int* tokens, int n) const = 0;

    // ---- capacity / stats ----
    virtual size_t capacity_tokens() const = 0;  // policy bound on total tokens
    virtual size_t used_tokens() const = 0;      // total stored across sequences
    virtual KVCacheStats stats() const = 0;
};

// Baseline backend: one flat [N_LAYERS][MAX_SEQ][KV_DIM] pair per sequence,
// allocated in full on begin_sequence, never evicted. This is the engine's
// original behaviour, relocated behind the interface.
class BaselineKVCache : public KVCache {
public:
    SeqId begin_sequence() override;
    void end_sequence(SeqId seq) override;
    void clear() override;

    void store(SeqId seq, int layer, int pos, int token,
               const float* k_row, const float* v_row) override;
    const float* key_row(SeqId seq, int layer, int pos) const override;
    const float* value_row(SeqId seq, int layer, int pos) const override;
    int stored_tokens(SeqId seq) const override;

    int longest_cached_prefix(SeqId seq, const int* tokens, int n) const override;

    size_t capacity_tokens() const override;
    size_t used_tokens() const override;
    KVCacheStats stats() const override;

private:
    struct Seq {
        std::vector<float> keys, vals;  // [N_LAYERS][MAX_SEQ][KV_DIM]
        std::vector<int> ids;           // token per position, for prefix checks
        int len = 0;                    // positions stored so far
    };
    std::unordered_map<SeqId, Seq> seqs_;
    SeqId next_ = 0;
    mutable size_t row_fetches_ = 0, prefix_checks_ = 0, prefix_hits_ = 0;
};

}  // namespace tl