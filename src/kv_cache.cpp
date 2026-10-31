// Baseline KV cache backend: per-sequence flat arrays, never evicted.
// This is the engine's original behaviour, relocated behind KVCache.
#include "kv_cache.hpp"

#include <cstring>
#include <stdexcept>

namespace tl {
namespace {

constexpr size_t SEQUENCE_ROWS = (size_t)N_LAYERS * MAX_SEQ * KV_DIM;
constexpr size_t ROW = (size_t)KV_DIM;

size_t row_offset(int layer, int pos) {
    return (size_t)layer * MAX_SEQ * KV_DIM + (size_t)pos * KV_DIM;
}

}  // namespace

SeqId BaselineKVCache::begin_sequence() {
    SeqId id = next_++;
    Seq& s = seqs_[id];
    s.keys.assign(SEQUENCE_ROWS, 0.0f);
    s.vals.assign(SEQUENCE_ROWS, 0.0f);
    s.ids.assign(MAX_SEQ, 0);
    return id;
}

void BaselineKVCache::end_sequence(SeqId seq) {
    seqs_.erase(seq);
}

void BaselineKVCache::clear() {
    seqs_.clear();
}

void BaselineKVCache::store(SeqId seq, int layer, int pos, int token,
                            const float* k_row, const float* v_row) {
    Seq& s = seqs_.at(seq);  // unknown seq -> engine bug, surface it
    if (layer < 0 || layer >= N_LAYERS || pos < 0 || pos >= MAX_SEQ)
        throw std::out_of_range("kv cache store out of range");
    if (pos > s.len)
        throw std::logic_error("kv cache store out of order: fill positions in order");
    const size_t off = row_offset(layer, pos);
    std::memcpy(&s.keys[off], k_row, ROW * sizeof(float));
    std::memcpy(&s.vals[off], v_row, ROW * sizeof(float));
    if (pos == s.len) s.len = pos + 1;  // extend; pos < len overwrites in place
    s.ids[pos] = token;
}

const float* BaselineKVCache::key_row(SeqId seq, int layer, int pos) const {
    ++row_fetches_;
    return &seqs_.at(seq).keys[row_offset(layer, pos)];
}

const float* BaselineKVCache::value_row(SeqId seq, int layer, int pos) const {
    ++row_fetches_;
    return &seqs_.at(seq).vals[row_offset(layer, pos)];
}

int BaselineKVCache::stored_tokens(SeqId seq) const {
    return seqs_.at(seq).len;
}

int BaselineKVCache::longest_cached_prefix(SeqId seq, const int* tokens, int n) const {
    const Seq& s = seqs_.at(seq);
    ++prefix_checks_;
    int k = 0;
    while (k < n && k < s.len && s.ids[k] == tokens[k]) k++;
    if (k > 0) ++prefix_hits_;
    return k;
}

size_t BaselineKVCache::capacity_tokens() const {
    return (size_t)MAX_SEQ;  // per-sequence positions; baseline does not evict
}

size_t BaselineKVCache::used_tokens() const {
    size_t total = 0;
    for (const auto& [_, s] : seqs_) total += (size_t)s.len;
    return total;
}

KVCacheStats BaselineKVCache::stats() const {
    return {seqs_.size(), used_tokens(),
            used_tokens() * 2 * SEQUENCE_ROWS / MAX_SEQ * sizeof(float),
            row_fetches_, prefix_checks_, prefix_hits_, 0};
}

}  // namespace tl