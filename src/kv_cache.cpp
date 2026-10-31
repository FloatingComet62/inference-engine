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

// ---------------------------------------------------------------------------
// PagedKVCache — PagedAttention-style physical blocks + logical block tables.
// See kv_cache.hpp for the behavioural contract.
// ---------------------------------------------------------------------------

namespace {

constexpr size_t P_BLOCK_ROWS = (size_t)N_LAYERS * PagedKVCache::BLOCK_SIZE * KV_DIM;

// Floats per block for both keys and values (payload only, for stats()).
size_t p_block_bytes() {
    return 2 * P_BLOCK_ROWS * sizeof(float);
}

}  // namespace

PagedKVCache::PagedKVCache(int pool_blocks) : blocks_(pool_blocks) {
    if (pool_blocks <= 0)
        throw std::invalid_argument("PagedKVCache: pool_blocks must be > 0");
    for (int i = 0; i < pool_blocks; i++) free_blocks_.push_back(i);
}

SeqId PagedKVCache::begin_sequence() {
    SeqId id = next_++;
    seqs_[id];  // empty sequence: no blocks, no ids, len 0
    return id;
}

void PagedKVCache::end_sequence(SeqId seq) {
    Seq& s = seqs_.at(seq);
    for (const BlockEntry& e : s.blocks) {
        PhysicalBlock& blk = blocks_[e.physical_block_id];
        if (--blk.ref_count == 0) {
            free_blocks_.push_back(e.physical_block_id);
            --allocated_blocks_;
        }
    }
    seqs_.erase(seq);
}

void PagedKVCache::clear() {
    seqs_.clear();
    free_blocks_.clear();
    for (size_t i = 0; i < blocks_.size(); i++) free_blocks_.push_back((int)i);
    allocated_blocks_ = 0;
    // Cumulative counters (row_fetches_ & co) persist, matching BaselineKVCache.
}

void PagedKVCache::store(SeqId seq, int layer, int pos, int token,
                         const float* k_row, const float* v_row) {
    Seq& s = seqs_.at(seq);  // unknown seq -> engine bug, surface it
    if (layer < 0 || layer >= N_LAYERS || pos < 0 || pos >= MAX_SEQ)
        throw std::out_of_range("kv cache store out of range");
    if (pos > s.len)
        throw std::logic_error("kv cache store out of order: fill positions in order");

    const int logical = pos / BLOCK_SIZE;
    const int offset  = pos % BLOCK_SIZE;
    while ((int)s.blocks.size() <= logical) {  // normally 0-1 iterations
        if (free_blocks_.empty())
            throw std::runtime_error(
                "paged kv cache: physical block pool exhausted "
                "(raise pool_blocks or end sequences)");
        const int id = free_blocks_.back();
        free_blocks_.pop_back();
        PhysicalBlock& blk = blocks_[id];
        if (blk.keys.empty()) {  // allocate payload once; never reallocates
            blk.keys.resize(P_BLOCK_ROWS);
            blk.values.resize(P_BLOCK_ROWS);
        }
        blk.ref_count = 1;
        ++allocated_blocks_;
        s.blocks.push_back({id, 0});
    }

    BlockEntry& e = s.blocks[logical];
    const size_t off = ((size_t)layer * BLOCK_SIZE + offset) * KV_DIM;
    std::memcpy(&blocks_[e.physical_block_id].keys[off], k_row, (size_t)KV_DIM * sizeof(float));
    std::memcpy(&blocks_[e.physical_block_id].values[off], v_row, (size_t)KV_DIM * sizeof(float));

    if (pos == s.len) {  // append: extend the valid range
        ++e.filled;
        s.len = pos + 1;
        s.ids.push_back(token);
    } else {             // overwrite of an already-stored position (ppl mode)
        s.ids[pos] = token;
    }
}

const float* PagedKVCache::key_row(SeqId seq, int layer, int pos) const {
    ++row_fetches_;
    const Seq& s = seqs_.at(seq);
    if (layer < 0 || layer >= N_LAYERS || pos < 0 || pos >= s.len) return nullptr;
    const PhysicalBlock& blk = blocks_[s.blocks[pos / BLOCK_SIZE].physical_block_id];
    return blk.keys.data() + ((size_t)layer * BLOCK_SIZE + pos % BLOCK_SIZE) * KV_DIM;
}

const float* PagedKVCache::value_row(SeqId seq, int layer, int pos) const {
    ++row_fetches_;
    const Seq& s = seqs_.at(seq);
    if (layer < 0 || layer >= N_LAYERS || pos < 0 || pos >= s.len) return nullptr;
    const PhysicalBlock& blk = blocks_[s.blocks[pos / BLOCK_SIZE].physical_block_id];
    return blk.values.data() + ((size_t)layer * BLOCK_SIZE + pos % BLOCK_SIZE) * KV_DIM;
}

int PagedKVCache::stored_tokens(SeqId seq) const {
    return seqs_.at(seq).len;
}

int PagedKVCache::longest_cached_prefix(SeqId seq, const int* tokens, int n) const {
    const Seq& s = seqs_.at(seq);
    ++prefix_checks_;
    int k = 0;
    while (k < n && k < s.len && s.ids[k] == tokens[k]) k++;
    if (k > 0) ++prefix_hits_;
    return k;
}

size_t PagedKVCache::capacity_tokens() const {
    return (size_t)BLOCK_SIZE * blocks_.size();
}

size_t PagedKVCache::used_tokens() const {
    size_t total = 0;
    for (const auto& [_, s] : seqs_) total += (size_t)s.len;
    return total;
}

KVCacheStats PagedKVCache::stats() const {
    return {seqs_.size(), used_tokens(),
            allocated_blocks_ * p_block_bytes(),
            row_fetches_, prefix_checks_, prefix_hits_, 0};
}

}  // namespace tl