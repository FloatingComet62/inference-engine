// KV cache interface for the TinyLlama engine.
//
// The engine stores one K row and one V row per (layer, position) — KV_DIM
// floats each — during prefill/decode and reads them back for attention.
// KVCache abstracts *where and how* those rows live so caching algorithms can
// be swapped and compared without touching the forward pass.
//
// Implementations (all live in kv_cache.cpp):
//   BaselineKVCache  flat per-layer arrays, no eviction
//   PagedKVCache     fixed-size token blocks + block tables (vLLM style)
//   RadixKVCache     global radix tree of token prefixes + shared KV pages,
//                    LRU leaf eviction (RadixAttention, the engine default)
// Planned: FreeOnFinish (drop a sequence's rows when it ends), BoundedGrowth.
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
#include <cstdint>
#include <mutex>
#include <set>
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

// Paged backend (vLLM PagedAttention style, SOSP'23): the KV cache is a fixed
// pool of physical blocks; each sequence maps its logical blocks to physical
// ones through a block table, so token positions are never assumed physically
// contiguous. Blocks are allocated lazily — begin_sequence() allocates
// nothing; store() grabs a block on demand — and released to the pool when a
// sequence ends. One block holds all layers (keys/values are
// [N_LAYERS][BLOCK_SIZE][KV_DIM] floats), which is what store() must be told
// in layer-major order but writes into the same block.
//
// Behavioural notes:
//   * capacity_tokens() = BLOCK_SIZE * pool_blocks (default: MAX_SEQ).
//   * store() throws std::runtime_error when the pool is exhausted (no
//     eviction yet; positions < stored_tokens stay readable by contract).
//   * Row pointers returned by key_row/value_row stay valid while the owning
//     sequence lives; a block's storage is fixed at first allocation and only
//     freed when the pool reuses it after end_sequence()/clear().
//   * key_row/value_row return nullptr for positions >= stored_tokens (never
//     read by the engine) or out-of-range layers; unknown sequences throw.
class PagedKVCache : public KVCache {
public:
    static constexpr int BLOCK_SIZE = 16;
    static constexpr int DEFAULT_POOL_BLOCKS = MAX_SEQ / BLOCK_SIZE;

    explicit PagedKVCache(int pool_blocks = DEFAULT_POOL_BLOCKS);

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
    struct PhysicalBlock {
        std::vector<float> keys, values;  // [N_LAYERS][BLOCK_SIZE][KV_DIM]
        int ref_count = 0;                // 1 while owned; 0 -> released
    };
    struct BlockEntry {
        int physical_block_id;
        int filled;    // tokens stored in this block's [0, filled)
    };
    struct Seq {
        std::vector<BlockEntry> blocks;   // logical block table
        std::vector<int> ids;             // token per position, for prefix checks
        int len = 0;                      // positions stored so far
    };
    std::vector<PhysicalBlock> blocks_;  // fixed pool, sized at construction
    std::vector<int> free_blocks_;       // pool ids available
    std::unordered_map<SeqId, Seq> seqs_;
    SeqId next_ = 0;
    size_t allocated_blocks_ = 0;        // for stats().bytes
    mutable size_t row_fetches_ = 0, prefix_checks_ = 0, prefix_hits_ = 0;
};

// RadixAttention (SGLang, arXiv:2312.07104) KV cache: a global compressed
// radix tree of token prefixes; the KV rows of a matched prefix are *shared*
// physical pages, so sequences never duplicate KV for an identical prefix and
// a new request can skip recomputing it. This is a global, content-keyed
// cache, unlike the per-sequence baseline/paged layouts.
//
// Storage model:
//   * One physical page per token position (all N_LAYERS × KV_DIM keys +
//     values): pages are lazy (allocated at first store) and reference-counted
//     by the tree edges that hold them.
//   * Tree edges are compressed: an edge is a run of tokens + the pages that
//     hold their KV. Inserting a diverging sequence splits the shared edge at
//     the divergence point (tokens and pages divide without copying).
//   * A page is in the tree while some edge lists it (ref_count == 1).
//     Active sequences hold *node* references on their matched prefix path,
//     which pins every page below it; a page referenced only by a tree edge is
//     evictable.
//
// Behavioural notes:
//   * capacity_tokens() = one page per token (constructor arg, default
//     MAX_SEQ); tree pages and active sequences share the same pool. When the
//     pool is full, store() evicts the least-recently-used *leaf* (pages that
//     no active sequence needs) before throwing std::runtime_error if nothing
//     is evictable.
//   * longest_cached_prefix(seq, tokens, n) is the reuse entry point: it
//     matches the caller's tokens against (1) the sequence's own already
//     stored content (chat-style growth on one sequence) and (2) the global
//     tree, splits an edge if the match cuts inside one, protects the matched
//     path from eviction, and returns the longest length k whose positions
//     [0, k) hold KV for exactly tokens[0..k). The engine then skips
//     recomputing positions < k. If the sequence had a previous match, its
//     protection is released first.
//   * store() for pos < matched_prefix is a no-op (the shared KV is already
//     correct; the header contract guarantees content equality there);
//     positions >= matched prefix are the sequence's own pages and may be
//     overwritten in place (ppl-style rewind). All layer calls for one
//     position write into the same page — one page per token, not per layer.
//   * end_sequence() does NOT drop rows: the sequence's position list is
//     inserted into the tree (duplicate prefixes are deduplicated — the
//     second copy's pages are released). Its KV therefore stays cached for
//     future requests; only pages not referenced by the tree are freed.
//   * Row pointers stay valid until the pool reuses the page (after an
//     eviction or clear()); the engine holds rows across one forward pass.
//   * Thread-safety matches the interface contract: reads under a shared
//     lock; writes single-threaded under the same lock.
class RadixKVCache : public KVCache {
public:
    explicit RadixKVCache(size_t capacity = MAX_SEQ);

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
    static constexpr size_t PAGE_BYTES = (size_t)2 * N_LAYERS * KV_DIM * sizeof(float);

    struct KVPage {
        std::vector<float> keys, values;  // [N_LAYERS][KV_DIM] each
        size_t ref_count = 0;             // tree edges referencing this page
    };
    struct RadixNode {
        size_t parent_edge = SIZE_MAX;    // incoming edge (SIZE_MAX = root)
        std::unordered_map<int, size_t> children;  // first token -> edge id
        size_t ref_count = 0;             // active sequences pinning this path
        uint64_t last_access = 0;         // LRU clock
        bool in_lru = false;              // tracked as evictable leaf
        std::multiset<std::pair<uint64_t, size_t>>::iterator lru_it;
    };
    struct RadixEdge {
        std::vector<int> tokens;          // token run (compressed edge)
        std::vector<size_t> pages;        // one page per token, 1:1
        size_t owner = SIZE_MAX;          // node the edge hangs under
        size_t child = SIZE_MAX;          // node at the end of the run
    };
    struct Sequence {
        std::vector<int> tokens;          // full stored position contents
        std::vector<size_t> pages;        // physical page per position
        size_t length = 0;                // positions stored (readable: [0,len))
        size_t matched_len = 0;           // positions covered by a prefix match
        size_t tree_node = 0;             // tree path end of last match (refs)
    };

    // --- tree helpers (caller holds the lock) ---
    // longest_cached_prefix is const in the public interface but must split
    // edges and pin/unpin paths (reference Option A: mutable tree metadata);
    // the helpers below are const and touch only mutable state. The page-pool
    // helpers (allocate_page/evict_edge) stay non-const — only store() and
    // end_sequence() use them.
    size_t alloc_node() const;            // fresh or reused node id
    void free_node(size_t nid) const;
    size_t alloc_edge() const;            // fresh or reused edge id
    void free_edge(size_t eid) const;
    void add_to_lru(size_t nid) const;    // if evictable leaf (not root, ref 0)
    void remove_from_lru(size_t nid) const;
    void touch(size_t nid) const;         // bump last_access (+ re-sort in LRU)
    size_t split_edge(size_t eid, int k) const;  // split at k; returns new node
    size_t match_tree(const int* tokens, int n, std::vector<size_t>* pages) const;
    void insert_tree(const std::vector<int>& tokens,
                     const std::vector<size_t>& pages, size_t n);
    void protect_path(size_t nid) const;  // ref-count root..nid (+ touch)
    void release_path(size_t nid) const;  // undo protect_path

    // --- pages ---
    size_t allocate_page();               // evicts on demand; may throw
    void evict_edge(size_t eid);          // free a leaf edge's pages + node

    std::vector<KVPage> pages_;           // grows to capacity, never moves
    std::vector<size_t> free_pages_;
    mutable std::vector<RadixNode> nodes_;   // [0] = root, never freed
    mutable std::vector<size_t> free_nodes_;
    mutable std::vector<RadixEdge> edges_;
    mutable std::vector<size_t> free_edges_;
    mutable std::multiset<std::pair<uint64_t, size_t>> lru_;  // (last_access, node)
    mutable std::unordered_map<SeqId, Sequence> seqs_;  // matched paths, per Option A
    SeqId next_ = 0;
    size_t capacity_;
    size_t used_pages_ = 0;               // pages held (active or tree)
    size_t evictions_ = 0;
    mutable uint64_t clock_ = 0;
    mutable std::mutex mu_;
    mutable size_t row_fetches_ = 0, prefix_checks_ = 0, prefix_hits_ = 0;
};

}  // namespace tl