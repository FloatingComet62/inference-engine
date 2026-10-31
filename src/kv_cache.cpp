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

// ---------------------------------------------------------------------------
// RadixKVCache — RadixAttention (SGLang, arXiv:2312.07104): a global radix
// tree of token prefixes backed by shared one-token KV pages. See kv_cache.hpp
// for the full behavioural contract. All methods take mu_; helpers assume the
// caller holds it.
// ---------------------------------------------------------------------------

RadixKVCache::RadixKVCache(size_t capacity) : capacity_(capacity ? capacity : 1) {
    nodes_.push_back(RadixNode{});   // node 0 = root
}

SeqId RadixKVCache::begin_sequence() {
    std::lock_guard<std::mutex> lk(mu_);
    SeqId id = next_++;
    seqs_[id] = Sequence{};
    return id;
}

void RadixKVCache::end_sequence(SeqId seq) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = seqs_.find(seq);
    if (it == seqs_.end()) return;   // unknown seq: nothing to do (baseline parity)
    Sequence s = std::move(it->second);
    seqs_.erase(it);

    // Insert the completed sequence into the tree; every position the tree
    // already covers (matched prefix or an identical stored prefix) keeps its
    // existing edge/pages — the incoming copy is dropped, never duplicated.
    insert_tree(s.tokens, s.pages, s.length);
    // Pages the tree did not adopt are this sequence's unshared copies -> free.
    for (size_t p : s.pages)
        if (pages_[p].ref_count == 0) { free_pages_.push_back(p); used_pages_--; }
    // Un-pin the matched tree path (the tree itself still owns its pages).
    if (s.tree_node != SIZE_MAX) release_path(s.tree_node);
}

void RadixKVCache::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    seqs_.clear();
    // Rebuild the node/edge arenas: every id ever created becomes free.
    free_nodes_.clear();
    for (size_t i = 1; i < nodes_.size(); i++) { nodes_[i] = RadixNode{}; free_nodes_.push_back(i); }
    nodes_[0] = RadixNode{};   // root stays, emptied
    free_edges_.clear();
    for (size_t i = 0; i < edges_.size(); i++) free_edges_.push_back(i);
    lru_.clear();
    // Pages: all rows become reusable (payload memory is kept for the pool).
    free_pages_.clear();
    for (size_t p = 0; p < pages_.size(); p++) { pages_[p].ref_count = 0; free_pages_.push_back(p); }
    used_pages_ = 0;
}

void RadixKVCache::store(SeqId seq, int layer, int pos, int token,
                         const float* k_row, const float* v_row) {
    std::lock_guard<std::mutex> lk(mu_);
    Sequence& s = seqs_.at(seq);   // throws for unknown seq
    if (layer < 0 || layer >= N_LAYERS)
        throw std::logic_error("radix kv cache: layer out of range");
    if (pos < (int)s.matched_len) return;   // shared prefix: KV already there
    if (pos > (int)s.length)
        throw std::logic_error("radix kv cache: store out of order");

    float* kdst;
    float* vdst;
    if (pos == (int)s.length) {              // append: new own page
        const size_t p = allocate_page();
        s.tokens.push_back(token);
        s.pages.push_back(p);
        s.length++;
        kdst = pages_[p].keys.data() + (size_t)layer * KV_DIM;
        vdst = pages_[p].values.data() + (size_t)layer * KV_DIM;
    } else {                                 // overwrite own position (ppl rewind)
        const size_t p = s.pages[pos];
        s.tokens[pos] = token;
        kdst = pages_[p].keys.data() + (size_t)layer * KV_DIM;
        vdst = pages_[p].values.data() + (size_t)layer * KV_DIM;
    }
    std::memcpy(kdst, k_row, KV_DIM * sizeof(float));
    std::memcpy(vdst, v_row, KV_DIM * sizeof(float));
}

const float* RadixKVCache::key_row(SeqId seq, int layer, int pos) const {
    std::lock_guard<std::mutex> lk(mu_);
    ++row_fetches_;
    const Sequence& s = seqs_.at(seq);   // throws for unknown seq
    if (layer < 0 || layer >= N_LAYERS || pos < 0 || (size_t)pos >= s.length)
        return nullptr;
    return pages_[s.pages[pos]].keys.data() + (size_t)layer * KV_DIM;
}

const float* RadixKVCache::value_row(SeqId seq, int layer, int pos) const {
    std::lock_guard<std::mutex> lk(mu_);
    ++row_fetches_;
    const Sequence& s = seqs_.at(seq);
    if (layer < 0 || layer >= N_LAYERS || pos < 0 || (size_t)pos >= s.length)
        return nullptr;
    return pages_[s.pages[pos]].values.data() + (size_t)layer * KV_DIM;
}

int RadixKVCache::stored_tokens(SeqId seq) const {
    std::lock_guard<std::mutex> lk(mu_);
    return (int)seqs_.at(seq).length;
}

int RadixKVCache::longest_cached_prefix(SeqId seq, const int* tokens, int n) const {
    std::lock_guard<std::mutex> lk(mu_);
    ++prefix_checks_;
    Sequence& s = seqs_.at(seq);   // mutable by design (records the match)

    // (1) The sequence's own already-stored content (chat-style growth: the
    //     engine keeps one sequence across turns and re-feeds the history).
    int k1 = 0;
    while (k1 < n && k1 < (int)s.tokens.size() && s.tokens[k1] == tokens[k1]) k1++;

    // (2) The global tree (fresh-request reuse). Matching inside a compressed
    //     edge splits it at the divergence point so the shared pages stay
    //     protected under a clean root..node path.
    std::vector<size_t> tpages;
    const size_t tnode = match_tree(tokens, n, &tpages);
    const int m2 = (int)tpages.size();

    if (s.tree_node != SIZE_MAX) { release_path(s.tree_node); s.tree_node = SIZE_MAX; }
    if (m2 > 0) { protect_path(tnode); s.tree_node = tnode; }

    // Tree positions beyond the sequence's own match become visible reads for
    // this sequence (pages/tokens reconciled); a stale own copy is discarded
    // on end_sequence.
    if (m2 > k1) {
        if ((int)s.tokens.size() < m2) { s.tokens.resize(m2); s.pages.resize(m2); }
        for (int i = k1; i < m2; i++) { s.tokens[i] = tokens[i]; s.pages[i] = tpages[i]; }
        if (s.length < (size_t)m2) s.length = m2;
    }

    const int matched = m2 > k1 ? m2 : k1;
    s.matched_len = matched;
    if (matched > 0) ++prefix_hits_;
    return matched;
}

size_t RadixKVCache::capacity_tokens() const {
    return capacity_;
}

size_t RadixKVCache::used_tokens() const {
    std::lock_guard<std::mutex> lk(mu_);
    return used_pages_;
}

KVCacheStats RadixKVCache::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return {seqs_.size(), used_pages_, used_pages_ * PAGE_BYTES,
            row_fetches_, prefix_checks_, prefix_hits_, evictions_};
}

// ---- tree helpers ----

size_t RadixKVCache::alloc_node() const {
    size_t id;
    if (!free_nodes_.empty()) { id = free_nodes_.back(); free_nodes_.pop_back(); nodes_[id] = RadixNode{}; }
    else { id = nodes_.size(); nodes_.push_back(RadixNode{}); }
    return id;
}

void RadixKVCache::free_node(size_t nid) const {
    RadixNode& n = nodes_[nid];
    if (n.in_lru) { lru_.erase(n.lru_it); n.in_lru = false; }
    n.children.clear();   // evicted nodes are leaves anyway
    free_nodes_.push_back(nid);
}

size_t RadixKVCache::alloc_edge() const {
    size_t id;
    if (!free_edges_.empty()) { id = free_edges_.back(); free_edges_.pop_back(); edges_[id] = RadixEdge{}; }
    else { id = edges_.size(); edges_.push_back(RadixEdge{}); }
    return id;
}

void RadixKVCache::free_edge(size_t eid) const {
    edges_[eid].tokens.clear();
    edges_[eid].pages.clear();
    free_edges_.push_back(eid);
}

void RadixKVCache::add_to_lru(size_t nid) const {
    RadixNode& n = nodes_[nid];
    if (nid == 0 || n.in_lru || n.ref_count != 0 || !n.children.empty()) return;
    n.in_lru = true;
    n.lru_it = lru_.insert({n.last_access, nid});
}

void RadixKVCache::remove_from_lru(size_t nid) const {
    RadixNode& n = nodes_[nid];
    if (!n.in_lru) return;
    lru_.erase(n.lru_it);
    n.in_lru = false;
}

void RadixKVCache::touch(size_t nid) const {
    RadixNode& n = nodes_[nid];
    n.last_access = ++clock_;
    if (n.in_lru) {                    // still a leaf: keep tracked at the new time
        lru_.erase(n.lru_it);
        n.lru_it = lru_.insert({n.last_access, nid});
    }
}

// Split edge `eid` at token index k (1 <= k < size): the first k tokens keep
// the edge id and owner; the remainder moves to a fresh edge under a new
// intermediate node, preserving the old child. Physical pages are moved, never
// copied. Returns the new node id.
size_t RadixKVCache::split_edge(size_t eid, int k) const {
    // Allocate before grabbing references: alloc_node/alloc_edge push to
    // vectors that may reallocate and invalidate held RadixEdge/RadixNode
    // references.
    const size_t nx = alloc_node();
    const size_t e2 = alloc_edge();
    RadixEdge& ed = edges_[eid];       // fetched after all growth
    RadixEdge& rem = edges_[e2];
    RadixNode& NX = nodes_[nx];
    const size_t old_child = ed.child;
    NX.parent_edge = eid;

    rem.tokens.assign(ed.tokens.begin() + k, ed.tokens.end());
    rem.pages.assign(ed.pages.begin() + k, ed.pages.end());
    rem.owner = nx;
    rem.child = old_child;
    NX.children[rem.tokens[0]] = e2;
    nodes_[old_child].parent_edge = e2;

    ed.tokens.resize(k);
    ed.pages.resize(k);
    ed.child = nx;
    return nx;
}

// Walk the tree matching tokens[0..n); split edges that match only partially
// (the prefix the caller keeps runs to the new node). Returns the node at the
// end of the matched path; if `pages` is non-null, appends the physical page
// of every matched token in order.
size_t RadixKVCache::match_tree(const int* tokens, int n, std::vector<size_t>* pages) const {
    size_t node = 0;
    int i = 0;
    while (i < n) {
        auto it = nodes_[node].children.find(tokens[i]);
        if (it == nodes_[node].children.end()) break;
        const size_t eid = it->second;
        const RadixEdge& ed = edges_[eid];
        int common = 0;
        while (common < (int)ed.tokens.size() && i + common < n &&
               ed.tokens[common] == tokens[i + common]) common++;
        if (pages)
            for (int c = 0; c < common; c++) pages->push_back(ed.pages[c]);
        i += common;
        if (common < (int)ed.tokens.size()) {      // match cuts inside the edge
            remove_from_lru(ed.owner);             // owner gains a child -> not a leaf
            node = split_edge(eid, common);
            break;
        }
        node = ed.child;
    }
    return node;
}

// Insert a completed sequence into the tree: walk existing edges while the
// tokens agree, split on divergence, and attach the remaining tokens/pages as
// a fresh edge (adopting their pages). Positions already covered by a matching
// edge are left alone — their incoming pages (matched by reference or a
// duplicate copy) are not adopted; the caller frees the unadopted copies.
void RadixKVCache::insert_tree(const std::vector<int>& tokens,
                               const std::vector<size_t>& pages, size_t n) {
    size_t node = 0;
    size_t i = 0;
    while (i < n) {
        auto it = nodes_[node].children.find(tokens[i]);
        if (it == nodes_[node].children.end()) {
            const size_t eid = alloc_edge();
            RadixEdge& ed = edges_[eid];
            ed.tokens.assign(tokens.begin() + i, tokens.begin() + n);
            ed.pages.assign(pages.begin() + i, pages.begin() + n);
            ed.owner = node;
            ed.child = alloc_node();
            nodes_[ed.child].parent_edge = eid;
            for (size_t c = 0; c < ed.pages.size(); c++) pages_[ed.pages[c]].ref_count++;
            touch(ed.child);
            add_to_lru(ed.child);
            nodes_[node].children[ed.tokens[0]] = eid;
            remove_from_lru(node);   // gained a child -> no longer a leaf
            return;
        }
        const size_t eid = it->second;
        const RadixEdge& ed = edges_[eid];
        int common = 0;
        while (common < (int)ed.tokens.size() && i + common < n &&
               ed.tokens[common] == tokens[i + common]) common++;
        if (common < (int)ed.tokens.size()) {      // diverge inside the edge
            size_t nx = split_edge(eid, common);
            if (i + (size_t)common < n) {          // attach the remaining tokens
                const size_t e2 = alloc_edge();
                RadixEdge& ne = edges_[e2];
                ne.tokens.assign(tokens.begin() + i + common, tokens.begin() + n);
                ne.pages.assign(pages.begin() + i + common, pages.begin() + n);
                ne.owner = nx;
                ne.child = alloc_node();
                nodes_[ne.child].parent_edge = e2;
                for (size_t c = 0; c < ne.pages.size(); c++) pages_[ne.pages[c]].ref_count++;
                touch(ne.child);
                add_to_lru(ne.child);
                nodes_[nx].children[ne.tokens[0]] = e2;
            }
            return;
        }
        node = ed.child;                           // full edge consumed; no adoption
        i += (size_t)common;
    }
}

void RadixKVCache::protect_path(size_t nid) const {
    size_t cur = nid;
    while (true) {
        touch(cur);
        nodes_[cur].ref_count++;
        remove_from_lru(cur);
        if (cur == 0) break;
        cur = edges_[nodes_[cur].parent_edge].owner;
    }
}

void RadixKVCache::release_path(size_t nid) const {
    size_t cur = nid;
    while (true) {
        if (--nodes_[cur].ref_count == 0) add_to_lru(cur);
        if (cur == 0) break;
        cur = edges_[nodes_[cur].parent_edge].owner;
    }
}

// ---- page pool ----

size_t RadixKVCache::allocate_page() {
    if (!free_pages_.empty()) {
        const size_t p = free_pages_.back(); free_pages_.pop_back();
        used_pages_++;
        return p;
    }
    if (pages_.size() < capacity_) {
        pages_.emplace_back();
        pages_.back().keys.resize((size_t)N_LAYERS * KV_DIM);
        pages_.back().values.resize((size_t)N_LAYERS * KV_DIM);
        used_pages_++;
        return pages_.size() - 1;
    }
    // At capacity: evict least-recently-used leaves until a page frees up.
    while (free_pages_.empty() && !lru_.empty()) {
        const auto it = lru_.begin();
        const uint64_t acc = it->first;
        const size_t nid = it->second;
        lru_.erase(it);
        RadixNode& n = nodes_[nid];
        if (n.in_lru && n.last_access == acc && n.ref_count == 0 &&
            n.children.empty() && n.parent_edge != SIZE_MAX) {
            n.in_lru = false;
            evict_edge(n.parent_edge);   // frees >= 1 page (updates evictions_)
        } else {
            n.in_lru = false;            // stale entry: drop
        }
    }
    if (!free_pages_.empty()) {
        const size_t p = free_pages_.back(); free_pages_.pop_back();
        used_pages_++;
        return p;
    }
    throw std::runtime_error("radix kv cache: capacity exhausted, no evictable leaf");
}

void RadixKVCache::evict_edge(size_t eid) {
    RadixEdge& ed = edges_[eid];
    const size_t parent = ed.owner;
    const size_t child = ed.child;
    free_node(child);
    for (size_t p : ed.pages) {
        if (--pages_[p].ref_count == 0) { free_pages_.push_back(p); used_pages_--; }
        evictions_++;
    }
    nodes_[parent].children.erase(ed.tokens[0]);
    free_edge(eid);
    add_to_lru(parent);   // parent may have become an evictable leaf
}

}  // namespace tl