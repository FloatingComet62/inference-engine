# Implementation Specification: RadixAttention and Prefix Caching in C++

Source: SGLang: Efficient Execution of Structured Language Model Programs — Zheng et al., arXiv:2312.07104v2, June 2024.

## Objective

Implement the KV cache reuse mechanism described in SGLang's RadixAttention using C++.

The implementation should integrate with the existing `KVCache` interface without changing its public API.

Unlike PagedAttention, which focuses on efficiently allocating physical KV memory, RadixAttention focuses on reusing previously computed KV cache entries across different sequences and requests.

The central idea is:

Maintain a global radix tree of token prefixes, where every tree edge represents a sequence of tokens and the corresponding KV cache data.

This allows multiple sequences to share cached prefixes without recomputing their KV tensors.

The initial implementation should focus exclusively on RadixAttention. The paper's other optimizations, including compressed finite-state machines and API speculative execution, are outside the scope of this KV cache implementation.

# 1. Understanding RadixAttention

## 1.1 Why prefix caching is necessary

During autoregressive inference, the model processes an input sequence through a forward pass called prefill.

For every token, it generates intermediate key and value tensors. These tensors are stored in the KV cache and reused during subsequent decoding.

An important property is:

The KV cache for a token depends on all preceding tokens, but not on future tokens.

Consequently, two sequences sharing an identical prefix can reuse the KV cache associated with that prefix.

For example:

```
Sequence A:
[10, 20, 30, 40, 50, 60]

Sequence B:
[10, 20, 30, 40, 70, 80]
```

The first four tokens are identical.

Without prefix caching:

```
Sequence A: Compute KV for 10, 20, 30, 40, 50, 60
Sequence B: Compute KV for 10, 20, 30, 40, 70, 80
```

With prefix caching:

```
Shared KV:  10, 20, 30, 40

Sequence A:                     50, 60
Sequence B:                     70, 80
```

Sequence B can reuse the first four tokens' KV tensors instead of recomputing them.

This reduces:

- Prefill computation.
- KV memory consumption.
- First-token latency.
- Overall inference time.

The paper applies this optimization to multiple generation calls, parallel branches, few-shot examples, multi-turn conversations, and other workflows.

Source: Section 3 and Appendix A.1.

## 1.2 Why a radix tree?

A simple hash map could associate complete token sequences with their KV cache.

However, this would not efficiently support arbitrary shared prefixes.

Consider:

```
Sequence A: [10, 20, 30, 40, 50]
Sequence B: [10, 20, 30, 40, 60]
Sequence C: [10, 20, 70, 80]
```

A radix tree represents these sequences as shared paths:

```
                         Root
                           |
                       [10, 20]
                           |
                          [30]
                           |
                          [40]
                         /    \
                       [50]  [60]

                           |
                         [70, 80]
```

The important distinction is that a radix tree is a compressed trie.

Unlike a conventional trie, an edge can contain multiple tokens instead of exactly one token.

For example:

```
Root
 |
 +---- [10, 20, 30, 40] ----+
                             |
                       +-----+-----+
                       |           |
                     [50]         [60]
```

This reduces the number of tree nodes and simplifies management.

Each edge represents:

- A sequence of token IDs.
- The KV cache corresponding to those tokens.

When two sequences share a prefix, they share the corresponding tree path and KV storage.

Source: Section 3, page 4.

# 2. Core Data Structures

The implementation should have four primary components:

1. Radix tree.
2. Radix tree nodes and edges.
3. KV memory allocator.
4. Sequence manager.

## 2.1 Radix tree

Maintain a global radix tree representing all cached token sequences.

The tree is shared across all active and completed sequences.

Conceptually:

```
class RadixTree {
public:
    // Find the longest matching prefix.
    // Insert a token sequence and its KV data.
    // Evict cached leaves.
};
```

The tree should support:

```
match_prefix(tokens, n);
insert(tokens, kv_data);
evict(required_tokens);
```

These operations will be discussed in detail later.

## 2.2 Radix tree nodes

A node represents a position in the radix tree.

Suggested structure:

```
struct RadixNode {
    RadixNode* parent = nullptr;

    std::unordered_map<int, RadixEdge*> children;

    size_t ref_count = 0;

    size_t last_access = 0;

    bool is_leaf() const {
        return children.empty();
    }
};
```

However, this structure alone is insufficient because the paper uses compressed edges.

A better design is to represent edges explicitly.

```
struct RadixEdge {
    std::vector<int> tokens;

    RadixNode* child = nullptr;

    std::vector<int> physical_blocks;
};
```

The root node represents the empty prefix.

Each edge stores a token sequence and leads to another node.

For example:

```
Root
 |
 +---- Edge: [10, 20, 30]
                  |
                Node A
                  |
                  +---- Edge: [40, 50]
                  |
                  +---- Edge: [60, 70]
```

### Important: Edge splitting

Suppose the tree contains:

```
Root
 |
 +---- [10, 20, 30, 40, 50]
```

Now insert:

```
[10, 20, 30, 40, 60]
```

The existing edge must be split at the divergence point:

```
Root
 |
 +---- [10, 20, 30, 40]
                  |
                Node A
                 /  \
                /    \
             [50]   [60]
```

The KV data corresponding to the shared prefix remains associated with the shared portion.

The remaining portions become separate branches.

This operation is essential to RadixAttention.

Without edge splitting, the implementation cannot efficiently represent arbitrary shared prefixes.

## 2.3 KV memory allocator

The paper stores KV tensors in a non-contiguous paged layout.

An important distinction from the previous PagedAttention implementation:

- PagedAttention uses blocks containing multiple token positions.
- RadixAttention describes pages with a size equivalent to one token.

Conceptually:

```
Physical KV memory:

Page 0 -> Token 10
Page 1 -> Token 20
Page 2 -> Token 30
Page 3 -> Token 40
Page 4 -> Token 50
```

Each page contains the key and value rows for one token across all transformer layers.

Suggested structure:

```
struct KVPage {
    std::vector<float> keys;
    std::vector<float> values;

    size_t ref_count = 0;
};
```

The page should contain:

```
Keys:
    Layer 0: [KV_DIM floats]
    Layer 1: [KV_DIM floats]
    ...
    Layer N: [KV_DIM floats]

Values:
    Same layout
```

Index calculation:

```
index = layer * KV_DIM;
```

Each page represents one token position, so there is no token offset inside the page.

The allocator should support:

```
int allocate_page();
void release_page(int page_id);
```

Physical pages are shared between sequences whenever they share a cached prefix.

Reference counting prevents a page from being released while it is still referenced.

## 2.4 Sequence manager

Maintain an unordered map:

```
std::unordered_map<SeqId, Sequence> sequences;
```

Suggested structure:

```
struct Sequence {
    std::vector<int> token_ids;

    int length = 0;

    RadixNode* matched_node = nullptr;

    int matched_prefix_length = 0;
};
```

The sequence manager tracks:

- Sequence identity.
- Token IDs.
- Current sequence length.
- Matched radix tree prefix.
- Active references to cached KV data.

Unlike the baseline implementation, the sequence should not own independent copies of all its KV rows.

Its KV data should be accessed through the radix tree and shared physical pages.

# 3. Prefix Matching

This is the most important operation in RadixAttention.

## 3.1 `match_prefix()`

Given an incoming token sequence, traverse the radix tree and find the longest matching prefix.

Example:

```
Cached sequences:

A: [10, 20, 30, 40, 50]
B: [10, 20, 30, 60, 70]

Incoming:

C: [10, 20, 30, 40, 99]
```

The longest matching prefix is:

```
[10, 20, 30, 40]

Length = 4
```

The algorithm should:

1. Start at the root.
2. Find the outgoing edge matching the next input token.
3. Compare the edge's token sequence against the input.
4. Continue traversing while tokens match.
5. Stop when:
   - No matching edge exists.
   - A token mismatch occurs.
   - The input sequence ends.
6. Return the number of matching tokens and the corresponding tree position.

Suggested interface:

```
struct PrefixMatch {
    RadixNode* node = nullptr;

    int matched_tokens = 0;

    std::vector<int> physical_pages;
};
```

Implementation pseudocode:

```
PrefixMatch match_prefix(
    const int* tokens,
    int n
) {
    PrefixMatch result;

    RadixNode* current = &root;

    int matched = 0;

    while (matched < n) {
        int next_token = tokens[matched];

        auto it = current->children.find(next_token);

        if (it == current->children.end()) {
            break;
        }

        RadixEdge* edge = it->second;

        int common = 0;

        while (
            common < edge->tokens.size() &&
            matched + common < n &&
            edge->tokens[common] == tokens[matched + common]
        ) {
            common++;
        }

        matched += common;

        if (common < edge->tokens.size()) {
            break;
        }

        current = edge->child;
    }

    result.node = current;
    result.matched_tokens = matched;

    return result;
}
```

This is conceptual pseudocode. The actual implementation must also handle a match ending in the middle of an edge.

In particular, if the input ends halfway through an edge, the result must still report the correct matched prefix length.

## 3.2 Matching inside compressed edges

Consider:

```
Cached:
[10, 20, 30, 40, 50]

Incoming:
[10, 20, 30]
```

The entire cached edge may contain five tokens.

However, the incoming sequence only contains three.

The result must be:

```
matched_tokens = 3
```

The implementation must not require a match to end exactly at a radix node.

This is important because radix compression allows edges to contain multiple tokens.

# 4. Insertion and Edge Splitting

After a sequence finishes processing, its token sequence and KV data should be inserted into the radix tree.

The tree must preserve existing shared prefixes.

## 4.1 Basic insertion

Suppose the tree contains:

```
Root
 |
 +---- [10, 20, 30]
```

Insert:

```
[10, 20, 30, 40, 50]
```

The existing edge is a complete prefix.

Append the remaining tokens:

```
Root
 |
 +---- [10, 20, 30]
                  |
                [40, 50]
```

The new edge stores the KV pages corresponding to tokens 40 and 50.

## 4.2 Insertion with divergence

Suppose the tree contains:

```
Root
 |
 +---- [10, 20, 30, 40, 50]
```

Insert:

```
[10, 20, 30, 40, 60]
```

The common prefix is:

```
[10, 20, 30, 40]
```

Split the existing edge:

```
Root
 |
 +---- [10, 20, 30, 40]
                  |
                Node A
                 /  \
                /    \
             [50]   [60]
```

The shared portion retains its original KV pages.

The new branch receives KV pages for token 60.

Do not duplicate the KV pages associated with the common prefix.

## 4.3 Insertion pseudocode

```
void insert(
    const int* tokens,
    int n,
    const std::vector<int>& pages
);
```

Conceptual algorithm:

1. Start at the root.
2. Find an outgoing edge matching the next token.
3. If no edge exists:
   - Create a new edge containing the remaining tokens.
   - Associate the corresponding KV pages.
   - Finish.
4. If an edge exists:
   - Calculate the common prefix length.
5. If the edge matches completely:
   - Continue traversal.
6. If the edge only partially matches:
   - Split the edge.
   - Create an intermediate node.
   - Preserve the existing branch.
   - Add the new branch if necessary.
7. Continue until the complete sequence is inserted.

The implementation must carefully maintain the association between tokens and physical KV pages during edge splitting.

# 5. LRU Eviction

KV memory is limited.

RadixAttention addresses this by implementing an LRU eviction policy on the radix tree.

However, it does not evict arbitrary nodes.

The policy evicts the least recently used leaf first.

## 5.1 Why evict leaves?

Consider:

```
Root
 |
 +---- [10, 20]
          |
        Node A
         /  \
        /    \
      [30]   [40]
       |      |
      [50]   [60]
```

Suppose the branch containing `[50]` is the least recently used.

Evict:

```
Root
 |
 +---- [10, 20]
          |
        Node A
          |
         [40]
          |
         [60]
```

The shared prefix `[10, 20]` remains available.

This is important because other sequences might still reuse it.

After removing a leaf, its parent might become a leaf.

That parent can subsequently become eligible for eviction.

This allows the cache to preserve shared prefixes for as long as possible.

Source: Section 3, page 4.

## 5.2 LRU tracking

Each node should maintain a last-access timestamp or equivalent LRU metadata.

For example:

```
struct RadixNode {
    RadixNode* parent = nullptr;

    std::unordered_map<int, RadixEdge*> children;

    size_t ref_count = 0;

    uint64_t last_access = 0;
};
```

Every successful prefix lookup should update the relevant access metadata.

When memory is needed:

1. Find an evictable leaf.
2. Select the least recently used one.
3. Release its associated KV pages.
4. Remove the leaf from the tree.
5. Check whether its parent has become a leaf.
6. Repeat until sufficient memory is available.

An efficient implementation should maintain an LRU structure rather than scanning the entire tree for every eviction.

One possible approach is an ordered set of evictable leaves.

However, the LRU structure must be updated whenever a node:

- Becomes a leaf.
- Stops being a leaf.
- Is accessed.
- Becomes referenced by an active request.
- Becomes unreferenced.

## 5.3 Reference counting

The paper introduces reference counters to prevent eviction of nodes used by currently running requests.

A node is evictable only when:

```
node->ref_count == 0
```

When a request starts using a cached prefix:

```
node->ref_count++;
```

When the request finishes:

```
node->ref_count--;
```

The implementation must ensure that active sequences protect every cached prefix region they depend on.

For a radix tree, it is useful to define precisely whether reference counts represent:

- References to the node itself.
- References to the entire root-to-node path.
- References to the associated edge and its KV pages.

A practical implementation should use explicit ownership rules and ensure that every physical page remains alive while any active sequence needs it.

Do not evict a page simply because one sequence has finished if another sequence still references it.

# 6. Cache-Aware Scheduling

RadixAttention also introduces a cache-aware scheduling policy.

This is not just a memory-management optimization. It improves cache reuse by choosing which requests should execute first.

## 6.1 Problem with FCFS scheduling

Suppose the waiting queue contains:

```
Request A: [10, 20, 30, 40, 50]
Request B: [10, 20, 30, 60, 70]
Request C: [90, 91, 92, 93]
```

Requests A and B share a prefix.

If the scheduler processes unrelated requests between them, cached prefixes may be evicted before they can be reused.

This leads to cache thrashing.

## 6.2 Longest-shared-prefix-first

The paper sorts waiting requests according to their matched prefix lengths.

Requests with longer cached prefixes receive priority.

For example:

| Request | Cached prefix length |
| ------- | -------------------- |
| A       | 4                    |
| B       | 3                    |
| C       | 0                    |

The scheduler prioritizes:

```
A -> B -> C
```

This improves cache reuse.

The paper establishes that, for an offline batch of requests, visiting the radix tree in depth-first search order achieves an optimal cache hit rate when the cache capacity is at least the maximum request length.

The longest-shared-prefix-first ordering is equivalent to this DFS ordering.

However, the paper also notes that greedy cache-aware scheduling can cause starvation.

Source: Section 3, page 5, and Appendix A.2–A.3.

## 6.3 Scheduling pseudocode from the paper

The paper provides Algorithm 1 for cache-aware scheduling with continuous batching.

Its main steps are:

```
Input:
    Radix tree T
    Memory pool P
    Current running batch B
    Waiting queue Q

1. Get all requests from Q.

2. For every waiting request:
       Find its matching radix prefix.
       Record its matched prefix node and length.

3. Sort requests by matched prefix length.

4. Calculate available memory:
       Evictable cached memory
       + currently available memory.

5. Select requests that fit in the available memory.

6. Increase reference counters for their matched prefixes.

7. Remove selected requests from the waiting queue.

8. Merge selected requests into the running batch.

9. Calculate required memory.

10. Allocate memory.

11. If allocation fails:
        Evict cached entries.
        Retry allocation.

12. Run the batch.

13. For completed requests:
        Decrease prefix reference counters.
        Insert their completed sequences into the radix tree.
```

This scheduling algorithm is part of the broader SGLang runtime.

Your current `KVCache` interface does not expose a waiting queue or batch scheduler, so implement prefix matching and memory management first. Scheduling can be added separately.

# 7. Mapping RadixAttention to the Existing KVCache Interface

The existing interface is:

```
class KVCache {
public:
    virtual ~KVCache() = default;

    virtual SeqId begin_sequence() = 0;
    virtual void end_sequence(SeqId seq) = 0;
    virtual void clear() = 0;

    virtual void store(
        SeqId seq,
        int layer,
        int pos,
        int token,
        const float* k_row,
        const float* v_row
    ) = 0;

    virtual const float* key_row(
        SeqId seq,
        int layer,
        int pos
    ) const = 0;

    virtual const float* value_row(
        SeqId seq,
        int layer,
        int pos
    ) const = 0;

    virtual int stored_tokens(SeqId seq) const = 0;

    virtual int longest_cached_prefix(
        SeqId seq,
        const int* tokens,
        int n
    ) const = 0;

    virtual size_t capacity_tokens() const = 0;
    virtual size_t used_tokens() const = 0;
    virtual KVCacheStats stats() const = 0;
};
```

The most important consideration is that RadixAttention is a global cache, whereas the existing interface is sequence-oriented.

The implementation must bridge these two models.

## 7.1 `begin_sequence()`

```
SeqId begin_sequence() override;
```

Create an empty sequence.

Do not allocate KV memory.

Do not create a radix tree branch yet.

The sequence will be associated with a cached prefix when `longest_cached_prefix()` is called.

## 7.2 `longest_cached_prefix()`

```
int longest_cached_prefix(
    SeqId seq,
    const int* tokens,
    int n
) const override;
```

This method becomes the entry point for RadixAttention.

Algorithm:

1. Find the sequence.
2. Search the global radix tree.
3. Find the longest matching prefix.
4. Record the matching tree location in the sequence.
5. Protect the matched cached prefix from eviction.
6. Return the matched prefix length.

Example:

```
Cached tree:

[10, 20, 30, 40, 50]

Incoming:

[10, 20, 30, 40, 60, 70]

Result:

4
```

### Important interface limitation

The method is declared `const`, but RadixAttention needs to associate the sequence with its matched tree location and update reference counters.

There are two possible approaches:

Option A: Mutable internal state

Use mutable sequence metadata and carefully synchronized tree metadata.

```
mutable std::unordered_map<SeqId, Seq> seqs_;
```

This preserves the public interface.

Option B: Change the interface

Introduce a non-const prefix lookup operation.

This is cleaner from a const-correctness perspective, but requires changing the existing interface.

Prefer Option A if preserving the interface is mandatory.

The implementation must also ensure that prefix matching and reference acquisition are atomic with respect to eviction.

## 7.3 `store()`

```
void store(
    SeqId seq,
    int layer,
    int pos,
    int token,
    const float* k_row,
    const float* v_row
) override;
```

The engine calls this method for each layer and token.

The important behavior is that writes should populate the shared KV storage associated with the sequence's current radix-tree path.

Example:

```
Cached prefix:
[10, 20, 30, 40]

Incoming sequence:
[10, 20, 30, 40, 50, 60]
```

The first four positions already exist.

The engine only needs to compute and store positions 4 and 5.

The implementation should:

1. Find the sequence.
2. Determine whether the position belongs to the matched cached prefix.
3. If it belongs to the cached prefix, do not overwrite the shared KV data.
4. Otherwise, allocate a new physical page.
5. Store the K/V rows.
6. Associate the new page with the sequence's pending suffix.
7. Record the token ID.

Do not allocate duplicate pages for cached prefix positions.

### Important: Pending suffixes

The tree should not necessarily be modified on every individual layer write.

A position's KV data is complete only after all layers have written their rows.

The implementation should track whether each position has received all its layer data.

Only complete token KV entries should become available for reuse.

The exact mechanism may use:

```
struct PendingToken {
    int token_id;

    std::vector<bool> written_layers;

    int physical_page_id;
};
```

Alternatively, track completion using a per-sequence position bitmap.

## 7.4 `key_row()` and `value_row()`

```
const float* key_row(
    SeqId seq,
    int layer,
    int pos
) const override;
```

These methods must support both:

- Cached prefix positions.
- Newly generated sequence positions.

The lookup process is:

1. Find the sequence.
2. Determine the physical page associated with the logical position.
3. Calculate the layer offset.
4. Return the appropriate row.

Conceptually:

```
int page_id = sequence.page_mapping[pos];

return pages[page_id].keys.data()
    + layer * KV_DIM;
```

The same applies to values.

Increment `row_fetches_` on every call.

Pointers must remain valid while the associated pages are retained.

## 7.5 `stored_tokens()`

Return the number of contiguous, fully valid token positions.

This should include:

- Cached prefix positions.
- Newly computed positions.

For example:

```
Cached prefix: 4 tokens
Newly computed: 3 tokens

stored_tokens() = 7
```

Do not count partially written token positions.

## 7.6 `end_sequence()`

```
void end_sequence(SeqId seq) override;
```

When a sequence finishes:

1. Release its active references to cached tree nodes.
2. Make eligible cached nodes available for eviction.
3. Insert its completed token sequence and KV mappings into the radix tree.
4. Remove temporary sequence metadata.
5. Preserve the physical KV pages that are now owned by the global cache.

Important: Unlike the baseline implementation, ending a sequence must not automatically release all its KV pages.

The completed sequence's KV data should remain cached for future requests.

Only pages that are no longer referenced by the radix tree or active sequences should be released.

## 7.7 `clear()`

Release:

- All active sequences.
- All radix tree nodes and edges.
- All cached KV pages.
- All temporary sequence metadata.
- All allocator bookkeeping.

Reset the statistics consistently with the baseline implementation.

## 7.8 Capacity and usage

```
size_t capacity_tokens() const override;
size_t used_tokens() const override;
```

The capacity is the configured maximum number of token pages.

Since the paper uses one page per token:

```
capacity_tokens() = maximum_number_of_pages;
```

`used_tokens()` should count the number of allocated physical token pages.

This includes both:

- Pages referenced by active sequences.
- Pages retained by the radix tree for future reuse.

A token shared by multiple sequences must only be counted once.

## 7.9 Statistics

The existing `KVCacheStats` structure contains:

```
struct KVCacheStats {
    size_t sequences = 0;
    size_t tokens = 0;
    size_t bytes = 0;
    size_t row_fetches = 0;
    size_t prefix_checks = 0;
    size_t prefix_hits = 0;
    size_t evictions = 0;
};
```

Recommended interpretation:

| Field           | Meaning                                                     |
| --------------- | ----------------------------------------------------------- |
| `sequences`     | Currently active sequences                                  |
| `tokens`        | Currently allocated physical token pages                    |
| `bytes`         | KV payload bytes allocated                                  |
| `row_fetches`   | Number of key/value row accesses                            |
| `prefix_checks` | Number of prefix lookup calls                               |
| `prefix_hits`   | Number of lookups returning more than zero                  |
| `evictions`     | Number of physical token positions removed through eviction |

The distinction between active sequences and cached token pages is important.

A completed sequence should no longer count toward `sequences`, but its KV pages can remain in the cache.

# 8. Important Memory Management Considerations

## 8.1 Active versus cached memory

The paper does not maintain a separate, permanently reserved cache memory pool.

Instead, cached tokens and currently running requests share the same memory pool.

Conceptually:

```
                 Shared KV Memory Pool
                          |
             +------------+------------+
             |                         |
       Active requests            Cached prefixes
             |                         |
       Currently used KV         Retained KV pages
             |                         |
             +------------+------------+
                          |
                    Physical pages
```

When memory is needed for new requests, cached entries can be evicted to make room.

This is different from simply reserving a fixed percentage of memory for prefix caching.

For your implementation, a fixed `capacity_tokens()` is still useful as a policy limit, but the pool must be shared between active and cached pages.

## 8.2 Page ownership

A physical page can be referenced by:

- An active sequence.
- A radix tree edge.
- Multiple logical sequences sharing a prefix.

Use explicit ownership or reference counting.

Never release a physical page while it is still needed by an active sequence or retained cache entry.

## 8.3 Edge splitting and page ownership

When splitting an edge, the original edge's token and page arrays must be divided consistently.

Example:

```
Before:

Edge:
Tokens: [10, 20, 30, 40, 50]
Pages:  [P0, P1, P2, P3, P4]
```

After splitting:

```
Shared edge:
Tokens: [10, 20, 30]
Pages:  [P0, P1, P2]

Branch:
Tokens: [40, 50]
Pages:  [P3, P4]
```

The physical pages must not be copied.

Only their ownership and tree references should change.

## 8.4 Eviction and active references

Suppose:

```
Root
 |
 +---- [10, 20]
          |
         [30]
          |
         [40]
```

An active sequence is using the prefix `[10, 20, 30]`.

The cache must not evict the shared prefix even if it is old.

Only eligible, unreferenced leaves can be evicted.

After a leaf is removed, its parent may become eligible for eviction.

## 8.5 Thread safety

The paper's runtime uses asynchronous execution and continuous batching.

Although your initial implementation may be single-threaded, the design should avoid unsafe concurrent modifications.

Operations that need synchronization include:

- Prefix matching.
- Reference counter updates.
- Edge splitting.
- Insertion.
- Eviction.
- Page allocation and release.

A global mutex is acceptable for the initial implementation.

More granular locking can be introduced later.

# 9. Implementation Architecture

Recommended internal architecture:

```
                         RadixKVCache
                               |
              +----------------+----------------+
              |                |                |
         Radix Tree       Page Allocator   Sequence Manager
              |                |                |
         Prefix lookup     Physical pages    SeqId mapping
         Edge splitting    Free page pool    Active references
         Insertion         Reference counts  Pending writes
         LRU eviction      Memory accounting  Token IDs
```

Suggested class organization:

```
class RadixKVCache : public KVCache {
public:
    SeqId begin_sequence() override;
    void end_sequence(SeqId seq) override;
    void clear() override;

    void store(
        SeqId seq,
        int layer,
        int pos,
        int token,
        const float* k_row,
        const float* v_row
    ) override;

    const float* key_row(
        SeqId seq,
        int layer,
        int pos
    ) const override;

    const float* value_row(
        SeqId seq,
        int layer,
        int pos
    ) const override;

    int stored_tokens(SeqId seq) const override;

    int longest_cached_prefix(
        SeqId seq,
        const int* tokens,
        int n
    ) const override;

    size_t capacity_tokens() const override;
    size_t used_tokens() const override;
    KVCacheStats stats() const override;

private:
    struct KVPage {
        std::vector<float> keys;
        std::vector<float> values;

        size_t ref_count = 0;
    };

    struct RadixNode {
        RadixNode* parent = nullptr;

        std::unordered_map<int, RadixEdge*> children;

        size_t ref_count = 0;

        uint64_t last_access = 0;
    };

    struct RadixEdge {
        std::vector<int> tokens;

        RadixNode* child = nullptr;

        std::vector<int> physical_pages;
    };

    struct Sequence {
        std::vector<int> token_ids;
        std::vector<int> page_mapping;

        int length = 0;
        int matched_prefix_length = 0;

        RadixNode* matched_node = nullptr;
    };

    RadixNode root_;

    std::vector<KVPage> pages_;
    std::vector<int> free_pages_;

    mutable std::unordered_map<SeqId, Sequence> sequences_;

    SeqId next_ = 0;

    size_t used_tokens_ = 0;
    size_t evictions_ = 0;

    mutable size_t row_fetches_ = 0;
    mutable size_t prefix_checks_ = 0;
    mutable size_t prefix_hits_ = 0;
};
```

This is a conceptual organization. The exact ownership model and edge representation can be adjusted to avoid unnecessary allocations and pointer instability.

In particular, use stable node and edge storage. Avoid storing raw pointers into vectors that may reallocate.

# 10. Implementation Order

Implement the following milestones in order.

## Phase 1: Radix tree

- Implement radix nodes.
- Implement compressed edges.
- Implement prefix matching.
- Handle matching inside compressed edges.
- Implement edge splitting.
- Implement insertion.
- Implement tree traversal and debugging output.

At this stage, use token IDs only. Do not integrate KV memory yet.

## Phase 2: KV page management

- Implement one-token physical KV pages.
- Implement physical page allocation.
- Implement page release.
- Implement reference counting.
- Implement layer-based row indexing.
- Implement stable row pointers.

## Phase 3: Interface integration

- Implement `begin_sequence()`.
- Implement `longest_cached_prefix()`.
- Associate sequences with matched radix paths.
- Implement cached-prefix row lookup.
- Implement incremental suffix storage.
- Implement `stored_tokens()`.
- Implement `end_sequence()`.
- Implement `clear()`.
- Implement accurate statistics.

## Phase 4: LRU eviction

- Track node access times.
- Maintain eligible leaf nodes.
- Implement least-recently-used leaf selection.
- Prevent eviction of referenced nodes.
- Release physical pages correctly.
- Prune newly eligible parent leaves.
- Implement capacity exhaustion handling.

## Phase 5: Benchmarking

- Compare against `BaselineKVCache`.
- Compare against the PagedAttention implementation.
- Benchmark identical prefixes.
- Benchmark partially shared prefixes.
- Benchmark unrelated sequences.
- Benchmark multi-turn conversations.
- Measure memory savings.
- Measure prefix lookup overhead.
- Measure eviction overhead.

# 11. Acceptance Criteria

The implementation must satisfy the following tests.

| Test                               | Expected result                                             |
| ---------------------------------- | ----------------------------------------------------------- |
| Insert first sequence              | Radix tree contains its token path                          |
| Insert identical sequence          | No duplicate KV pages                                       |
| Insert sequence with shared prefix | Existing prefix pages are reused                            |
| Insert sequence with divergence    | Existing edge is split correctly                            |
| Match complete sequence            | Full sequence length returned                               |
| Match partial sequence             | Correct partial prefix length returned                      |
| Match no prefix                    | Zero returned                                               |
| Match ending inside edge           | Correct prefix length returned                              |
| Read shared KV row                 | Correct original K/V values                                 |
| Insert multiple layers             | One page per token, not per layer                           |
| End sequence                       | KV pages remain cached                                      |
| Multiple sequences share prefix    | Shared pages remain valid                                   |
| Evict leaf                         | Only eligible leaf pages are released                       |
| Evict shared branch                | Shared ancestors remain available                           |
| Active sequence references node    | Node cannot be evicted                                      |
| Capacity exhausted                 | LRU eviction is attempted                                   |
| Clear cache                        | Tree and page allocator are empty                           |
| Stats                              | Active sequences, physical pages and evictions are accurate |

## Particularly important tests

### Test 1: Shared prefix

Insert:

```
A = [10, 20, 30, 40, 50]
```

Then insert:

```
B = [10, 20, 30, 40, 60]
```

Expected:

```
Shared KV pages: 10, 20, 30, 40

A-specific page: 50
B-specific page: 60

Total physical pages: 6
```

Not 10.

### Test 2: Multiple levels of sharing

Insert:

```
A = [10, 20, 30, 40, 50]
B = [10, 20, 30, 40, 60]
C = [10, 20, 70, 80]
```

Expected tree:

```
Root
 |
 +---- [10, 20]
          |
          +---- [30, 40]
          |        |
          |       Node
          |       /  \
          |     [50] [60]
          |
          +---- [70, 80]
```

Expected physical pages:

```
10, 20, 30, 40, 50, 60, 70, 80
```

Total: 8 pages.

### Test 3: Edge splitting

Insert:

```
A = [10, 20, 30, 40, 50]
```

Then insert:

```
B = [10, 20, 30, 60, 70]
```

Expected:

```
Root
 |
 +---- [10, 20, 30]
              |
             Node
             /  \
            /    \
         [40,50] [60,70]
```

Verify that the original pages for `[10, 20, 30]` are retained.

### Test 4: LRU eviction

Insert:

```
A = [10, 20, 30, 40]
B = [10, 20, 30, 50]
```

Access A more recently than B.

Force capacity exhaustion.

Expected:

- B's exclusive leaf is evicted first.
- The shared prefix `[10, 20, 30]` remains cached.
- A remains accessible.
- Physical pages belonging exclusively to B are released.

### Test 5: Active references

Start a sequence that reuses:

```
[10, 20, 30]
```

Force memory pressure.

Expected:

- The active sequence's required prefix cannot be evicted.
- Other eligible cached leaves can be evicted.
- The active sequence continues to return correct KV rows.

# 12. Important Distinction: PagedAttention vs. RadixAttention

These two papers describe complementary optimizations.

| Property             | PagedAttention                             | RadixAttention                          |
| -------------------- | ------------------------------------------ | --------------------------------------- |
| Primary purpose      | Efficient KV memory allocation             | KV cache reuse                          |
| Main data structure  | Physical block pool and block tables       | Radix tree                              |
| Memory organization  | Fixed-size blocks                          | One-token pages in the described design |
| Sharing              | Copy-on-write and shared blocks            | Arbitrary shared prefixes               |
| Prefix matching      | Not its primary focus                      | Core functionality                      |
| Eviction             | Preemption and swapping mechanisms         | LRU leaf eviction                       |
| Memory fragmentation | Reduces internal/external fragmentation    | Reuses existing cached pages            |
| Scheduling           | General request scheduling                 | Longest-shared-prefix-first             |
| Cross-request reuse  | Supported through block sharing mechanisms | Automatic tree-based prefix reuse       |

Recommended architecture for your inference engine:

Eventually, these two mechanisms should work together.

```
                     Inference Engine
                            |
                         KVCache
                            |
                       RadixKVCache
                            |
                       Radix Tree
                            |
                     Physical KV Pages
                            |
                   Paged Memory Allocator
```

The radix tree determines which token positions can be reused.

The physical allocator determines where their KV rows reside.

The important point is that RadixAttention does not replace the need for physical memory management.

# 13. Scope Restrictions

Do not implement the entire SGLang system.

The paper describes three major runtime optimizations:

1. RadixAttention — KV cache reuse.
2. Compressed finite-state machines — faster constrained decoding.
3. API speculative execution — reducing latency and input-token costs for API-only models.

Only the first is relevant to the current `KVCache` interface.

Additionally, cache-aware scheduling belongs outside the initial KV cache implementation because the interface does not expose a waiting queue or batch scheduler.

Implement the radix tree, prefix reuse, page ownership, and LRU eviction first.

# 14. Final Implementation Recommendation

The most important design decision is to separate the global cache from individual sequences.

The baseline implementation treats every sequence as the owner of its own KV memory.

RadixAttention changes this model:

- The radix tree owns cached token prefixes.
- Physical pages contain the actual KV tensors.
- Active sequences reference existing cached prefixes.
- Newly computed suffixes receive new pages.
- Completed sequences insert their KV data into the global tree.
- LRU eviction removes unreferenced leaves when memory is required.

The implementation should prioritize:

1. Correct radix tree construction.
2. Correct edge splitting.
3. Correct longest-prefix matching.
4. Shared physical KV pages.
5. Correct sequence-to-tree references.
6. LRU leaf eviction.
7. Accurate memory accounting.

Do not begin with cache-aware scheduling or complex concurrency.

First establish that multiple sequences can share KV pages correctly, that divergent sequences split tree edges correctly, and that eviction never invalidates an active sequence.

The ultimate correctness criterion is that two sequences with an identical token prefix must be able to read the exact same physical KV data for that prefix, without recomputing or duplicating those KV tensors.

Reference: Zheng et al., SGLang: Efficient Execution of Structured Language Model Programs, arXiv:2312.07104v2 (2024). Particularly Section 3, Figure 3, and Appendix A.1–A.3.