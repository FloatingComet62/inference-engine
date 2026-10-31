# Implementation Specification: PagedAttention KV Cache in C++

Source: Efficient Memory Management for Large Language Model Serving with PagedAttention — vLLM, SOSP 2023.

## Objective

Implement a paged KV cache in C++ that replaces the existing `BaselineKVCache` implementation while preserving the existing `KVCache` interface.

The implementation should follow the memory-management principles described in the PagedAttention paper.

The central idea is:

Logical token positions must not correspond to contiguous physical memory positions.

Instead, divide KV memory into fixed-size blocks and maintain a mapping between logical blocks and physical blocks.

The implementation should initially focus on basic paging, dynamic allocation, memory management, and correctness. Advanced features such as copy-on-write, swapping, and eviction can be implemented later.

# 1. Understanding PagedAttention

## 1.1 Problem with the baseline implementation

The baseline implementation allocates the entire KV cache for every sequence upfront.

Conceptually:

```
Sequence 0:
┌──────────────────────────────────────────────┐
│ Layer 0: Token 0, Token 1, ... MAX_SEQ       │
│ Layer 1: Token 0, Token 1, ... MAX_SEQ       │
│ ...                                          │
│ Layer N: Token 0, Token 1, ... MAX_SEQ       │
└──────────────────────────────────────────────┘
```

This creates two problems:

- Internal fragmentation: memory is allocated for tokens that might never be generated.
- External fragmentation: contiguous allocations make it difficult to utilize available memory efficiently.

## 1.2 Fixed-size KV blocks

PagedAttention divides the KV cache into fixed-size blocks.

For example, with a block size of 4:

```
Logical sequence (10 tokens):

Block 0: [0] [1] [2] [3]
Block 1: [4] [5] [6] [7]
Block 2: [8] [9] [ ] [ ]
```

These logical blocks can map to arbitrary physical blocks:

```
Logical block     Physical block
--------------------------------
0                 7
1                 2
2                 11
```

Physical memory:

```
Physical block 2:  Tokens 4-7
Physical block 7:  Tokens 0-3
Physical block 11: Tokens 8-9
```

The sequence sees a continuous logical sequence of tokens, but the underlying physical memory is not necessarily contiguous.

This is analogous to virtual memory paging in operating systems.

The cache manager translates logical block indices into physical block indices whenever KV data is accessed.

Important: Never assume that consecutive token positions reside in consecutive physical memory locations.

Source: Paper, Sections 3.1 and 4.1–4.3, pages 4–6.

# 2. Data Structures

The implementation should have three primary components:

1. Physical block allocator.
2. Logical block tables.
3. Sequence manager.

## 2.1 Physical block allocator

This component manages a fixed pool of physical KV blocks.

Each physical block contains:

- Key rows for a fixed number of tokens.
- Value rows for a fixed number of tokens.
- A reference count for potential future block sharing.

Suggested structure:

```
struct PhysicalBlock {
    std::vector<float> keys;
    std::vector<float> values;

    int ref_count = 0;
};
```

The actual memory layout should be contiguous.

For example, assuming:

- `BLOCK_SIZE = 16`
- `N_LAYERS = 22`
- `KV_DIM = 2048`

One possible layout is:

```
Physical block:

Keys:
    Layer 0:
        Token 0: [2048 floats]
        Token 1: [2048 floats]
        ...
        Token 15: [2048 floats]

    Layer 1:
        ...

Values:
    Same layout
```

The corresponding index calculation is:

```
index = (layer * BLOCK_SIZE + offset) * KV_DIM;
```

Where:

```
offset = pos % BLOCK_SIZE;
```

Maintain a free-block pool, preferably using a vector of available block IDs.

Required internal operations:

```
int allocate_block();
void release_block(int block_id);
```

Allocation removes an ID from the free pool.

Release returns it to the free pool only when its reference count reaches zero.

### Allocation requirements

- Physical blocks should be allocated dynamically from a fixed-size pool.
- Blocks should not be allocated for unused future tokens.
- Each physical block must contain storage for all transformer layers.
- Allocation must fail gracefully when no physical blocks are available.

## 2.2 Logical block table

Each sequence maintains a mapping between its logical blocks and physical blocks.

Suggested structures:

```
struct BlockEntry {
    int physical_block_id;
    int filled;
};

struct Sequence {
    std::vector<BlockEntry> blocks;

    std::vector<int> token_ids;

    int length = 0;
};
```

Example:

```
Sequence 0:

Logical block     Physical block     Filled positions
-----------------------------------------------------
0                 7                  4
1                 2                  4
2                 11                 2
```

To access token position 9:

```
int logical_block = 9 / BLOCK_SIZE;
int offset = 9 % BLOCK_SIZE;

int physical_block =
    sequence.blocks[logical_block].physical_block_id;
```

The physical block ID is then used to locate the actual KV data.

The logical block table is the central abstraction of the implementation.

All KV accesses must go through this mapping.

## 2.3 Sequence manager

Maintain an unordered map:

```
std::unordered_map<SeqId, Sequence> sequences;
```

This is similar to the existing `BaselineKVCache`.

The sequence manager coordinates:

- Sequence creation.
- Logical block allocation.
- Physical block allocation.
- Token storage.
- Sequence destruction.
- Memory accounting.

# 3. Mapping the Existing KVCache Interface

The implementation must preserve the existing public interface.

## 3.1 `begin_sequence()`

```
SeqId begin_sequence() override;
```

Create a new empty sequence.

Unlike the baseline implementation:

- Do not allocate the entire KV cache.
- Do not allocate any physical blocks initially.
- Initialize an empty block table.
- Assign a unique `SeqId`.

Memory should only be allocated when tokens are stored.

## 3.2 `store()`

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

This is the most important method.

Algorithm:

1. Find the sequence.
2. Calculate its logical block index:
   ```
   int logical = pos / BLOCK_SIZE;
   ```
3. Calculate its offset inside the block:
   ```
   int offset = pos % BLOCK_SIZE;
   ```
4. If the required logical block does not exist, allocate a physical block.
5. Record the logical-to-physical mapping in the sequence's block table.
6. Copy the key and value rows into the corresponding physical block.
7. Record the token ID at that position.
8. Update the block's filled-position count and sequence length.

Pseudocode:

```
if (logical >= sequence.blocks.size()) {
    allocate_and_append_block(sequence);
}

int physical =
    sequence.blocks[logical].physical_block_id;

copy(k_row, physical_block.keys, layer, offset);
copy(v_row, physical_block.values, layer, offset);

sequence.token_ids[pos] = token;
```

### Important: Layered storage

`store()` is called separately for each layer.

However, block allocation must be shared across all layers.

Do not allocate a separate physical block for every layered `store()` call.

For example:

```
store(seq, 0, 0, ...)
store(seq, 1, 0, ...)
store(seq, 2, 0, ...)
...
store(seq, N, 0, ...)
```

All these calls must write into the same physical block, at different layer offsets.

The paper explicitly allows managing all layers within one physical block. This is the preferred arrangement for this interface.

### Sequential writes

The paper assumes blocks are filled from left to right.

The implementation should enforce sequential writes or explicitly handle out-of-order writes.

Do not report a longer valid prefix when there are holes in the stored positions.

## 3.3 `key_row()`

```
const float* key_row(
    SeqId seq,
    int layer,
    int pos
) const override;
```

Translate the logical position into a physical position.

Algorithm:

```
int logical = pos / BLOCK_SIZE;
int offset = pos % BLOCK_SIZE;

int physical =
    sequence.blocks[logical].physical_block_id;

return physical_block.keys.data()
    + (layer * BLOCK_SIZE + offset) * KV_DIM;
```

The method must return a pointer to the correct key row.

Increment `row_fetches_` on every invocation.

Return `nullptr` for invalid or unavailable positions, following the conventions established for the cache.

## 3.4 `value_row()`

```
const float* value_row(
    SeqId seq,
    int layer,
    int pos
) const override;
```

Identical to `key_row()`, except that it accesses the values allocation.

Increment `row_fetches_` on every invocation.

## 3.5 `stored_tokens()`

```
int stored_tokens(SeqId seq) const override;
```

Return the number of valid leading positions.

For example:

```
Block 0: 16 filled
Block 1: 16 filled
Block 2:  7 filled

stored_tokens() = 39
```

For a normal, non-evicted sequence, this is simply its length.

## 3.6 `longest_cached_prefix()`

```
int longest_cached_prefix(
    SeqId seq,
    const int* tokens,
    int n
) const override;
```

This method checks how much of an incoming prompt is already cached.

The cache stores token IDs associated with each position.

Algorithm:

```
int limit = std::min(n, stored_tokens(seq));

int matched = 0;

while (
    matched < limit &&
    sequence.token_ids[matched] == tokens[matched]
) {
    matched++;
}

return matched;
```

Example:

```
Cached:  [10, 20, 30, 40, 50]
Incoming:[10, 20, 30, 99, 50]

Result: 3
```

The inference engine can start prefill at position 3.

Statistics:

- Increment `prefix_checks` on every invocation.
- Increment `prefix_hits` when the result is greater than zero.

Note that the paper's prefix-sharing mechanism is broader than this interface. It supports sharing actual physical blocks between sequences. The current interface exposes prefix lookup, but does not expose explicit cross-sequence prefix sharing.

# 4. Lifecycle and Memory Management

## 4.1 `end_sequence()`

```
void end_sequence(SeqId seq) override;
```

When a sequence finishes:

1. Iterate over its logical block table.
2. Decrement the reference count of each physical block.
3. Return blocks with reference count zero to the free pool.
4. Remove the sequence from the sequence map.

Do not simply erase the sequence's block table without releasing its physical allocations.

## 4.2 `clear()`

```
void clear() override;
```

Release all active sequences and their physical blocks.

Reset:

- Sequence mappings.
- Physical block allocations.
- Free-block pool.
- Used-token counters.

Decide whether benchmark counters should reset on `clear()` and maintain consistent behavior across cache implementations.

## 4.3 Capacity

For block size \\(B\\) and \\(N\\) physical blocks:

\\[ \text{capacity\\\_tokens} = B \times N \\]

Actual payload memory:

\\[ \text{bytes} = 2 \times N \times B \times L \times D \times sizeof(float) \\]

Where:

- \\(L\\): Number of transformer layers.
- \\(D\\): KV row dimension.
- 2: Keys and values.

### Important distinction

`used_tokens()` should report the number of occupied logical token positions across active sequences, not the number of allocated physical slots.

`stats().bytes` should report actual KV payload bytes allocated for physical blocks, excluding bookkeeping.

The paper's design limits internal fragmentation to unused positions in the final block of each sequence.

# 5. Advanced Features in the Paper

These features are important to understand, but they should not all be part of the initial implementation.

## 5.1 Reference counting and copy-on-write

The paper supports multiple sequences sharing the same physical blocks.

For example, parallel sampling:

```
                Original prompt
                      |
                Shared KV blocks
                   /     \
                  /       \
             Sample A   Sample B
                 |          |
             New tokens  New tokens
```

Both sequences initially reference the same physical blocks.

When one sequence needs to modify a shared block:

1. Check its reference count.
2. If the reference count is greater than one, allocate a new physical block.
3. Copy the old block into the new block.
4. Update the sequence's mapping.
5. Decrement the old block's reference count.
6. Perform the write on the new block.

This is called copy-on-write (COW).

It allows sharing prompt KV states without copying the entire prompt for every output sequence.

The paper also applies this concept to beam search, where sequences can share multiple blocks and diverge dynamically.

Source: Paper, Section 4.4, pages 7–8.

## 5.2 Eviction and preemption

When GPU memory runs out, vLLM can preempt sequences.

The paper uses:

- FCFS scheduling.
- Preemption of the latest-arriving requests first.
- All-or-nothing eviction of a sequence's blocks.
- Gang scheduling for sequences belonging to the same request.

Two recovery mechanisms are described:

| Mechanism     | Operation                                                                            |
| ------------- | ------------------------------------------------------------------------------------ |
| Swapping      | Move KV blocks from GPU memory to CPU memory and restore them later.                 |
| Recomputation | Discard KV blocks and regenerate them from the original prompt and generated tokens. |

These mechanisms are not directly expressible through the current interface because it does not expose suspension, restoration, or an eviction callback.

For the initial implementation, leave eviction out rather than silently dropping arbitrary blocks and breaking the contiguous-prefix invariant.

If capacity is exhausted, define an explicit allocation-failure policy.

Source: Paper, Section 4.5, page 8.

# 6. Recommended C++ Architecture

Separate the implementation into two internal components, even if they live in one `.cpp` file.

```
                    PagedKVCache
                          |
             +------------+------------+
             |                         |
       Sequence Manager          Physical Allocator
             |                         |
       SeqId -> Sequence          Physical block pool
       Logical block tables      Free-block list
       Token IDs                  Reference counts
       Sequence lengths           Allocation accounting
```

Suggested class structure:

```
class PagedKVCache : public KVCache {
public:
    // All KVCache interface methods.

private:
    struct PhysicalBlock {
        std::vector<float> keys;
        std::vector<float> values;
        int ref_count = 0;
    };

    struct BlockEntry {
        int physical_block_id;
        int filled;
    };

    struct Seq {
        std::vector<BlockEntry> blocks;
        std::vector<int> ids;
        int len = 0;
    };

    std::vector<PhysicalBlock> blocks_;
    std::vector<int> free_blocks_;

    std::unordered_map<SeqId, Seq> seqs_;

    SeqId next_ = 0;

    size_t used_tokens_ = 0;

    mutable size_t row_fetches_ = 0;
    mutable size_t prefix_checks_ = 0;
    mutable size_t prefix_hits_ = 0;

    size_t evictions_ = 0;
};
```

This is a suggested organization, not code copied from the paper.

# 7. Implementation Order

Implement the following milestones in order.

## Phase 1: Basic paging

- Physical block allocator.
- Logical block tables.
- Dynamic allocation in `store()`.
- Logical-to-physical translation in `key_row()` and `value_row()`.
- `stored_tokens()`.
- `longest_cached_prefix()`.
- `end_sequence()` and `clear()`.
- Accurate statistics.

## Phase 2: Correctness and benchmarking

- Test sequences crossing block boundaries.
- Test multiple sequences sharing the physical pool.
- Test partially filled blocks.
- Test capacity exhaustion.
- Compare results against `BaselineKVCache`.
- Measure memory usage and row-fetch overhead.

## Phase 3: Advanced paging

- Reference counting.
- Sequence forking.
- Copy-on-write.
- Shared-prefix caching.

## Phase 4: Advanced memory management

- Sequence eviction.
- CPU swapping.
- Recomputation.

# 8. Important Design Considerations

These are important implementation details specific to the existing interface.

### 8.1 Stable row pointers

`key_row()` and `value_row()` return raw pointers.

Physical block storage must not relocate while those pointers are expected to remain valid.

Preallocate the physical block pool rather than continually resizing individual blocks.

### 8.2 Layered storage

Allocate one logical physical block containing storage for all layers.

The paper explicitly discusses this arrangement and also allows separate block tables per layer.

The current interface is simpler with one shared mapping.

### 8.3 Token consistency

The same position receives the same token ID across layered `store()` calls.

Only record the token ID once logically. Do not count each layer's write as another token.

### 8.4 Position validity

The interface promises that positions in `[0, stored_tokens)` are readable.

Enforce sequential writes or explicitly handle out-of-order writes.

Do not report a longer valid prefix when there are holes.

### 8.5 Accounting distinction

Allocated block capacity and valid token positions are different quantities.

- `used_tokens()` counts valid logical positions.
- Physical memory usage accounts for complete allocated blocks.
- `stats().bytes` reports KV payload allocation, excluding bookkeeping.

### 8.6 No implicit sharing initially

Reference counting is useful for future copy-on-write support.

Initially, every sequence can own its blocks exclusively.

# 9. Acceptance Criteria

The implementation must satisfy the following tests:

| Test                         | Expected result                             |
| ---------------------------- | ------------------------------------------- |
| Begin empty sequence         | Zero physical blocks allocated              |
| Store first token            | One block allocated                         |
| Store B tokens               | Exactly one block allocated                 |
| Store B+1 tokens             | Exactly two blocks allocated                |
| Store across multiple layers | No duplicate block allocations              |
| Read a stored row            | Exact original K/V values                   |
| Read across block boundary   | Correct physical block translation          |
| Compare cached prefix        | Correct longest matching prefix             |
| End sequence                 | All exclusively owned blocks released       |
| Clear cache                  | No active sequences or allocated blocks     |
| Multiple sequences           | Physical blocks independently allocated     |
| Capacity exhausted           | Defined allocation-failure behavior         |
| Stats                        | Counters and memory accounting are accurate |

Most important correctness test:

Create two sequences, store different data in them, deliberately arrange for their logical blocks to map to non-consecutive physical blocks, and verify that every `key_row()` and `value_row()` returns the correct data.

# 10. Final Architectural Recommendation

Do not try to implement the entire vLLM system.

The current interface is a KV storage abstraction, not a complete LLM scheduler.

Implement the core PagedAttention memory-management mechanism:

1. Fixed-size physical blocks.
2. Per-sequence logical block tables.
3. Dynamic block allocation.
4. Logical-to-physical address translation.
5. Correct lifecycle management.
6. Accurate memory accounting.

Leave scheduling, swapping, recomputation, and sequence forking outside the initial scope.

This will provide a proper paged KV cache that can replace `BaselineKVCache` without requiring changes to the inference engine's public interface.

Reference: Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023, particularly Sections 3.1, 4.1–4.5, 5.2, and 7.2.