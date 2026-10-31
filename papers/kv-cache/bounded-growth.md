# Implementation Specification: StreamingLLM and Attention Sinks in C++

Source: Efficient Streaming Language Models with Attention Sinks — Guangxuan Xiao et al., MIT, Meta AI and CMU, arXiv:2309.17453v1, September 2023.

## Objective

Implement the StreamingLLM KV cache mechanism in the existing C++ inference engine.

The goal is to enable autoregressive generation over arbitrarily long input streams while maintaining:

- Bounded KV cache memory.
- Constant-size attention context.
- Stable language-modeling performance beyond the model's training context length.
- Efficient incremental decoding without recomputing the entire sliding window.

The implementation should integrate with the existing `KVCache` interface without changing its public API.

The central idea is:

Keep the KV states of the first few tokens permanently, while maintaining a rolling cache of the most recent tokens.

These initial tokens are called attention sinks. They collect substantial attention scores even when their semantic content is irrelevant.

The paper demonstrates that retaining four initial tokens alongside the recent-token window is generally sufficient for pretrained models.

This is fundamentally different from RadixAttention:

- RadixAttention reuses KV data across different sequences.
- StreamingLLM retains selected KV data within one continuously growing sequence.

The two techniques can eventually be combined, but they solve different problems.

## 1. Understanding StreamingLLM

### 1.1 The problem with ordinary KV caching

During autoregressive inference, every generated token produces key and value tensors that are stored for subsequent attention operations.

Ordinary KV caching retains all previous tokens:

```
Input: [10, 20, 30, 40, 50, 60, 70, 80]

KV cache:
[10, 20, 30, 40, 50, 60, 70, 80]
```

As generation continues, the cache grows indefinitely.

This creates two problems:

1. Memory consumption increases with sequence length.
2. Attention computation becomes increasingly expensive.

A natural solution is sliding-window attention.

### 1.2 Why ordinary sliding-window attention fails

Suppose the cache capacity is eight tokens.

A rolling cache might look like this:

```
Initially:
[0, 1, 2, 3, 4, 5, 6, 7]

Generate token 8:
[1, 2, 3, 4, 5, 6, 7, 8]

Generate token 9:
[2, 3, 4, 5, 6, 7, 8, 9]
```

This keeps memory bounded, but the paper demonstrates a serious problem.

When the initial tokens are evicted, language-modeling performance collapses.

The authors observed that pretrained autoregressive models assign unusually high attention scores to initial tokens, regardless of their semantic relevance.

These tokens function as attention sinks.

Removing their KV states changes the attention-score distribution and therefore changes model behavior.

The paper demonstrates this in Section 3.1 and Figures 2–3.

### 1.3 The StreamingLLM solution

Instead of retaining only the most recent tokens, divide the cache into two regions:

1. Attention sinks: the first few tokens.
2. Rolling KV cache: the most recent tokens.

For a cache capacity of eight and four sink tokens:

```
Initial:
[0, 1, 2, 3, 4, 5, 6, 7]

Generate token 8:
[0, 1, 2, 3, 5, 6, 7, 8]

Generate token 9:
[0, 1, 2, 3, 6, 7, 8, 9]

Generate token 10:
[0, 1, 2, 3, 7, 8, 9, 10]
```

Illustration based on the rolling-cache mechanism in Figure 4, page 5.

The first four tokens remain permanently cached.

The remaining four slots are continuously overwritten by newer tokens.

This gives a fixed memory footprint regardless of the total generated sequence length.

The paper reports that four initial tokens generally restore performance close to the sliding-window-with-recomputation baseline.

Source: Sections 3.1–3.2, pages 4–5.

# 2. Core Data Structures

The implementation needs four primary components:

1. Fixed-capacity KV storage.
2. Attention-sink storage.
3. Rolling KV buffer.
4. Sequence position manager.

Unlike RadixAttention, there is no radix tree, no cross-sequence prefix matching, and no LRU eviction.

## 2.1 Cache configuration

Introduce two configuration parameters:

```
struct StreamingConfig {
    int sink_tokens = 4;
    int window_size = 1024;
};
```

Here, `window_size` is the total cache capacity, including sink tokens.

Therefore:

```
rolling_capacity = window_size - sink_tokens;
```

For example:

```
Total capacity:    1024 tokens
Attention sinks:      4 tokens
Rolling cache:     1020 tokens
```

Validate:

```
assert(sink_tokens >= 1);
assert(window_size > sink_tokens);
```

The paper uses four initial tokens as its default configuration for pretrained models.

It also demonstrates that the required number of sink tokens depends on the model and its training configuration.

## 2.2 Physical KV storage

Each cached token needs its key and value vectors across all transformer layers.

A conceptual representation:

```
struct KVSlot {
    std::vector<float> keys;
    std::vector<float> values;
};
```

A slot contains:

```
Keys:
    Layer 0: [KV_DIM]
    Layer 1: [KV_DIM]
    ...
    Layer N: [KV_DIM]

Values:
    Layer 0: [KV_DIM]
    Layer 1: [KV_DIM]
    ...
    Layer N: [KV_DIM]
```

The implementation should preferably use contiguous allocations rather than individual `std::vector` allocations for every slot.

For example:

```
std::vector<float> key_storage;
std::vector<float> value_storage;
```

An element can be addressed using:

```
offset = (
    (slot * num_layers + layer) * kv_dim
);
```

This is a simple layout for a CPU inference engine.

## 2.3 Attention-sink storage

The first `sink_tokens` positions must never be overwritten.

Use a dedicated region:

```
struct SinkCache {
    // First sink_tokens positions.
    // Never overwritten during generation.
};
```

These positions correspond to the initial tokens of the original sequence.

For a four-token configuration:

```
Sink slots:

Slot 0 -> Original token 0
Slot 1 -> Original token 1
Slot 2 -> Original token 2
Slot 3 -> Original token 3
```

Once populated, they remain unchanged until the sequence is cleared.

## 2.4 Rolling KV cache

The remaining slots hold the most recent tokens.

Suggested representation:

```
struct RollingCache {
    int capacity = 0;

    int size = 0;

    int next_write = 0;
};
```

A circular buffer is appropriate because it avoids physically shifting large KV tensors every time a token is generated.

The rolling cache should support:

```
void append_token(...);

void overwrite_oldest(...);

int physical_slot(int logical_position) const;
```

However, physical circular-buffer positions must not be confused with attention positions.

The attention implementation must see the retained tokens in chronological order.

For example:

```
Physical slots:

0  1  2  3  4  5  6  7
S  S  S  S  A  B  C  D
```

After another token arrives:

```
0  1  2  3  4  5  6  7
S  S  S  S  E  B  C  D
```

The physical location of the oldest rolling token can be reused.

A logical-to-physical mapping is therefore necessary.

For a first implementation, you can also use a simple contiguous buffer and shift rolling KV rows when full. This is easier to debug, although less efficient.

# 3. Rolling Cache Algorithm

This is the central part of StreamingLLM.

## 3.1 Initial prefill

Suppose the model receives:

```
Tokens:
[10, 20, 30, 40, 50, 60, 70, 80]
```

With:

```
Sink tokens = 4
Cache capacity = 8
```

After prefill:

```
Sink cache:
[10, 20, 30, 40]

Rolling cache:
[50, 60, 70, 80]
```

The initial four tokens become permanent attention sinks.

The remaining tokens form the rolling window.

If the input is shorter than the configured capacity, simply retain all available tokens.

If the input is longer than the capacity, retain the first four and the final four tokens.

For a large prompt:

```
Original:
[10, 20, 30, 40, ... 500, 600, 700, 800]

Retained:
[10, 20, 30, 40, 500, 600, 700, 800]
```

This is an important distinction: the initial tokens must be captured from the original sequence, not from the end of prefill.

## 3.2 Appending a token

Suppose the cache is full:

```
[10, 20, 30, 40, 50, 60, 70, 80]
```

The model generates token 90.

The oldest rolling token, 50, must be evicted.

Result:

```
[10, 20, 30, 40, 60, 70, 80, 90]
```

The operation is:

1. Preserve the sink region.
2. Identify the oldest rolling entry.
3. Reuse its physical storage.
4. Write the new token's K/V rows.
5. Update the logical-to-physical mapping.
6. Increment the absolute sequence position.

No historical tokens need to be recomputed.

## 3.3 Suggested implementation

```
class StreamingKVCache : public KVCache {
private:
    int sink_tokens_;
    int capacity_;
    int rolling_capacity_;

    int rolling_size_ = 0;
    int next_write_ = 0;

    // Physical KV storage.
    std::vector<float> keys_;
    std::vector<float> values_;

    // Absolute token positions represented by each slot.
    std::vector<int> position_mapping_;

public:
    void append_token(
        int absolute_position,
        const float* k_rows,
        const float* v_rows
    );
};
```

Conceptual append algorithm:

```
void append_token(
    int absolute_position,
    const float* k_rows,
    const float* v_rows
) {
    int slot;

    if (absolute_position < sink_tokens_) {
        slot = absolute_position;
    } else {
        slot = sink_tokens_ + next_write_;

        next_write_ =
            (next_write_ + 1) % rolling_capacity_;

        rolling_size_ = std::min(
            rolling_size_ + 1,
            rolling_capacity_
        );
    }

    write_kv(slot, k_rows, v_rows);

    position_mapping_[slot] = absolute_position;
}
```

This is conceptual pseudocode. In particular, prefill must handle the transition between initial sink tokens and the most recent prompt tokens.

The implementation must also distinguish prefill writes from decode writes so that it does not overwrite sink positions or retain an incorrect portion of the prompt.

## 3.4 Chronological iteration

The attention computation must read the retained rolling tokens in chronological order.

A circular buffer does not necessarily store them in chronological physical order.

For example:

```
Physical storage:

Sink:   [0, 1, 2, 3]
Rolling:[8, 5, 6, 7]
```

Logical attention order:

```
[0, 1, 2, 3, 5, 6, 7, 8]
```

The implementation needs a function:

```
int physical_slot_for_attention(
    int logical_position
) const;
```

This maps an attention position to the corresponding physical KV slot.

Do not pass the physical ring-buffer order directly to the attention kernel.

# 4. Position Encoding and RoPE

Critical implementation detail

The paper explicitly identifies position handling as essential to StreamingLLM.

Simply evicting old KV entries and keeping the remaining tensors unchanged is not sufficient for RoPE-based models.

## 4.1 Cache positions versus original positions

Consider the following original sequence:

```
Original positions:

[0, 1, 2, 3, 4, 5, 6, 7, 8]
```

After evicting token 4, the retained cache is:

```
Original positions:

[0, 1, 2, 3, 5, 6, 7, 8]
```

StreamingLLM assigns positions according to the current cache layout:

```
Cache positions:

[0, 1, 2, 3, 4, 5, 6, 7]
```

This means the attention computation uses contiguous cache positions instead of the original sequence positions.

The distinction is critical.

If the original positions are used, the cache has discontinuous positional information, which does not follow the method described in the paper.

Source: Section 3.2, page 5.

## 4.2 RoPE implementation

With RoPE, the paper recommends caching keys before applying the rotary transformation.

The process is:

```
Model produces raw Q, K, V
           |
           +---- Q -> Apply RoPE
           |
           +---- K -> Store unrotated K
           |
           +---- V -> Store V
```

During attention:

1. Determine the current cache positions.
2. Apply RoPE to the query using its cache position.
3. Apply RoPE to the retained keys using their current cache positions.
4. Compute attention using the transformed Q and K.
5. Use the corresponding values.

The important point is that the keys in the rolling cache may need their rotary transformation reapplied when their cache positions change.

For example:

```
Before eviction:

Original positions:
[0, 1, 2, 3, 4, 5, 6, 7]

Cache positions:
[0, 1, 2, 3, 4, 5, 6, 7]
```

After eviction:

```
Original positions:
[0, 1, 2, 3, 5, 6, 7, 8]

Cache positions:
[0, 1, 2, 3, 4, 5, 6, 7]
```

The retained keys must be transformed according to the new cache positions.

Do not store already-rotated keys and then rotate them again.

A practical implementation can either:

- Reapply RoPE to the retained keys when constructing the attention inputs.
- Maintain separate raw-key and transformed-key buffers.

The first approach is simpler but performs additional work.

The second approach can be optimized later.

## 4.3 ALiBi implementation

For ALiBi, the paper describes a more direct implementation.

Instead of applying a discontinuous positional bias based on original sequence positions, use a contiguous linear bias based on cache positions.

Conceptually:

```
Cache positions:
[0, 1, 2, 3, 4, 5, 6, 7]
```

The ALiBi attention bias should use these positions.

This avoids introducing positional gaps after eviction.

Source: Section 3.2, page 5.

## 4.4 Model compatibility

The paper demonstrates StreamingLLM with models using:

- RoPE: Llama-2, Falcon and Pythia.
- ALiBi: MPT.

The technique is designed for autoregressive models employing relative positional encoding.

Do not assume that the same position-rebasing procedure works unchanged for every positional encoding scheme.

Absolute positional embeddings, for example, require separate consideration.

# 5. Mapping StreamingLLM to the Existing KVCache Interface

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

The existing interface is sequence-oriented, which works well for StreamingLLM.

However, there is an important distinction between an absolute sequence position and a position inside the retained cache.

The implementation needs to account for this distinction.

## 5.1 `begin_sequence()`

```
SeqId begin_sequence() override;
```

Create an empty streaming cache for the new sequence.

Initialize:

```
sink_size = 0;
rolling_size = 0;
next_write = 0;
total_processed = 0;
```

No KV pages need to be allocated until tokens are stored.

## 5.2 `store()`

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

This is responsible for writing KV rows into the correct physical slot.

The `pos` argument represents the original sequence position.

The algorithm should be:

```
If pos < sink_tokens:
    Write to the corresponding permanent sink slot.

Otherwise:
    Write to the rolling cache.

    If the rolling cache is full:
        Overwrite the oldest rolling entry.

Record the original token position.
Update the logical-to-physical mapping.
```

Remember that `store()` is called separately for each transformer layer.

Consequently, a token position must be mapped to the same physical slot across every layer.

Do not independently advance the circular buffer for each layer.

The slot assignment should happen at the sequence/token level, not at the layer level.

## 5.3 `key_row()` and `value_row()`

These methods should return the KV rows for retained cache positions.

For StreamingLLM, define the `pos` argument of these read methods as the logical cache position:

```
0 ... sink_size - 1
    Attention sinks

sink_size ... stored_tokens - 1
    Rolling tokens, chronological order
```

Internally, translate this logical position into a physical circular-buffer slot.

For example:

```
const float* key_row(
    SeqId seq,
    int layer,
    int pos
) const override {
    const Sequence& sequence = sequences_.at(seq);

    int slot = sequence.physical_slot(pos);

    return key_storage_.data() + offset(slot, layer);
}
```

The value lookup follows the same procedure.

This allows the attention implementation to iterate over a contiguous cache even though the physical storage is circular.

## 5.4 `stored_tokens()`

Return the number of currently retained tokens.

```
int stored_tokens(SeqId seq) const override {
    return sink_size + rolling_size;
}
```

For example:

```
Original sequence length: 100,000
Sink tokens:              4
Rolling capacity:         1020

stored_tokens() = 1024
```

Do not return the total number of tokens ever processed.

The attention computation must operate on the retained cache, not the entire historical sequence.

Track the absolute sequence length separately.

## 5.5 `longest_cached_prefix()`

StreamingLLM does not implement cross-request prefix caching.

This method is therefore not applicable.

For compatibility, return:

```
return 0;
```

Do not confuse attention sinks with prefix caching.

Attention sinks are retained because of their role in the model's attention distribution, not because another sequence might reuse them.

## 5.6 `end_sequence()`

Release all storage associated with the sequence.

Unlike RadixAttention, there is no global cache into which completed sequences are inserted.

Once a streaming sequence finishes, its KV cache can be discarded.

## 5.7 `clear()`

Release all sequence caches and reset allocator bookkeeping.

Reset:

- Sink storage.
- Rolling storage.
- Position mappings.
- Sequence metadata.
- Memory accounting.
- Statistics.

## 5.8 Capacity and usage

```
size_t capacity_tokens() const override;
size_t used_tokens() const override;
```

For a cache with capacity 1024:

```
capacity_tokens() == 1024;
```

`used_tokens()` should count the physical KV slots currently occupied by active sequences.

For a single sequence:

```
Before filling:
used_tokens() = number of retained tokens

After filling:
used_tokens() = 1024
```

If the cache uses a preallocated buffer, distinguish allocated memory from occupied token slots. Follow the existing `KVCacheStats` accounting convention.

## 5.9 Statistics

The existing statistics structure is:

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

| Field           | Meaning                                             |
| --------------- | --------------------------------------------------- |
| `sequences`     | Currently active sequences                          |
| `tokens`        | Currently retained physical token slots             |
| `bytes`         | KV storage bytes                                    |
| `row_fetches`   | Number of key/value row accesses                    |
| `prefix_checks` | Always zero                                         |
| `prefix_hits`   | Always zero                                         |
| `evictions`     | Number of rolling-cache token positions overwritten |

An eviction here is different from RadixAttention's LRU eviction.

StreamingLLM evicts the oldest rolling token deterministically whenever the rolling window is full.

# 6. Attention Computation Integration

This is where the KV cache implementation alone is insufficient.

StreamingLLM changes which KV entries participate in attention and how positional information is assigned.

The attention implementation must be modified accordingly.

## 6.1 Attention input construction

For every decoding step, the attention operation should receive:

```
Attention sinks:
    First S tokens

Recent tokens:
    Most recent W-S tokens

Current token:
    The token currently being processed
```

The resulting attention context is:

\\[ C = S + W\_{\text{rolling}} \\]

Where:

- \\(S\\) is the number of sink tokens.
- \\(W\_{\text{rolling}}\\) is the number of retained recent tokens.

The attention complexity is proportional to the configured cache size rather than the total sequence length.

For a fixed cache size:

\\[ O(TW) \\]

where \\(T\\) is the total generated sequence length and \\(W\\) is the cache capacity.

This is the primary efficiency advantage over full-context attention.

## 6.2 Attention implementation pseudocode

```
void streaming_attention(
    Sequence& seq,
    int layer,
    const float* query,
    float* output
) {
    int n = seq.cache.stored_tokens();

    for (int i = 0; i < n; i++) {
        const float* k = seq.cache.key_row(
            layer, i
        );

        const float* v = seq.cache.value_row(
            layer, i
        );

        // Compute attention score.
        // Apply the appropriate positional transformation.
        // Accumulate weighted values.
    }
}
```

This is conceptual pseudocode.

For RoPE, the actual implementation must transform queries and keys according to the cache positions before calculating attention scores.

For ALiBi, it must use the contiguous cache-relative attention bias.

## 6.3 Prefill and decoding

There are two distinct execution stages.

Prefill

- Process the incoming prompt.
- Capture its initial sink tokens.
- Maintain the most recent tokens in the rolling cache.
- Preserve the KV rows needed for subsequent decoding.

Decoding

- Compute the next token.
- Append its KV rows.
- Overwrite the oldest rolling entry when necessary.
- Construct attention using the sinks and recent tokens.

Be particularly careful with long prompts.

The initial sink tokens must come from the beginning of the original prompt, while the rolling cache must represent its most recent tokens.

Do not simply retain the final `window_size` prompt tokens.

## 6.4 Important interface limitation

The current `KVCache` interface does not explicitly expose cache-relative position IDs or the RoPE transformation operation.

Therefore, preserving the public interface does not mean the entire StreamingLLM implementation can be confined to `StreamingKVCache`.

The attention kernel must also be updated to use the retained cache's logical positions.

If the existing attention implementation already obtains rows through `key_row()` and `value_row()` using positions from zero to `stored_tokens() - 1`, the proposed logical-to-physical mapping can preserve the interface.

If it instead assumes that cache positions are original sequence positions, that assumption must be changed internally.

# 7. Memory Management

## 7.1 Fixed memory consumption

StreamingLLM uses a fixed number of KV slots.

For one sequence:

\\[ M = W \times N_L \times 2 \times D \times B \\]

Where:

- \\(W\\): cache capacity in tokens.
- \\(N_L\\): number of transformer layers.
- \\(D\\): number of floats per K/V row.
- \\(B\\): bytes per float.

The factor of two accounts for keys and values.

For example, suppose:

```
Cache capacity:       1024
Transformer layers:     22
KV dimensions:         256
Data type:           FP32
```

Then:

\\[ M = 1024 \times 22 \times 2 \times 256 \times 4 \\]

\\[ M = 46,137,344 \text{ bytes} \\]

Approximately 44 MiB per sequence.

This memory consumption does not increase as the conversation grows.

## 7.2 No LRU implementation

Do not implement:

- Radix tree nodes.
- Reference counters.
- LRU timestamps.
- Leaf selection.
- Prefix matching.
- Shared-page ownership.

Those are RadixAttention features, not StreamingLLM features.

## 7.3 Circular-buffer optimization

A circular buffer avoids copying large KV tensors every time a new token arrives.

The implementation should maintain:

```
int next_write;
int rolling_size;
int rolling_capacity;
```

Physical slot selection:

```
int slot =
    sink_tokens +
    next_write;
```

Advance:

```
next_write =
    (next_write + 1) % rolling_capacity;
```

This makes token insertion constant-time, excluding the KV row copies themselves.

However, attention must still read entries in chronological order.

# 8. Pretraining with a Dedicated Sink Token

The paper introduces an additional technique in Section 3.3.

Instead of relying on several naturally occurring initial tokens as attention sinks, a model can be trained with a dedicated learnable sink token.

The training procedure prepends this token to every training sample.

The model then learns to direct unnecessary attention scores towards this token.

The paper compares three approaches:

1. Vanilla attention.
2. Zero Sink: a zero-valued placeholder in the attention computation.
3. Learnable Sink Token: a dedicated trainable token.

The learnable sink token achieves stable streaming performance while requiring only one sink token.

The authors report that this does not negatively affect normal language-model performance.

Source: Section 3.3, page 6, Table 3 and Section 4.2, page 7.

Implementation decision: Do not implement sink-token pretraining. Your objective is to support existing pretrained models without fine-tuning.

Use the first four tokens as attention sinks.

# 9. Implementation Architecture

Recommended organization:

```
                    StreamingKVCache
                           |
             +-------------+-------------+
             |             |             |
         Sink Cache    Rolling Cache   Sequence Manager
             |             |             |
         Permanent     Circular       Absolute position
         KV storage     KV storage     Tracking
             |             |             |
             +-------------+-------------+
                           |
                     KV Allocator
                           |
                     Attention Kernel
                           |
                   RoPE / ALiBi handling
```

Suggested class structure:

```
class StreamingKVCache : public KVCache {
public:
    StreamingKVCache(
        int num_layers,
        int kv_dim,
        int capacity_tokens,
        int sink_tokens = 4
    );

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
    struct Sequence {
        int total_processed = 0;

        int sink_size = 0;
        int rolling_size = 0;

        int next_write = 0;

        std::vector<int> token_ids;

        std::vector<int> position_mapping;
    };

    int num_layers_;
    int kv_dim_;

    int capacity_;
    int sink_tokens_;
    int rolling_capacity_;

    std::vector<float> keys_;
    std::vector<float> values_;

    std::unordered_map<SeqId, Sequence> sequences_;

    SeqId next_seq_id_ = 0;

    size_t row_fetches_ = 0;
    size_t evictions_ = 0;

    int physical_slot(
        const Sequence& seq,
        int logical_position
    ) const;
};
```

This is a starting architecture, not a complete implementation.

The storage layout can be adjusted to match the existing baseline allocator.

# 10. Implementation Order

Implement the following milestones in order.

## Phase 1: Fixed-capacity KV cache

- Implement configurable cache capacity.
- Implement sink-token storage.
- Implement rolling-token storage.
- Implement per-sequence metadata.
- Implement contiguous KV allocations.
- Implement basic KV row lookup.

At this stage, do not implement RoPE changes.

First verify that the correct KV entries are retained.

## Phase 2: Rolling cache

- Implement initial prompt processing.
- Capture the first four tokens.
- Retain the most recent rolling tokens.
- Implement circular-buffer insertion.
- Implement overwrite behavior.
- Implement logical-to-physical mapping.
- Implement chronological KV iteration.
- Track absolute sequence positions.

## Phase 3: Attention integration

- Modify attention to read sink and rolling KV entries.
- Ensure attention excludes evicted tokens.
- Implement cache-relative positions.
- Integrate RoPE handling.
- Ensure keys are stored before rotary transformation.
- Apply rotary transformations using cache positions.
- Implement contiguous ALiBi bias if supporting ALiBi models.

This phase is essential. A rolling KV cache without the corresponding attention-position changes is not a complete StreamingLLM implementation.

## Phase 4: Interface integration

- Implement `begin_sequence()`.
- Implement `store()`.
- Implement `key_row()`.
- Implement `value_row()`.
- Implement `stored_tokens()`.
- Implement `end_sequence()`.
- Implement `clear()`.
- Implement memory accounting.
- Implement statistics.

## Phase 5: Testing and benchmarking

- Test sink-token preservation.
- Test rolling-cache eviction.
- Test circular-buffer wrapping.
- Test logical-to-physical mapping.
- Test RoPE position rebasing.
- Test long sequences.
- Compare perplexity against ordinary window attention.
- Compare performance against sliding-window recomputation.
- Measure memory consumption.

# 11. Acceptance Criteria

The implementation must satisfy the following tests.

| Test                       | Expected result                          |
| -------------------------- | ---------------------------------------- |
| Empty sequence             | Zero occupied slots                      |
| Initial prefill            | Correct sink and rolling entries         |
| Short prompt               | All prompt tokens retained               |
| Long prompt                | Initial sinks and recent tokens retained |
| Append before capacity     | Rolling cache grows                      |
| Append after capacity      | Oldest rolling token overwritten         |
| Sink preservation          | Initial tokens never overwritten         |
| Circular-buffer wraparound | Correct chronological ordering           |
| KV row lookup              | Correct physical K/V values              |
| Multiple layers            | Consistent slot mapping across layers    |
| Long generation            | Memory usage remains bounded             |
| RoPE reindexing            | Keys use cache-relative positions        |
| ALiBi                      | Bias uses contiguous cache positions     |
| End sequence               | All sequence storage released            |
| Clear                      | All storage and metadata reset           |

## Particularly important tests

### Test 1: Sink preservation

Configuration:

```
Capacity: 8
Sink tokens: 4
```

Generate:

```
[10, 20, 30, 40, 50, 60, 70, 80]
```

Expected cache:

```
[10, 20, 30, 40, 50, 60, 70, 80]
```

Generate another token:

```
90
```

Expected:

```
[10, 20, 30, 40, 60, 70, 80, 90]
```

Verify that the KV data for tokens 10, 20, 30 and 40 is unchanged.

### Test 2: Multiple rolling evictions

Generate:

```
[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
```

Expected:

```
[10, 20, 30, 40, 70, 80, 90, 100]
```

The sink region remains unchanged, and the rolling region contains the latest four tokens.

### Test 3: Circular-buffer wraparound

Use a rolling capacity of four.

Generate:

```
[50, 60, 70, 80]
```

Then:

```
90
100
110
120
```

Expected rolling cache:

```
[90, 100, 110, 120]
```

Verify that the physical circular-buffer layout is not mistaken for chronological attention order.

### Test 4: RoPE position rebasing

Original positions:

```
[0, 1, 2, 3, 5, 6, 7, 8]
```

Expected cache positions:

```
[0, 1, 2, 3, 4, 5, 6, 7]
```

Verify that the attention kernel uses the cache-relative positions when applying RoPE.

### Test 5: Bounded memory

Generate increasingly long sequences:

```
1,000 tokens
10,000 tokens
100,000 tokens
1,000,000 tokens
```

Expected:

```
KV cache capacity remains unchanged.
KV memory usage remains bounded.
```

The cache should never allocate additional KV storage simply because the absolute sequence position increases.

### Test 6: Numerical correctness

Compare the model's output against a sliding-window recomputation baseline.

The objective is to verify that StreamingLLM preserves the model's streaming behavior while avoiding repeated context recomputation.

Use perplexity rather than relying exclusively on exact token-by-token generation matches.

# 12. Performance Expectations

The paper evaluates StreamingLLM against sliding-window recomputation.

Its key findings:

- StreamingLLM maintains stable perplexity over extremely long sequences.
- The method supports sequences exceeding four million tokens in the reported experiments.
- Decoding latency grows linearly with cache size for the tested configuration.
- Sliding-window recomputation has substantially higher decoding latency.
- The reported maximum speedup is 22.2× per token.

Source: Sections 4.1 and 4.5, pages 6 and 9; Figures 5 and 10.

These are results reported by the paper, not expected performance guarantees for your CPU implementation.

# 13. StreamingLLM vs. RadixAttention

Since these are now two cache implementations in your inference engine, their differences are important.

| Property               | RadixAttention                           | StreamingLLM                                |
| ---------------------- | ---------------------------------------- | ------------------------------------------- |
| Primary purpose        | Cross-request KV reuse                   | Long-running sequence inference             |
| Main data structure    | Radix tree                               | Fixed-size rolling buffer                   |
| Cache growth           | Limited by available memory and eviction | Fixed capacity                              |
| Initial tokens         | No special treatment                     | Permanently retained                        |
| Recent tokens          | Stored as part of cached prefixes        | Continuously retained                       |
| Prefix matching        | Yes                                      | No                                          |
| Cross-sequence sharing | Yes                                      | No                                          |
| Eviction               | LRU leaf eviction                        | Oldest rolling token                        |
| Position handling      | Model's normal position handling         | Cache-relative positions                    |
| RoPE modifications     | Not inherently required                  | Required by the described method            |
| Primary optimization   | Avoid recomputing shared prefixes        | Avoid unbounded KV growth and recomputation |

The two techniques are complementary.

A future implementation could combine them:

```
                    Inference Engine
                           |
                         KVCache
                           |
                  StreamingRadixKVCache
                           |
                  +--------+--------+
                  |                 |
              Radix Tree       Rolling Cache
                  |                 |
             Prefix Reuse      Attention Sinks
                  |                 |
                  +--------+--------+
                           |
                       KV Storage
                           |
                      Attention
```

However, this combination introduces additional complexity around shared-page ownership, eviction, and sequence position management.

Do not implement it yet.

# 14. Final Implementation Recommendation

For your current C++ inference engine, implement StreamingLLM in this order:

1. Fixed-capacity KV storage.
2. Permanent attention-sink region.
3. Rolling circular buffer.
4. Logical-to-physical position mapping.
5. Attention integration.
6. RoPE position rebasing.
7. Benchmarking.

The most important correctness requirement is:

The attention computation must always see the first four tokens of the original sequence together with the most recent tokens, using contiguous cache-relative positions.

There are two separate things to verify:

- The cache retains the correct physical KV rows.
- The attention kernel interprets those rows using the correct positional information.

Getting only the first part right will produce a rolling KV cache, but not a correct implementation of StreamingLLM.

Reference: Xiao et al., Efficient Streaming Language Models with Attention Sinks, arXiv:2309.17453v1 (2023). Particularly Figure 4 and Sections 3.1–3.3.

bounded-growth.pdf

One architectural consideration: Keep `BaselineKVCache`, `RadixKVCache`, and `StreamingKVCache` as separate implementations behind the same interface. This will let you benchmark the three approaches independently before considering a combined implementation.