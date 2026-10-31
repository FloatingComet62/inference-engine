#!/usr/bin/env python3
"""Benchmark the TinyLlama C++ engine against Ollama's tinyllama:1.1b.

Fairness rules applied to both engines:
  * identical prompts and the same Zephyr chat template (Ollama gets raw=true)
  * greedy decoding (temperature 0), same max new tokens, same thread count
  * a warm-up run first (loads weights into RAM), then RUNS measured runs
  * each run uses a different prompt, so Ollama's prompt cache cannot skip prefill
Metrics: decode/prefill tok/s, time to first token, peak RAM, model size,
CPU utilisation, energy (Intel/AMD RAPL, needs read access: make rapl-perms),
and quality (perplexity + top-1 agreement vs the fp32 reference, --quality).
Medians are reported. Every run is saved to bench/results/ and appended to
bench/results/history.csv so you can track optimisation progress over time.
"""
import argparse
import time
import csv
import datetime
import json
import os
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM = "You are a helpful AI assistant. Follow the requested structure carefully and provide complete, detailed answers."
# Each run gets a unique tag at the very start of the system prompt, so Ollama's
# server-side prompt cache cannot reuse KV from earlier runs or earlier sessions
# (without it, repeat runs reported 18,000-27,000 tok/s "prefill" in ~17 ms).
RUN_ID = os.urandom(4).hex()
_run_counter = 0


def tagged_system():
    global _run_counter
    _run_counter += 1
    return f"[session {RUN_ID}-{_run_counter}] {SYSTEM}"
PROMPTS = [
    """You are the principal systems architect responsible for designing a production-grade CPU-only inference engine for decoder-only transformer models. You are preparing an exhaustive technical architecture specification that will be used by multiple engineering teams to implement the system independently.

    The target is a locally hosted model with approximately 1.1 billion parameters. The engine must support model loading, tokenization, tensor execution, transformer layers, attention, KV caching, sampling, multiple independent sequences, streaming generation, cancellation, and deterministic execution.

    Write an extremely detailed architecture document containing the following sections.

    SECTION 1: SYSTEM ARCHITECTURE
    Define every major subsystem and its responsibilities:
    - Model loader and model metadata parser.
    - Tokenizer and vocabulary manager.
    - Tensor representation and tensor memory management.
    - CPU operator library.
    - Linear algebra implementation.
    - Normalization operators.
    - Rotary positional embedding implementation.
    - Attention implementation.
    - Transformer block.
    - Feed-forward network.
    - Logits processor.
    - Sampling implementation.
    - Sequence manager.
    - KV-cache manager.
    - Scheduler.
    - Generation controller.
    - Streaming output interface.
    - Memory allocator.
    - Thread pool.

    For each component, define its public interface, private state, ownership responsibilities, error conditions, dependencies, and invariants. Explain which components should be independent libraries and which should remain internal implementation details.

    Describe at least three alternative architectural decompositions. Compare a monolithic inference engine, a layered library architecture, and a graph-oriented execution architecture. Explain the tradeoffs in testing, performance, maintainability, debugging, and future hardware backends.

    SECTION 2: MODEL LOADING
    Specify a complete model-loading pipeline. Discuss model configuration parsing, vocabulary loading, tensor metadata, tensor names, tensor dimensions, quantization metadata, tensor offsets, alignment, file mapping, lazy loading, eager loading, and validation.

    Explain how the engine should detect:
    - Missing embeddings.
    - Missing normalization weights.
    - Missing attention projections.
    - Incompatible query and key dimensions.
    - Invalid grouped-query attention configuration.
    - Incorrect feed-forward dimensions.
    - Unsupported activation functions.
    - Incorrect tensor ordering.
    - Duplicate tensor names.
    - Overlapping tensor offsets.
    - Truncated tensor data.
    - Invalid quantization metadata.
    - Integer overflow in tensor-size calculations.
    - Incorrect vocabulary size.
    - Incompatible tokenizer and model configuration.

    Define the validation order and explain why some errors must be detected before allocating large amounts of memory.

    SECTION 3: TENSOR ENGINE
    Design a tensor abstraction for C++ that supports multidimensional shapes, strides, offsets, data types, alignment, ownership, non-owning views, slicing, transposition, reshaping, and contiguous storage.

    Define the behavior of tensor copying, moving, aliasing, and destruction. Discuss whether tensors should use shared ownership, unique ownership, or explicit borrowed views.

    Explain how tensor shape arithmetic must be implemented safely. Include examples of multiplication overflow, offset overflow, invalid strides, non-contiguous views, and incorrect assumptions about contiguous memory.

    Compare row-major and column-major storage. Explain the impact of each on matrix multiplication, attention, and normalization.

    Define a testing strategy for every tensor operation, including randomized dimensions, unusual strides, empty tensors, singleton dimensions, and boundary dimensions.

    SECTION 4: TRANSFORMER EXECUTION
    Walk through the complete execution of one transformer layer for a single input token.

    Explain input normalization, query projection, key projection, value projection, head splitting, grouped-query head mapping, rotary embeddings, cache append, attention score computation, causal masking, stable softmax, value aggregation, output projection, residual addition, feed-forward normalization, gate projection, up projection, activation, down projection, and final residual addition.

    Give exact tensor dimensions for a hypothetical model with:
    - Hidden dimension: 2048.
    - Query heads: 16.
    - KV heads: 4.
    - Head dimension: 128.
    - Feed-forward dimension: 5632.
    - Transformer layers: 22.

    Explain how these dimensions determine the sizes of intermediate tensors.

    Describe the execution differences between prompt prefill and autoregressive decoding. Explain why prefill is generally more parallelizable and why decoding frequently becomes memory-bandwidth-bound.

    SECTION 5: KV CACHE
    Develop an exhaustive KV-cache design.

    Compare:
    - Per-sequence contiguous allocation.
    - Global contiguous allocation.
    - Fixed-size block allocation.
    - PagedAttention-style allocation.
    - Prefix-shared immutable blocks.
    - Copy-on-write prefix sharing.

    For each strategy, define its data structures, allocation algorithm, append operation, lookup operation, growth strategy, deletion operation, and memory reclamation strategy.

    Define the exact memory calculation for a model with L layers, Hkv KV heads, D head dimensions, T cached tokens, and B bytes per element.

    Calculate the expected memory for several combinations of token counts, layers, head counts, and data types.

    Explain fragmentation, allocation overhead, cache-line utilization, alignment, and block-table overhead.

    Discuss reference counting, block ownership, partially filled blocks, concurrent append operations, sequence cancellation, and stale block references.

    Explain how the implementation should behave when memory capacity is exhausted. Compare rejecting requests, evicting inactive sequences, evicting least-recently-used prefixes, and reducing context length.

    SECTION 6: CPU PERFORMANCE
    Analyze the effects of:
    - Memory bandwidth.
    - L1, L2, and L3 cache capacity.
    - SIMD instruction sets.
    - AVX2.
    - AVX-512 where available.
    - Cache-line alignment.
    - False sharing.
    - NUMA placement.
    - Thread scheduling.
    - Thread oversubscription.
    - Matrix multiplication blocking.
    - Operator fusion.
    - Memory allocation frequency.
    - Page faults.
    - Huge pages.
    - Quantized weights.

    Explain why a faster matrix multiplication kernel does not necessarily improve end-to-end token generation speed.

    SECTION 7: TESTING
    Specify unit tests, integration tests, numerical tests, concurrency tests, malformed-model tests, long-context tests, memory exhaustion tests, and cancellation tests.

    Define numerical tolerances for comparisons against a trusted implementation. Explain how floating-point accumulation order affects reproducibility.

    SECTION 8: IMPLEMENTATION ROADMAP
    Divide implementation into independently testable milestones. Define dependencies, deliverables, failure conditions, performance targets, and acceptance criteria for every milestone.

    The final document must be comprehensive, technically precise, and internally consistent. Do not summarize sections. Do not omit implementation details merely because they are complicated. When a design choice has alternatives, analyze each alternative before making a recommendation.""",

    """You are the senior reliability engineer investigating a severe production incident in a CPU inference server hosting a decoder-only transformer model.

    The server initially performs well. After several hours of sustained usage, memory consumption rises continuously, time to first token increases, decode throughput falls, and long-running conversations become dramatically slower.

    The service supports multiple simultaneous users, streaming responses, cancellation, conversation history, prefix caching, and dynamic request scheduling.

    Produce an exhaustive incident investigation report.

    SECTION 1: INCIDENT TIMELINE
    Construct a hypothetical but technically realistic timeline covering:
    - Initial deployment.
    - Normal operation.
    - First signs of memory growth.
    - Increased request latency.
    - First failed allocations.
    - Increased cancellation frequency.
    - Service degradation.
    - Detection.
    - Mitigation.
    - Recovery.

    Explain what telemetry should have identified the issue earlier.

    SECTION 2: ROOT-CAUSE INVESTIGATION
    Investigate each of these possibilities independently:
    1. KV-cache memory leaks.
    2. Sequence deletion not releasing blocks.
    3. Cancellation leaving partially allocated cache state.
    4. Incorrect reference counting.
    5. Shared prefix blocks being freed prematurely.
    6. Prefix blocks never being reclaimed.
    7. Cache fragmentation.
    8. Repeated prompt prefill.
    9. Unbounded conversation histories.
    10. Context-window overflow.
    11. Incorrect token accounting.
    12. Scheduler starvation.
    13. Lock contention.
    14. Thread oversubscription.
    15. CPU memory-bandwidth saturation.
    16. Allocator fragmentation.
    17. Memory mapping and page faults.
    18. Accidental copying of entire KV caches.
    19. Stale references after vector reallocation.
    20. Concurrent modification of block tables.

    For every hypothesis, provide:
    - The underlying mechanism.
    - A realistic example.
    - Expected symptoms.
    - Metrics to inspect.
    - A confirming experiment.
    - A rejecting experiment.
    - The corrective implementation.
    - Regression tests.

    SECTION 3: OBSERVABILITY
    Design a complete metrics system.

    Define metrics for:
    - Requests received.
    - Requests completed.
    - Requests cancelled.
    - Active sequences.
    - Queued sequences.
    - Prompt token count.
    - Generated token count.
    - Cached token count.
    - Prefix cache hits.
    - Prefix cache misses.
    - KV blocks allocated.
    - KV blocks occupied.
    - KV blocks free.
    - KV blocks shared.
    - KV blocks evicted.
    - Allocation failures.
    - Cache growth events.
    - Cache copy operations.
    - Cache reclamation latency.
    - Prefill duration.
    - Decode duration.
    - Time to first token.
    - Inter-token latency.
    - Tokens per second.
    - Queue waiting time.
    - CPU utilization.
    - Resident memory.
    - Virtual memory.
    - Page faults.
    - Context switches.
    - Thread count.

    For every metric, define its unit, aggregation strategy, cardinality constraints, and interpretation.

    Explain which measurements must be collected per request, per sequence, per layer, and globally.

    SECTION 4: LOAD TESTING
    Design a detailed load-test matrix.

    Test prompt lengths of:
    128, 512, 1024, 2048, 4096, 8192, 16384, and 32768 tokens.

    Test generation lengths of:
    16, 64, 128, 256, 512, 1024, and 2048 tokens.

    Test concurrency levels of:
    1, 2, 4, 8, 16, 32, and 64 sequences.

    Vary request arrival rates, cancellation frequencies, repeated-prefix ratios, and conversation reuse.

    Explain how to isolate prefill cost from decode cost.

    Explain how to identify memory-bandwidth saturation independently from compute saturation.

    SECTION 5: REMEDIATION
    Propose fixes in priority order.

    Include immediate mitigation, short-term engineering fixes, architectural improvements, and long-term performance work.

    Define a staged rollout, canary deployment, rollback conditions, and post-deployment verification.

    Define numerical success criteria for memory growth, allocation failures, p95 latency, p99 latency, throughput, and cache reclamation.

    Do not assume that adding threads improves performance. Explain how the server should determine its optimal concurrency and worker-thread configuration.

    Make the report suitable for a production reliability review. Every recommendation must have an observable validation method.""",

    """Prepare an exhaustive implementation comparison of three transformer attention-state architectures.

    The architectures are:
    A. Conventional per-sequence contiguous KV cache.
    B. Fixed-size paged/block-based KV cache.
    C. Prefix-sharing KV cache with immutable shared blocks and copy-on-write continuation.

    Do not provide a superficial comparison. Develop the actual data models, algorithms, memory layouts, ownership rules, and complexity implications.

    SECTION 1: FOUNDATIONAL DATA MODEL
    Define:
    - Logical sequence identity.
    - Logical token position.
    - Physical cache location.
    - Transformer layer identity.
    - KV head identity.
    - Head dimension.
    - Block identity.
    - Block ownership.
    - Shared block references.
    - Valid-token count.
    - Block capacity.
    - Sequence-to-block mapping.

    Explain how a logical sequence position maps to a physical memory address.

    SECTION 2: CONTIGUOUS CACHE
    Design a contiguous cache.

    Explain allocation, capacity growth, append, lookup, attention reads, sequence deletion, and memory reclamation.

    Compare allocating the maximum context length immediately against geometric growth.

    Discuss reallocations, pointer invalidation, copying costs, capacity slack, fragmentation, and failure handling.

    Show how the design behaves when multiple sequences have radically different lengths.

    SECTION 3: BLOCK-BASED CACHE
    Design a paged KV cache using fixed-size token blocks.

    Consider block sizes of 8, 16, 32, 64, and 128 tokens.

    Explain the tradeoffs between internal fragmentation, block-table size, allocation frequency, and attention lookup overhead.

    Define a free-list allocator and explain how blocks are returned safely.

    Describe how to implement an append operation when:
    - The current block has free space.
    - The current block is full.
    - No blocks are available.
    - Allocation partially succeeds.
    - The sequence is concurrently cancelled.

    Explain how attention traverses blocks without accidentally interpreting unused block slots as valid tokens.

    SECTION 4: PREFIX SHARING
    Design prefix caching for requests with common initial prompts.

    Consider these sequences:

    Sequence A:
    "You are a systems engineer. Design a CPU inference engine. Explain memory allocation and KV-cache behavior."

    Sequence B:
    "You are a systems engineer. Design a CPU inference engine. Explain memory allocation and KV-cache behavior. Now focus on concurrency."

    Sequence C:
    "You are a systems engineer. Design a CPU inference engine. Explain memory allocation and KV-cache behavior. Now focus on memory fragmentation."

    Explain which tokens can share cache state and which cannot.

    Define prefix identification using token hashes, model identity, tokenizer identity, positional settings, and inference configuration.

    Explain why hashing raw text is insufficient.

    SECTION 5: OWNERSHIP AND CORRECTNESS
    Develop detailed ownership rules.

    Explain:
    - Atomic reference counts.
    - Block pinning.
    - Copy-on-write.
    - Partial final blocks.
    - Shared immutable blocks.
    - Sequence-specific mutable blocks.
    - Concurrent append.
    - Cancellation.
    - Block reclamation.
    - Cache invalidation.
    - Hash collisions.
    - Model changes.
    - Tokenizer changes.
    - RoPE configuration changes.

    Provide invariants for every state transition.

    SECTION 6: MEMORY ANALYSIS
    Derive the KV memory equation.

    Analyze the effects of:
    - Layer count.
    - KV-head count.
    - Head dimension.
    - Data type.
    - Context length.
    - Number of active sequences.
    - Block size.
    - Prefix reuse.

    Explain why GQA changes the cache size and why query-head count is not the correct quantity for calculating KV storage.

    Calculate illustrative memory usage for several hypothetical model configurations.

    SECTION 7: PERFORMANCE ANALYSIS
    Compare all three architectures in terms of:
    - Allocation latency.
    - Append latency.
    - Attention read overhead.
    - Memory locality.
    - Fragmentation.
    - Cache growth.
    - Deletion.
    - Prefix reuse.
    - Lock contention.
    - Atomic operations.
    - Allocator complexity.

    SECTION 8: FINAL RECOMMENDATIONS
    Give independent recommendations for:
    - A small single-process CPU engine.
    - A high-concurrency inference server.
    - An embedded inference runtime.
    - A system with highly repetitive prompts.

    Specify benchmark experiments that could prove each recommendation wrong.""",

    """Create a rigorous and extremely detailed benchmark specification for a C++ transformer inference engine running on a CPU-only laptop.

    The benchmark must determine exactly how model size, prompt length, generation length, KV-cache allocation policy, prefix reuse, memory bandwidth, and concurrency affect performance.

    The objective is to compare multiple implementations fairly and identify the exact point at which a particular cache strategy becomes advantageous.

    SECTION 1: WORKLOAD DEFINITIONS

    Define the following workload classes:
    - Very short prompt and very short generation.
    - Very short prompt and extremely long generation.
    - Extremely long prompt and very short generation.
    - Extremely long prompt and extremely long generation.
    - Repeated prompts with identical prefixes.
    - Prompts with partially overlapping prefixes.
    - Completely unrelated prompts.
    - Many independent short sequences.
    - A small number of extremely long sequences.
    - Sequences with frequent cancellation.
    - Sequences that grow gradually over several thousand tokens.
    - Sequences that approach the model context limit.

    Explain what each workload isolates.

    SECTION 2: TOKEN LENGTH MATRIX

    Test prompt lengths:
    32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, and 32768 tokens.

    Test generation lengths:
    16, 32, 64, 128, 256, 512, 1024, 2048, and 4096 tokens.

    Test concurrent sequence counts:
    1, 2, 4, 8, 16, 24, 32, and 64.

    Explain which combinations are particularly important for identifying KV-cache scaling problems.

    SECTION 3: METRICS

    Define:
    - Prompt processing time.
    - Time to first token.
    - Decode duration.
    - Inter-token latency.
    - Prefill tokens per second.
    - Decode tokens per second.
    - Total request latency.
    - CPU time.
    - Wall-clock time.
    - Peak RSS.
    - KV allocated bytes.
    - KV occupied bytes.
    - KV wasted bytes.
    - Block allocation count.
    - Block release count.
    - Allocation failures.
    - Cache hit rate.
    - Cache growth events.
    - Page faults.
    - CPU utilization.
    - Memory bandwidth.
    - Energy consumption where available.

    Define exact formulas, units, and aggregation methods.

    SECTION 4: EXPERIMENTAL CONTROLS

    Explain how to control:
    - CPU frequency.
    - Turbo boost.
    - Thermal throttling.
    - Thread count.
    - CPU affinity.
    - Background activity.
    - NUMA placement.
    - Model loading.
    - Page-cache state.
    - Memory allocator.
    - Compiler optimization flags.
    - SIMD availability.
    - Quantization.
    - Output token count.
    - Sampling configuration.
    - Random seeds.

    Explain why both implementations must receive identical token IDs rather than merely identical text.

    SECTION 5: KV-CACHE EXPERIMENTS

    Design experiments that isolate:
    - Initial allocation.
    - Incremental growth.
    - Reallocation.
    - Block allocation.
    - Block release.
    - Sequence deletion.
    - Cache reuse.
    - Prefix sharing.
    - Copy-on-write.
    - Fragmentation.
    - Long-running cache occupancy.

    Include a test that creates hundreds of sequences with different lengths, cancels a random subset, continues generating on the remaining sequences, and repeats this cycle many times.

    Explain how to detect memory leaks and fragmentation from these experiments.

    SECTION 6: STATISTICS

    Define warm-up runs, measurement repetitions, median, mean, p50, p90, p95, p99, standard deviation, confidence intervals, and outlier investigation.

    Explain how many repetitions are appropriate for different benchmark durations.

    Explain why a single tokens-per-second result is insufficient.

    SECTION 7: RESULTS

    Provide a sample results table with units and an interpretation guide.

    Define a reproducibility checklist.

    Finish with a prioritized experiment sequence that starts with isolated cache microbenchmarks and progresses toward realistic concurrent workloads.

    Make this a complete experimental protocol, not a general discussion.""",

    """Act as the principal C++ engineer reviewing a submitted transformer inference implementation.

    The implementation works for short prompts and single sequences, but occasionally produces incorrect output after several thousand tokens. It also experiences intermittent crashes when multiple sequences are active.

    Prepare an exhaustive code-review and debugging guide.

    SECTION 1: MEMORY SAFETY
    Review:
    - Tensor ownership.
    - Buffer lifetime.
    - Move construction.
    - Move assignment.
    - Copy semantics.
    - Non-owning views.
    - Aliasing.
    - Alignment.
    - Bounds checking.
    - Integer overflow.
    - Pointer arithmetic.
    - Vector reallocation.
    - Exception safety.
    - Destruction order.

    Give concrete failure scenarios and corrected implementations.

    SECTION 2: KV-CACHE INVARIANTS
    Define all invariants required before and after KV append.

    Include:
    - Sequence identity.
    - Layer identity.
    - Token position.
    - Block identity.
    - Physical offset.
    - Block capacity.
    - Valid-token count.
    - Allocation state.
    - Ownership state.
    - Reference count.
    - Sequence length.

    Explain how every invariant should be validated.

    SECTION 3: ATTENTION CORRECTNESS
    Review:
    - Query dimensions.
    - Key dimensions.
    - Value dimensions.
    - Query-head mapping.
    - KV-head mapping.
    - Rotary embeddings.
    - Position indices.
    - Causal masking.
    - Softmax stability.
    - Scaling factors.
    - Accumulation precision.
    - Current-token handling.
    - Historical-token handling.

    Include examples of errors that produce plausible output for short prompts but fail at longer contexts.

    SECTION 4: BLOCK BOUNDARIES
    Examine block sizes of 8, 16, 32, and 64 tokens.

    For each block size, investigate sequence lengths immediately before, at, and after a block boundary.

    Explain what should happen at lengths:
    7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, and 129.

    Include partially filled final blocks, full blocks, allocation of the next block, and sequence deletion.

    SECTION 5: CONCURRENCY
    Investigate:
    - Concurrent appends.
    - Concurrent reads.
    - Concurrent deletion.
    - Reference-count races.
    - Block reuse races.
    - Sequence-ID reuse.
    - Lock ordering.
    - Deadlocks.
    - False sharing.
    - Data races.
    - Cancellation during allocation.

    Define a safe locking strategy and discuss when atomics are appropriate.

    SECTION 6: DEBUGGING
    Provide a progression:
    1. Empty prompt.
    2. One input token.
    3. Two input tokens.
    4. One generation step.
    5. Short prompt.
    6. Short generation.
    7. One block boundary.
    8. Multiple block boundaries.
    9. Long prompt.
    10. Long generation.
    11. Multiple sequences.
    12. Cancellation.
    13. Prefix sharing.
    14. Memory exhaustion.

    Explain the use of AddressSanitizer, UndefinedBehaviorSanitizer, ThreadSanitizer, Valgrind, GDB watchpoints, allocation tracing, and reference implementations.

    SECTION 7: TESTING
    Define unit tests, integration tests, property-based tests, randomized tests, differential tests, and stress tests.

    Provide pseudocode for a randomized test that creates sequences, appends tokens, deletes random sequences, and continuously verifies cache invariants.

    SECTION 8: MERGE CRITERIA
    Define mandatory acceptance criteria before merging the implementation.

    The document must be specific enough that a reviewer can use it as a practical checklist during an actual code review.""",

    """You are designing an extremely long-context stress test for a CPU-only transformer inference engine.

    The purpose is to discover the exact failure point of its KV-cache implementation. The test must systematically push memory usage, context length, allocation frequency, and sequence concurrency toward their limits.

    Produce a complete stress-testing specification.

    SECTION 1: OBJECTIVES
    Define tests for:
    - Maximum context length.
    - Maximum active sequence count.
    - Maximum total cached tokens.
    - Maximum KV allocation.
    - Maximum block-table size.
    - Maximum sequence count with fragmented allocations.
    - Repeated growth.
    - Repeated deletion.
    - Cancellation during allocation.
    - Allocation failure.
    - Cache reuse.
    - Prefix sharing.
    - Memory reclamation.

    SECTION 2: EXTREME CONTEXT TESTS
    Define prompt lengths of:
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, and 262144 tokens.

    Explain how to adapt these tests to models with smaller context limits without confusing model-level context rejection with cache implementation failure.

    Design tests where the sequence starts small and grows continuously to the maximum supported length.

    Explain how to measure memory after every 256 generated tokens and identify unexpected allocation jumps.

    SECTION 3: CONCURRENCY TESTS
    Test 1, 2, 4, 8, 16, 32, 64, 128, and 256 active sequences where memory permits.

    Consider mixed sequence lengths, synchronized growth, staggered growth, random cancellation, and random deletion.

    Define tests that deliberately create fragmentation by alternating short-lived and long-lived sequences.

    SECTION 4: MEMORY EXHAUSTION
    Define behavior when the cache reaches:
    50%, 70%, 85%, 95%, and 100% of its configured capacity.

    Explain what must happen when a new block cannot be allocated.

    The implementation must not:
    - Corrupt existing sequences.
    - Return uninitialized cache data.
    - Lose ownership information.
    - Leave partially updated block tables.
    - Leak blocks.
    - Deadlock.
    - Silently truncate sequence history.

    Define the transactional behavior expected from a failed append.

    SECTION 5: LONG-RUNNING TEST
    Design a test lasting several hours.

    Repeatedly:
    - Create a set of sequences.
    - Populate each with a long prompt.
    - Generate different numbers of tokens.
    - Cancel a random subset.
    - Complete the remaining sequences.
    - Delete finished sequences.
    - Reuse freed blocks.
    - Repeat.

    Explain how to identify a leak when RSS does not immediately decrease because the system allocator retains freed memory.

    SECTION 6: VALIDATION
    Define consistency checks for every layer, every sequence, every block, and every valid token.

    Include checksums or deterministic test patterns to identify data corruption.

    SECTION 7: FAILURE REPORTING
    Define the logs and diagnostic information captured when:
    - Allocation fails.
    - A sequence is cancelled.
    - A block is released.
    - A block is reused.
    - An invariant fails.
    - A sequence reaches maximum context.

    SECTION 8: ACCEPTANCE CRITERIA
    Define numerical pass/fail criteria for correctness, memory growth, reclamation, allocation failures, latency, and concurrency.

    Make this suitable for validating a KV cache implementation before it is used by an inference server.""",

    """You are a researcher investigating the relationship between prompt length, KV-cache size, and autoregressive decoding performance.

    Prepare an exhaustive experimental report explaining how inference cost evolves as context length increases.

    SECTION 1: MATHEMATICAL MODEL
    Derive the KV-cache memory formula:

    KV_bytes = 2 * layers * KV_heads * head_dimension * cached_tokens * bytes_per_element

    Explain every term and derive the formula from the actual key and value tensor dimensions.

    Explain the differences between MHA, GQA, and MQA.

    Calculate cache memory for several hypothetical configurations, including:
    - 12 layers, 4 KV heads, dimension 64.
    - 22 layers, 4 KV heads, dimension 128.
    - 32 layers, 8 KV heads, dimension 128.
    - 40 layers, 8 KV heads, dimension 128.

    Use FP32, FP16, BF16, and 8-bit cache storage where supported.

    SECTION 2: SCALING
    Analyze context lengths of:
    128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, and 131072 tokens.

    Explain linear KV memory growth.

    Distinguish memory allocated from memory actually occupied.

    Explain capacity slack, block fragmentation, and metadata overhead.

    SECTION 3: ATTENTION COMPUTATION
    Explain why the attention operation for each generated token must consider historical keys and values.

    Analyze how the amount of attention work grows as context increases.

    Compare full attention against sliding-window attention.

    Explain why KV caching avoids recomputing previous projections but does not eliminate historical attention reads.

    SECTION 4: CPU MEMORY BANDWIDTH
    Estimate the amount of KV data read for a decoding step.

    Explain how batch size and active sequence count affect memory traffic.

    Analyze the relationship between cache storage format and memory bandwidth.

    Discuss whether quantizing the KV cache helps on a laptop CPU with limited memory bandwidth.

    SECTION 5: EXPERIMENTS
    Design experiments that measure decode latency at each context length.

    Hold the number of generated tokens constant while varying initial context length.

    Then hold initial context length constant while varying generated token count.

    Explain why these experiments isolate different scaling effects.

    SECTION 6: RESULTS INTERPRETATION
    Provide hypothetical results and explain how to interpret:
    - Increasing memory usage.
    - Increasing inter-token latency.
    - Falling tokens per second.
    - Stable prefill speed but slower decoding.
    - Sudden latency jumps at block boundaries.
    - Decreasing throughput at higher concurrency.

    SECTION 7: CONCLUSIONS
    Provide implementation recommendations for contiguous, paged, and prefix-sharing caches.

    Identify the experiments that should be performed before adopting each optimization.""",

    """You are the technical lead for a team building a production-quality CPU inference runtime.

    You must prepare a complete engineering implementation plan covering the entire development lifecycle, from initial model loading through high-concurrency generation and long-context memory management.

    The project is divided into multiple engineers working independently. The resulting interfaces must be sufficiently precise to allow parallel implementation.

    SECTION 1: PROJECT REQUIREMENTS
    Define functional and non-functional requirements.

    Include:
    - Model compatibility.
    - Tokenizer compatibility.
    - Deterministic execution.
    - Numerical correctness.
    - Long-context support.
    - Multiple independent sequences.
    - Streaming.
    - Cancellation.
    - Bounded memory.
    - Predictable allocation.
    - Thread safety.
    - Reproducible benchmarking.

    SECTION 2: PUBLIC INTERFACES
    Specify C++ interfaces for:
    - Model.
    - ModelConfig.
    - Tensor.
    - TensorView.
    - Tokenizer.
    - Operator.
    - TransformerLayer.
    - Attention.
    - KVCache.
    - Sequence.
    - SequenceManager.
    - Scheduler.
    - Sampler.
    - Generator.

    For each interface, explain method responsibilities, parameter semantics, ownership, return values, exceptions, and thread-safety guarantees.

    SECTION 3: KV CACHE API
    Develop an especially detailed KV-cache interface.

    Define operations for:
    - Creating a sequence.
    - Appending a token.
    - Appending multiple tokens.
    - Reading keys.
    - Reading values.
    - Querying sequence length.
    - Querying allocated capacity.
    - Growing storage.
    - Cloning a prefix.
    - Sharing immutable blocks.
    - Detaching shared blocks.
    - Deleting a sequence.
    - Cancelling a sequence.
    - Clearing all sequences.
    - Reporting memory usage.

    Explain atomicity requirements and failure behavior for every operation.

    SECTION 4: IMPLEMENTATION ORDER
    Define a dependency graph.

    Identify which modules can be implemented independently and which require completed upstream interfaces.

    Propose milestones for:
    - Tensor foundations.
    - File loading.
    - Tokenization.
    - CPU operators.
    - Single-layer execution.
    - Full transformer execution.
    - Greedy decoding.
    - Sampling.
    - KV caching.
    - Multiple sequences.
    - Scheduling.
    - Streaming.
    - Prefix caching.
    - Performance optimization.

    SECTION 5: INTEGRATION
    Explain how the team should integrate independently developed modules.

    Define API stability requirements, test fixtures, numerical reference outputs, mock components, and integration test stages.

    SECTION 6: QUALITY ASSURANCE
    Specify static analysis, compiler warnings, sanitizers, fuzzing, unit tests, differential testing, concurrency testing, and performance regression tests.

    SECTION 7: PERFORMANCE GOALS
    Define measurable targets for model loading, prefill throughput, decode throughput, memory utilization, allocation overhead, cache lookup, and concurrent throughput.

    SECTION 8: FINAL DELIVERABLES
    List every required design document, interface specification, implementation, test suite, benchmark, and operational guide.

    Make the result detailed enough to serve as the team's primary engineering execution plan."""
]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EVAL_TEXT = HERE / "data" / "eval.txt"


# ---------------------------------------------------------------- energy + process stats
def rapl_domains():
    """Top-level package counters (intel-rapl:0, intel-rapl:1, ...) that we can read."""
    out = []
    for d in sorted(Path("/sys/class/powercap").glob("intel-rapl:*")):
        if d.name.count(":") != 1:          # skip sub-domains (core, uncore) to avoid double counting
            continue
        try:
            int((d / "energy_uj").read_text())
            out.append((d / "energy_uj", int((d / "max_energy_range_uj").read_text())))
        except (OSError, ValueError):
            pass
    return out


def rapl_read(domains):
    return [int(p.read_text()) for p, _ in domains]


def rapl_joules(domains, before, after):
    total = 0
    for (_, wrap), b, a in zip(domains, before, after):
        total += a - b if a >= b else a + wrap - b
    return total / 1e6


class Energy:
    """Package energy over a block of code; .joules is None if RAPL is unreadable."""
    domains = None

    def __enter__(self):
        if Energy.domains is None:
            Energy.domains = rapl_domains()
        self.joules = None
        self.b = rapl_read(Energy.domains) if Energy.domains else None
        return self

    def __exit__(self, *exc):
        if self.b is not None:
            self.joules = rapl_joules(Energy.domains, self.b, rapl_read(Energy.domains))


CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def ollama_pids():
    """PIDs of Ollama processes (server + model runner)."""
    pids = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if b"ollama" in (d / "cmdline").read_bytes():
                    pids.append(int(d.name))
            except OSError:
                pass
    return pids


def cpu_seconds(pids):
    t = 0
    for pid in pids:
        try:
            f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            t += int(f[11]) + int(f[12])     # utime + stime
        except (OSError, IndexError):
            pass
    return t / CLK_TCK


def peak_rss_mb(pids):
    """Largest VmHWM (lifetime peak RSS) among the pids; for Ollama that is the runner."""
    best = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    best = max(best, int(line.split()[1]) / 1024)
        except OSError:
            pass
    return best


def chat_template(system, user):
    return f"<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n"


# ---------------------------------------------------------------- our engine
def run_engine(a, user):
    cmd = [a.bin, "bench", "--model", a.model, "--tokenizer", a.tokenizer,
           "-p", user, "--system", tagged_system(), "-n", str(a.n), "--threads", str(a.threads)] + a.engine_args.split()
    with Energy() as e:
        p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"engine failed:\n{p.stderr}")
    r = json.loads(p.stdout.strip().splitlines()[-1])
    r["energy_j"] = e.joules     # whole process, includes weight load
    return r


# ---------------------------------------------------------------- ollama
def http(url, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ollama_model_size(a):
    try:
        tags = http(f"{a.ollama_url}/api/tags", timeout=5)
    except (urllib.error.URLError, OSError):
        sys.exit("Ollama is not running. In another terminal run:  make ollama-serve")
    for m in tags.get("models", []):
        if m.get("name") == a.ollama_model or m.get("model") == a.ollama_model:
            return m.get("size", 0)
    sys.exit(f"Ollama has no model {a.ollama_model}. Run:  make ollama-pull")


def run_ollama(a, user):
    pids = ollama_pids()
    c0, w0 = cpu_seconds(pids), time.perf_counter()
    with Energy() as e:
        r = http(f"{a.ollama_url}/api/generate", {
        "model": a.ollama_model,
        "prompt": chat_template(tagged_system(), user),
        "raw": True,
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": a.n,
                    "num_thread": a.threads, "num_ctx": 2048},
        })
    pids = ollama_pids()                     # the runner may have just started
    wall = time.perf_counter() - w0
    cpu = cpu_seconds(pids) - c0
    pe, ped = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 0) / 1e6
    ec, ed = r.get("eval_count", 0), r.get("eval_duration", 0) / 1e6
    return {"prompt_tokens": pe, "gen_tokens": ec, "prefill_ms": ped, "decode_ms": ed,
            "prefill_tps": pe / (ped / 1000) if ped else 0.0,
            "decode_tps": ec / (ed / 1000) if ed else 0.0,
            "ttft_ms": ped + r.get("load_duration", 0) / 1e6,
            "peak_rss_mb": peak_rss_mb(pids),
            "cpu_util_pct": 100 * cpu / wall if wall else 0.0,
            "energy_j": e.joules}


# ---------------------------------------------------------------- helpers
def physical_cores():
    """Same default as the Makefile (scripts/physical_cores.sh), not os.cpu_count()."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "physical_cores.sh"
    try:
        return int(subprocess.run([str(script)], capture_output=True, text=True, check=True).stdout)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return os.cpu_count() or 4


def measure_bandwidth(a):
    if not a.bandwidth_bin or not Path(a.bandwidth_bin).exists():
        return None
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
    p = subprocess.run([a.bandwidth_bin], capture_output=True, text=True, env=env)
    return json.loads(p.stdout)["read_gbps"] if p.returncode == 0 else None


def median_of(runs):
    keys = ["prompt_tokens", "gen_tokens", "prefill_tps", "decode_tps", "ttft_ms",
            "peak_rss_mb", "cpu_util_pct"]
    m = {k: statistics.median(r[k] for r in runs) for k in keys}
    if all(r["energy_j"] is not None for r in runs):
        m["energy_j"] = statistics.median(r["energy_j"] for r in runs)
        m["j_per_tok"] = statistics.median(r["energy_j"] / max(r["gen_tokens"], 1) for r in runs)
    else:
        m["energy_j"] = m["j_per_tok"] = None
    return m


# ---------------------------------------------------------------- quality
def run_ppl(a, model):
    cmd = [a.bin, "ppl", "--model", model, "--tokenizer", a.tokenizer, "--text", str(a.eval_text),
           "--max-tokens", str(a.ppl_tokens), "--ctx", str(a.ppl_ctx), "--threads", str(a.threads)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"ppl failed:\n{p.stderr}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def reference_ppl(a):
    """fp32 reference result, cached by file size+mtime so it is computed once."""
    st = Path(a.ref_model).stat()
    key = f"{Path(a.ref_model).name}-{st.st_size}-{int(st.st_mtime)}-{a.ppl_tokens}-{a.ppl_ctx}"
    cache = RESULTS / f"ppl-ref-{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print("  quality: scoring fp32 reference (cached after the first time)", flush=True)
    r = run_ppl(a, a.ref_model)
    RESULTS.mkdir(exist_ok=True)
    cache.write_text(json.dumps(r))
    return r


def quality(a):
    ref = reference_ppl(a)
    if Path(a.model).resolve() == Path(a.ref_model).resolve():
        cur = ref
    else:
        print("  quality: scoring model under test", flush=True)
        cur = run_ppl(a, a.model)
    n = min(len(ref["top1"]), len(cur["top1"]))
    agree = sum(x == y for x, y in zip(ref["top1"], cur["top1"])) / n if n else 0.0
    return {"tokens": cur["tokens"], "ppl": cur["ppl"], "ref_ppl": ref["ppl"],
            "ppl_delta_pct": 100 * (cur["ppl"] / ref["ppl"] - 1), "top1_agree_pct": 100 * agree}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def bench(name, fn, a):
    print(f"  {name}: warm-up", flush=True)
    fn(a, "Warm-up: explain what a key-value cache stores during autoregressive transformer inference.")
    runs = []
    for i in range(a.runs):
        user = PROMPTS[i % len(PROMPTS)]
        r = fn(a, user)
        runs.append(r)
        print(f"  {name}: run {i + 1}/{a.runs}  decode {r['decode_tps']:.2f} tok/s  "
              f"prefill {r['prefill_tps']:.1f} tok/s", flush=True)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["engine", "ollama", "both"], default="both")
    ap.add_argument("--bin", default="build/tinyllama")
    ap.add_argument("--model", default="models/tinyllama-q4.bin")
    ap.add_argument("--tokenizer", default="models/tokenizer.bin")
    ap.add_argument("--bandwidth-bin", default="build/bandwidth")
    ap.add_argument("--ollama-model", default="tinyllama:1.1b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--threads", type=int, default=physical_cores(),
                    help="default: scripts/physical_cores.sh, the same as the Makefile")
    ap.add_argument("-n", type=int, default=512, help="max new tokens per run (long benchmark default)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--engine-args", default="", help='extra engine flags, e.g. "--kv q8 --spec ngram"')
    ap.add_argument("--quality", action="store_true", help="also measure perplexity vs fp32 reference")
    ap.add_argument("--ref-model", default="models/tinyllama-f32.bin", help="fp32 reference for quality")
    ap.add_argument("--eval-text", default=str(EVAL_TEXT))
    ap.add_argument("--ppl-tokens", type=int, default=512, help="tokens scored for perplexity")
    ap.add_argument("--ppl-ctx", type=int, default=256, help="perplexity window length")
    a = ap.parse_args()

    print(f"benchmark: {a.runs} runs, up to {a.n} new tokens, {a.threads} threads, greedy")
    gbps = measure_bandwidth(a)
    if gbps:
        print(f"  memory read probe: {gbps:.1f} GB/s (pure read loop; on laptops in powersave it can read"
              f" below what a real workload sustains, so it is shown for reference only)")
    if not rapl_domains():
        print("  energy: RAPL not readable, skipped (run: make rapl-perms)")

    rows = []
    if a.only in ("engine", "both"):
        runs = bench("tinyllama-cpp", run_engine, a)
        rows.append(("tinyllama-cpp", median_of(runs), runs[0]["bytes_per_token"], runs))
    if a.only in ("ollama", "both"):
        size = ollama_model_size(a)
        runs = bench("ollama", run_ollama, a)
        rows.append((f"ollama {a.ollama_model}", median_of(runs), size, runs))

    qual = quality(a) if a.quality and a.only in ("engine", "both") else None

    def fmt(v, spec):
        return "n/a" if v is None else format(v, spec)

    cols = [("prompt tok", lambda m, nb: fmt(m["prompt_tokens"], ".0f")),
            ("prefill tok/s", lambda m, nb: fmt(m["prefill_tps"], ".1f")),
            ("decode tok/s", lambda m, nb: fmt(m["decode_tps"], ".2f")),
            ("TTFT ms", lambda m, nb: fmt(m["ttft_ms"], ".0f")),
            ("peak RAM MB", lambda m, nb: fmt(m["peak_rss_mb"], ".0f")),
            ("model MB", lambda m, nb: fmt(nb / 1e6, ".0f")),
            ("CPU %", lambda m, nb: fmt(m["cpu_util_pct"], ".0f")),
            ("J/token", lambda m, nb: fmt(m["j_per_tok"], ".2f"))]
    cols.append(("read GB/s", lambda m, nb: fmt(m["decode_tps"] * nb / 1e9 if nb else None, ".1f")))
    print()
    hdr = f"{'engine':<24}" + "".join(f"{c:>14}" for c, _ in cols)
    print(hdr)
    print("-" * len(hdr))
    for name, m, nb, _ in rows:
        print(f"{name:<24}" + "".join(f"{f(m, nb):>14}" for _, f in cols))
    if len(rows) == 2 and rows[1][1]["decode_tps"]:
        ratio = rows[0][1]["decode_tps"] / rows[1][1]["decode_tps"]
        print(f"\ndecode speed vs Ollama: {ratio:.2f}x")
    if qual:
        print(f"quality ({qual['tokens']} tokens of {Path(a.eval_text).name}): ppl {qual['ppl']:.3f} "
              f"vs fp32 {qual['ref_ppl']:.3f} ({qual['ppl_delta_pct']:+.2f}%), "
              f"top-1 agreement {qual['top1_agree_pct']:.1f}%")
    print(f"CPU % = CPU time / wall time, 100% per busy core ({a.threads} threads -> max {100 * a.threads}%)")
    print("TTFT  = prompt in -> first token (Ollama: + its model load time)")
    print("peak RAM includes mmap'd weight pages (engine) / runner lifetime peak (Ollama)")
    print("read GB/s = decode tok/s x weight bytes per token (effective memory bandwidth used by decode)")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = git_commit()
    record = {"time": stamp, "commit": commit, "suite": "longer", "model": Path(a.model).name,
              "engine_args": a.engine_args, "threads": a.threads, "n": a.n, "bandwidth_gbps": gbps, "quality": qual,
              "results": {name: {"median": m, "model_bytes": nb, "runs": r} for name, m, nb, r in rows}}
    (RESULTS / f"{stamp}.json").write_text(json.dumps(record, indent=2))
    hist = RESULTS / "history.csv"
    new = not hist.exists()
    with hist.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "commit", "engine", "threads", "decode_tps", "prefill_tps", "model_mb",
                        "ttft_ms", "peak_rss_mb", "cpu_util_pct", "j_per_tok", "ppl", "ppl_delta_pct",
                        "top1_agree_pct"])
        for name, m, nb, _ in rows:
            q = qual if qual and name == "tinyllama-cpp" else {}
            w.writerow([stamp, commit, name, a.threads, f"{m['decode_tps']:.2f}",
                        f"{m['prefill_tps']:.1f}", f"{nb / 1e6:.0f}", f"{m['ttft_ms']:.0f}",
                        f"{m['peak_rss_mb']:.0f}", f"{m['cpu_util_pct']:.0f}",
                        "" if m["j_per_tok"] is None else f"{m['j_per_tok']:.3f}",
                        f"{q['ppl']:.3f}" if q else "", f"{q['ppl_delta_pct']:.2f}" if q else "",
                        f"{q['top1_agree_pct']:.1f}" if q else ""])
    print(f"saved bench/results/{stamp}.json")


if __name__ == "__main__":
    main()
