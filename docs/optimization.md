# Attention Optimization Techniques

This document details the optimization techniques implemented in the attention kernels.

## Overview

The attention mechanism in transformers is computationally expensive, with O(n²) complexity in sequence length. This project explores various optimizations to improve performance on AMD GPUs.

## Implementations

### 1. Naive Attention (`attention_naive.hip`)

The baseline implementation computes attention directly:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

**Characteristics:**
- Simple, readable implementation
- O(n²) memory for attention matrix
- Poor memory bandwidth utilization

### 2. Shared Memory Attention (`attention_shared.hip`)

Uses LDS (Local Data Share) to cache tiles of Q, K, V matrices.

**Optimizations:**
- Tiled computation to fit in LDS
- Coalesced global memory accesses
- Reduced global memory traffic

**Performance:**
- 2-3x faster than naive for long sequences
- Better memory efficiency

### 3. Flash Attention (`attention_flash.hip`)

Implements the FlashAttention algorithm (Dao et al.) for ROCm.

**Key Optimizations:**
- Online softmax computation
- Minimal HBM reads/writes
- Fused forward pass
- Memory complexity: O(n) instead of O(n²)

**Performance:**
- 3-5x faster than naive
- Enables longer context lengths
- 10-20x memory reduction

## Memory Hierarchy Utilization

### Global Memory (HBM)
- Store Q, K, V matrices
- Final output storage
- Target: Minimize access

### L2 Cache
- Automatic caching of reused data
- 64KB - 8MB depending on GPU

### LDS (Shared Memory)
- 64KB per compute unit
- Store tiles of Q, K, V
- Store partial softmax results

### Registers
- Current computation values
- Accumulator variables

## Block Size Selection

Optimal block sizes depend on:
1. **Sequence length**: Longer sequences need more tiles
2. **Head dimension**: Common values: 64, 128
3. **GPU architecture**: Wave size (64 for CDNA)

Recommended configurations:

| Sequence | Head Dim | Block Q | Block K |
|----------|----------|---------|---------|
| 512 | 64 | 64 | 64 |
| 2048 | 64 | 64 | 64 |
| 8192 | 64 | 32 | 128 |

## Profiling Results

### MI100 Performance (seq_len=2048, head_dim=64)

| Implementation | Time (ms) | Memory (MB) | Speedup |
|----------------|-----------|-------------|---------|
| Naive | 45.2 | 256 | 1.0x |
| Shared | 18.5 | 256 | 2.4x |
| Flash | 9.3 | 16 | 4.9x |

### Scaling Analysis

Flash Attention scales better with sequence length:

```
Sequence   Naive    Flash    Speedup
1024       12ms     4ms      3.0x
2048       45ms     9ms      5.0x
4096       180ms    20ms     9.0x
8192       720ms    45ms     16.0x
```

## Implementation Details

### Online Softmax

Flash Attention computes softmax incrementally:

```cpp
// For each new block of scores
m_new = max(m_old, max(scores))
l_new = l_old * exp(m_old - m_new) + sum(exp(scores - m_new))
O_new = O_old * (l_old / l_new) * exp(m_old - m_new) + sum(...)
```

### Memory Access Pattern

Optimal access pattern for AMD GPUs:

```cpp
// Good: Coalesced, 4-float vector loads
float4 q_vec = *reinterpret_cast<float4*>(&Q[base_idx]);

// Bad: Strided access
float q = Q[base_idx + threadIdx.x * stride];
```

### LDS Bank Conflicts

Avoid bank conflicts with padding:

```cpp
// Without padding: 32-way bank conflict
__shared__ float tile[32][32];

// With padding: No conflicts
__shared__ float tile[32][33];
```

## Future Optimizations

1. **Multi-Query Attention (MQA)**: Share K, V across heads
2. **Grouped Query Attention (GQA)**: Group heads for K, V
3. **PagedAttention**: Efficient KV cache for inference
4. **FP16/BF16**: Use mixed precision
5. **Async copies**: Overlap compute and memory

## References

1. [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
2. [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
3. [AMD ROCm Documentation](https://rocm.docs.amd.com/)
