---
title: Case Study 1 - Fused Attention (FlashAttention)
---

## Module 4.2: Case Study 1 - Fused Attention (FlashAttention)

### The Attention Mechanism

Self-attention is the core operation in Transformer models. For a sequence of length *N*, standard attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
* $$Q, K, V$$ are query, key, and value matrices
* $$d_k$$ is the key dimension
* The result is an $$N \times N$$ attention matrix followed by multiplication with $$V$$

### The Memory Problem

Standard attention implementation has severe memory constraints:

#### Quadratic Memory Complexity

* Attention matrix is $$N \times N$$
* For sequence length 2048: 4,194,304 elements per head
* With 32 heads: ~134 million elements per layer
* **Problem**: Cannot fit in fast GPU memory for long sequences

#### Multiple Memory Reads/Writes

Traditional implementation requires multiple passes:

1. Compute $$QK^T$$ → write to HBM (high-bandwidth memory)
2. Read $$QK^T$$, apply softmax → write to HBM
3. Read softmax result, multiply by $$V$$ → write to HBM

**Total**: 3 writes and 3 reads to slow global memory

### FlashAttention: IO-Aware Exact Attention

FlashAttention (Dao et al., 2022) achieves exact attention with dramatically reduced memory usage through clever algorithm redesign.

#### Key Innovations

1. **Tiling**: Break computation into blocks that fit in SRAM (shared memory)
2. **Recomputation**: Recompute attention scores in backward pass instead of storing
3. **Online Softmax**: Compute softmax incrementally without full matrix
4. **Fused Kernel**: All operations in single kernel

### The Tiling Strategy

Instead of computing the full $$N \times N$$ attention matrix:

1. **Divide Q, K, V into blocks**
   * Block size chosen to fit in shared memory (~100-200 KB)
   * Typical block: 128x128 elements

2. **Load blocks into shared memory**
   * Load block of Q: $$Q_i$$
   * Load block of K: $$K_j$$
   * Compute partial attention scores

3. **Incrementally update output**
   * Maintain running statistics for softmax
   * Update output block as new scores computed

4. **Never materialize full attention matrix**
   * Only block-level matrices in shared memory
   * Massive memory savings

### Online Softmax Algorithm

Computing softmax incrementally is non-trivial:

**Challenge**: Softmax requires max and sum over entire row

**Solution**: Use a numerically stable incremental algorithm:

```python
# Simplified version of online softmax
for each block:
    # Compute local max and sum
    local_max = max(scores_block)
    local_sum = sum(exp(scores_block - local_max))
    
    # Update global max and rescale
    if local_max > global_max:
        rescale_factor = exp(global_max - local_max)
        global_sum = global_sum * rescale_factor + local_sum
        global_max = local_max
    else:
        global_sum += local_sum * exp(local_max - global_max)
    
    # Update output with properly scaled values
    output += exp(scores_block - global_max) @ values_block
```

### Performance Improvements

FlashAttention achieves dramatic speedups:

#### Memory Usage

* **Standard Attention**: O(N²) memory for attention matrix
* **FlashAttention**: O(N) memory (only Q, K, V stored)
* **Enables**: 4-8x longer sequences on same hardware

#### Speed

* **Training**: 2-4x faster than standard attention
* **Inference**: 1.5-3x faster
* **Longer Sequences**: Greater speedup (5-10x for N=4096+)

#### Why Faster?

1. **Reduced HBM Access**: ~7x fewer bytes transferred
2. **Better SRAM Utilization**: Data stays on-chip
3. **Fused Operations**: No kernel launch overhead
4. **No Intermediate Storage**: Attention matrix never written

### FlashAttention-2

The second version (2023) added further optimizations:

* **Improved Work Partitioning**: Better load balancing across SMs
* **Reduced Non-Matmul Operations**: Optimized softmax and rescaling
* **Tuned for A100/H100**: Leverages newer GPU features
* **Additional 2x Speedup**: Total 4-8x vs. baseline

### Code Structure

Simplified FlashAttention kernel structure:

```cuda
__global__ void flashAttentionKernel(
    float* Q, float* K, float* V, float* O, 
    int N, int d, int blockSize) {
    
    __shared__ float Qi[BLOCK_SIZE][D];
    __shared__ float Kj[BLOCK_SIZE][D];
    __shared__ float Vj[BLOCK_SIZE][D];
    
    // Load Q block into shared memory
    loadBlock(Q, Qi, blockIdx.x, N, d);
    
    // Initialize running statistics
    float maxScore = -INFINITY;
    float sumExp = 0.0f;
    float output[D] = {0};
    
    // Loop over K, V blocks
    for (int j = 0; j < numBlocks; j++) {
        loadBlock(K, Kj, j, N, d);
        loadBlock(V, Vj, j, N, d);
        
        // Compute attention scores for this block
        float scores[BLOCK_SIZE];
        computeScores(Qi, Kj, scores);
        
        // Update online softmax statistics
        updateSoftmax(scores, &maxScore, &sumExp, output, Vj);
        
        __syncthreads();
    }
    
    // Finalize and write output
    finalizeOutput(output, maxScore, sumExp, O, blockIdx.x);
}
```

### Impact on LLM Training

FlashAttention has become essential for modern LLM training:

* **Longer Context**: Train on 8K-32K token sequences
* **Larger Batches**: Fit more samples per GPU
* **Lower Cost**: Same model with fewer GPUs or less time
* **Wider Adoption**: Used in GPT-4, LLaMA, PaLM, and others

### Lessons Learned

1. **Algorithm-Hardware Co-Design**: Understanding hardware constraints enables better algorithms
2. **IO Awareness**: Accounting for memory hierarchy is crucial
3. **Recomputation Trade-off**: Sometimes recomputing is faster than storing
4. **Block-Level Parallelism**: Tiling enables massive parallelism within memory constraints

### Further Reading

* Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
* Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
* Triton implementations and tutorials
