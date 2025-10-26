---
title: Foundational Optimization Techniques
---

## Module 3.2: Foundational Optimization Techniques

### Memory Access Patterns

Optimizing memory access is the most critical aspect of GPU performance. Poor memory access patterns can reduce effective bandwidth by 10-100x.

### Memory Coalescing

**Memory coalescing** occurs when threads in a warp access consecutive memory locations in a single transaction.

#### Coalesced Access Pattern

```cuda
// GOOD: Coalesced access
__global__ void coalescedAccess(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[tid];  // Consecutive threads access consecutive addresses
}
```

#### Uncoalesced Access Pattern

```cuda
// BAD: Strided access (stride = 32)
__global__ void stridedAccess(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[tid * 32];  // Large gaps between accesses
}
```

#### Performance Impact

* **Coalesced Access**: 1 memory transaction per warp (32 threads)
* **Uncoalesced Access**: Up to 32 separate transactions
* **Speedup**: 10-30x for well-coalesced patterns

#### Best Practices

* Arrange data structures for unit-stride access
* Consider transpose operations if necessary
* Use array-of-structures (AoS) to structure-of-arrays (SoA) transformation
* Pad arrays to avoid bank conflicts and partition camping

### Shared Memory Utilization

**Shared memory** acts as a programmer-managed cache, enabling fast data sharing between threads in a block.

#### Use Cases

1. **Tiling**: Break computation into smaller tiles that fit in shared memory
2. **Reduction Operations**: Efficiently combine values across threads
3. **Data Reuse**: Load data once, use multiple times
4. **Thread Cooperation**: Enable communication between threads

#### Tiling Example

```cuda
__global__ void tiledMatrixMul(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // Load tile into shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        
        __syncthreads();  // Wait for all threads to load
        
        // Compute using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();  // Wait before loading next tile
    }
    
    C[row * N + col] = sum;
}
```

### Bank Conflicts

Shared memory is divided into **banks** (typically 32) for parallel access.

#### What Are Bank Conflicts?

* When multiple threads in a warp access the same bank (but different addresses), accesses are serialized
* Reduces effective bandwidth proportionally to conflict degree
* **Broadcast**: All threads accessing the *same* address is conflict-free

#### Avoiding Bank Conflicts

```cuda
// BAD: Bank conflicts
__shared__ float data[32][32];
float value = data[threadIdx.x][threadIdx.y];  // Column access causes conflicts

// GOOD: Padding eliminates conflicts
__shared__ float data[32][33];  // Extra column shifts banks
float value = data[threadIdx.x][threadIdx.y];
```

#### Conflict Detection

* Use NVIDIA Nsight Compute profiler
* Look for `shared_load_transactions` and `shared_store_transactions`
* Ideal ratio: transactions per request = 1.0

### Optimization Strategy Summary

#### Step 1: Maximize Memory Bandwidth

* Ensure coalesced global memory accesses
* Use shared memory for frequently accessed data
* Eliminate bank conflicts in shared memory

#### Step 2: Maximize Occupancy

* Balance thread count, register usage, and shared memory
* Use occupancy calculator to find sweet spot
* Don't over-optimize - 50% occupancy often sufficient

#### Step 3: Minimize Divergence

* Avoid branch divergence within warps
* Restructure code to reduce if-else statements
* Use warp-level primitives when possible

#### Step 4: Profile and Iterate

* Use NVIDIA Nsight Compute and Nsight Systems
* Focus on bottlenecks identified by profiler
* Measure actual performance, not theoretical

### Common Pitfalls

* **Over-optimization**: Premature optimization without profiling
* **Ignoring Algorithmic Complexity**: O(N²) is still O(N²) on GPU
* **Insufficient Work**: GPU excels at massive parallelism (10K+ threads)
* **Memory Patterns**: Assuming CPU optimization techniques apply to GPU
