---
title: The CUDA Execution and Memory Model
---

## Module 3.1: The CUDA Execution and Memory Model

### Introduction to CUDA

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model. It enables developers to harness the massive parallelism of GPUs for general-purpose computing.

### The GPU Architecture

Modern GPUs consist of:

* **Streaming Multiprocessors (SMs)**: Independent processing units
* **CUDA Cores**: Execution units within each SM
* **Memory Hierarchy**: Multiple levels from registers to global memory
* **Thread Schedulers**: Hardware units that manage thread execution

### Thread Hierarchy

CUDA organizes computation into a hierarchical structure:

* **Thread**: The basic unit of execution
  * Executes a kernel function
  * Has unique thread ID within its block
  * Has access to thread-local registers

* **Block** (Thread Block): Group of threads
  * Threads in a block can cooperate via shared memory
  * Can synchronize using barriers
  * Scheduled to run on a single SM
  * Maximum size typically 1024 threads

* **Grid**: Collection of thread blocks
  * All blocks execute the same kernel
  * Blocks execute independently
  * Can span multiple SMs

#### Thread Indexing

CUDA provides built-in variables for thread identification:

```cuda
// 1D indexing
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D indexing
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

### Memory Hierarchy

CUDA GPUs have multiple memory spaces with different characteristics:

* **Registers**: Fastest, per-thread, limited (~64KB per SM)
  * Automatic variables in kernel code
  * Spilling to local memory if registers exhausted

* **Shared Memory**: Fast, per-block, programmable cache (~48-164KB per SM)
  * Explicitly managed by programmer
  * Enables thread cooperation within block
  * 100x faster than global memory

* **Local Memory**: Per-thread, actually in global memory
  * Used for register spills and large arrays
  * Same latency as global memory despite the name

* **Global Memory**: Large, high-latency, accessible by all threads
  * Main device memory (GB to TB)
  * ~400-800 cycles latency
  * Persistent across kernel launches

* **Constant Memory**: Read-only, cached, 64KB
  * Broadcast capability for same address reads
  * Optimal for uniform access patterns

* **Texture Memory**: Read-only, cached, optimized for spatial locality
  * 2D/3D locality optimization
  * Hardware interpolation support

### Memory Performance Characteristics

| Memory Type | Latency | Bandwidth | Scope | Lifetime |
|-------------|---------|-----------|-------|----------|
| Registers | 1 cycle | Highest | Thread | Kernel |
| Shared Memory | ~20 cycles | Very High | Block | Kernel |
| Global Memory | ~400 cycles | High | All threads | Application |
| Constant Memory | ~20 cycles (cached) | Medium | All threads | Application |

### Execution Model

#### Warps

* A warp is a group of 32 threads
* Threads in a warp execute in lockstep (SIMT - Single Instruction Multiple Thread)
* All threads in a warp execute the same instruction on different data
* Branch divergence within a warp causes serialization

#### Occupancy

**Occupancy** is the ratio of active warps to maximum possible warps per SM:

* Higher occupancy helps hide memory latency
* Limited by registers, shared memory, and threads per block
* Not always optimal - resource usage matters too

### Kernel Launch Configuration

```cuda
// Determine block and grid dimensions
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

// Launch kernel
myKernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);
```

### Key Takeaways

* CUDA's hierarchical thread organization enables massive parallelism
* Memory hierarchy requires careful optimization for performance
* Understanding warps is crucial for avoiding divergence
* Occupancy and resource usage balance affects performance
