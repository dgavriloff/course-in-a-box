---
title: Deep Dive - The Ring-AllReduce Algorithm
---

## Module 5.2: Deep Dive - The Ring-AllReduce Algorithm

### Why Ring-AllReduce?

The Ring-AllReduce algorithm is a bandwidth-optimal approach to performing AllReduce across $$N$$ processes. It solves a fundamental challenge: how to sum values across all GPUs without creating a bottleneck.

### Naive AllReduce Approaches

#### Approach 1: Centralized Reduction

```
All GPUs → Send to GPU 0 → GPU 0 sums → GPU 0 broadcasts result → All GPUs
```

**Problems**:
* GPU 0's network becomes bottleneck
* Time complexity: $$O(N \cdot M)$$ where M is message size
* Doesn't scale to large clusters

#### Approach 2: Tree Reduction

```
GPUs organize in binary tree, reduce up tree, broadcast down tree
```

**Problems**:
* Better than centralized but still has bottlenecks at tree nodes
* Time: $$O(\log N)$$ steps but each step waits for slowest link

### The Ring-AllReduce Algorithm

Ring-AllReduce achieves optimal bandwidth usage by:
* Organizing GPUs in a logical ring
* Breaking data into chunks
* Simultaneously utilizing all network links

### The Two-Phase Algorithm

Ring-AllReduce operates in two phases:

* **Phase 1: Reduce-Scatter**
  * Each GPU reduces received data with its own data
  * After $$N-1$$ steps, each GPU has a different chunk of the final reduced result

* **Phase 2: AllGather**
  * Each GPU forwards its reduced chunk around the ring
  * After $$N-1$$ steps, each GPU has all chunks (complete AllReduce result)

### Detailed Algorithm Walkthrough

Consider 4 GPUs with arrays to reduce:

```
GPU 0: [a0, a1, a2, a3]
GPU 1: [b0, b1, b2, b3]
GPU 2: [c0, c1, c2, c3]
GPU 3: [d0, d1, d2, d3]

Goal: Each GPU should have [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
```

#### Phase 1: Reduce-Scatter (3 steps for 4 GPUs)

**Step 1**: Each GPU sends one chunk to next GPU in ring

```
GPU 0 sends a3 → GPU 1, receives d0 ← GPU 3
GPU 1 sends b0 → GPU 2, receives a3 ← GPU 0
GPU 2 sends c1 → GPU 3, receives b0 ← GPU 1
GPU 3 sends d2 → GPU 0, receives c1 ← GPU 2

After reduction:
GPU 0: [a0, a1, a2, a3+d3]
GPU 1: [a1+b1, b1, b2, b3]
GPU 2: [c0, b2+c2, c2, c3]
GPU 3: [d0, d1, c3+d3, d3]
```

**Step 2**: Continue passing and reducing

```
GPU 0: [a0, a1, a2+d2, a3+d3] (received and reduced d2)
GPU 1: [a1+b1, b1, b2, a3+b3+d3] (received and reduced a3+d3)
GPU 2: [c0, b2+c2, c2, b3+c3] (received and reduced b3)
GPU 3: [c0+d0, d1, c3+d3, d3] (received and reduced c0)
```

**Step 3**: Final reduce-scatter step

```
GPU 0: [a0+b0+c0+d0, a1, a2+d2, a3+d3]
GPU 1: [a1+b1, a1+b1+c1+d1, b2, a3+b3+d3]
GPU 2: [c0, b2+c2, a2+b2+c2+d2, b3+c3]
GPU 3: [c0+d0, d1, c3+d3, a3+b3+c3+d3]
```

Each GPU now has one fully reduced chunk!

#### Phase 2: AllGather (3 steps for 4 GPUs)

Now each GPU shares its reduced chunk:

**Steps 4-6**: Pass completed chunks around ring

```
After 3 more steps, each GPU has all chunks:
GPU 0: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
GPU 1: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
GPU 2: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
GPU 3: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
```

### Complexity Analysis

For $$N$$ processes with message size $$M$$:

* **Steps**: $$2(N-1)$$ communication steps
* **Data per step**: $$M/N$$ (one chunk)
* **Total data transferred per GPU**: $$2(N-1) \cdot M/N \approx 2M$$
* **Total time**: $$2(N-1) \cdot (\alpha + M/(N \cdot \beta))$$

Where α is latency and β is bandwidth.

### Why It's Optimal

Ring-AllReduce is bandwidth-optimal because:

1. **All links used simultaneously**: Every GPU sends and receives in parallel
2. **No bottlenecks**: No single GPU handles more traffic than others
3. **Minimal data transfer**: Each GPU sends/receives only $$2M$$ bytes total
4. **Scales linearly**: Adding GPUs doesn't increase per-GPU traffic significantly

### Implementation Considerations

#### Chunk Size Selection

```python
# Optimal chunk size balances overhead and pipeline efficiency
chunk_size = max(min_chunk_size, tensor_size // num_gpus)
```

* Too small: High overhead from frequent messages
* Too large: Poor pipeline utilization

#### Handling Non-Divisible Sizes

```python
if tensor_size % num_gpus != 0:
    # Last chunk is slightly larger
    last_chunk_size = chunk_size + (tensor_size % num_gpus)
```

#### Pipeline Optimization

Modern implementations pipeline communication and computation:

```python
# While sending chunk i, compute reduction for chunk i-1
send_async(chunk[i])
reduce(chunk[i-1])
```

### Lab Project: Implementing Ring-AllReduce

Students will implement a simplified version of Ring-AllReduce to understand:

* How data is chunked and routed
* Synchronization between communication and computation
* Performance characteristics vs. naive approaches

#### Starter Code Framework

```python
def ring_allreduce(tensor, rank, world_size):
    """
    Implements Ring-AllReduce algorithm
    
    Args:
        tensor: Local tensor to reduce (will be modified in-place)
        rank: This process's rank (0 to world_size-1)
        world_size: Total number of processes
    """
    chunk_size = len(tensor) // world_size
    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1 + world_size) % world_size
    
    # Phase 1: Reduce-Scatter
    for step in range(world_size - 1):
        send_chunk_idx = ???  # Student implements
        recv_chunk_idx = ???  # Student implements
        
        # Send and receive chunks
        send_data = tensor[send_chunk_idx]
        recv_data = receive_from(recv_rank)
        send_to(send_rank, send_data)
        
        # Reduce received data
        tensor[recv_chunk_idx] += recv_data
    
    # Phase 2: AllGather
    for step in range(world_size - 1):
        send_chunk_idx = ???  # Student implements
        recv_chunk_idx = ???  # Student implements
        
        # Send and receive completed chunks
        send_data = tensor[send_chunk_idx]
        recv_data = receive_from(recv_rank)
        send_to(send_rank, send_data)
        
        # Copy received data (no reduction in AllGather phase)
        tensor[recv_chunk_idx] = recv_data
    
    return tensor
```

#### Expected Outcomes

Students should observe:

* Ring-AllReduce completes in $$2(N-1)$$ steps
* All network links utilized simultaneously
* 10-100x speedup vs. naive centralized approach for large N
* Near-linear scaling as number of GPUs increases

### Real-World Applications

Ring-AllReduce is used in:

* **Horovod**: Uber's distributed training framework
* **PyTorch DDP**: Data-parallel training
* **TensorFlow Distribution Strategy**: Multi-GPU training
* **DeepSpeed**: Microsoft's training optimization library

### Advanced Variants

* **Hierarchical Ring-AllReduce**: Multiple rings for multi-node clusters
* **2D-Ring**: Separate rings for intra-node and inter-node communication
* **Butterfly AllReduce**: Alternative topology for specific network architectures

### Key Takeaways

* Ring-AllReduce achieves bandwidth-optimal AllReduce
* Two phases (Reduce-Scatter + AllGather) complete in $$2(N-1)$$ steps
* All network links utilized simultaneously with no bottlenecks
* Understanding this algorithm is crucial for distributed training optimization
* Modern frameworks use variants of this algorithm under the hood
