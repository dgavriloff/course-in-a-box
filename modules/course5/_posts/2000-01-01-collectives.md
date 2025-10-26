---
title: The Language of Parallelism - Collectives
---

## Module 5.1: The Language of Parallelism - Collectives

### Introduction to Collective Communications

In distributed computing, **collective operations** are communication patterns where multiple processes coordinate to exchange data. These operations are fundamental to distributed deep learning.

### NCCL: NVIDIA Collective Communications Library

**NCCL** (pronounced "Nickel") is NVIDIA's optimized library for multi-GPU and multi-node communication:

* Implements collective operations optimized for NVIDIA GPUs
* Supports both single-node and multi-node configurations
* Automatically selects optimal algorithms based on hardware topology
* Integrates with major deep learning frameworks (PyTorch, TensorFlow, JAX)

### Core Collective Primitives

The fundamental collective operations used in distributed training:

* **AllReduce**: Each process contributes data, all processes receive the combined result
  * **Use Case**: Gradient synchronization in data parallelism
  * **Operation**: Sum, average, min, max, product
  * **Result**: All ranks have identical output

* **Broadcast**: One process sends data to all other processes
  * **Use Case**: Distributing model parameters, configuration
  * **Operation**: Copy from source rank to all ranks
  * **Result**: All ranks have copy of source data

* **Reduce**: All processes contribute data, one process receives the combined result
  * **Use Case**: Collecting metrics, aggregating statistics
  * **Operation**: Sum, average, min, max, product
  * **Result**: Only root rank has the result

* **AllGather**: Each process contributes data, all processes receive all contributions concatenated
  * **Use Case**: Collecting embeddings, gathering predictions
  * **Operation**: Concatenate data from all ranks
  * **Result**: All ranks have full combined data

* **ReduceScatter**: Combines AllReduce with Scatter; each rank gets a portion of the reduced result
  * **Use Case**: Splitting reduced gradients across ranks
  * **Operation**: Reduce and partition output
  * **Result**: Each rank gets a different chunk

* **Scatter**: One process sends different data to each process
  * **Use Case**: Distributing data batches, partitioning work
  * **Operation**: Split data and send chunks to ranks
  * **Result**: Each rank receives unique data

### AllReduce: The Most Important Collective

AllReduce is the workhorse of distributed training:

```python
# PyTorch example
import torch.distributed as dist

# Each GPU has its own gradient tensor
local_gradient = compute_gradient()  # Shape: [model_size]

# Sum gradients across all GPUs, result on all GPUs
dist.all_reduce(local_gradient, op=dist.ReduceOp.SUM)

# Average gradient
local_gradient /= world_size
```

#### Why AllReduce Dominates

In data-parallel training:
1. Each GPU processes a different batch
2. Each computes gradients independently
3. Gradients must be averaged across all GPUs
4. All GPUs need the averaged gradient to update parameters

**Frequency**: Every training step (millions of times during training)

### Communication Complexity

Understanding costs helps optimize training:

| Operation | Data Sent | Data Received | Total Traffic |
|-----------|-----------|---------------|---------------|
| AllReduce | $$O(M)$$ | $$O(M)$$ | $$O(M \cdot N)$$ naive, $$O(M)$$ optimized |
| Broadcast | $$O(M)$$ (from root) | $$O(M)$$ | $$O(M \cdot N)$$ naive, $$O(M)$$ optimized |
| AllGather | $$O(M)$$ | $$O(M \cdot N)$$ | $$O(M \cdot N)$$ |
| ReduceScatter | $$O(M)$$ | $$O(M/N)$$ | $$O(M)$$ |

Where:
* $$M$$ = message size
* $$N$$ = number of processes

### Latency vs Bandwidth

Two key metrics for collective operations:

* **Latency** (α): Fixed overhead per message
  * Network initialization, protocol overhead
  * Typically 1-50 microseconds

* **Bandwidth** (β): Throughput for data transfer
  * Measured in GB/s or Gbps
  * Depends on interconnect technology

#### Cost Model

```
Time = α + (Message_Size / β)
```

For small messages: latency dominates
For large messages: bandwidth dominates

### Algorithmic Optimizations

NCCL employs sophisticated algorithms:

#### Ring Algorithm

* Processes arranged in logical ring
* Data passed around ring in chunks
* Optimal bandwidth utilization
* Used for AllReduce, AllGather, ReduceScatter

#### Tree Algorithm

* Processes arranged in tree structure
* Faster for small messages (lower latency)
* Used for Broadcast, Reduce

#### Double-Binary Tree

* Hybrid approach combining two tree structures
* Balances latency and bandwidth
* Used for AllReduce on certain topologies

### Integration with Deep Learning Frameworks

#### PyTorch Distributed

```python
import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend='nccl', init_method='env://')

# Wrap model for data parallelism
model = torch.nn.parallel.DistributedDataParallel(model)

# Training loop - gradients automatically synchronized via AllReduce
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # AllReduce happens here
    optimizer.step()
```

#### TensorFlow/Horovod

```python
import horovod.tensorflow as hvd

hvd.init()

# Scale learning rate by number of GPUs
optimizer = tf.train.AdamOptimizer(lr * hvd.size())

# Wrap optimizer to average gradients
optimizer = hvd.DistributedOptimizer(optimizer)
```

### Performance Considerations

* **Message Size**: Larger messages amortize latency overhead
* **Frequency**: Fewer, larger collectives better than many small ones
* **Overlap**: Overlap communication with computation when possible
* **Topology Awareness**: Algorithm selection depends on network topology

### Monitoring and Debugging

Tools for analyzing collective performance:

* **NCCL Tests**: Benchmark suite for collective operations
* **NVIDIA Nsight Systems**: Timeline view of GPU and network activity
* **Profilers**: PyTorch Profiler, TensorFlow Profiler show communication overhead

### Key Takeaways

* Collective operations are the foundation of distributed deep learning
* AllReduce is the most critical operation for data-parallel training
* NCCL provides optimized implementations with automatic algorithm selection
* Understanding communication patterns enables better system design
* Modern frameworks abstract collectives but understanding them aids debugging and optimization
