---
title: Model Parallelism - Splitting the Model
---

## Module 6.2: Model Parallelism - Splitting the Model

### When Data Parallelism Isn't Enough

Even with ZeRO, some models are too large to fit on available GPUs. **Model parallelism** splits the model itself across multiple devices.

Two primary approaches:
1. **Tensor Parallelism**: Split individual layers across GPUs
2. **Pipeline Parallelism**: Split model vertically into stages

### Tensor Parallelism

**Tensor parallelism** partitions individual weight matrices and operations across GPUs.

#### How It Works

For a linear layer $$Y = XW$$:

**Column Parallelism**: Split W column-wise
```
W = [W_1 | W_2]  # Split across 2 GPUs

GPU 1: Y_1 = X @ W_1
GPU 2: Y_2 = X @ W_2

Y = [Y_1 | Y_2]  # Concatenate results
```

**Row Parallelism**: Split W row-wise
```
W = [W_1]  # Split across 2 GPUs
    [W_2]

GPU 1: Y = X_1 @ W_1
GPU 2: Y = X_2 @ W_2

Y = Y_1 + Y_2  # Sum results (requires AllReduce)
```

#### Attention Layer Partitioning

For multi-head attention with H heads:

```python
# Each GPU handles H/N heads
GPU 1: Heads 1 to H/N
GPU 2: Heads (H/N + 1) to 2H/N
...
GPU N: Heads ((N-1)H/N + 1) to H
```

**Communication**: 
* AllReduce after attention output projection
* Broadcast input activations if needed

#### Feed-Forward Network Partitioning

For FFN: $$\text{FFN}(x) = \text{GeLU}(xW_1)W_2$$

```python
# Split first layer column-wise, second layer row-wise
GPU 1: 
  h_1 = GeLU(x @ W1_1)  # W1 split column-wise
  y_1 = h_1 @ W2_1      # W2 split row-wise

GPU 2:
  h_2 = GeLU(x @ W1_2)
  y_2 = h_2 @ W2_2

# AllReduce sum for final output
y = y_1 + y_2
```

### Megatron-LM Tensor Parallelism

NVIDIA's Megatron-LM implements optimized tensor parallelism:

```python
# Simplified Megatron tensor parallel linear layer
class ColumnParallelLinear(torch.nn.Module):
    def forward(self, input):
        # input is replicated across GPUs
        # Each GPU computes its column slice
        output = torch.matmul(input, self.weight_slice)
        # output is partitioned across GPUs
        return output

class RowParallelLinear(torch.nn.Module):
    def forward(self, input):
        # input is partitioned across GPUs
        # Each GPU computes with its row slice
        output_parallel = torch.matmul(input, self.weight_slice)
        # AllReduce to get final output
        output = reduce_from_parallel_region(output_parallel)
        return output
```

#### Communication Costs

For a Transformer layer with hidden size $$h$$ and sequence length $$s$$:

* **AllReduce calls per layer**: 2 (after attention and FFN)
* **Data volume per AllReduce**: $$s \times h$$ elements
* **Total communication**: $$4sh$$ bytes (FP32) per layer

### Pipeline Parallelism

**Pipeline parallelism** splits the model into stages, with each stage on a different GPU.

#### Basic Pipeline Parallelism

```
GPU 1: Layers 1-6
GPU 2: Layers 7-12
GPU 3: Layers 13-18
GPU 4: Layers 19-24
```

**Problem**: Naive implementation has severe GPU under-utilization (bubble overhead).

```
Time →
GPU 1: [F1]     [F2]     [F3]     [F4]
GPU 2:      [F1]     [F2]     [F3]     [F4]
GPU 3:           [F1]     [F2]     [F3]     [F4]
GPU 4:                [F1]     [F2]     [F3]     [F4]
```

Most of the time, most GPUs are idle!

#### GPipe: Micro-Batching

**GPipe** improves efficiency by splitting batches into micro-batches:

```
Batch of size 8 → 4 micro-batches of size 2

Time →
GPU 1: [F1][F2][F3][F4]    [B1][B2][B3][B4]
GPU 2:     [F1][F2][F3][F4][B1][B2][B3][B4]
GPU 3:         [F1][F2][F3][F4][B1][B2][B3][B4]
GPU 4:             [F1][F2][F3][F4][B1][B2][B3][B4]
```

**Bubble overhead**: $$(P-1) / (M + P - 1)$$

Where P = number of pipeline stages, M = number of micro-batches

#### PipeDream: Interleaved Schedules

**PipeDream** further optimizes by interleaving forward and backward passes:

```
Time →
GPU 1: [F1][F2][F3][F4][B4][B3][B2][B1]
GPU 2:  [F1][F2][F3][F4][B4][B3][B2][B1]
GPU 3:   [F1][F2][F3][F4][B4][B3][B2][B1]
GPU 4:    [F1][F2][F3][F4][B4][B3][B2][B1]
```

#### Virtual Pipeline Parallelism

Interleave model chunks across GPUs:

```
GPU 1: Layers [1-3, 13-15]
GPU 2: Layers [4-6, 16-18]
GPU 3: Layers [7-9, 19-21]
GPU 4: Layers [10-12, 22-24]
```

**Benefit**: Reduces bubble overhead by 2-4×

### Communication Patterns

#### Tensor Parallelism
* **Pattern**: AllReduce within each layer
* **Frequency**: 2× per Transformer layer
* **Volume**: Proportional to batch size and hidden dimension
* **Bandwidth Requirement**: High (requires fast interconnect)

#### Pipeline Parallelism
* **Pattern**: Point-to-point between adjacent stages
* **Frequency**: Once per micro-batch per stage
* **Volume**: Proportional to batch size, sequence length, and hidden dimension
* **Bandwidth Requirement**: Moderate (fewer GPUs communicate)

### When to Use Each Approach

#### Tensor Parallelism
* **Best for**: Intra-node parallelism (fast NVLink)
* **Scales to**: 4-8 GPUs typically
* **Advantages**: Fine-grained parallelism, low bubble overhead
* **Disadvantages**: High communication, requires fast interconnect

#### Pipeline Parallelism
* **Best for**: Inter-node parallelism (slower Infiniband)
* **Scales to**: Dozens to hundreds of stages
* **Advantages**: Reduced communication frequency, works over slower interconnects
* **Disadvantages**: Bubble overhead, requires micro-batching

### Hybrid Approaches

Combining tensor and pipeline parallelism:

```
Node 1:
  GPU 1-4: Layers 1-12 (tensor parallel within)
Node 2:
  GPU 5-8: Layers 13-24 (tensor parallel within)
Node 3:
  GPU 9-12: Layers 25-36 (tensor parallel within)
```

**Benefits**:
* Tensor parallelism within nodes (fast NVLink)
* Pipeline parallelism across nodes (slower network)
* Optimal use of hardware topology

### Memory Requirements

#### Tensor Parallelism
* Each GPU stores: 1/N of each layer's parameters
* Activations: Depends on partitioning scheme
* **Total memory per GPU**: $$P / N + \text{activations}$$

#### Pipeline Parallelism
* Each GPU stores: Complete parameters for its stages
* Activations: Only for current micro-batch
* **Total memory per GPU**: $$P / N_{\text{stages}} + \text{micro-batch activations}$$

### Implementation Frameworks

* **Megatron-LM**: NVIDIA's tensor parallelism implementation
* **DeepSpeed**: Pipeline parallelism with ZeRO
* **FairScale**: Facebook's model parallelism library
* **Alpa**: Automatic parallelization with compiler optimizations
* **Colossal-AI**: Unified parallelism framework

### Example Configuration

For a 175B parameter model on 128 GPUs:

```python
# Hybrid parallelism setup
tensor_parallel_size = 8    # 8-way tensor parallelism
pipeline_parallel_size = 16  # 16 pipeline stages
data_parallel_size = 1       # No data parallelism (or use ZeRO)

# Total GPUs = TP × PP × DP = 8 × 16 × 1 = 128
```

### Key Takeaways

* **Tensor parallelism** splits individual layers, requires fast interconnect
* **Pipeline parallelism** splits model into stages, tolerates slower interconnect
* Micro-batching and scheduling optimizations reduce pipeline bubble overhead
* Hybrid approaches leverage hardware topology for optimal performance
* Choice of parallelism strategy depends on model size, hardware, and interconnect
* Modern frameworks abstract parallelism details but understanding principles aids debugging
