---
title: Data Parallelism and its Memory Limits
---

## Module 6.1: Data Parallelism and its Memory Limits

### Data Parallelism Fundamentals

**Data parallelism** is the simplest and most common approach to distributed training:

* Each GPU holds a complete copy of the model
* Each GPU processes a different batch of data
* Gradients are averaged across GPUs using AllReduce
* All GPUs update parameters with the averaged gradient

### Data Parallelism in Action

```python
# PyTorch DistributedDataParallel example
model = MyLargeModel()
model = torch.nn.parallel.DistributedDataParallel(model)

for batch in dataloader:
    # Each GPU processes different batch
    outputs = model(batch)  # Forward pass
    loss = criterion(outputs, targets)
    loss.backward()  # Backward pass
    # DDP automatically does AllReduce on gradients here
    optimizer.step()  # Update local parameters
```

### Memory Requirements

For a model with $$P$$ parameters:

#### Training Memory Breakdown

* **Model Parameters**: $$P$$ values (FP32: 4P bytes, FP16: 2P bytes)
* **Gradients**: $$P$$ values (same size as parameters)
* **Optimizer State**: Varies by optimizer
  * SGD: No additional state (0P)
  * Momentum SGD: 1× parameters (4P bytes)
  * Adam: 2× parameters (8P bytes for first and second moments)
* **Activations**: Depends on batch size and model architecture
  * Largest component for deep networks
  * Scales with batch size and sequence length

#### Example: 175B Parameter Model (GPT-3 scale)

Using Adam optimizer with FP32:

* Parameters: 175B × 4 bytes = 700 GB
* Gradients: 175B × 4 bytes = 700 GB  
* Optimizer state: 175B × 8 bytes = 1,400 GB
* **Total**: 2,800 GB (2.8 TB) per GPU

**Problem**: Single A100 GPU has 80 GB memory—model doesn't fit!

### The Memory Wall

Data parallelism hits a wall when:

$$\text{Model Memory} + \text{Activation Memory} > \text{GPU Memory}$$

For modern LLMs:
* 7B parameters: Fits on single GPU with careful optimization
* 13B parameters: Challenging on single GPU
* 70B+ parameters: Impossible on single GPU with standard data parallelism

### ZeRO: Zero Redundancy Optimizer

**ZeRO** (Zero Redundancy Optimizer) by Microsoft DeepSpeed eliminates redundant storage in data parallelism.

#### Key Insight

In standard data parallelism, every GPU stores:
* Complete model parameters
* Complete gradients
* Complete optimizer state

But each GPU only needs:
* Complete parameters for forward pass
* Its own gradients for backward pass
* Its own optimizer state for parameter update

### ZeRO Optimization Stages

ZeRO progressively partitions training state across GPUs:

* **ZeRO Stage 1** ($$P_{os}$$): Partition Optimizer State
  * Each GPU stores 1/N of optimizer state
  * Memory reduction: 4× for Adam (optimizer state is 50% of memory)
  * Communication overhead: Minimal (only during optimizer step)

* **ZeRO Stage 2** ($$P_{os+g}$$): Partition Optimizer State + Gradients
  * Each GPU stores 1/N of optimizer state and gradients
  * Memory reduction: 8× compared to standard data parallelism
  * Communication overhead: Moderate (gradient reduce-scatter)

* **ZeRO Stage 3** ($$P_{os+g+p}$$): Partition Optimizer State + Gradients + Parameters
  * Each GPU stores 1/N of everything
  * Memory reduction: Proportional to number of GPUs (N×)
  * Communication overhead: Higher (parameters gathered for forward/backward)

### Memory Savings Calculation

For 175B parameter model with Adam on 64 GPUs:

#### Standard Data Parallelism
* Per GPU: 2.8 TB (model doesn't fit!)

#### ZeRO Stage 1 ($$P_{os}$$)
* Per GPU: 1.4 TB / 64 + 700 GB (params) + 700 GB (grads) = ~1.42 TB (still doesn't fit)

#### ZeRO Stage 2 ($$P_{os+g}$$)
* Per GPU: 1.4 TB / 64 + 700 GB / 64 + 700 GB (params) = ~722 GB (tight!)

#### ZeRO Stage 3 ($$P_{os+g+p}$$)
* Per GPU: 2.8 TB / 64 = ~44 GB (fits comfortably!)

### Communication Patterns

#### ZeRO Stage 1
* **Forward/Backward**: No extra communication (parameters local)
* **Optimizer Step**: Gather updated parameters with AllGather

#### ZeRO Stage 2
* **Forward**: No extra communication
* **Backward**: ReduceScatter for gradients
* **Optimizer Step**: AllGather for parameters

#### ZeRO Stage 3
* **Forward**: AllGather parameters for each layer before computation
* **Backward**: ReduceScatter gradients after each layer
* **Optimizer Step**: Parameters already distributed

### Performance Implications

Trade-offs for each stage:

| Stage | Memory Savings | Communication Overhead | Throughput Impact |
|-------|---------------|----------------------|-------------------|
| Stage 1 | 4× | Minimal (<5%) | ~3-5% slowdown |
| Stage 2 | 8× | Moderate (~10%) | ~5-10% slowdown |
| Stage 3 | N× | Significant (~20-30%) | ~10-20% slowdown |

### When to Use Each Stage

* **Stage 1**: Large models that almost fit in GPU memory
* **Stage 2**: Models that definitely don't fit, but training is memory-bound
* **Stage 3**: Extremely large models where memory is the primary constraint

### ZeRO-Offload

Extension of ZeRO that offloads to CPU memory:

* Optimizer state stored in CPU RAM
* Parameters and gradients remain on GPU
* Useful when GPU memory is scarce but CPU RAM is abundant

**Use case**: Training large models on consumer GPUs

### ZeRO-Infinity

Further extension supporting:
* NVMe SSD storage for optimizer state
* Enables training trillion-parameter models
* High latency tolerance through careful scheduling

### Implementation Example

```python
# DeepSpeed ZeRO configuration
from deepspeed import initialize

ds_config = {
    "zero_optimization": {
        "stage": 3,  # ZeRO Stage 3
        "offload_optimizer": {
            "device": "cpu"  # Offload optimizer to CPU
        },
        "offload_param": {
            "device": "cpu"  # Offload parameters to CPU
        }
    }
}

# Initialize model with DeepSpeed
model_engine, optimizer, _, _ = initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)
```

### Real-World Adoption

ZeRO is used in training:
* **Megatron-DeepSpeed**: Combines with model parallelism
* **BLOOM**: 176B parameter multilingual model
* **Stability AI**: Stable Diffusion training
* **Microsoft Research**: Large-scale experiments

### Key Takeaways

* Data parallelism is simple but memory-limited
* ZeRO eliminates memory redundancy while maintaining data parallelism semantics
* Three stages offer different memory/communication trade-offs
* ZeRO Stage 3 enables training models N× larger on N GPUs
* Essential technique for training modern large language models
* Communication overhead is acceptable given memory savings
