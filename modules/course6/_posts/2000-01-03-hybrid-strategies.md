---
title: Hybrid Parallelism Strategies
---

## Module 6.3: Hybrid Parallelism Strategies

### The Need for Multiple Parallelism Dimensions

Modern LLM training requires combining multiple parallelism strategies:

* Single parallelism approach insufficient for extreme-scale models
* Different hardware topologies favor different strategies
* Workload characteristics (batch size, sequence length) influence optimal configuration

### 3D Parallelism

**3D Parallelism** combines all three parallelism dimensions:

1. **Data Parallelism (DP)**: Replicate model across data-parallel workers
2. **Tensor Parallelism (TP)**: Split layers within each data-parallel copy
3. **Pipeline Parallelism (PP)**: Split model stages across pipeline ranks

#### Configuration Example

For 512 GPUs training GPT-3 (175B):

```
Total GPUs: 512
├─ Data Parallel: 4
├─ Tensor Parallel: 8
└─ Pipeline Parallel: 16

Hierarchy:
- 4 independent training groups (data parallel)
- Each group has 128 GPUs
  - Organized as 16 pipeline stages
  - Each stage uses 8 GPUs for tensor parallelism
```

### Parallelism Strategy Decision Matrix

<table>
<thead>
<tr>
<th>Strategy</th>
<th>Key Idea</th>
<th>Memory Efficiency</th>
<th>Compute Efficiency</th>
<th>Communication</th>
<th>Best Use Case</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Data Parallelism</strong></td>
<td>Replicate model, split data</td>
<td>Low (redundant storage)</td>
<td>High (minimal overhead)</td>
<td>AllReduce per step</td>
<td>Models that fit in GPU memory</td>
</tr>
<tr>
<td><strong>ZeRO Stage 1</strong></td>
<td>Partition optimizer state</td>
<td>Medium (4× savings)</td>
<td>High (~3% overhead)</td>
<td>AllGather optimizer state</td>
<td>Large models, fast interconnect</td>
</tr>
<tr>
<td><strong>ZeRO Stage 2</strong></td>
<td>Partition optimizer + gradients</td>
<td>High (8× savings)</td>
<td>Medium (~8% overhead)</td>
<td>ReduceScatter gradients</td>
<td>Very large models</td>
</tr>
<tr>
<td><strong>ZeRO Stage 3</strong></td>
<td>Partition everything</td>
<td>Very High (N× savings)</td>
<td>Medium (~15% overhead)</td>
<td>AllGather per layer</td>
<td>Extremely large models</td>
</tr>
<tr>
<td><strong>Tensor Parallelism</strong></td>
<td>Split layer weights</td>
<td>High (1/N per layer)</td>
<td>Medium (communication cost)</td>
<td>AllReduce per layer</td>
<td>Intra-node, fast NVLink</td>
</tr>
<tr>
<td><strong>Pipeline Parallelism</strong></td>
<td>Split model into stages</td>
<td>High (1/N per stage)</td>
<td>Low (bubble overhead)</td>
<td>P2P between stages</td>
<td>Inter-node, many layers</td>
</tr>
<tr>
<td><strong>3D Parallelism</strong></td>
<td>Combine DP + TP + PP</td>
<td>Very High</td>
<td>Medium (balanced overhead)</td>
<td>Hierarchical</td>
<td>Extreme scale (1000+ GPUs)</td>
</tr>
<tr>
<td><strong>Mixture of Experts (MoE)</strong></td>
<td>Sparse activation, expert routing</td>
<td>High (sparse computation)</td>
<td>Variable (load imbalance)</td>
<td>Expert routing, AlltoAll</td>
<td>Scaling model capacity</td>
</tr>
</tbody>
</table>

### Mixture of Experts (MoE)

**MoE** is a sparse model architecture that dramatically increases model capacity without proportional compute increase.

#### How MoE Works

Instead of dense feed-forward networks:

```python
# Dense FFN: All neurons always active
output = FFN(input)  # Processes all input through all parameters
```

MoE uses sparse routing:

```python
# MoE FFN: Route to subset of experts
expert_weights = Router(input)  # Determine which experts to use
selected_experts = top_k(expert_weights, k=2)  # Select top-2 experts
output = sum(expert_weights[i] * Expert_i(input) 
             for i in selected_experts)
```

#### MoE Components

* **Router**: Learned gating function that decides which experts to activate
* **Experts**: Specialized feed-forward networks (typically 8-128 experts)
* **Top-K Gating**: Only activate K experts per token (typically K=1 or K=2)
* **Load Balancing**: Auxiliary loss ensures experts receive equal amounts of work

#### Capacity Scaling

With MoE, model capacity scales dramatically:

* **Dense Model**: 13B parameters, 100% active
* **MoE Model**: 1.6T parameters, ~5% active per token (80B active parameters)

**Result**: ~100× more parameters with ~10× compute increase

### Expert Parallelism

MoE requires additional parallelism dimension:

```
Each GPU stores: 1/N of total experts
During forward pass: 
  - Router determines expert assignment
  - Tokens routed to appropriate GPUs (AlltoAll)
  - Experts process their tokens
  - Results routed back (AlltoAll)
```

#### Communication Pattern

```python
# Pseudo-code for MoE forward pass
def moe_forward(tokens):
    # Each GPU has different experts
    expert_assignment = router(tokens)  # Local computation
    
    # AlltoAll: Send tokens to expert locations
    tokens_for_my_experts = all_to_all(tokens, expert_assignment)
    
    # Process with local experts
    processed = [expert_i(t) for expert_i, t in tokens_for_my_experts]
    
    # AlltoAll: Return results to original locations
    output = all_to_all(processed, reverse_routing)
    
    return output
```

### Combined 3D + MoE Parallelism

State-of-the-art training uses all parallelism dimensions:

```
Example: Training a 1.7T MoE model on 2048 GPUs

Configuration:
- Data Parallel: 4
- Tensor Parallel: 8
- Pipeline Parallel: 16
- Expert Parallel: 4

Organization:
- 4 data-parallel groups (independent training)
- Each group: 512 GPUs
  - 16 pipeline stages
  - Each stage: 32 GPUs
    - 8 tensor-parallel replicas
    - 4 expert-parallel partitions
```

### Topology-Aware Parallelism

Optimal parallelism depends on hardware topology:

#### Single Node (8x A100 with NVLink)

```
Tensor Parallel: 8 (use fast NVLink)
Pipeline Parallel: 1 (unnecessary within node)
Data Parallel: Use across nodes
```

#### Multi-Node (InfiniBand)

```
Tensor Parallel: 4-8 (within node)
Pipeline Parallel: 4-16 (across nodes)
Data Parallel: Across pipeline groups
```

#### Hierarchical Network (Fat-Tree)

```
Tensor Parallel: Within rack (highest bandwidth)
Pipeline Parallel: Across racks
Data Parallel: Across pods
```

### Automatic Parallelism

Modern tools automatically search for optimal parallelism strategies:

#### Alpa (Automatic Parallelization)

```python
# Alpa automatically determines parallelism
import alpa

@alpa.parallelize
def train_step(batch):
    logits = model(batch)
    loss = criterion(logits, labels)
    return loss

# Alpa finds optimal TP/PP/DP configuration
```

Alpa uses:
* **Intra-operator parallelism**: Tensor parallelism within operators
* **Inter-operator parallelism**: Pipeline parallelism between operators
* **Cost model**: Estimates execution time for different strategies
* **Dynamic programming**: Finds optimal partition

### Performance Optimization

#### Overlapping Communication and Computation

```python
# Pipeline communication with computation
with torch.cuda.Stream(compute_stream):
    # Compute on current layer
    output = layer_forward(input)

with torch.cuda.Stream(comm_stream):
    # Meanwhile, communicate next layer's inputs
    all_gather_async(next_layer_params)
```

#### Gradient Accumulation

Increase effective batch size without memory overhead:

```python
for micro_batch in range(num_micro_batches):
    loss = model(micro_batch)
    loss.backward()  # Accumulate gradients
    
# Synchronize and optimize once
synchronize_gradients()
optimizer.step()
```

### Debugging Hybrid Parallelism

Common issues and solutions:

* **Deadlocks**: Ensure consistent communication ordering across ranks
* **Memory Overflow**: Check activation checkpointing, reduce micro-batch size
* **Load Imbalance**: Monitor per-GPU utilization, adjust expert capacity
* **Gradient Explosion**: Use gradient clipping, check learning rate
* **Communication Bottlenecks**: Profile with Nsight Systems, optimize parallelism config

### Real-World Examples

#### GPT-3 (OpenAI)

* 175B parameters
* Pipeline parallelism: 16 stages
* Tensor parallelism: 8 per stage
* Data parallelism: Across pipeline groups

#### Switch Transformer (Google)

* 1.6T parameters (MoE)
* 2048 experts
* Expert parallelism across 2048 TPUs
* Only ~10B parameters active per token

#### Megatron-Turing NLG (Microsoft/NVIDIA)

* 530B parameters
* 3D parallelism: DP=8, TP=8, PP=8
* 4096 A100 GPUs

### Best Practices

* **Start simple**: Begin with data parallelism, add complexity as needed
* **Profile first**: Use profiling tools to identify bottlenecks before optimizing
* **Match hardware**: Align parallelism strategy with network topology
* **Test configurations**: Grid search over small set of promising configs
* **Monitor actively**: Watch GPU utilization, memory usage, communication time
* **Document thoroughly**: Track which configuration works for which model size

### Key Takeaways

* No single parallelism strategy optimal for all scenarios
* 3D parallelism combines data, tensor, and pipeline parallelism
* MoE enables massive model scaling with sparse activation
* Topology-aware configuration critical for performance
* Modern tools provide automatic parallelization
* Hybrid strategies essential for training frontier models at scale
