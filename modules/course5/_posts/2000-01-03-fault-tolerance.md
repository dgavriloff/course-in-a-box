---
title: Fault Tolerance in Large-Scale Training
---

## Module 5.3: Fault Tolerance in Large-Scale Training

### The Scale Challenge

Training large language models requires thousands of GPUs running for weeks or months:

* **GPT-3**: 10,000+ GPUs, several weeks
* **PaLM**: 6,000+ TPUs, months
* **LLaMA-2 70B**: 2,000+ GPUs, weeks

At this scale, hardware failures are inevitable, not exceptional.

### Probability of Failure

Using Mean Time Between Failures (MTBF) statistics:

```
P(failure) = 1 - (1 - 1/MTBF)^N
```

Example calculation:
* Single GPU MTBF: 6 years (52,560 hours)
* Training time: 1 month (720 hours)
* Cluster size: 10,000 GPUs

```
P(at least one failure) ≈ 1 - (1 - 720/52560)^10000 ≈ 99.9%
```

**Reality**: Failures are guaranteed in large-scale training.

### Types of Failures

#### Hardware Failures

* **GPU failures**: Memory errors, compute errors, complete failure
* **Network failures**: Switch failures, cable disconnections, congestion
* **Host failures**: CPU errors, power supply issues, cooling problems
* **Storage failures**: Disk crashes, filesystem corruption

#### Software Failures

* **Out-of-Memory (OOM)**: Gradient accumulation, activation checkpointing issues
* **Numerical instability**: NaN/Inf in forward or backward pass
* **Deadlocks**: Synchronization bugs in distributed code
* **Framework bugs**: Issues in PyTorch, TensorFlow, or custom kernels

### Checkpoint-Based Recovery

The primary fault tolerance mechanism is **checkpointing**: periodically saving training state to persistent storage.

#### What to Checkpoint

A complete checkpoint includes:

* **Model parameters**: All weights and biases
* **Optimizer state**: Momentum, variance statistics (for Adam, etc.)
* **Training metadata**: Step number, epoch, random seeds
* **Learning rate schedule**: Current LR and schedule state
* **Data loader state**: Position in dataset to avoid repeating/skipping data

#### Checkpoint Frequency Trade-offs

```
Cost_per_checkpoint = Time_to_save + Storage_space
Cost_of_failure = Average_time_since_last_checkpoint

Total_cost = Cost_per_checkpoint × Checkpoints_per_training + 
             Cost_of_failure × Expected_failures
```

**Common practice**: Checkpoint every 1-4 hours of training

### Checkpointing Challenges

At scale, checkpointing faces several challenges:

* **Size**: Modern models have hundreds of billions of parameters
  * GPT-3 (175B): ~350 GB for parameters alone
  * With optimizer state (Adam): ~1.4 TB per checkpoint
  * Storage requirements: 10-50 TB for full training run

* **Time**: Writing large checkpoints is slow
  * 1 TB checkpoint at 10 GB/s: ~100 seconds
  * During this time, GPUs are idle (wasted compute)
  * Can lose 1-5% of training time to checkpointing

* **I/O Bottlenecks**: Shared filesystem contention
  * Many GPUs writing simultaneously
  * Filesystem performance degrades
  * May require dedicated checkpoint infrastructure

* **Coordination**: All ranks must synchronize
  * Need barrier to ensure consistent state
  * Stragglers delay entire checkpoint operation
  * Increases critical path latency

### Optimization Strategies

#### Asynchronous Checkpointing

Move checkpoint writing off critical path:

```python
# Copy state to CPU memory (fast)
checkpoint_data = {
    'model': model.state_dict().cpu(),
    'optimizer': optimizer.state_dict().cpu(),
}

# Write to disk asynchronously in background thread
checkpoint_thread = Thread(target=save_checkpoint, args=(checkpoint_data,))
checkpoint_thread.start()

# Continue training immediately
```

**Benefit**: Reduces GPU idle time by 50-90%

**Trade-off**: May lose progress if failure occurs before write completes

#### Incremental Checkpointing

Only save changes since last checkpoint:

```python
# Track which parameters changed
changed_params = track_parameter_changes(model, last_checkpoint)

# Only save changed parameters
save_incremental_checkpoint(changed_params, step)
```

**Benefit**: Faster saves, less storage

**Challenge**: Need to reconstruct full state during recovery

#### Hierarchical Checkpointing

Different checkpoint frequencies at different levels:

* **L1 (Frequent)**: Fast local SSD checkpoints every 10-30 minutes
* **L2 (Regular)**: Network filesystem checkpoints every 1-2 hours  
* **L3 (Infrequent)**: Long-term storage (S3, tape) every 6-12 hours

**Benefit**: Balance fast recovery with long-term durability

#### Distributed Checkpointing

Each rank saves only its portion of the model:

```python
# With model parallelism, each GPU has different parameters
if rank == 0:
    save_checkpoint('checkpoint_rank0.pt', my_model_slice)
elif rank == 1:
    save_checkpoint('checkpoint_rank1.pt', my_model_slice)
```

**Benefit**: Parallel I/O, faster overall checkpoint time

**Challenge**: Must carefully manage recovery when failure occurs

### Elastic Training

Modern frameworks support **elastic training**: dynamically adjusting to resource availability.

#### Key Features

* **Dynamic membership**: GPUs can join/leave without stopping training
* **Automatic re-sharding**: Redistribute work across available GPUs
* **Health monitoring**: Detect and exclude faulty GPUs automatically

#### Implementation Example (PyTorch Elastic)

```python
import torch.distributed.elastic as elastic

# Training loop automatically handles failures
with elastic.agent():
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        # Framework detects failures and re-initializes automatically
```

### Redundancy and Replication

#### Gradient Checksum Validation

Detect silent data corruption:

```python
# Compute checksum of gradients
gradient_checksum = torch.sum(gradient)

# All-reduce checksums across ranks
dist.all_reduce(gradient_checksum)

# Verify all ranks have same checksum
if gradient_checksum != expected_checksum:
    raise ValueError("Gradient corruption detected!")
```

#### Shadow Training

Run redundant computation on subset of GPUs:

* Primary training uses N GPUs
* Shadow training uses M < N GPUs
* Compare results periodically to detect hardware issues

**Trade-off**: Uses extra resources but provides early failure detection

### Failure Recovery Workflow

1. **Detect Failure**: Monitor system health, GPU status, job status
2. **Initiate Recovery**: Identify failed components, reallocate resources
3. **Load Checkpoint**: Restore model, optimizer, and training state
4. **Validate State**: Verify checkpoint integrity, check for corruption
5. **Resume Training**: Restart from checkpoint step with new resources

### Monitoring and Alerting

Essential monitoring for large-scale training:

* **GPU health**: Temperature, memory errors, compute errors
* **Network health**: Bandwidth utilization, packet loss, latency
* **Training metrics**: Loss curves, gradient norms, throughput
* **Resource usage**: Memory consumption, disk I/O, CPU usage

### Best Practices

* **Checkpoint frequently enough** to minimize lost work but not so frequently that I/O dominates
* **Test recovery regularly** to ensure checkpoints are valid and recovery works
* **Monitor aggressively** to detect issues before they cause failures
* **Use multiple checkpoint layers** (local + remote) for resilience
* **Implement asynchronous checkpointing** to minimize training overhead
* **Version checkpoints** to enable rollback if checkpoint corruption occurs
* **Log everything** to help debug failures post-mortem

### Real-World Examples

#### Meta's OPT-175B

* Training interrupted dozens of times due to hardware failures
* Automatic recovery from checkpoints enabled continuous progress
* Checkpointing overhead: ~2% of total training time

#### Google's PaLM

* Multi-week training runs on 6,000+ TPUs
* Robust checkpointing and recovery essential for success
* Hierarchical storage strategy for checkpoint management

### Key Takeaways

* Failures are inevitable at scale—plan for them from the start
* Checkpointing is the foundation of fault tolerance
* Optimization strategies (async, incremental, hierarchical) reduce overhead
* Elastic training enables graceful degradation with resource changes
* Monitoring and automated recovery are essential for production training
* The cost of fault tolerance (checkpointing) is far less than restarting from scratch
