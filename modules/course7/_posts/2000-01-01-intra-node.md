---
title: Intra-Node Interconnects (Scale-Up)
---

## Module 7.1: Intra-Node Interconnects (Scale-Up)

### The Intra-Node Challenge

Within a single server, multiple GPUs must communicate efficiently. The interconnect choice dramatically impacts training performance.

### PCIe: The Universal Standard

**PCI Express (PCIe)** is the standard interface for connecting GPUs to CPUs and other peripherals.

#### PCIe Generations

* **PCIe Gen 3**: 16 GB/s per x16 slot (128 Gbps)
* **PCIe Gen 4**: 32 GB/s per x16 slot (256 Gbps)
* **PCIe Gen 5**: 64 GB/s per x16 slot (512 Gbps)
* **PCIe Gen 6**: 128 GB/s per x16 slot (expected 2024-2025)

#### PCIe Topology

Typical server with 8 GPUs:

```
             CPU
              |
        ┌─────┴─────┐
        |           |
    PCIe Switch  PCIe Switch
        |           |
    ────┼────     ──┼────
    |   |   |     |   |   |
   GPU0 GPU1 GPU2 GPU3 GPU4 GPU5
```

**Problem**: GPU-to-GPU communication must traverse PCIe switches and potentially CPU, adding latency and reducing bandwidth.

#### PCIe Limitations for AI

* **Bandwidth**: Insufficient for frequent collective operations
* **Latency**: 5-10 microseconds (too high for tight synchronization)
* **Scalability**: Shared bus creates contention with multiple GPUs
* **CPU Bottleneck**: CPU becomes intermediary for GPU-to-GPU transfers

### NVLink: NVIDIA's High-Speed Interconnect

**NVLink** provides direct GPU-to-GPU communication bypassing PCIe and CPU.

#### NVLink Generations

* **NVLink 1.0** (P100): 160 GB/s bidirectional
* **NVLink 2.0** (V100): 300 GB/s bidirectional
* **NVLink 3.0** (A100): 600 GB/s bidirectional
* **NVLink 4.0** (H100): 900 GB/s bidirectional

#### NVLink Topology

Each GPU has multiple NVLink connections:

**8x A100 with NVLink:**
```
GPU0 ─────┬───── GPU1
    \     │     /
     \    │    /
      \   │   /
       \  │  /
        \ │ /
         \│/
        GPU2
         ...
```

* Each A100 has 12 NVLink connections
* Can create various topologies: mesh, torus, full connectivity
* Direct peer-to-peer without CPU involvement

#### NVLink Benefits

* **10-20× faster** than PCIe for GPU-GPU transfers
* **Lower latency**: ~1-2 microseconds
* **Higher bandwidth**: 600 GB/s per GPU (A100)
* **CPU offload**: Frees CPU and PCIe for other tasks

### NVSwitch: Full GPU Connectivity

**NVSwitch** is a physical switch enabling all-to-all GPU connectivity.

#### DGX A100 Architecture

NVIDIA DGX A100 uses 6 NVSwitches:

```
        NVSwitch Fabric (6 switches)
        ┌────────────────┐
        │  All-to-All    │
        │  Connectivity  │
        └────────────────┘
         ╱│╲    ╱│╲    ╱│╲
       GPU0 GPU1 GPU2 ... GPU7
```

* **Full bandwidth** between any pair of GPUs
* **No bottlenecks**: Every GPU can communicate with every other simultaneously
* **600 GB/s** per GPU bidirectional to NVSwitch fabric
* **4.8 TB/s** aggregate bisection bandwidth (8 GPUs)

#### DGX H100 Improvements

* 4th generation NVSwitch
* 18 NVLink 4.0 connections per H100 GPU
* 900 GB/s per GPU to NVSwitch fabric
* **7.2 TB/s** aggregate bisection bandwidth

### Comparison Table

| Technology | Bandwidth (bidirectional) | Latency | Scalability | Use Case |
|-----------|---------------------------|---------|-------------|----------|
| PCIe Gen 3 | 16 GB/s | 5-10 μs | Limited | Legacy systems |
| PCIe Gen 4 | 32 GB/s | 3-5 μs | Limited | Entry-level AI |
| PCIe Gen 5 | 64 GB/s | 2-4 μs | Moderate | Future standard |
| NVLink 3.0 | 600 GB/s | 1-2 μs | 8 GPUs | A100 systems |
| NVLink 4.0 | 900 GB/s | <1 μs | 8 GPUs | H100 systems |
| NVSwitch | 4.8-7.2 TB/s aggregate | <1 μs | Excellent | DGX systems |

### AMD Infinity Fabric

AMD's equivalent to NVLink for MI series accelerators:

* **Infinity Fabric Link**: Up to 200 GB/s per GPU (MI250X)
* **Supports 8 GPUs** with direct connectivity
* Used in Frontier supercomputer (world's first exascale system)

### Intel Xe Link

Intel's upcoming high-speed GPU interconnect:

* **Ponte Vecchio** architecture
* Multiple tiles connected within package
* Details emerging with Intel GPU offerings

### Scale-Up Architecture Benefits

High-speed intra-node interconnects enable:

#### Efficient Tensor Parallelism

```python
# AllReduce within node is fast with NVLink
# Can use fine-grained tensor parallelism
for layer in model.layers:
    output = layer(input)
    output = all_reduce(output)  # Fast over NVLink
```

#### Reduced Communication Overhead

* 10-20× faster AllReduce within node
* Enables larger tensor parallel groups
* Lower latency for gradient synchronization

#### Higher Effective Throughput

* Less time waiting for communication
* Higher GPU utilization (85-95% vs. 60-70% with PCIe)
* Better scaling efficiency

### Real-World Impact

#### Training Performance

On 8x A100 with NVLink vs PCIe:

* **GPT-2 (1.5B)**: 3.5× faster training
* **GPT-3 (13B)**: 6× faster with tensor parallelism
* **BLOOM (176B)**: Essential for tensor parallelism (wouldn't be practical with PCIe)

#### Memory Pooling

NVLink enables treating multiple GPUs as single memory space:

```python
# With NVLink, can access peer GPU memory directly
# Enables models larger than single GPU memory
torch.cuda.set_device(0)
tensor_on_gpu1 = torch.zeros(size).cuda(1)  # Allocated on GPU 1
result = compute(tensor_on_gpu1)  # Accessed from GPU 0 via NVLink
```

### Selection Criteria

#### When PCIe is Sufficient

* Inference workloads with minimal communication
* Data parallelism with large batch sizes
* Models that fit on single GPU
* Budget-constrained deployments

#### When NVLink is Essential

* Training with tensor parallelism
* Models requiring frequent AllReduce (e.g., Transformers)
* Multi-GPU inference with small batches
* Research and development environments

#### When NVSwitch is Worth It

* Maximum performance requirements
* Largest models with complex parallelism
* Production training clusters
* Multi-user shared infrastructure

### Future Trends

* **Higher Bandwidth**: Each generation doubles bandwidth
* **Lower Latency**: Sub-microsecond becoming standard
* **Optical Interconnects**: Moving to photonics for even higher speeds
* **Coherent Memory**: True shared memory across GPUs
* **Universal Standards**: Industry convergence on high-speed protocols

### Key Takeaways

* Intra-node interconnect is critical for multi-GPU performance
* NVLink provides 10-20× better bandwidth than PCIe
* NVSwitch enables full bisection bandwidth for 8 GPUs
* Fast interconnects enable tensor parallelism and reduce communication overhead
* Choice of interconnect impacts model architecture and parallelism strategy
* Modern LLM training requires NVLink-class interconnects for efficiency
