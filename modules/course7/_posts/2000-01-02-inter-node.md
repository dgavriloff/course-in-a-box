---
title: Inter-Node Interconnects (Scale-Out)
---

## Module 7.2: Inter-Node Interconnects (Scale-Out)

### The Scale-Out Challenge

Training frontier LLMs requires thousands of GPUs across hundreds of servers. Inter-node networking becomes the critical bottleneck.

### Ethernet: The Traditional Choice

**Ethernet** has been the standard datacenter network for decades.

#### Ethernet Speeds

* **10 GbE**: 10 Gbps (1.25 GB/s) - legacy
* **25 GbE**: 25 Gbps (3.125 GB/s) - common
* **100 GbE**: 100 Gbps (12.5 GB/s) - modern standard
* **400 GbE**: 400 Gbps (50 GB/s) - emerging
* **800 GbE**: 800 Gbps (100 GB/s) - future

#### Ethernet Limitations for AI

* **High Latency**: 5-50 microseconds (TCP/IP overhead)
* **Software Overhead**: Kernel network stack processing
* **Congestion**: TCP congestion control adds unpredictability
* **Small Message Performance**: Poor for frequent small collectives

### InfiniBand: HPC Standard

**InfiniBand** is the de facto standard for high-performance computing and increasingly for AI.

#### InfiniBand Speeds

* **FDR**: 56 Gbps (7 GB/s)
* **EDR**: 100 Gbps (12.5 GB/s)
* **HDR**: 200 Gbps (25 GB/s)
* **NDR**: 400 Gbps (50 GB/s)
* **XDR**: 800 Gbps (100 GB/s) - upcoming

#### InfiniBand Advantages

* **Low Latency**: ~1-2 microseconds
* **RDMA Support**: Remote Direct Memory Access
* **Hardware Offload**: Minimal CPU involvement
* **Reliable Transport**: Hardware-level reliability
* **Quality of Service**: Traffic prioritization

### RDMA: Remote Direct Memory Access

**RDMA** is the key technology enabling high-performance networking.

#### How RDMA Works

Traditional networking:

```
GPU Memory → Host Memory → Kernel → NIC → Network
           (PCIe)       (copy)
```

With RDMA:

```
GPU Memory → NIC → Network
         (direct, zero-copy)
```

#### RDMA Benefits

* **Zero-Copy**: Data transferred directly without CPU copying
* **Kernel Bypass**: No operating system overhead
* **CPU Offload**: Network processing in hardware
* **Low Latency**: Microsecond-level transfers
* **High Bandwidth**: Near wire-speed throughput

### GPUDirect RDMA

**GPUDirect RDMA** enables direct GPU-to-GPU transfers across network:

```
GPU 0 (Node 1) → InfiniBand NIC → Network → InfiniBand NIC → GPU 0 (Node 2)
              (direct path, no CPU)
```

#### Without GPUDirect RDMA

```
GPU → CPU Memory → NIC → Network → NIC → CPU Memory → GPU
    (PCIe)      (DMA)                  (DMA)      (PCIe)
```

**4 memory copies**, high latency, CPU bottleneck

#### With GPUDirect RDMA

```
GPU → NIC → Network → NIC → GPU
```

**Zero copies**, low latency, no CPU involvement

#### Performance Impact

* **2-3× lower latency** for GPU-to-GPU across nodes
* **40-60% higher bandwidth** utilization
* **CPU freed** for other tasks
* **Essential** for efficient distributed training

### RoCE: RDMA over Converged Ethernet

**RoCE** (RDMA over Converged Ethernet) brings RDMA capabilities to Ethernet.

#### RoCE Versions

* **RoCE v1**: Non-routable, same L2 network
* **RoCE v2**: Routable, supports L3 networking (most common)

#### RoCE Advantages

* **Ethernet Infrastructure**: Use existing switches and cables
* **RDMA Benefits**: Low latency, zero-copy
* **Cost**: Lower than InfiniBand
* **Familiarity**: Existing Ethernet expertise

#### RoCE Challenges

* **Lossless Ethernet Required**: Priority Flow Control (PFC), ECN
* **Configuration Complexity**: More tuning than InfiniBand
* **Congestion Management**: Careful DCQCN/ECN tuning needed
* **Switch Requirements**: Not all Ethernet switches support lossless mode

### NVIDIA Quantum InfiniBand

**NVIDIA Quantum** series for AI-optimized InfiniBand:

* **Quantum-2**: 400 Gbps (NDR) switches
  * 64-port switches
  * In-network computing capabilities
  * SHARP (Scalable Hierarchical Aggregation Protocol)
* **Quantum-3**: 800 Gbps (XDR) - upcoming
  * Higher port density
  * Enhanced in-network computing

#### SHARP: In-Network Computing

**SHARP** performs collective operations in the network switches:

```
Traditional AllReduce: All data goes through network multiple times

With SHARP: Switch aggregates data inline
  GPU1 →  \
  GPU2 →   Switch (aggregates) → Result to all GPUs
  GPU3 →  /
```

**Benefits**:
* 3-5× faster AllReduce
* Reduced network congestion
* Lower CPU/GPU overhead

### Network Bandwidth Requirements

For distributed training, estimate bandwidth needs:

```
Bandwidth = (Model_Size × 2) / (Time_Per_Step × Num_GPUs_Per_Node)
```

Example for GPT-3 (175B parameters):
* Model size: 175B × 4 bytes = 700 GB
* 100 ms per step target
* 8 GPUs per node

```
Bandwidth = (700 GB × 2) / (0.1s × 8) ≈ 1.75 TB/s per node

Per NIC: 1.75 TB/s / 8 = 219 GB/s = 1752 Gbps
```

**Reality**: Need multiple 200-400 Gbps NICs per node

### Network Interface Card (NIC) Architecture

Modern AI servers use multiple NICs:

#### ConnectX-7 (NVIDIA/Mellanox)

* **400 Gbps InfiniBand (NDR)**
* **GPUDirect RDMA** support
* **Hardware acceleration** for collectives
* **Adaptive routing**
* **Telemetry** for monitoring

#### Typical Configuration

* **8x A100 server**: 4-8 NICs (1.6-3.2 Tbps total)
* **8x H100 server**: 8 NICs (3.2 Tbps total)
* Each GPU connected to multiple NICs (multi-rail)

### Multi-Rail Networking

**Multi-rail** uses multiple independent network paths:

```
GPU0 → NIC0 → Network 0
GPU0 → NIC1 → Network 1
GPU0 → NIC2 → Network 2
```

**Benefits**:
* Aggregate bandwidth from multiple NICs
* Fault tolerance (failover)
* Load balancing across rails
* Reduced per-NIC congestion

### Latency Comparison

| Technology | Latency | Bandwidth | Use Case |
|-----------|---------|-----------|----------|
| Ethernet (10G) | 50-100 μs | 1.25 GB/s | Legacy |
| Ethernet (100G) | 10-30 μs | 12.5 GB/s | General purpose |
| RoCE v2 (100G) | 5-15 μs | 12.5 GB/s | Cost-effective AI |
| InfiniBand EDR | 1-2 μs | 12.5 GB/s | HPC |
| InfiniBand HDR | 1-2 μs | 25 GB/s | AI training |
| InfiniBand NDR | 0.6-1 μs | 50 GB/s | Frontier AI |

### Real-World Deployments

#### Meta AI Research SuperCluster (RSC)

* 16,000 A100 GPUs
* InfiniBand HDR network
* 3-rail connectivity per server
* Optimized for LLM training

#### Microsoft Azure ND A100 v4

* InfiniBand HDR (200 Gbps × 8 = 1.6 Tbps per node)
* GPUDirect RDMA
* Quantum InfiniBand switches
* Used for training Turing-Megatron, DeepSpeed models

#### Google TPU v4 Pods

* Custom high-speed interconnect
* 3D torus topology
* 10+ Tbps per chip interconnect bandwidth
* Proprietary but demonstrates importance of networking

### Choosing the Right Interconnect

#### InfiniBand HDR/NDR

**Best for**:
* Maximum performance
* Large-scale training (1000+ GPUs)
* Frequent collectives
* Budget allows

#### RoCE

**Best for**:
* Existing Ethernet infrastructure
* Medium-scale training (100-1000 GPUs)
* Cost-sensitive deployments
* Expertise available for tuning

#### High-Speed Ethernet (200/400G)

**Best for**:
* Future-proofing
* Mixed workloads (AI + other)
* Simplified operations
* Campus-wide deployments

### Future Directions

* **800 Gbps and Beyond**: XDR InfiniBand, 800GbE
* **Co-Packaged Optics**: Optical transceivers integrated with switch chips
* **Ultra-Low Latency**: Sub-microsecond becoming standard
* **In-Network Computing**: More SHARP-like capabilities
* **AI-Specific Protocols**: Optimizations for collective operations

### Key Takeaways

* Inter-node networking is the bottleneck for scale-out training
* InfiniBand with RDMA provides lowest latency and highest bandwidth
* GPUDirect RDMA essential for efficient GPU-to-GPU across nodes
* Multiple NICs per server (multi-rail) aggregate bandwidth
* RoCE offers RDMA over Ethernet as cost-effective alternative
* Network bandwidth requirements scale with model size and training speed
* In-network computing (SHARP) accelerates collective operations
