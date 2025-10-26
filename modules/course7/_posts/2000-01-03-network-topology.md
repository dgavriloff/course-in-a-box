---
title: Network Topology for AI Clusters
---

## Module 7.3: Network Topology for AI Clusters

### Why Topology Matters

Network topology determines:
* **Bandwidth availability** between any pair of nodes
* **Latency** for communication patterns
* **Fault tolerance** and redundancy
* **Cost** of the overall system
* **Scalability** limits

For AI training with frequent collective operations, topology is critical.

### Traditional Topologies

#### Star Topology

```
        Switch
       /  |  \
      /   |   \
   Node1 Node2 Node3
```

**Characteristics**:
* Simple, low cost
* Single point of failure
* Limited scalability (switch port count)
* Good for small clusters (<48 nodes)

#### Tree Topology

```
         Root
        /    \
     SW1      SW2
    /  \     /  \
   N1  N2   N3  N4
```

**Problems for AI**:
* Oversubscription at higher levels
* Root switch bottleneck
* Poor for all-to-all communication

### Fat-Tree Topology

**Fat-tree** is the dominant topology for large-scale AI clusters.

#### Basic Fat-Tree Structure

```
         Spine Switches
        /  |    |   \
       /   |    |    \
    Leaf Switches
    /  |       |    \
Servers (with GPUs)
```

#### Key Properties

* **Full Bisection Bandwidth**: Any half of servers can communicate with other half at full speed
* **Multiple Paths**: Redundancy and load balancing
* **Uniform Latency**: Consistent hop count between any pair
* **Scalability**: Can scale to tens of thousands of nodes

#### Fat-Tree Variants

**2-Tier Fat-Tree** (Leaf-Spine):
* Leaf switches connect to servers
* Spine switches connect all leaf switches
* Common for 500-2000 node clusters

**3-Tier Fat-Tree**:
* Access (Leaf) → Aggregation → Core (Spine)
* Scales to 10,000+ nodes
* Used in mega-scale datacenters

### Clos Network

**Clos** is a generalization of fat-tree, widely used in modern datacenters:

```
       Core Switches (Tier 3)
            |    |
      Aggregation (Tier 2)
         /    |    \
    Leaf Switches (Tier 1)
    /    |    |    \
  Servers with GPUs
```

#### Configuration Parameters

* **k**: Number of ports per switch
* **m**: Number of parallel paths
* **n**: Number of nodes per leaf

**Scaling**: Can build network with $$k^3/4$$ nodes using k-port switches

### Network Oversubscription

**Oversubscription ratio**: Ratio of potential demand to available bandwidth.

```
Oversubscription = (Total Downlink Bandwidth) / (Total Uplink Bandwidth)
```

#### Oversubscription Examples

**1:1 (Non-oversubscribed)**:
* Leaf: 32 × 100G down, 32 × 100G up
* Full bisection bandwidth
* Ideal for AI training
* Most expensive

**2:1 Oversubscribed**:
* Leaf: 32 × 100G down, 16 × 100G up
* Half bisection bandwidth
* Acceptable for some AI workloads
* 50% cost reduction

**4:1 or Higher**:
* Leaf: 32 × 100G down, 8 × 100G up
* Significant congestion possible
* Not recommended for distributed training
* Common in general-purpose datacenters

### Rail-Optimized Topologies

Modern AI clusters use **multi-rail** designs for maximum bandwidth:

#### Design Principle

Each GPU connects to separate network rail:

```
Node 1:
  GPU0 → NIC0 → Network Rail 0
  GPU1 → NIC1 → Network Rail 1
  GPU2 → NIC2 → Network Rail 2
  ...

Node 2:
  GPU0 → NIC0 → Network Rail 0
  GPU1 → NIC1 → Network Rail 1
  GPU2 → NIC2 → Network Rail 2
  ...
```

#### Benefits

* **Aggregate Bandwidth**: 8 rails × 400 Gbps = 3.2 Tbps per node
* **Fault Isolation**: Failure in one rail doesn't affect others
* **Load Balancing**: Distribute traffic across rails
* **Reduced Congestion**: Separate planes for different communication patterns

### Dragonfly Topology

**Dragonfly** is an alternative high-radix topology:

```
Group 0          Group 1          Group 2
  ╱╲╲             ╱╲╲             ╱╲╲
 ╱  ╲╲           ╱  ╲╲           ╱  ╲╲
├────┤───────────├────┤───────────├────┤
Routers          Routers          Routers
  │                │                │
Nodes            Nodes            Nodes
```

#### Characteristics

* **High-radix switches** connect many nodes
* **Groups** connected by inter-group links
* **Lower cost** than fat-tree for same bisection bandwidth
* **Non-uniform latency** (intra-group vs inter-group)

#### AI Suitability

* **Good for**: Sparse communication patterns
* **Challenges**: Non-uniform latency complicates optimization
* **Used in**: Some HPC systems (Cray Slingshot)

### Interconnect Comparison Table

<table>
<thead>
<tr>
<th>Interconnect</th>
<th>Domain</th>
<th>Max Bandwidth</th>
<th>Latency</th>
<th>Topology Support</th>
<th>AI Training Suitability</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>PCIe 4.0</strong></td>
<td>Intra-node</td>
<td>32 GB/s</td>
<td>3-5 μs</td>
<td>Bus</td>
<td>Moderate (limited GPU-GPU)</td>
</tr>
<tr>
<td><strong>NVLink 3.0</strong></td>
<td>Intra-node</td>
<td>600 GB/s</td>
<td>1-2 μs</td>
<td>Custom mesh</td>
<td>Excellent (tensor parallel)</td>
</tr>
<tr>
<td><strong>NVLink 4.0</strong></td>
<td>Intra-node</td>
<td>900 GB/s</td>
<td><1 μs</td>
<td>Custom mesh</td>
<td>Excellent (H100)</td>
</tr>
<tr>
<td><strong>NVSwitch</strong></td>
<td>Intra-node</td>
<td>7.2 TB/s aggregate</td>
<td><1 μs</td>
<td>All-to-all</td>
<td>Excellent (DGX)</td>
</tr>
<tr>
<td><strong>InfiniBand HDR</strong></td>
<td>Inter-node</td>
<td>25 GB/s</td>
<td>1-2 μs</td>
<td>Fat-tree, dragonfly</td>
<td>Excellent</td>
</tr>
<tr>
<td><strong>InfiniBand NDR</strong></td>
<td>Inter-node</td>
<td>50 GB/s</td>
<td>0.6-1 μs</td>
<td>Fat-tree, dragonfly</td>
<td>Excellent (frontier)</td>
</tr>
<tr>
<td><strong>RoCE v2 (200G)</strong></td>
<td>Inter-node</td>
<td>25 GB/s</td>
<td>5-15 μs</td>
<td>Fat-tree, leaf-spine</td>
<td>Good (with tuning)</td>
</tr>
<tr>
<td><strong>100G Ethernet</strong></td>
<td>Inter-node</td>
<td>12.5 GB/s</td>
<td>10-30 μs</td>
<td>Fat-tree, leaf-spine</td>
<td>Moderate (inference)</td>
</tr>
</tbody>
</table>

### Hardware-Software Co-Design

Optimal performance requires co-designing network topology with parallelism strategy:

#### Intra-Node: Tensor Parallelism

```
8 GPUs in node with NVLink
→ Use 8-way tensor parallelism
→ AllReduce within node is fast (600 GB/s × 8)
```

#### Inter-Node: Pipeline Parallelism

```
Nodes connected via InfiniBand
→ Use pipeline parallelism across nodes
→ Minimize inter-node communication
→ Point-to-point between pipeline stages
```

#### Multi-Tier Hierarchy

```
Level 1: GPUs within node (NVLink) → Tensor Parallel
Level 2: Nodes within rack (InfiniBand) → Pipeline Parallel
Level 3: Racks in cluster → Data Parallel
```

### Network Telemetry and Monitoring

Critical for maintaining AI cluster health:

#### Metrics to Monitor

* **Bandwidth Utilization**: Per-link throughput
* **Packet Loss**: Should be near-zero
* **Latency**: P50, P95, P99 latency
* **Queue Depths**: Congestion indicators
* **Errors**: CRC errors, link flaps

#### Tools

* **DCGM** (Data Center GPU Manager): GPU and NIC metrics
* **InfiniBand Subnet Manager**: Network health
* **What About (WA)**: NVIDIA's network monitoring
* **Prometheus + Grafana**: Time-series visualization

### Failure Handling

Network failures are inevitable at scale:

#### Redundancy Strategies

* **Multi-path routing**: Automatic failover
* **Link redundancy**: Multiple NICs per server
* **Switch redundancy**: Dual spine switches
* **Checkpoint resilience**: Tolerate temporary network issues

#### Graceful Degradation

```python
# Training can continue with reduced network capacity
if network_bandwidth < threshold:
    reduce_tensor_parallel_degree()
    increase_pipeline_stages()
    reduce_batch_size()
```

### Cost Optimization

Network often represents 20-40% of total cluster cost:

#### Trade-offs

* **Oversubscription**: 2:1 saves 50% of cost but reduces performance by 20-30%
* **Link Speed**: 200G vs 400G has different cost/performance
* **Switch Tier**: Merchant silicon vs custom (InfiniBand)

#### Cost-Effective Designs

* **Leaf-spine** for <2000 nodes (2-tier sufficient)
* **Rail optimization** for specific communication patterns
* **Hybrid topologies** for different traffic classes

### Future Trends

* **800G and Beyond**: Optical interconnects
* **In-Network Computing**: SmartNICs, switch computing
* **CXL** (Compute Express Link): Cache-coherent device interconnection
* **Photonics**: Silicon photonics for chip-to-chip
* **Quantum Networking**: Long-term research

### Key Takeaways

* Fat-tree/Clos topology provides full bisection bandwidth
* Non-oversubscribed networks essential for large-scale training
* Multi-rail design aggregates bandwidth and provides fault tolerance
* Topology should match parallelism strategy (tensor/pipeline/data)
* Network represents significant portion of cluster cost
* Hardware-software co-design critical for optimal performance
* Monitoring and telemetry essential for maintaining cluster health
