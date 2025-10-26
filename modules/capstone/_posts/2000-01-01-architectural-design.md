---
title: Architectural Design for a Frontier Model
---

## Capstone Project: Architectural Design for a Frontier Model

### Project Overview

In this capstone project, you will design a complete system architecture for training a frontier-scale large language model. This project synthesizes knowledge from all four pillars of the curriculum and requires you to make informed trade-offs across multiple dimensions.

### Problem Statement

Your organization plans to train a 500B parameter dense Transformer model (comparable to PaLM or MT-NLG). The model will be trained on 2 trillion tokens using a cluster of 4,096 NVIDIA H100 GPUs distributed across 512 servers (8 GPUs per server).

**Your task**: Design the complete training infrastructure, including parallelism strategy, network architecture, datacenter requirements, and operational considerations.

### Constraints and Requirements

#### Hardware Budget

* **Compute**: 512 servers with 8× H100 GPUs each (fixed)
* **Network**: Budget for high-speed interconnects
* **Storage**: Parallel filesystem and object storage
* **Datacenter**: Power, cooling, and space considerations

#### Performance Targets

* **Time to train**: Complete training in 60 days
* **GPU utilization**: Target >85% MFU (Model FLOPs Utilization)
* **Checkpoint frequency**: Every 2 hours maximum data loss
* **Fault tolerance**: Resume from failure within 15 minutes

#### Operational Requirements

* **Monitoring**: Real-time metrics and alerting
* **Scalability**: Ability to expand to 1T parameters in future
* **Sustainability**: Minimize carbon footprint
* **Cost efficiency**: Optimize TCO over 3 years

### Required Deliverables

Your submission must include detailed specifications for each component:

* **Parallelism Strategy**
  * Data parallelism degree
  * Tensor parallelism degree
  * Pipeline parallelism stages
  * Micro-batch configuration
  * Activation checkpointing strategy
  * Justification for configuration choices
  * Expected training throughput (tokens/sec)
  * Memory usage per GPU calculation

* **Kernel Requirements**
  * List of critical kernels requiring optimization
  * FlashAttention configuration (tile sizes, data types)
  * Fused operations (LayerNorm+Activation, etc.)
  * Mixed precision strategy (FP16, BF16, FP8)
  * Expected speedup from kernel optimizations
  * Memory bandwidth utilization targets

* **Network Architecture**
  * Intra-node interconnect (NVLink topology)
  * Inter-node interconnect (InfiniBand/RoCE specs)
  * Network topology (fat-tree, rail configuration)
  * Bandwidth requirements per link
  * Expected communication overhead (% of total time)
  * Redundancy and failover strategy

* **Datacenter Requirements**
  * Power consumption calculation (per rack, total)
  * Cooling solution (air, liquid, hybrid)
  * PUE target and justification
  * Physical space requirements (sq ft, rack count)
  * Environmental considerations (location, climate)
  * Estimated energy cost over training period

* **Storage Architecture**
  * Training dataset size and format
  * Parallel filesystem specifications (Lustre/BeeGFS)
  * Local NVMe caching strategy
  * Checkpoint storage plan (size, frequency, retention)
  * Data loading pipeline design
  * Estimated I/O bandwidth requirements

* **Fault Tolerance Strategy**
  * Checkpoint frequency and size
  * Incremental vs full checkpoints
  * Hierarchical checkpoint storage
  * Expected time between failures (MTBF)
  * Recovery time objective (RTO)
  * Automated failure detection and recovery

* **Monitoring and Observability**
  * GPU metrics to track (utilization, temperature, memory)
  * Network metrics (bandwidth, latency, packet loss)
  * Training metrics (loss, throughput, gradient norms)
  * Alerting thresholds and escalation
  * Logging and telemetry infrastructure
  * Dashboard design for operators

* **Budget and Cost Analysis**
  * Hardware costs (GPUs, servers, networking)
  * Infrastructure costs (power, cooling, space)
  * Operational costs (electricity, personnel, maintenance)
  * Storage costs (filesystem, object storage, backup)
  * Total Cost of Ownership (TCO) over 3 years
  * Cost per training run
  * Cost per parameter

* **Risk Assessment and Mitigation**
  * Technical risks (hardware failure, software bugs)
  * Operational risks (power outage, cooling failure)
  * Timeline risks (delays, optimization challenges)
  * Mitigation strategies for each risk
  * Contingency plans

* **Sustainability Plan**
  * Carbon footprint calculation
  * Renewable energy strategy
  * Heat reuse opportunities
  * E-waste and recycling plan
  * Comparison with alternatives

### Evaluation Criteria

Your design will be evaluated on:

#### Technical Soundness (40%)

* **Correctness**: Calculations are accurate and realistic
* **Completeness**: All components specified in detail
* **Feasibility**: Design can actually be built and operated
* **Integration**: Components work together coherently
* **Scalability**: Architecture can grow to future requirements

#### Performance Optimization (25%)

* **Parallelism**: Optimal configuration for hardware
* **Communication**: Minimal overhead, good topology mapping
* **Memory**: Efficient use of GPU memory
* **I/O**: Storage doesn't bottleneck training
* **Utilization**: High GPU and network efficiency

#### Operational Excellence (20%)

* **Reliability**: Fault tolerance and recovery
* **Monitoring**: Comprehensive observability
* **Maintainability**: System can be operated long-term
* **Documentation**: Clear specifications and procedures
* **Automation**: Minimal manual intervention

#### Cost Effectiveness (10%)

* **Budget**: Within reasonable cost envelope
* **TCO**: Optimized total cost of ownership
* **Trade-offs**: Justified cost-performance decisions
* **Efficiency**: Minimal waste (power, cooling, hardware)

#### Sustainability (5%)

* **Carbon**: Minimized environmental impact
* **Energy**: Efficient power usage
* **Location**: Considered climate and energy sources
* **Longevity**: Hardware reuse and recycling

### Submission Guidelines

#### Format

Submit a technical report (20-40 pages) with:

1. **Executive Summary** (1-2 pages)
   * High-level architecture overview
   * Key design decisions and trade-offs
   * Expected performance and costs

2. **Detailed Design** (15-30 pages)
   * Each deliverable section with specifications
   * Diagrams, tables, and calculations
   * Justifications for choices

3. **Appendices**
   * Detailed calculations
   * Configuration files (YAML, TOML)
   * Network topology diagrams
   * Cost breakdown spreadsheets
   * Risk matrix

#### Supporting Materials

* **Architecture diagrams**: System overview, network topology, data flow
* **Configuration files**: Example configs for frameworks (DeepSpeed, Megatron)
* **Calculations**: Show your work for throughput, memory, costs
* **Code snippets**: Pseudocode for critical components
* **Comparison tables**: Alternative designs considered

### Example Calculation: Training Time

Show how you estimate training time:

```
Model: 500B parameters
Training data: 2T tokens
Hardware: 4096 H100 GPUs

Step 1: Compute requirement
FLOPs per token = 6 × N (forward + backward)
= 6 × 500B = 3 × 10^12 FLOPs/token

Total FLOPs = 3 × 10^12 × 2 × 10^12 = 6 × 10^24 FLOPs

Step 2: Hardware capability
H100: 989 TFLOPS (FP16 with sparsity)
Assume 60% MFU: 593 TFLOPS per GPU
Total: 593 TFLOPS × 4096 = 2.4 × 10^18 FLOPS

Step 3: Training time
Time = 6 × 10^24 / 2.4 × 10^18 = 2.5 × 10^6 seconds
= 29 days (achievable within 60-day target)

Note: Includes buffer for checkpointing, failures, ramp-up
```

### Tips for Success

* **Start with back-of-envelope calculations**: Sanity-check before detailed design
* **Research real systems**: Study papers on GPT-3, PaLM, BLOOM, OPT
* **Consult datasheets**: H100 specs, InfiniBand bandwidth, power requirements
* **Consider trade-offs**: No single optimal design, justify your choices
* **Be realistic**: Account for overhead, failures, non-ideal conditions
* **Think operationally**: Design must be runnable, not just theoretical
* **Quantify everything**: Provide numbers, not just qualitative descriptions

### Resources

* NVIDIA H100 Datasheet
* Megatron-LM and DeepSpeed documentation
* Research papers: GPT-3, PaLM, BLOOM architecture papers
* InfiniBand and network equipment specifications
* Datacenter design guides
* Published training costs and times for large models

### Questions to Guide Your Design

* How many tokens per second can your system process?
* What percentage of time is spent on communication vs computation?
* How long does a checkpoint take, and how often should you checkpoint?
* What happens when a GPU fails? A node? A rack?
* Can you achieve 60% MFU? What limits you?
* Is your network a bottleneck? How do you know?
* What's your largest single point of failure?
* How much does training cost per day?
* Could you train a 1T parameter model with minor changes?

### Submission Format

Submit as:
* **PDF report** (main deliverable)
* **ZIP archive** with:
  * Report PDF
  * Configuration files
  * Calculation spreadsheets
  * Diagrams (source files if possible)
  * Any code/scripts

Upload to: [Project Submission Portal] (details provided separately)

### Timeline

* **Week 1-2**: Research and planning
* **Week 3-4**: Detailed design and calculations
* **Week 5-6**: Documentation and review
* **Week 7**: Final submission

### Getting Help

* **Office hours**: Weekly Q&A sessions
* **Discussion forum**: Ask questions, share ideas
* **Peer review**: Optional draft review with classmates
* **Instructor feedback**: Submit outline for early feedback

---

This capstone represents the culmination of your journey through the full-stack LLM engineering curriculum. It requires synthesizing knowledge from statistical foundations, GPU programming, distributed systems, and datacenter engineering. Good luck, and we look forward to seeing your innovative designs!
