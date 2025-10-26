---
title: Power and Density
---

## Module 8.1: Power and Density

### The Power Challenge

Modern AI accelerators consume unprecedented amounts of power, fundamentally changing datacenter design.

### GPU Power Consumption Evolution

Historical progression of NVIDIA GPUs:

* **Tesla K80** (2014): 300W per GPU
* **Tesla P100** (2016): 300W per GPU
* **Tesla V100** (2017): 300W per GPU
* **A100** (2020): 400W per GPU (up to 500W with boost)
* **H100** (2022): 700W per GPU (up to 1000W in some configs)
* **H200** (2023): 700W per GPU

**Trend**: Power consumption doubling every 2-3 generations.

### Rack-Level Power

#### Traditional Enterprise Rack

```
42U rack: 5-10 kW typical
- 20-30 1U servers
- Each server: 200-400W
- Total: 8.4-12.6 kW
```

#### Modern AI Rack (8x A100)

```
42U rack: 15-20 kW
- 4-6 GPU servers (8-16U each)
- Each server: 3-4 kW (8×500W GPUs + CPUs + overhead)
- Total: 12-24 kW per rack
```

#### Next-Gen AI Rack (8x H100)

```
42U rack: 30-40 kW
- 2-4 GPU servers
- Each server: 8-10 kW (8×1000W GPUs + CPUs)
- Total: 16-40 kW per rack
```

**Challenge**: 3-4× higher power density than traditional datacenters!

### Power Distribution

#### Electrical Infrastructure Requirements

**For 1000 H100 GPUs** (typical small cluster):

```
GPU Power: 1000 × 1000W = 1 MW
CPU Power: 125 servers × 500W = 62.5 kW
Networking: ~100 kW
Storage: ~50 kW
Overhead (cooling, PDU losses): 30% = ~360 kW

Total: ~1.57 MW
```

**Equivalent to**: 1,000-1,500 average homes

#### Power Distribution Units (PDU)

* **Rack PDU**: Distributes power within rack
  * Traditional: 208V, 30A = 6.2 kW
  * High-density: 480V, 60A = 28.8 kW
  * Ultra-high-density: 480V, 100A = 48 kW

* **Busway**: High-capacity power distribution
  * Overhead or underfloor busways
  * 400-600A capacity per busway
  * Tap boxes every few racks

### Power Usage Effectiveness (PUE)

**PUE** measures datacenter efficiency:

```
PUE = Total Facility Power / IT Equipment Power
```

#### PUE Categories

* **PUE 2.0**: Poor (legacy datacenters)
  * For every 1W of IT equipment, 1W for cooling/overhead

* **PUE 1.5**: Average
  * 50% overhead for cooling and infrastructure

* **PUE 1.2**: Good (modern facilities)
  * 20% overhead - achievable with efficient cooling

* **PUE 1.1**: Excellent (state-of-art)
  * 10% overhead - requires advanced cooling, hot climates

#### Example Calculation

1 MW IT load with PUE 1.3:

```
Total Facility Power = 1 MW × 1.3 = 1.3 MW
Cooling/Overhead = 1.3 MW - 1 MW = 300 kW
```

### Density Challenges

#### Floor Space Limitations

```
Traditional datacenter: 100-150W per sq ft
AI datacenter: 400-800W per sq ft
```

**Problem**: Existing facilities can't support AI workloads without major upgrades.

#### Electrical Capacity

Many older datacenters limited by:
* **Substation capacity**: Total power available
* **Distribution infrastructure**: Can't deliver power where needed
* **Circuit breaker sizing**: Protection equipment inadequate

### Thermal Density

Power consumption = Heat generation

```
8x H100 server: 10 kW = 34,000 BTU/hr
```

For comparison:
* Average home HVAC: 24,000-48,000 BTU/hr
* **One GPU server produces as much heat as an entire house!**

### Cooling Requirements

Cooling capacity must match or exceed heat generation:

```
1 MW IT load requires:
- 1 MW cooling capacity (for PUE 1.0, theoretical minimum)
- 1.3 MW cooling capacity (for PUE 1.3, realistic)
- Higher for peak conditions
```

### Power Quality

AI training requires stable, clean power:

#### Voltage Stability

* **Tolerance**: ±5% of nominal voltage
* **Sags/surges**: Can cause GPU errors or crashes
* **Solution**: UPS systems, voltage regulators

#### Power Factor

```
Power Factor = Real Power / Apparent Power
```

* **Good**: 0.95+ (efficient power use)
* **Poor**: <0.85 (wasted reactive power)
* **Modern PSUs**: 0.99+ power factor

### Energy Costs

Training cost dominated by energy:

#### Example: GPT-3 Training

```
GPU Hours: 175B params × 300B tokens × efficiency factors
≈ 1,000,000 GPU-hours (A100)

Power: 1,000,000 hrs × 500W = 500,000 kWh
Cost at $0.10/kWh: $50,000 (electricity only)
Cost at $0.20/kWh: $100,000

Total TCO including hardware amortization: $5-10M
```

#### Operational Cost Breakdown

For large AI datacenter:

* **Electricity**: 50-60% (IT equipment + cooling)
* **Hardware amortization**: 25-35%
* **Personnel**: 5-10%
* **Networking**: 5-10%
* **Other**: 5%

### Peak vs. Sustained Power

GPUs have different power states:

```
H100 Power Modes:
- Idle: 50-100W
- Training (sustained): 600-700W
- Training (peak): 1000W (brief spikes)
```

**Design consideration**: Plan for sustained power, not just peak.

### Power Redundancy

Critical AI infrastructure requires redundancy:

#### N+1 Redundancy

```
10 MW required → Install 11 MW capacity
Extra 1 MW for maintenance and failover
```

#### 2N Redundancy

```
10 MW required → Install 20 MW capacity
Complete duplicate power train
Zero downtime for maintenance
```

**Cost**: 2N redundancy adds 50-100% to infrastructure cost.

### Sustainability Considerations

#### Carbon Footprint

```
1 MW datacenter for 1 year:
= 8,760 MWh consumed
= 3,500-7,000 tons CO2 (depending on grid mix)
```

#### Renewable Energy

Strategies to reduce carbon:
* **On-site solar**: 10-30% of load in sunny climates
* **PPAs** (Power Purchase Agreements): Buy renewable energy
* **Location selection**: Choose regions with clean grids (hydroelectric, wind)
* **Time-shifting**: Train during high renewable generation periods

#### Example: Iceland

Many AI companies train in Iceland:
* **100% renewable** (geothermal + hydro)
* **Cool climate** (reduced cooling costs)
* **Low PUE** (1.1-1.2 achievable)
* **Carbon-free AI training**

### Future Trends

#### Increasing Power Requirements

```
Current: 1-10 MW per AI cluster
Near future (2025): 20-50 MW per cluster
Future (2030): 100+ MW "AI factories"
```

#### Efficiency Improvements

* **More efficient chips**: Better FLOP/Watt
* **Sparsity**: Compute only on non-zero values
* **Precision reduction**: FP8, INT4 training
* **Algorithm efficiency**: Better models require less compute

#### Infrastructure Evolution

* **Dedicated AI substations**: Purpose-built power infrastructure
* **Modular datacenters**: Faster deployment
* **Edge AI**: Distributed inference reduces centralized power needs
* **Liquid cooling**: Enables higher density (covered in Module 8.2)

### Planning Considerations

When designing AI datacenter:

1. **Power capacity**: Ensure 2-3× current load for growth
2. **Distribution**: Multiple paths for redundancy
3. **Efficiency**: Target PUE <1.3
4. **Sustainability**: Renewable energy sources
5. **Location**: Power cost, climate, grid reliability
6. **Scalability**: Modular expansion capability

### Key Takeaways

* Modern AI GPUs consume 700-1000W, 3× more than previous generation
* AI racks require 30-40 kW, 4× traditional datacenter density
* Power infrastructure is major constraint for AI deployments
* PUE of 1.2-1.3 achievable with efficient design
* Energy costs dominate operational expenses for training
* Sustainability requires renewable energy and efficient cooling
* Future AI clusters will require 100+ MW of power
* Infrastructure planning must consider 5-10 year growth trajectory
