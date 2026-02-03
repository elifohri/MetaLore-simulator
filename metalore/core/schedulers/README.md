# Schedulers Module

The schedulers module handles how a base station divides its bandwidth among connected entities (UEs, sensors). It determines **how much bandwidth each entity gets** and computes the resulting **data rates**.

## Overview

When multiple entities are connected to the same base station, they share the BS's total bandwidth. The scheduler decides how to split it.

```
BS_0 (20 MHz total)
 |
 +-- UE_0:  gets 10 MHz  -->  rate = 10 MHz * log2(1 + SNR_0)
 +-- UE_1:  gets 10 MHz  -->  rate = 10 MHz * log2(1 + SNR_1)
```

The environment uses **two separate scheduler instances** — one for UEs and one for sensors — so each type can use a different scheduling strategy.

## How It Works

The scheduling pipeline runs in two steps via `compute_rates()`:

### Step 1: Allocate bandwidth (`share`)

The scheduler splits the available bandwidth among connected entities. This is the part that differs between scheduler implementations.

**ResourceFair** splits equally:

```
BS_0 has 20 MHz, 3 connected UEs

UE_0: 20 / 3 = 6.67 MHz
UE_1: 20 / 3 = 6.67 MHz
UE_2: 20 / 3 = 6.67 MHz
```

**RoundRobin** distributes in fixed-size chunks (quantum), rotating the starting position each timestep for fairness:

```
BS_0 has 20 MHz, quantum = 5 MHz, 3 connected UEs

Timestep 0 (start_index=2):  UE_2: 5, UE_0: 5, UE_1: 5, UE_2: 5  -->  [5, 5, 10]
Timestep 1 (start_index=2):  UE_2: 5, UE_0: 5, UE_1: 5, UE_2: 5  -->  [5, 5, 10]
```

### Step 2: Compute data rates (Shannon capacity)

For each entity, the channel model provides the SNR, and the Shannon capacity formula converts bandwidth + SNR into a data rate:

```
rate = allocated_bw * log2(1 + SNR) / 1e6    (in Mbps)
```

This step is handled by the base class `compute_rates()` and is the same for all schedulers.

## Architecture

```
Scheduler (base class)
    |
    +-- ResourceFair     (equal split)
    +-- RoundRobin       (quantum-based rotation)
```

### `Scheduler` (base.py)

Abstract base class. Subclasses must implement `share()`. Provides `compute_rates()` which combines `share()` with the Shannon capacity formula.

### `ResourceFair` (resource_fair.py)

Divides bandwidth equally among all connected entities. Stateless — no memory between timesteps.

### `RoundRobin` (round_robin.py)

Distributes bandwidth in fixed-size chunks (`quantum`), cycling through entities. Tracks `last_served_index` per BS to rotate the starting position each timestep, ensuring long-term fairness.

Parameters:
- `quantum` (float, default 1.0) — chunk size for each allocation round
- `offset` (int, default 3) — how far to shift the starting index between timesteps

## Key Methods

| Method | Class | Description |
|--------|-------|-------------|
| `share(bs, conns, total_resources)` | All | Split bandwidth among entities (abstract) |
| `compute_rates(bs, entities, bandwidth, channel)` | Base | Full pipeline: share + SNR + Shannon rate |
| `reset()` | All | Clear scheduler state |

## Usage

```python
from metalore.core.schedulers.resource_fair import ResourceFair
from metalore.core.schedulers.round_robin import RoundRobin

# Create scheduler
scheduler = ResourceFair()

# Compute rates for entities connected to a BS
rates = scheduler.compute_rates(bs, connected_ues, bandwidth=10e6, channel=channel)
# returns: {(bs, ue_0): 12.5, (bs, ue_1): 8.3, ...}  (Mbps)
```

## Adding a New Scheduler

1. Create a new file in `metalore/core/schedulers/`
2. Subclass `Scheduler` and implement `share()`
3. Register it in your config

```python
from metalore.core.schedulers.base import Scheduler

class ProportionalFair(Scheduler):

    def share(self, bs, conns, total_resources):
        # Your allocation logic here
        # Must return a list of floats, one per entity
        ...
```
