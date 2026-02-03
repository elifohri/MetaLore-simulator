# Schedulers Module

The schedulers module handles how a base station divides its bandwidth among connected entities (UEs, sensors). It determines **how much bandwidth each entity gets**.

## Overview

When multiple entities are connected to the same base station, they share the BS's total bandwidth. The scheduler decides how to split it.

```
BS_0 (20 MHz total)
 |
 +-- UE_0:  gets 10 MHz
 +-- UE_1:  gets 10 MHz
```

The environment uses **two separate scheduler instances** — one for UEs and one for sensors — so each type can use a different scheduling strategy.

## How It Works

The scheduler splits the available bandwidth among connected entities via `share()`. This is the part that differs between scheduler implementations.

**ResourceFair** splits equally:

```
BS_0 has 20 MHz, 3 connected UEs

UE_0: 20 / 3 = 6.67 MHz
UE_1: 20 / 3 = 6.67 MHz
UE_2: 20 / 3 = 6.67 MHz
```

## Architecture

```
Scheduler (base class)
    |
    +-- ResourceFair     (equal split)
```

### `Scheduler` (base.py)

Abstract base class. Subclasses must implement `share()`.

### `ResourceFair` (resource_fair.py)

Divides bandwidth equally among all connected entities. Stateless — no memory between timesteps.

## Key Methods

| Method | Class | Description |
|--------|-------|-------------|
| `share(bs, conns, total_resources)` | All | Split bandwidth among entities (abstract) |
| `reset()` | All | Clear scheduler state |

## Usage

```python
from metalore.core.schedulers import ResourceFair

# Create scheduler
scheduler = ResourceFair(seed=42, reset_rng_episode=False)

# Allocate bandwidth among entities connected to a BS
allocations = scheduler.share(bs, connected_ues, total_resources=10e6)
# returns: [5e6, 5e6]  (Hz per entity)
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
