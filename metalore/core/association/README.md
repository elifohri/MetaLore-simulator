# Association Module

The association module handles how entities (UEs, sensors) connect to base stations in the simulation. It determines **who connects to whom**.

## Overview

There are three types of entity in the simulation:

- **Base Station (BS)** - fixed infrastructure that provides wireless coverage
- **User Equipment (UE)** - mobile devices (phones, laptops, etc.)
- **Sensor** - IoT devices deployed in the environment

The association module manages three relationships:

```
UE  ----->  BS       (each UE connects to one BS)
Sensor -->  BS       (each sensor connects to one BS)
UE  ----->  Sensor   (each UE links to one sensor)
```

## How It Works

The association process runs in three steps via `update_association()`:

### Step 1: Associate UEs to closest BS

A pairwise distance matrix is computed between all UEs and all BSs:

```
              BS_0     BS_1     BS_2
UE_0       [ 150.2,  320.5,  410.0 ]
UE_1       [ 280.1,   90.3,  500.7 ]
UE_2       [ 410.7,  200.8,  120.4 ]
```

For each UE (row), `argmin` picks the BS (column) with the smallest distance:

```
UE_0 -> BS_0  (150.2 is smallest in row 0)
UE_1 -> BS_1  (90.3 is smallest in row 1)
UE_2 -> BS_2  (120.4 is smallest in row 2)
```

Results are stored in `connections_ue`: a dict mapping each BS to the set of UEs connected to it.

### Step 2: Associate sensors to closest BS

Same process as Step 1 but for sensors.

### Step 3: Update each UE's nearest sensor

Same distance-based assignment. Each UE stores a reference to its nearest sensor (`ue.nearest_sensor`).

### Connection Validation (called separately from the environment)

After association (and scheduling), the environment calls `validate_connections(channel)` to filter out connections where the signal is too weak. For each BS-entity pair, the channel model computes the SNR. If the SNR falls below the entity's threshold, the connection is dropped.

```
BS_0: {UE_0, UE_1, UE_2}  -->  BS_0: {UE_0, UE_1}   (UE_2 dropped, SNR too low)
```

## Architecture

```
Association (base class)
    |
    +-- ClosestAssociation
```

### `Association` (base.py)

Abstract base class that defines the interface. Stores two connection dicts:

- `connections_ue: Dict[BS, Set[UE]]`
- `connections_sensor: Dict[BS, Set[Sensor]]`

### `ClosestAssociation` (closest.py)

The concrete implementation. Takes two arguments:

- `env` - the simulation environment (holds all entities)
- `channel` - a `Channel` model

## Key Methods

| Method | Description |
|--------|-------------|
| `update_association()` | Runs the 3-step association cycle |
| `reset()` | Clears all connections |
| `validate_connections(channel)` | Filters connections based on SNR threshold |
| `get_connected_ues(bs)` | Returns set of UEs connected to a BS |
| `get_connected_sensors(bs)` | Returns set of sensors connected to a BS |
| `get_bs_for_entity(entity)` | Returns which BS an entity is connected to |

## Usage

```python
from metalore.core.association import ClosestAssociation

association = ClosestAssociation(env, channel)

# Run association cycle
association.update_association()

# Validate connections
association.validate_connections(channel)

# Query results
ues = association.get_connected_ues(some_bs)
bs = association.get_bs_for_entity(some_ue)
```
