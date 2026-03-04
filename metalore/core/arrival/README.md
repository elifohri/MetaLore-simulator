# Arrival

Controls when entities enter and leave the simulation. Each arrival model assigns an `stime` (arrival timestep) and `extime` (departure timestep) to every entity at the start of an episode.


## Base Class: `Arrival`

Abstract base class all arrival models must subclass. Manages the episode RNG and defines the required interface.

**Constructor parameters** (passed via `env_params`):

| Parameter | Type | Description |
|---|---|---|
| `ep_max_time` | `int` | Episode length in timesteps |
| `seed` | `int` | RNG seed for reproducibility |
| `reset_rng_episode` | `bool` | If `True`, RNG resets to the same seed each episode |

**Interface:**

```python
def arrival(self, entities: Dict) -> None:
    """Assign stime to each entity."""

def departure(self, entities: Dict) -> None:
    """Assign extime to each entity."""
```

Both methods receive the full entity dict (`{id: entity}`). After calling them, every entity must have `stime < extime` and both values within `[0, ep_max_time]`.

**`reset()`** is called at the start of each episode by the environment. It reinitialises the RNG if `reset_rng_episode` is set, or initialises it on the first call.


## `NoDeparture`

The default arrival pattern. All entities are present for the entire episode, no arrivals or departures mid-episode.

```python
from metalore.core.arrival import NoDeparture
```

**Behaviour:**
- `arrival`: sets `stime = 0` for every entity
- `departure`: sets `extime = ep_max_time` for every entity


## Adding a new arrival model

Subclass `Arrival` and implement both methods:

```python
from metalore.core.arrival.base import Arrival

class MyArrival(Arrival):

    def arrival(self, entities: Dict) -> None:
        for entity in entities.values():
            entity.stime = ...   # int in [0, ep_max_time)

    def departure(self, entities: Dict) -> None:
        for entity in entities.values():
            entity.extime = ...  # int in (entity.stime, ep_max_time]
```

Register it in the config:

```python
config['environment']['arrival_ue'] = MyArrival
```

The environment will instantiate it with `env_params` (which includes `ep_max_time`, `seed`, `reset_rng_episode`, `width`, `height`).
