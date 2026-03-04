# Movement

Controls how entities (UEs and sensors) move through the simulation area each timestep. The movement model is called once per entity per timestep and returns the entity's new `(x, y)` position.


## Base Class: `Movement`

Abstract base class all movement models must subclass. Manages the episode RNG and defines the required interface.

**Constructor parameters** (passed via `env_params`):

| Parameter | Type | Description |
|---|---|---|
| `width` | `float` | Simulation area width |
| `height` | `float` | Simulation area height |
| `seed` | `int` | RNG seed for reproducibility |
| `reset_rng_episode` | `bool` | If `True`, RNG resets to the same seed each episode |

**Interface:**

```python
def move(self, entity) -> Tuple[float, float]:
    """Move entity for one timestep. Returns new (x, y) position."""
```

**`reset()`** is called at the start of each episode by the environment. It reinitialises the RNG if `reset_rng_episode` is set, or initialises it on the first call.


## `StaticMovement`

The default movement model. Entities remain at their initial position for the entire episode.

```python
from metalore.core.movement import StaticMovement
```

**Behaviour:** returns `(entity.x, entity.y)` unchanged every timestep.


## `RandomWaypointMovement`

Entities move toward a randomly generated waypoint. When the waypoint is reached, a new random waypoint is generated.

```python
from metalore.core.movement import RandomWaypointMovement
```

**Behaviour:**
- On first call for an entity, a waypoint `(wx, wy)` is sampled uniformly within `[0, width] × [0, height]`
- Each timestep the entity moves `entity.velocity` units toward its waypoint
- When the remaining distance is ≤ `entity.velocity`, the entity moves directly to the waypoint and a new one is assigned next call
- Waypoints are stored per entity id: `Dict[int, Tuple[float, float]]`


## Adding a new movement model

Subclass `Movement` and implement `move()`:

```python
from metalore.core.movement.base import Movement
from typing import Tuple

class MyMovement(Movement):

    def move(self, entity) -> Tuple[float, float]:
        new_x = ...
        new_y = ...
        return new_x, new_y
```

If your model has per-episode state, override `reset()` and call `super().reset()` first:

```python
    def reset(self) -> None:
        super().reset()
        self.my_state.clear()
```

Register it in the config:

```python
config['environment']['movement_ue'] = MyMovement
```

The environment will instantiate it with `env_params` (which includes `width`, `height`, `seed`, `reset_rng_episode`, `ep_max_time`).
