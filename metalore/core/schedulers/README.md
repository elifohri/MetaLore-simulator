# Schedulers

Controls how bandwidth is divided among entities connected to a base station. The scheduler is called once per base station per timestep and returns a per-entity bandwidth allocation in Hz.


## Base Class: `Scheduler`

Abstract base class all schedulers must subclass. Manages the episode RNG and defines the required interface.

**Constructor parameters** (passed via `env_params`):

| Parameter | Type | Description |
|---|---|---|
| `seed` | `int` | RNG seed for reproducibility |
| `reset_rng_episode` | `bool` | If `True`, RNG resets to the same seed each episode |

**Interface:**

```python
def share(self, bs: BaseStation, conns: List, total_resources: float) -> List[float]:
    """Allocate resources among connected entities. Returns one value per entity."""
```

The returned list must be the same length as `conns`. The sum of allocations should not exceed `total_resources`.

**`reset()`** is called at the start of each episode by the environment. It reinitialises the RNG if `reset_rng_episode` is set, or initialises it on the first call. Stateful schedulers should override `reset()` and call `super().reset()`.


## `ResourceFair`

The default scheduler. Splits bandwidth equally among all connected entities.

```python
from metalore.core.schedulers import ResourceFair
```

**Behaviour:** each entity receives `total_resources / num_connected` Hz.


## Adding a new scheduler

Subclass `Scheduler` and implement `share()`:

```python
from metalore.core.schedulers.base import Scheduler
from metalore.core.entities.base_station import BaseStation
from typing import List

class MyScheduler(Scheduler):

    def share(self, bs: BaseStation, conns: List, total_resources: float) -> List[float]:
        allocations = []
        for entity in conns:
            allocations.append(...)
        return allocations
```

If your scheduler is stateful, override `reset()` and call `super().reset()` first:

```python
    def reset(self) -> None:
        super().reset()
        self.my_state.clear()
```

Register it in the config:

```python
config['environment']['scheduler_ue'] = MyScheduler
config['environment']['scheduler_sensor'] = MyScheduler
```

The environment maintains separate scheduler instances for UEs and sensors (`scheduler_ue`, `scheduler_sensor`). The environment will instantiate each with `env_params`.
