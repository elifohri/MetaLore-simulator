# Association

Controls how entities are connected to base stations and to each other. The association model is called once per timestep and updates three mappings:

| Mapping | Type | Description |
|---|---|---|
| `connections_ue` | `Dict[BaseStation, Set[UserEquipment]]` | Which UEs are connected to each BS |
| `connections_sensor` | `Dict[BaseStation, Set[Sensor]]` | Which sensors are connected to each BS |
| `nearest_sensor` | `Dict[UserEquipment, Sensor]` | The closest sensor for each UE |


## Base Class: `Association`

Abstract base class all association models must subclass. Manages episode RNG and owns the three connection mappings.

**Constructor parameters** (passed via `env_params`):

| Parameter | Type | Description |
|---|---|---|
| `seed` | `int` | RNG seed for reproducibility |
| `reset_rng_episode` | `bool` | If `True`, RNG resets to the same seed each episode |

**Interface:**

```python
def update_association(self, stations: Dict, users: Dict, sensors: Dict) -> None:
    """Perform full association update. Must populate connections_ue,
    connections_sensor and nearest_sensor before returning."""
```

**Convenience query methods** (available to all subclasses):

```python
def get_connected_ues(self, bs: BaseStation) -> Set[UserEquipment]
def get_connected_sensors(self, bs: BaseStation) -> Set[Sensor]
def get_bs_for_entity(self, entity: Entity) -> Optional[BaseStation]
```

**`reset()`** is called at the start of each episode. It reinitialises the RNG if `reset_rng_episode` is set. Subclasses should override it to clear their connection mappings, calling `super().reset()` first.


## `ClosestAssociation`

The default association model. Associates each entity to its geometrically closest base station and each UE to its closest sensor.

```python
from metalore.core.association import ClosestAssociation
```

**Behaviour:**
- Computes pairwise distances using vectorised numpy operations
- Each UE and sensor is assigned to the nearest BS (by Euclidean distance)
- Each UE's nearest sensor is updated every timestep
- Connections are fully rebuilt at each call


## Adding a new association model

Subclass `Association`, implement `update_association()`, and clear the mappings at episode start:

```python
from metalore.core.association.base import Association
from typing import Dict

class MyAssociation(Association):

    def reset(self) -> None:
        super().reset()
        self.connections_ue.clear()
        self.connections_sensor.clear()
        self.nearest_sensor.clear()

    def update_association(self, stations: Dict, users: Dict, sensors: Dict) -> None:
        # populate self.connections_ue, self.connections_sensor, self.nearest_sensor
        ...
```

Register it in the config:

```python
config['environment']['association'] = MyAssociation
```

The environment will instantiate it with `env_params`.
