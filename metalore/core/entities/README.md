# Entities

Network entities in the MetaLore simulation. Each entity represents a physical node in the wireless network.

## Classes

### `BaseStation`
A cellular tower with Mobile Edge Computing (MEC) capability.

Provides wireless connectivity and compute resources for connected devices. Each BS maintains two independent processing queues, one for UE jobs and one for sensor jobs, accessed via `proc_queues['UE']` and `proc_queues['SENSOR']`.

**Key attributes:**
| Attribute | Type | Description |
|---|---|---|
| `id` | `int` | Unique identifier |
| `position` | `(float, float)` | Fixed `(x, y)` coordinates |
| `bandwidth` | `float` | Total available bandwidth in Hz |
| `frequency` | `float` | Carrier frequency in MHz |
| `tx_power` | `float` | Transmission power in dBm |
| `compute_capacity` | `float` | MEC processing rate in CPU |
| `proc_queues` | `Dict[str, ProcessQueue]` | MEC job queues keyed by `DEVICE_TYPE` |

**Methods:**
| Method | Description |
|---|---|
| `reset_queue()` | Clears all entries in `proc_queues` at the start of a new episode |

---

### `UserEquipment`
A mobile device that moves through the simulation area and generates jobs for service requests.

Jobs are generated stochastically (Bernoulli process) and held in `tx_queue` until transmitted to the connected BS. Position is updated each timestep by the movement model.

**Key attributes:**
| Attribute | Type | Description |
|---|---|---|
| `id` | `int` | Unique identifier |
| `DEVICE_TYPE` | `str` | `'UE'` |
| `position` | `(float, float)` | Current `(x, y)` coordinates (mutable) |
| `velocity` | `float` | Movement speed |
| `height` | `float` | Antenna height in meters |
| `snr_threshold` | `float` | Minimum SNR required for connectivity |
| `noise` | `float` | Receiver noise power in Watts |
| `tx_queue` | `JobQueue` | Jobs waiting to be transmitted to BS |
| `stime` / `extime` | `Optional[int]` | Arrival / departure timestep |
| `is_mobile` | `bool` | `True` if `velocity > 0` |

**Methods:**
| Method | Description |
|---|---|
| `reset_queue()` | Clears `tx_queue` at the start of a new episode |

---

### `Sensor`
A stationary IoT device that periodically transmits environmental data.

Sensors emit jobs on a fixed schedule to maintain digital twin synchronisation. They are typically static but support movement via config.

**Key attributes:**
| Attribute | Type | Description |
|---|---|---|
| `id` | `int` | Unique identifier |
| `DEVICE_TYPE` | `str` | `'SENSOR'` |
| `position` | `(float, float)` | Current `(x, y)` coordinates (mutable) |
| `sensing_range` | `float` | Detection radius in meters |
| `update_interval` | `int` | Job emission period in timesteps |
| `tx_queue` | `JobQueue` | Jobs waiting to be transmitted to BS |
| `stime` / `extime` | `Optional[int]` | Arrival / departure timestep |

**Methods:**
| Method | Description |
|---|---|
| `reset_queue()` | Clears `tx_queue` at the start of a new episode |

`Sensor` shares `velocity`, `height`, `snr_threshold`, and `noise` with `UserEquipment`.