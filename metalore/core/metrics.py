"""
Metrics tracking for MetaLore simulation.

Records per-timestep simulation data organized into four categories:
- Environment: entity counts and time info
- Topology: connections and nearest-sensor assignments
- Performance: datarates, macro datarates, and utilities per entity
- Actions: agent bandwidth and compute allocation splits
"""

from collections import defaultdict
from typing import Dict, List


class MetricsTracker:
    """Tracks simulation metrics across timesteps within an episode."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear all recorded metrics for a new episode."""
        # Environment
        self.environment: Dict[str, List] = {
            "time": [],
            "num_active_ues": [],
            "num_active_sensors": [],
            "throughput_ue": [],
            "throughput_sensor": [],
        }

        # Topology (snapshot per timestep)
        self.topology: Dict[str, List] = {
            "connections_ue": [],
            "connections_sensor": [],
            "nearest_sensor": [],
            "bs_load_ue": [],
            "bs_load_sensor": [],
        }

        # Performance (per-entity time series)
        self.performance: Dict[str, defaultdict] = {
            "datarates_ue": defaultdict(list),
            "datarates_sensor": defaultdict(list),
            "macro_ue": defaultdict(list),
            "macro_sensor": defaultdict(list),
            "utilities_ue": defaultdict(list),
            "utilities_sensor": defaultdict(list),
        }

        # Actions & Outputs
        self.actions: Dict[str, List] = {
            "bw_split": [],
            "comp_split": [],
            "reward": [],
            "observation": [],
        }

    def record(self, env, bw_split: float, comp_split: float, reward: float, observation) -> None:
        """Record a snapshot of the current simulation state.

        Should be called once per step, after utilities are computed.
        """
        # Environment
        self.environment["time"].append(env.time)
        self.environment["num_active_ues"].append(len(env.active_ues))
        self.environment["num_active_sensors"].append(len(env.active_sensors))
        self.environment["throughput_ue"].append(sum(env.macro_ue.values()))
        self.environment["throughput_sensor"].append(sum(env.macro_sensor.values()))

        # Topology — connections
        conn_ue = {
            bs.id: sorted(ue.id for ue in ues)
            for bs, ues in env.connections_ue.items()
            if ues
        }
        conn_sensor = {
            bs.id: sorted(s.id for s in sensors)
            for bs, sensors in env.connections_sensor.items()
            if sensors
        }
        nearest = {
            ue.id: ue.nearest_sensor.id
            for ue in env.active_ues
            if hasattr(ue, "nearest_sensor") and ue.nearest_sensor is not None
        }
        bs_load_ue = {bs.id: len(ues) for bs, ues in env.connections_ue.items()}
        bs_load_sensor = {bs.id: len(sensors) for bs, sensors in env.connections_sensor.items()}

        self.topology["connections_ue"].append(conn_ue)
        self.topology["connections_sensor"].append(conn_sensor)
        self.topology["nearest_sensor"].append(nearest)
        self.topology["bs_load_ue"].append(bs_load_ue)
        self.topology["bs_load_sensor"].append(bs_load_sensor)

        # Performance — per-entity metrics
        for (_, ue), rate in env.datarates_ue.items():
            self.performance["datarates_ue"][ue.id].append(rate)

        for (_, sensor), rate in env.datarates_sensor.items():
            self.performance["datarates_sensor"][sensor.id].append(rate)

        for ue, macro in env.macro_ue.items():
            self.performance["macro_ue"][ue.id].append(macro)

        for sensor, macro in env.macro_sensor.items():
            self.performance["macro_sensor"][sensor.id].append(macro)

        for ue, util in env.utilities_ue.items():
            self.performance["utilities_ue"][ue.id].append(util)

        for sensor, util in env.utilities_sensor.items():
            self.performance["utilities_sensor"][sensor.id].append(util)

        # Actions & Outputs
        self.actions["bw_split"].append(bw_split)
        self.actions["comp_split"].append(comp_split)
        self.actions["reward"].append(reward)
        self.actions["observation"].append(observation.tolist() if hasattr(observation, 'tolist') else observation)

    @property
    def num_steps(self) -> int:
        """Number of timesteps recorded so far."""
        return len(self.environment["time"])

    def latest(self, category: str, metric: str):
        """Get the most recent value for a metric, or None if empty."""
        data = getattr(self, category, {}).get(metric)
        if data is None:
            return None
        if isinstance(data, list):
            return data[-1] if data else None
        # defaultdict(list) — return dict of latest values
        return {k: v[-1] for k, v in data.items() if v}

    def mean(self, category: str, metric: str):
        """Get the running mean for a scalar metric, or None if empty."""
        data = getattr(self, category, {}).get(metric)
        if not isinstance(data, list) or not data:
            return None
        return sum(data) / len(data)
