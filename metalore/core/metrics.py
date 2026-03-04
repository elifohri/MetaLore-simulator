"""
Metrics tracking for MetaLore simulation.

Records per-timestep simulation data organized into five categories:
- Environment: entity counts and time info
- Topology: connections and nearest-sensor assignments
- Performance: datarates, utilities, and queue lengths per entity
- Actions: agent bandwidth and compute allocation splits
- Jobs: queue lengths, transmission and processing throughput
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
            # Wireless channel
            "datarates_ue":           defaultdict(list),
            "datarates_sensor":       defaultdict(list),
            "utilities_ue":           defaultdict(list),
            "utilities_sensor":       defaultdict(list),
            # Queue state
            "tx_queue_length_ue":     defaultdict(list),
            "tx_queue_length_sensor": defaultdict(list),
            # Job activity
            "jobs_generated_ue":      defaultdict(list),
            "bits_transmitted_ue":    defaultdict(list),
        }

        # Actions & Outputs
        self.actions: Dict[str, List] = {
            "bw_split": [],
            "comp_split": [],
            "reward": [],
            "observation": [],
        }

        # Jobs — per-step queue backlog and completion counts
        self.jobs: Dict[str, List] = {
            # Raw backlog (for analysis/reward computation)
            "tx_queue_bits_ue": [],
            "tx_queue_bits_sensor": [],
            "proc_queue_cycles_ue": [],
            "proc_queue_cycles_sensor": [],
            # Job counts (for visualization)
            "tx_queue_jobs_ue": [],
            "tx_queue_jobs_sensor": [],
            "proc_queue_jobs_ue": [],
            "proc_queue_jobs_sensor": [],
            # Step-level events
            "jobs_generated": [],
            "jobs_transmitted": [],
            "jobs_processed": [],
            "bits_transmitted": [],
            "cycles_processed": [],
        }

    def record(self, env, bw_split: float, comp_split: float, reward: float, observation) -> None:
        """Record a snapshot of the current simulation state.

        Should be called once per step, after utilities are computed.
        """
        # Environment
        self.environment["time"].append(env.time)
        self.environment["num_active_ues"].append(len(env.active_ues))
        self.environment["num_active_sensors"].append(len(env.active_sensors))
        self.environment["throughput_ue"].append(sum(env.datarates_ue.values()))
        self.environment["throughput_sensor"].append(sum(env.datarates_sensor.values()))

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
            ue.id: sensor.id
            for ue, sensor in env.association.nearest_sensor.items()
            if ue in env.active_ues
        }
        bs_load_ue = {bs.id: len(ues) for bs, ues in env.connections_ue.items()}
        bs_load_sensor = {bs.id: len(sensors) for bs, sensors in env.connections_sensor.items()}

        self.topology["connections_ue"].append(conn_ue)
        self.topology["connections_sensor"].append(conn_sensor)
        self.topology["nearest_sensor"].append(nearest)
        self.topology["bs_load_ue"].append(bs_load_ue)
        self.topology["bs_load_sensor"].append(bs_load_sensor)

        # Performance — per entity per step
        datarate_ue_map = {ue.id: rate for (_, ue), rate in env.datarates_ue.items()}
        datarate_sensor_map = {s.id: rate for (_, s), rate in env.datarates_sensor.items()}
        util_ue_map = {ue.id: util for ue, util in env.utilities_ue.items()}
        util_sensor_map = {s.id: util for s, util in env.utilities_sensor.items()}

        for ue_id, ue in env.users.items():
            self.performance["datarates_ue"][ue_id].append(datarate_ue_map.get(ue_id, float('nan')))
            self.performance["utilities_ue"][ue_id].append(util_ue_map.get(ue_id, float('nan')))
            self.performance["tx_queue_length_ue"][ue_id].append(ue.tx_queue.length)
            self.performance["jobs_generated_ue"][ue_id].append(env.job_tracker.step_entity_generated.get(('UE', ue_id), 0))
            self.performance["bits_transmitted_ue"][ue_id].append(env.job_tracker.step_entity_bits_transmitted.get(('UE', ue_id), 0.0))

        for sensor_id, sensor in env.sensors.items():
            self.performance["datarates_sensor"][sensor_id].append(datarate_sensor_map.get(sensor_id, float('nan')))
            self.performance["utilities_sensor"][sensor_id].append(util_sensor_map.get(sensor_id, float('nan')))
            self.performance["tx_queue_length_sensor"][sensor_id].append(sensor.tx_queue.length)

        # Actions & Outputs
        self.actions["bw_split"].append(bw_split)
        self.actions["comp_split"].append(comp_split)
        self.actions["reward"].append(reward)
        self.actions["observation"].append(observation.tolist() if hasattr(observation, 'tolist') else observation)

        # Jobs — queue backlog and step-level completion counts
        self.jobs["tx_queue_bits_ue"].append(
            sum(ue.tx_queue.total_bits for ue in env.users.values())
        )
        self.jobs["tx_queue_bits_sensor"].append(
            sum(s.tx_queue.total_bits for s in env.sensors.values())
        )
        self.jobs["proc_queue_cycles_ue"].append(
            sum(bs.proc_queues['UE'].total_cycles for bs in env.stations.values())
        )
        self.jobs["proc_queue_cycles_sensor"].append(
            sum(bs.proc_queues['SENSOR'].total_cycles for bs in env.stations.values())
        )
        self.jobs["tx_queue_jobs_ue"].append(
            sum(ue.tx_queue.length for ue in env.users.values())
        )
        self.jobs["tx_queue_jobs_sensor"].append(
            sum(s.tx_queue.length for s in env.sensors.values())
        )
        self.jobs["proc_queue_jobs_ue"].append(
            sum(bs.proc_queues['UE'].length for bs in env.stations.values())
        )
        self.jobs["proc_queue_jobs_sensor"].append(
            sum(bs.proc_queues['SENSOR'].length for bs in env.stations.values())
        )
        self.jobs["jobs_generated"].append(env.job_tracker.step_generated)
        self.jobs["jobs_transmitted"].append(env.job_tracker.step_transmitted)
        self.jobs["jobs_processed"].append(env.job_tracker.step_processed)
        self.jobs["bits_transmitted"].append(env.job_tracker.step_bits_transmitted)
        self.jobs["cycles_processed"].append(env.job_tracker.step_cycles_processed)

    def summary(self, job_tracker) -> Dict:
        """Return episode-level aggregate statistics.

        Should be called at the end of an episode (after truncation).
        """
        rewards = self.actions["reward"]
        throughput_ue = self.environment["throughput_ue"]
        throughput_sensor = self.environment["throughput_sensor"]

        return {
            # Episode job totals
            "total_generated":        job_tracker.total_generated,
            "total_transmitted":      job_tracker.total_transmitted,
            "total_processed":        job_tracker.total_processed,
            "total_bits_transmitted": job_tracker.total_bits_transmitted,
            "total_cycles_processed": job_tracker.total_cycles_processed,
            "completion_rate":        job_tracker.total_processed / job_tracker.total_generated
                                      if job_tracker.total_generated > 0 else 0.0,
            # Per-entity episode totals (keyed by (entity_type, entity_id))
            "entity_generated":         dict(job_tracker.entity_generated),
            "entity_transmitted":       dict(job_tracker.entity_transmitted),
            "entity_processed":         dict(job_tracker.entity_processed),
            "entity_bits_transmitted":  dict(job_tracker.entity_bits_transmitted),
            "entity_cycles_processed":  dict(job_tracker.entity_cycles_processed),
            # Derived episode-level stats
            "mean_reward":            sum(rewards) / len(rewards) if rewards else None,
            "total_reward":           sum(rewards) if rewards else None,
            "mean_throughput_ue":     sum(throughput_ue) / len(throughput_ue) if throughput_ue else None,
            "mean_throughput_sensor": sum(throughput_sensor) / len(throughput_sensor) if throughput_sensor else None,
        }

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
