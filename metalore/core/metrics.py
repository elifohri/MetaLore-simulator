"""
Metrics tracking for MetaLore simulation.

Records simulation data at five granularities:
  - step_totals     : scalar aggregates across all entities, one value per timestep
  - step_per_entity : per-(entity_type, entity_id) time series, one value per timestep
  - step_per_bs     : per-BS time series — load, queue state (multi-cell)
  - step_topology   : connection-map snapshots, one dict per timestep
  - ep_totals       : episode-level scalar aggregates
  - ep_per_entity   : per-entity episode totals
"""

from collections import defaultdict
from typing import Dict, List, Optional


class MetricsTracker:
    """Tracks simulation metrics across timesteps within an episode."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear all recorded metrics for a new episode."""

        self.step_totals: Dict[str, List] = {
            "time": [], "num_active_ues": [], "num_active_sensors": [],
            "ue_tx_queue_bits": [], "sensor_tx_queue_bits": [],
            "ue_tx_queue_jobs": [], "sensor_tx_queue_jobs": [],
            "jobs_generated": [], "jobs_transmitted": [], "jobs_processed": [],
            "bits_transmitted": [], "cycles_processed": [],
            "ue_bits_transmitted": [], "sensor_bits_transmitted": [],
            "ue_cycles_processed": [], "sensor_cycles_processed": [],
            "bw_split": [], "comp_split": [], "reward": [], "observation": [],
            "mean_aoi": [], "mean_aori": [], "mean_aosi": [],
        }

        self.step_per_entity: Dict[str, defaultdict] = {
            k: defaultdict(list) for k in (
                "datarate", "tx_queue_jobs",
                "jobs_generated", "jobs_transmitted", "jobs_processed",
                "bits_transmitted", "cycles_processed",
            )
        }

        self.step_topology: Dict[str, List] = {
            "ue_connections": [], "sensor_connections": [], "nearest_sensor": [],
        }

        self.step_per_bs: Dict[str, defaultdict] = {
            k: defaultdict(list) for k in (
                "ue_connections", "sensor_connections",
                "ue_proc_queue_jobs", "sensor_proc_queue_jobs",
                "ue_proc_queue_cycles", "sensor_proc_queue_cycles",
            )
        }

        self.ep_totals: Dict = {}
        self.ep_per_entity: Dict = {}

    # --- Public API ---

    def record(self, env, bw_split: float, comp_split: float, reward: float, observation) -> None:
        """Record a snapshot of the current simulation state. Called once per step."""
        jt = env.job_tracker
        self._record_topology(env)
        self._record_per_bs(env)
        self._record_per_entity(env, jt)
        self._record_step_totals(env, jt, bw_split, comp_split, reward, observation)

    def finalize(self, job_tracker) -> Dict:
        """Compute and store episode-level aggregates. Call once at the end of an episode."""
        st   = self.step_totals
        ep   = job_tracker.ep_totals

        self.ep_totals = {
            "jobs_generated":       ep.jobs_generated,
            "jobs_transmitted":     ep.jobs_transmitted,
            "jobs_processed":       ep.jobs_processed,
            "bits_transmitted":     ep.bits_transmitted,
            "cycles_processed":     ep.cycles_processed,
            "job_completion_rate":  ep.jobs_processed / ep.jobs_generated if ep.jobs_generated > 0 else 0.0,
            "total_reward":         sum(st["reward"]) if st["reward"] else None,
            "mean_aoi":             self._safe_mean([j.aoi  for j in job_tracker.completed_jobs if j.entity_type == 'UE' and j.aoi  is not None]),
            "mean_aori":            self._safe_mean([j.aori for j in job_tracker.completed_jobs if j.entity_type == 'UE' and j.aori is not None]),
            "mean_aosi":            self._safe_mean([j.aosi for j in job_tracker.completed_jobs if j.entity_type == 'UE' and j.aosi is not None]),
        }

        fields = ["jobs_generated", "jobs_transmitted", "jobs_processed", "bits_transmitted", "cycles_processed"]
        self.ep_per_entity = {
            field: {k: getattr(v, field) for k, v in job_tracker.ep_per_entity.items()}
            for field in fields
        }

        return {**self.ep_totals, "per_entity": self.ep_per_entity}

    @property
    def num_steps(self) -> int:
        """Number of timesteps recorded so far."""
        return len(self.step_totals["time"])

    def latest(self, metric: str, per_entity: bool = False, per_bs: bool = False):
        """Return the most recent value for a step metric."""
        if per_entity:
            data = self.step_per_entity.get(metric)
            return {k: v[-1] for k, v in data.items() if v} if data else None
        if per_bs:
            data = self.step_per_bs.get(metric)
            return {k: v[-1] for k, v in data.items() if v} if data else None
        data = self.step_totals.get(metric)
        return data[-1] if data else None

    def mean(self, metric: str) -> Optional[float]:
        """Return the running mean for a scalar step metric (skips None values)."""
        data = self.step_totals.get(metric)
        return self._safe_mean([v for v in data if v is not None]) if data else None

    # --- Private helpers ---

    @staticmethod
    def _safe_mean(values) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def _record_topology(self, env) -> None:
        self.step_topology["ue_connections"].append({
            bs.id: sorted(ue.id for ue in ues)
            for bs, ues in env.connections_ue.items() if ues
        })
        self.step_topology["sensor_connections"].append({
            bs.id: sorted(s.id for s in sensors)
            for bs, sensors in env.connections_sensor.items() if sensors
        })
        self.step_topology["nearest_sensor"].append({
            ue.id: sensor.id
            for ue, sensor in env.association.nearest_sensor.items()
        })

    def _record_per_bs(self, env) -> None:
        for bs in env.stations.values():
            self.step_per_bs["ue_connections"][bs.id].append(len(env.connections_ue[bs]))
            self.step_per_bs["sensor_connections"][bs.id].append(len(env.connections_sensor[bs]))
            self.step_per_bs["ue_proc_queue_jobs"][bs.id].append(bs.proc_queues['UE'].length)
            self.step_per_bs["sensor_proc_queue_jobs"][bs.id].append(bs.proc_queues['SENSOR'].length)
            self.step_per_bs["ue_proc_queue_cycles"][bs.id].append(bs.proc_queues['UE'].total_cycles)
            self.step_per_bs["sensor_proc_queue_cycles"][bs.id].append(bs.proc_queues['SENSOR'].total_cycles)

    def _record_per_entity(self, env, jt) -> None:
        datarate_map = {('UE', ue.id): r for (_, ue), r in env.datarates_ue.items()}
        datarate_map.update({('SENSOR', s.id): r for (_, s), r in env.datarates_sensor.items()})
        all_entities = (
            [('UE',     eid, e.tx_queue) for eid, e in env.users.items()] +
            [('SENSOR', eid, e.tx_queue) for eid, e in env.sensors.items()]
        )
        for entity_type, eid, tx_queue in all_entities:
            key    = (entity_type, eid)
            counts = jt.step_per_entity[key]
            self.step_per_entity["datarate"][key].append(datarate_map.get(key, float('nan')))
            self.step_per_entity["tx_queue_jobs"][key].append(tx_queue.length)
            self.step_per_entity["jobs_generated"][key].append(counts.jobs_generated)
            self.step_per_entity["jobs_transmitted"][key].append(counts.jobs_transmitted)
            self.step_per_entity["bits_transmitted"][key].append(counts.bits_transmitted)
            self.step_per_entity["jobs_processed"][key].append(counts.jobs_processed)
            self.step_per_entity["cycles_processed"][key].append(counts.cycles_processed)

    def _record_step_totals(self, env, jt, bw_split: float, comp_split: float, reward: float, observation) -> None:
        st = jt.step_totals

        # Per-type bits/cycles (single pass over step_per_entity)
        ue_bits = ue_cycles = sensor_bits = sensor_cycles = 0.0
        for (etype, _), counts in jt.step_per_entity.items():
            if etype == 'UE':
                ue_bits   += counts.bits_transmitted
                ue_cycles += counts.cycles_processed
            else:
                sensor_bits   += counts.bits_transmitted
                sensor_cycles += counts.cycles_processed

        # AoI over UE jobs completed this step
        step_ue_jobs = [j for j in jt.step_completed_jobs if j.entity_type == 'UE']

        updates = {
            "time":                     env.time,
            "num_active_ues":           len(env.active_ues),
            "num_active_sensors":       len(env.active_sensors),
            "ue_tx_queue_bits":         sum(ue.tx_queue.total_bits for ue in env.active_ues),
            "sensor_tx_queue_bits":     sum(s.tx_queue.total_bits  for s  in env.active_sensors),
            "ue_tx_queue_jobs":         sum(ue.tx_queue.length     for ue in env.active_ues),
            "sensor_tx_queue_jobs":     sum(s.tx_queue.length      for s  in env.active_sensors),
            "jobs_generated":           st.jobs_generated,
            "jobs_transmitted":         st.jobs_transmitted,
            "jobs_processed":           st.jobs_processed,
            "bits_transmitted":         st.bits_transmitted,
            "cycles_processed":         st.cycles_processed,
            "ue_bits_transmitted":      ue_bits,
            "sensor_bits_transmitted":  sensor_bits,
            "ue_cycles_processed":      ue_cycles,
            "sensor_cycles_processed":  sensor_cycles,
            "bw_split":                 bw_split,
            "comp_split":               comp_split,
            "reward":                   reward,
            "observation":              observation.tolist() if hasattr(observation, 'tolist') else observation,
            "mean_aoi":                 self._safe_mean([j.aoi  for j in step_ue_jobs if j.aoi  is not None]),
            "mean_aori":                self._safe_mean([j.aori for j in step_ue_jobs if j.aori is not None]),
            "mean_aosi":                self._safe_mean([j.aosi for j in step_ue_jobs if j.aosi is not None]),
        }
        for key, val in updates.items():
            self.step_totals[key].append(val)
