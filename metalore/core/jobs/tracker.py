"""
Job Tracker for MetaLore simulation.

Accumulates statistics on job generation, transmission and processing at four levels:
  - Episode totals          (ep_totals)
  - Per-entity episode      (ep_per_entity)
  - Per-step totals         (step_totals)
  - Per-entity per-step     (step_per_entity)
"""

import os
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from metalore.core.jobs.job import Job

EntityKey = Tuple[str, int]  # (entity_type, entity_id)


@dataclass
class JobCounts:
    """Aggregated job statistics for one scope (episode or step, global or per-entity)."""
    jobs_generated: int = 0
    jobs_transmitted: int = 0
    jobs_processed: int = 0
    bits_transmitted: float = 0.0
    cycles_processed: float = 0.0


class JobTracker:
    """Tracks job statistics at four levels of granularity."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all counters for a new episode."""
        self.ep_totals = JobCounts()
        self.ep_per_entity: Dict[EntityKey, JobCounts] = defaultdict(JobCounts)

        self.step_totals = JobCounts()
        self.step_per_entity: Dict[EntityKey, JobCounts] = defaultdict(JobCounts)

        # All fully-processed jobs this episode
        self.completed_jobs: List[Job] = []

        # Fully-processed jobs this step only
        self.step_completed_jobs: List[Job] = []

        # Latest processed sensor job per sensor
        self.sensor_latest_processed_job: Dict[int, Job] = {}

        # Departure tracking
        self.ue_extime_at_departure: Dict[int, int] = {}
        self.residual_jobs_at_departure: List[int] = []

    def begin_step(self) -> None:
        """Reset per-step counters at the start of each timestep."""
        self.step_totals = JobCounts()
        self.step_per_entity = defaultdict(JobCounts)
        self.step_completed_jobs = []

    def on_generated(self, job: Job) -> None:
        """Record that a job was generated this step."""
        key = (job.entity_type, job.entity_id)
        self.step_totals.jobs_generated += 1
        self.ep_totals.jobs_generated += 1
        self.step_per_entity[key].jobs_generated += 1
        self.ep_per_entity[key].jobs_generated += 1

    def on_transmitted(self, entity_key: EntityKey, jobs: List[Job], bits: float) -> None:
        """Record transmission progress for one entity this step."""
        n = len(jobs)
        self.step_totals.jobs_transmitted += n
        self.ep_totals.jobs_transmitted += n
        self.step_totals.bits_transmitted += bits
        self.ep_totals.bits_transmitted += bits
        self.step_per_entity[entity_key].jobs_transmitted += n
        self.ep_per_entity[entity_key].jobs_transmitted += n
        self.step_per_entity[entity_key].bits_transmitted += bits
        self.ep_per_entity[entity_key].bits_transmitted += bits

    def on_processed(self, jobs: List[Job], cycles: float) -> None:
        """Record that `jobs` were fully processed consuming `cycles` compute cycles."""
        n = len(jobs)
        self.step_totals.jobs_processed += n
        self.ep_totals.jobs_processed += n
        self.step_totals.cycles_processed += cycles
        self.ep_totals.cycles_processed += cycles

        for job in jobs:
            key = (job.entity_type, job.entity_id)
            self.completed_jobs.append(job)
            self.step_completed_jobs.append(job)
            self.step_per_entity[key].jobs_processed += 1
            self.ep_per_entity[key].jobs_processed += 1
            self.step_per_entity[key].cycles_processed += job.compute_size
            self.ep_per_entity[key].cycles_processed += job.compute_size

    def on_ue_departed(self, ue_id: int, extime: int, residual_count: int) -> None:
        """Record departure stats for a UE: its scheduled extime and how many jobs it left behind."""
        self.ue_extime_at_departure[ue_id] = extime
        self.residual_jobs_at_departure.append(residual_count)

    def update_ue_sensor_sync(self, job: Job) -> None:
        """Track latest processed sensor job for sensor-UE synchronization."""
        if job.entity_type == 'SENSOR':
            self.sensor_latest_processed_job[job.entity_id] = job

    def is_sensor_ready(self, sensor_id: int) -> bool:
        """Return True if at least one job from this sensor has been processed."""
        return self.sensor_latest_processed_job.get(sensor_id) is not None

    def get_sensor_snapshot_time(self, sensor_id: int) -> Optional[int]:
        """Return the generation time of the latest processed job for this sensor, or None."""
        sensor_job = self.sensor_latest_processed_job.get(sensor_id)
        return sensor_job.generated_at if sensor_job is not None else None

    def get_step_completed_ue_jobs(self) -> List[Job]:
        """Return completed UE jobs from this timestep."""
        return [job for job in self.step_completed_jobs if job.entity_type == 'UE']

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with one row per completed job and all lifecycle columns."""
        rows = [
            {
                "job_id":             job.id,
                "entity_id":          job.entity_id,
                "entity_type":        job.entity_type,
                "data_size":          job.data_size,
                "compute_size":       job.compute_size,
                "generated_at":       job.generated_at,
                "tx_start_at":        job.tx_start_at,
                "tx_end_at":          job.tx_end_at,
                "proc_start_at":      job.proc_start_at,
                "proc_end_at":        job.proc_end_at,
                "tx_queue_wait":      job.tx_queue_wait,
                "tx_duration":        job.tx_duration,
                "proc_queue_wait":    job.proc_queue_wait,
                "proc_duration":      job.proc_duration,
                "nearest_sensor_id":  job.nearest_sensor_id,
                "sensor_snapshot_at": job.sensor_snapshot_at,
                "aoi":                job.aoi,
                "aori":               job.aori,
                "aosi":               job.aosi,
            }
            for job in self.completed_jobs
        ]
        columns = [
            "job_id", "entity_id", "entity_type", "data_size", "compute_size",
            "generated_at", "tx_start_at", "tx_end_at", "proc_start_at", "proc_end_at",
            "tx_queue_wait", "tx_duration", "proc_queue_wait", "proc_duration",
            "nearest_sensor_id", "sensor_snapshot_at", "aoi", "aori", "aosi",
        ]
        return pd.DataFrame(rows, columns=columns)

    def save_log(self, path: str) -> None:
        """Save the completed job log to a CSV file."""
        if dir_path := os.path.dirname(path):
            os.makedirs(dir_path, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

    def __repr__(self) -> str:
        return (
            f"JobTracker(ep_jobs_generated={self.ep_totals.jobs_generated}, "
            f"ep_jobs_transmitted={self.ep_totals.jobs_transmitted}, "
            f"ep_jobs_processed={self.ep_totals.jobs_processed})"
        )