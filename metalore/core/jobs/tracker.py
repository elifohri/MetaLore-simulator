"""
Job Tracker for MetaLore simulation.

Accumulates statistics on job generation, transmission and processing at four levels:
  - Episode totals
  - Per-step totals
  - Per-entity episode totals
  - Per-entity per-step totals
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from metalore.core.jobs.job import Job


class JobTracker:
    """Tracks job statistics at episode, step, entity and entity-per-step levels."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all counters for a new episode."""

        # --- Episode totals ---
        self.total_generated: int = 0
        self.total_transmitted: int = 0
        self.total_processed: int = 0
        self.total_bits_transmitted: float = 0.0
        self.total_cycles_processed: float = 0.0

        # --- Per-entity episode totals ---
        self.entity_generated: Dict[Tuple[str, int], int] = defaultdict(int)
        self.entity_transmitted: Dict[Tuple[str, int], int] = defaultdict(int)
        self.entity_processed: Dict[Tuple[str, int], int] = defaultdict(int)
        self.entity_bits_transmitted: Dict[Tuple[str, int], float] = defaultdict(float)
        self.entity_cycles_processed: Dict[Tuple[str, int], float] = defaultdict(float)

        # --- Per-step totals ---
        self.step_generated: int = 0
        self.step_transmitted: int = 0
        self.step_processed: int = 0
        self.step_bits_transmitted: float = 0.0
        self.step_cycles_processed: float = 0.0

        # --- Per-entity per-step totals ---
        self.step_entity_generated: Dict[Tuple[str, int], int] = defaultdict(int)
        self.step_entity_transmitted: Dict[Tuple[str, int], int] = defaultdict(int)
        self.step_entity_processed: Dict[Tuple[str, int], int] = defaultdict(int)
        self.step_entity_bits_transmitted: Dict[Tuple[str, int], float] = defaultdict(float)
        self.step_entity_cycles_processed: Dict[Tuple[str, int], float] = defaultdict(float)

        # All fully processed jobs this episode
        self._jobs: List[Job] = []

        # Latest fully processed sensor job per sensor (sensor_id → Job)
        # Used by processor.py to stamp sensor context on completed UE jobs
        self.sensor_latest_job: Dict[int, Job] = {}

    def begin_step(self) -> None:
        """Reset per-step counters at the start of each timestep."""
        self.step_generated = 0
        self.step_transmitted = 0
        self.step_processed = 0
        self.step_bits_transmitted = 0.0
        self.step_cycles_processed = 0.0
        self.step_entity_generated = defaultdict(int)
        self.step_entity_transmitted = defaultdict(int)
        self.step_entity_processed = defaultdict(int)
        self.step_entity_bits_transmitted = defaultdict(float)
        self.step_entity_cycles_processed = defaultdict(float)

    def on_generated(self, job: Job) -> None:
        """Record that a job was generated this step."""
        key = (job.entity_type, job.entity_id)
        self.step_generated += 1
        self.total_generated += 1
        self.step_entity_generated[key] += 1
        self.entity_generated[key] += 1

    def on_transmitted(self, jobs: List[Job], bits: float) -> None:
        """Record that `jobs` were fully transmitted and `bits` were sent."""
        self.step_transmitted += len(jobs)
        self.step_bits_transmitted += bits
        self.total_transmitted += len(jobs)
        self.total_bits_transmitted += bits
        for job in jobs:
            key = (job.entity_type, job.entity_id)
            self.step_entity_transmitted[key] += 1
            self.step_entity_bits_transmitted[key] += job.data_size
            self.entity_transmitted[key] += 1
            self.entity_bits_transmitted[key] += job.data_size

    def on_processed(self, jobs: List[Job], cycles: float) -> None:
        """Record that `jobs` were fully processed and `cycles` were consumed."""
        self.step_processed += len(jobs)
        self.step_cycles_processed += cycles
        self.total_processed += len(jobs)
        self.total_cycles_processed += cycles
        for job in jobs:
            key = (job.entity_type, job.entity_id)
            self._jobs.append(job)
            self.step_entity_processed[key] += 1
            self.step_entity_cycles_processed[key] += job.compute_size
            self.entity_processed[key] += 1
            self.entity_cycles_processed[key] += job.compute_size
            if job.entity_type == 'UE':
                # Stamp sensor context from n-1 (UE queue is processed before sensor queue)
                sensor_job = self.sensor_latest_job.get(job.nearest_sensor_id)
                if sensor_job is not None:
                    job.sensor_snapshot_at = sensor_job.generated_at
                    job.is_context_synced = True
            elif job.entity_type == 'SENSOR':
                self.sensor_latest_job[job.entity_id] = job

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with one row per job and all lifecycle columns."""
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
                "is_context_synced":  job.is_context_synced,
                "aoi":                job.proc_end_at - job.sensor_snapshot_at if job.sensor_snapshot_at is not None else None,
                "aori":               job.total_latency,
                "aosi":               abs(job.sensor_snapshot_at - job.generated_at) if job.sensor_snapshot_at is not None else None,
            }
            for job in self._jobs
        ]
        return pd.DataFrame(rows)

    def save_log(self, path: str) -> None:
        """Save the job lifecycle log to a CSV file."""
        if dir_path := os.path.dirname(path):
            os.makedirs(dir_path, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

    def __repr__(self) -> str:
        return (
            f"JobTracker(generated={self.total_generated}, "
            f"transmitted={self.total_transmitted}, "
            f"processed={self.total_processed})"
        )