"""
Job Monitor for MetaLore simulation.

Tracks all jobs generated during an episode and exposes their
current state as a pandas DataFrame at any point during the episode.

Since Job objects are mutable, the DataFrame always reflects the
latest state of each job (e.g. None for fields not yet reached).

Usage:
    obs, reward, terminated, truncated, info = env.step(action)
    df = env.monitor.dataframe()   # all jobs generated so far, current state
"""

from typing import List

import pandas as pd

from metalore.core.jobs.job import Job

COLUMNS = [
    "job_id", "entity_id", "entity_type",
    "data_size", "compute_size", "generated_at",
    "is_transmitted", "is_processed",
    "tx_start_at", "tx_end_at",
    "proc_start_at", "proc_end_at",
    "tx_queue_wait", "tx_duration",
    "proc_queue_wait", "proc_duration",
    "nearest_sensor_id", "sensor_snapshot_at",
    "aoi", "aori", "aosi",
]


class JobMonitor:
    """Tracks all generated jobs and snapshots their state on demand."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all tracked jobs for a new episode."""
        self.all_jobs: List[Job] = []

    def on_generated(self, job: Job) -> None:
        """Register a newly generated job."""
        self.all_jobs.append(job)

    def dataframe(self) -> pd.DataFrame:
        """
        Return a DataFrame of all jobs generated this episode with their current state.
        Fields not yet reached (e.g. tx_end_at for a job still transmitting) are None.
        """
        rows = [
            {
                "job_id":             job.id,
                "entity_id":          job.entity_id,
                "entity_type":        job.entity_type,
                "data_size":          job.data_size,
                "compute_size":       job.compute_size,
                "generated_at":       job.generated_at,
                "is_transmitted":     job.is_transmitted,
                "is_processed":       job.is_processed,
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
            for job in self.all_jobs
        ]
        return pd.DataFrame(rows, columns=COLUMNS)

    def save(self, path: str) -> None:
        """Append the current job snapshot to a CSV file."""
        import os
        if dir_path := os.path.dirname(path):
            os.makedirs(dir_path, exist_ok=True)
        write_header = not os.path.exists(path)
        self.dataframe().to_csv(path, mode='a', index=False, header=write_header)
