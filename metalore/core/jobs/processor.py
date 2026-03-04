"""
MEC Processing logic for MetaLore simulation.

process() consumes CPU cycles from a ProcessQueue at the rate given by allocated compute capacity.
Supports partial processing: a compute-heavy job may span multiple timesteps. 
Completed jobs are returned for tracking.
"""

from typing import List, Tuple

from metalore.core.jobs.job import Job
from metalore.core.jobs.queue import ProcessQueue


def process(queue: ProcessQueue, compute_capacity: float, timestep: int) -> Tuple[float, List[Job]]:
    """
    Process jobs from the MEC queue for one timestep.

    Args:
        queue:            The BS's processing queue (UE or sensor side).
        compute_capacity: Compute rate in CPU cycles/second allocated to this queue.
        timestep:         Current simulation timestep (used for job lifecycle timestamps).

    Returns:
        cycles_consumed:  Total CPU cycles consumed this timestep.
        completed_jobs:   List of Job objects whose cycles_remaining reached 0
                          (fully processed, end-to-end lifecycle complete).
    """

    cycles_consumed = 0.0
    completed: List[Job] = []

    while queue.length > 0 and compute_capacity > 0.0:
        job = queue.head()

        if job.proc_start_at is None:
            job.proc_start_at = timestep

        drain = min(job.cycles_remaining, compute_capacity)
        job.cycles_remaining -= drain
        cycles_consumed += drain
        compute_capacity -= drain

        if job.cycles_remaining <= 0.0:
            job.proc_end_at = timestep
            job.is_processed = True
            queue.dequeue()
            completed.append(job)
        else:
            break  # capacity exhausted, job continues next timestep

    return cycles_consumed, completed