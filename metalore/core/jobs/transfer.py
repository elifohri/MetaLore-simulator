"""
Wireless Transmission logic for MetaLore simulation.

transmit() drains bits from the head of a TxQueue at the datarate given by the wireless channel.
Supports partial transmission: a large job may span multiple timesteps. 
Completed jobs are returned so can be forwardedto the BS processing queue.
"""

from typing import List, Tuple

from metalore.core.jobs.job import Job
from metalore.core.jobs.queue import TxQueue


def transmit(queue: TxQueue, datarate: float, timestep: int) -> Tuple[float, List[Job]]:
    """
    Uses the available channel capacity (datarate) to send bits from the head of the TxQueue.

    Args:
        queue:    The entity's transmission queue.
        datarate: Channel data rate in allocated to this entity.
        timestep: Current simulation timestep (used for job lifecycle timestamps).

    Returns:
        bits_sent:      Total bits transmitted this timestep.
        completed_jobs: List of Job objects fully transmitted, ready for processing.
    """

    bits_sent = 0.0
    completed: List[Job] = []

    while queue.length > 0 and datarate > 0.0:
        job = queue.head()

        if job.tx_start_at is None:
            job.tx_start_at = timestep

        drain = min(job.bits_remaining, datarate)
        job.bits_remaining -= drain
        bits_sent += drain
        datarate -= drain

        if job.bits_remaining <= 0.0:
            job.tx_end_at = timestep
            job.is_transmitted = True
            queue.dequeue()
            completed.append(job)
        else:
            break  # capacity exhausted, job continues next timestep

    return bits_sent, completed