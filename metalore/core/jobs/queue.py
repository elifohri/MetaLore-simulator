"""
Job Queues for MetaLore simulation.

Two queue types model the two-phase lifecycle of a job:
  1. TxQueue:      Per-entity FIFO queue for packets awaiting wireless transmission.
  2. ProcessQueue: Per-BS FIFO queue for packets awaiting MEC computation.
"""

from collections import deque
from typing import Optional

from metalore.core.jobs.job import Job


class JobQueue:
    """Base FIFO queue for jobs."""

    def __init__(self) -> None:
        self._q: deque[Job] = deque()

    def enqueue(self, job: Job) -> None:
        """Add a new job to the tail of the queue."""
        self._q.append(job)

    def head(self) -> Optional[Job]:
        """Peek at the job at the head without removing it."""
        return self._q[0] if self._q else None

    def dequeue(self) -> Job:
        """Remove and return the job at the head."""
        return self._q.popleft()

    def clear(self) -> None:
        """Discard all pending jobs."""
        self._q.clear()

    @property
    def length(self) -> int:
        """Number of jobs in the queue."""
        return len(self._q)


class TxQueue(JobQueue):
    """
    FIFO transmission queue at a UE/Sensor.

    Holds jobs waiting to be transmitted over the wireless channel.
    """

    @property
    def total_bits(self) -> float:
        """Sum of bits_remaining across all queued jobs."""
        return sum(j.bits_remaining for j in self._q)

    def __repr__(self) -> str:
        return f"TxQueue(length={self.length}, bits={self.total_bits:.1f})"


class ProcessQueue(JobQueue):
    """
    FIFO processing queue at a BS/MEC server.

    Holds jobs waiting for compute resources.
    """

    @property
    def total_cycles(self) -> float:
        """Sum of cycles_remaining across all queued jobs."""
        return sum(j.cycles_remaining for j in self._q)

    def __repr__(self) -> str:
        return f"ProcessQueue(length={self.length}, cycles={self.total_cycles:.1f})"
