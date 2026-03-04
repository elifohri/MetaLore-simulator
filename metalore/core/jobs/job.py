"""
Job - Represents a single data packet/task in the simulation.

A job follows two phases:
  1. Transmission: sent from the entity to the BS over the wireless channel.
  2. Processing: computed at the MEC server.
"""

from typing import Optional


class Job:

    def __init__(
        self,
        job_id: int,
        entity_id: int,
        entity_type: str,
        data_size: float,
        compute_size: float,
        generated_at: int,
    ) -> None:
        self._id = job_id
        self._entity_id = entity_id
        self._entity_type = entity_type
        self._data_size = data_size
        self._compute_size = compute_size
        self._generated_at = generated_at

        # Tracks how much work is left as the job progresses
        self.bits_remaining: float = data_size
        self.cycles_remaining: float = compute_size

        # Tracks whether each phase has completed
        self.is_transmitted: bool = False
        self.is_processed: bool = False

        # Tracks when each phase started and ended
        self.tx_start_at: Optional[int] = None
        self.tx_end_at: Optional[int] = None
        self.proc_start_at: Optional[int] = None
        self.proc_end_at: Optional[int] = None


    @property
    def id(self) -> int:
        """Unique job identifier."""
        return self._id

    @property
    def entity_id(self) -> int:
        """ID of the entity that generated this job."""
        return self._entity_id

    @property
    def entity_type(self) -> str:
        """Device type of the generating entity ('UE' or 'SENSOR')."""
        return self._entity_type

    @property
    def data_size(self) -> float:
        """Total bits to transmit over the wireless channel."""
        return self._data_size

    @property
    def compute_size(self) -> float:
        """Total CPU cycles required for MEC processing."""
        return self._compute_size

    @property
    def generated_at(self) -> int:
        """Simulation timestep when the job was created."""
        return self._generated_at

    @property
    def tx_queue_wait(self) -> Optional[int]:
        """Timesteps spent waiting in the UE tx queue before transmission started."""
        if self.tx_start_at is None:
            return None
        return self.tx_start_at - self._generated_at

    @property
    def tx_duration(self) -> Optional[int]:
        """Timesteps spent transmitting to the BS."""
        if self.tx_start_at is None or self.tx_end_at is None:
            return None
        return self.tx_end_at - self.tx_start_at

    @property
    def proc_queue_wait(self) -> Optional[int]:
        """Timesteps spent waiting in the BS queue before processing started."""
        if self.tx_end_at is None or self.proc_start_at is None:
            return None
        return self.proc_start_at - self.tx_end_at

    @property
    def proc_duration(self) -> Optional[int]:
        """Timesteps spent processing at the MEC server."""
        if self.proc_start_at is None or self.proc_end_at is None:
            return None
        return self.proc_end_at - self.proc_start_at

    @property
    def total_latency(self) -> Optional[int]:
        """Total end-to-end latency from generation to processing complete."""
        if self.proc_end_at is None:
            return None
        return self.proc_end_at - self._generated_at

    def __repr__(self) -> str:
        return (
            f"Job(id={self._id}, entity={self._entity_type}:{self._entity_id}, "
            f"data={self._data_size:.1f}, compute={self._compute_size:.1f}, "
            f"t={self._generated_at})"
        )
