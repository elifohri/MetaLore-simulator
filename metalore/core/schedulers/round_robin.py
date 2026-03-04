"""
Round Robin Scheduler for MetaLore.

Allocates resources in a round-robin fashion with configurable quantum.
"""

from typing import List, Dict

from metalore.core.schedulers.base import Scheduler
from metalore.core.entities.base_station import BaseStation


class RoundRobin(Scheduler):

    def __init__(self, quantum: float = 1.0, offset: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.quantum = quantum
        self.offset = offset
        self.last_served_index: Dict[int, int] = {}

    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.last_served_index.clear()

    def share(self, bs: BaseStation, conns: List, total_resources: float) -> List[float]:
        """Allocate bandwidth using round-robin."""
        num_devices = len(conns)

        if num_devices == 0:
            return []

        # Initialize last served index for this BS
        if bs.id not in self.last_served_index:
            self.last_served_index[bs.id] = -1

        allocation = [0.0] * num_devices
        remaining_resources = total_resources

        # Update starting index for fairness
        self.last_served_index[bs.id] = (self.last_served_index[bs.id] + self.offset) % num_devices
        start_index = self.last_served_index[bs.id]

        while remaining_resources > 0:
            done = True
            for i in range(num_devices):
                current_index = (start_index + i) % num_devices

                if remaining_resources <= 0:
                    break

                done = False
                alloc_amount = min(self.quantum, remaining_resources)
                allocation[current_index] += alloc_amount
                remaining_resources -= self.quantum

            if done:
                break

        return allocation