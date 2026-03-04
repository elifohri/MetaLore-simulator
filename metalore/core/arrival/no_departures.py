"""
No Departure Arrival Pattern for MetaLore.

All entities are active from the start and never depart.
This is the default and simplest arrival pattern.
"""

from typing import Dict

from metalore.core.arrival.base import Arrival


class NoDeparture(Arrival):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def arrival(self, entities: Dict) -> None:
        """All entities arrive at timestep 0."""
        for entity in entities.values():
            entity.stime = 0

    def departure(self, entities: Dict) -> None:
        """All entities stay until episode ends."""
        for entity in entities.values():
            entity.extime = self.ep_max_time
