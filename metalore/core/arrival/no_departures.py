"""
No Departure Arrival Pattern for MetaLore.

All entities are active from the start and never depart.
This is the default/simplest arrival pattern.
"""

from metalore.core.arrival.base import Arrival


class NoDeparture(Arrival):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def arrival(self, entity) -> int:
        """All entities arrive at timestep 0."""
        return 0
    
    def departure(self, entity) -> int:
        """All entities stay until episode ends."""
        return self.ep_time