"""
Random Vehicle Count Arrival for MetaLore.

At each episode reset, draws a random number of active vehicles from
[min_vehicles, max_vehicles] (inclusive). A random subset of that size
is selected from the full vehicle pool; the rest are inactive for the
entire episode. All active vehicles arrive at t=0 and stay until episode end.

Usage in config:
    cfg['environment']['arrival_vehicle'] = RandomVehicleCount
    cfg['environment']['num_isac_vehicles'] = MAX_VEHICLES   # upper bound of pool
    cfg['arrival_vehicle_min'] = 10
    cfg['arrival_vehicle_max'] = 20
"""

from typing import Dict
import numpy as np

from metalore.core.arrival.base import Arrival


class RandomVehicleCount(Arrival):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_entities = 10
        self.max_entities = 30
        self.n_active = self.max_entities

    def reset(self) -> None:
        super().reset()
        self.n_active = int(self.rng.integers(self.min_entities, self.max_entities + 1))

    def arrival(self, entities: Dict) -> None:
        """Randomly activate n_active entities; the rest stay inactive."""
        all_ids = list(entities.keys())
        n = min(self.n_active, len(all_ids))
        active_ids = set(
            int(i) for i in self.rng.choice(all_ids, size=n, replace=False)
        )
        for eid, entity in entities.items():
            entity.stime = 0 if eid in active_ids else self.ep_max_time

    def departure(self, entities: Dict) -> None:
        """All entities (active or not) exit at episode end."""
        for entity in entities.values():
            entity.extime = self.ep_max_time
