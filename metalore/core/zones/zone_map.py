"""
ZoneMap - Partitions the simulation area into a grid and tracks Age of Zone Information (AoZI).

AoZ for zone (i, j) at time t = t - last_sensed[i, j].
A zone is sensed when an ISAC vehicle in sensing mode is within its sensing radius.
"""

from typing import List, Tuple
import numpy as np


class ZoneMap:

    def __init__(self, width: float, height: float, num_zones_x: int, num_zones_y: int):
        self.width = width
        self.height = height
        self.num_zones_x = num_zones_x
        self.num_zones_y = num_zones_y
        self.zone_w = width / num_zones_x
        self.zone_h = height / num_zones_y
        self.last_sensed = np.full((num_zones_x, num_zones_y), -np.inf)          # Last timestep when each zone was sensed (-inf = never)

    def reset(self) -> None:
        self.last_sensed.fill(-np.inf)

    def zone_center(self, i: int, j: int):
        """Return (x, y) center of zone (i, j)."""
        return (i + 0.5) * self.zone_w, (j + 0.5) * self.zone_h

    def covered_zones(self, x: float, y: float, sensing_range: float) -> List[Tuple[int, int]]:
        """Return zone indices whose center falls within sensing_range of (x, y)."""
        zones = []
        for i in range(self.num_zones_x):
            for j in range(self.num_zones_y):
                cx, cy = self.zone_center(i, j)
                if (x - cx) ** 2 + (y - cy) ** 2 <= sensing_range ** 2:
                    zones.append((i, j))
        return zones

    def update_zone_freshness(self, zone: Tuple[int, int], observed_at: int) -> None:
        """Update last_sensed for a zone using the timestep when the observation was taken."""
        i, j = zone
        self.last_sensed[i, j] = max(self.last_sensed[i, j], observed_at)

    def get_zone(self, x: float, y: float) -> tuple:
        """Return (i, j) zone index for position (x, y)."""
        i = min(int(x / self.zone_w), self.num_zones_x - 1)
        j = min(int(y / self.zone_h), self.num_zones_y - 1)
        return (i, j)

    def initialize_sensor_coverage(self, entities) -> None:
        """Set last_sensed to 0 for all zones covered by the given entities at their current positions."""
        for entity in entities:
            for zone in self.covered_zones(entity.x, entity.y, entity.sensing_range):
                i, j = zone
                self.last_sensed[i, j] = 0

    def aozi(self, timestep: int) -> np.ndarray:
        """Return (num_zones_x, num_zones_y) array of AoZI values."""
        return timestep - self.last_sensed
