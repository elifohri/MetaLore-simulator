"""
Closest Entity Association model for MetaLore.

Associates entities to their closest counterpart (UE to closest BS, etc.).
"""

from typing import Dict, List
import numpy as np

from metalore.core.association.base import Association


class ClosestAssociation(Association):
    """Associates entities to their closest counterpart."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self) -> None:
        """Reset for new episode."""
        super().reset()
        self.connections_ue.clear()
        self.connections_sensor.clear()
        self.nearest_sensor.clear()

    @staticmethod
    def _compute_distances(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix between two position arrays."""
        return np.linalg.norm(pos_a[:, np.newaxis, :] - pos_b[np.newaxis, :, :], axis=2)

    def _associate_to_bs(self, bs_list: List, entities: List, connections: Dict) -> None:
        """Associate entities to their closest BS."""
        for bs in connections:
            connections[bs].clear()
            
        if not entities or not bs_list:
            return

        entity_positions = np.array([[e.x, e.y] for e in entities])
        bs_positions = np.array([[bs.x, bs.y] for bs in bs_list])

        distances = self._compute_distances(entity_positions, bs_positions)
        closest_indices = np.argmin(distances, axis=1)

        for entity_idx, bs_idx in enumerate(closest_indices):
            connections[bs_list[bs_idx]].add(entities[entity_idx])

    def _update_nearest_sensor(self, ue_list: List, sensor_list: List) -> None:
        """Populate self.nearest_sensor mapping each UE to its closest sensor."""
        self.nearest_sensor.clear()

        if not ue_list or not sensor_list:
            return

        ue_positions = np.array([[ue.x, ue.y] for ue in ue_list])
        sensor_positions = np.array([[s.x, s.y] for s in sensor_list])

        distances = self._compute_distances(ue_positions, sensor_positions)
        closest_indices = np.argmin(distances, axis=1)

        for ue_idx, sensor_idx in enumerate(closest_indices):
            self.nearest_sensor[ue_list[ue_idx]] = sensor_list[sensor_idx]

    def update_association(self, stations: Dict, users: Dict, sensors: Dict) -> None:
        """
        Perform full association update cycle.

        1. Associate UEs to closest BS
        2. Associate sensors to closest BS
        3. Update each UE's nearest sensor
        """
        self._associate_to_bs(list(stations.values()), list(users.values()), self.connections_ue)
        self._associate_to_bs(list(stations.values()), list(sensors.values()), self.connections_sensor)
        self._update_nearest_sensor(list(users.values()), list(sensors.values()))
