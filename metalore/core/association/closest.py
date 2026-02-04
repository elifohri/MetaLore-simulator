"""
Closest Entity Association model for MetaLore.

Associates entities to their closest counterpart (UE to closest BS, etc.).
"""

from typing import Dict, List
import numpy as np

from metalore.core.association.base import Association
from metalore.core.channels.base import Channel


class ClosestAssociation(Association):
    """Associates entities to their closest counterpart."""

    def __init__(self, env, channel: Channel, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.channel = channel

    def reset(self) -> None:
        """Reset all connections."""
        self.connections_ue.clear()
        self.connections_sensor.clear()

    def compute_distances(self, pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix between two position arrays."""
        return np.linalg.norm(pos_a[:, np.newaxis, :] - pos_b[np.newaxis, :, :], axis=2)

    # --- Association ---

    def associate_entities_to_bs(self, entities: List, positions: np.ndarray, connections: Dict) -> None:
        """Associate entities to their closest BS."""
        if not entities or not self.env.stations:
            return

        bs_list = list(self.env.stations.values())
        bs_positions = np.array([[bs.x, bs.y] for bs in bs_list])

        # distances[i, j] = distance from entity i to BS j
        distances = self.compute_distances(positions, bs_positions)

        # For each entity, find the index of the closest BS
        closest_indices = np.argmin(distances, axis=1)

        for bs in connections:
            connections[bs].clear()

        # Map each entity to its closest BS
        for entity_idx, bs_idx in enumerate(closest_indices):
            entity = entities[entity_idx]
            bs = bs_list[bs_idx]
            connections[bs].add(entity)

    def associate_ues_to_bs(self) -> None:
        """Associate each UE to the closest BS."""
        ue_list = list(self.env.users.values())
        ue_positions = np.array([[ue.x, ue.y] for ue in ue_list])
        self.associate_entities_to_bs(ue_list, ue_positions, self.connections_ue)

    def associate_sensors_to_bs(self) -> None:
        """Associate each sensor to its closest BS."""
        sensor_list = list(self.env.sensors.values())
        sensor_positions = np.array([[s.x, s.y] for s in sensor_list])
        self.associate_entities_to_bs(sensor_list, sensor_positions, self.connections_sensor)

    def update_nearest_sensor(self) -> None:
        """Update each UE's nearest_sensor with the closest sensor."""
        ue_list = list(self.env.users.values())
        sensor_list = list(self.env.sensors.values())

        if not ue_list or not sensor_list:
            return

        ue_positions = np.array([[ue.x, ue.y] for ue in ue_list])
        sensor_positions = np.array([[s.x, s.y] for s in sensor_list])

        distances = self.compute_distances(ue_positions, sensor_positions)
        closest_indices = np.argmin(distances, axis=1)

        for ue_idx, sensor_idx in enumerate(closest_indices):
            ue_list[ue_idx].nearest_sensor = sensor_list[sensor_idx]

    # --- Main Update ---

    def update_association(self) -> None:
        """
        Perform full association update cycle.

        1. Associate UEs to closest BS
        2. Associate sensors to closest BS
        3. Update each UE's nearest sensor
        """
        self.associate_ues_to_bs()
        self.associate_sensors_to_bs()
        self.update_nearest_sensor()
