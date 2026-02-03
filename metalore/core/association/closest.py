"""
Closest Entity Association model for MetaLore.

Associates entities to their closest counterpart (UE to closest BS, etc.).
"""

from typing import Dict, Set, Optional, List, Union
import numpy as np

from metalore.core.association.base import Association
from metalore.core.channels.okumura_hata import OkumuraHata
from metalore.core.entities.base_station import BaseStation
from metalore.core.entities.user_equipment import UserEquipment
from metalore.core.entities.sensor import Sensor

Entity = Union[UserEquipment, Sensor]


class ClosestAssociation(Association):
    """Associates entities to their closest counterpart."""

    def __init__(self, env, channel: OkumuraHata):
        super().__init__(env)
        self.channel = channel

    def reset(self) -> None:
        """Reset all connections."""
        self.connections_ue.clear()
        self.connections_sensor.clear()

    # --- Properties ---

    @property
    def bs_list(self) -> List[BaseStation]:
        return list(self.env.stations.values())

    @property
    def ue_list(self) -> List[UserEquipment]:
        return list(self.env.users.values())

    @property
    def sensor_list(self) -> List[Sensor]:
        return list(self.env.sensors.values())

    @property
    def bs_positions(self) -> np.ndarray:
        return np.array([[bs.x, bs.y] for bs in self.bs_list])

    @property
    def ue_positions(self) -> np.ndarray:
        return np.array([[ue.x, ue.y] for ue in self.ue_list])

    @property
    def sensor_positions(self) -> np.ndarray:
        return np.array([[s.x, s.y] for s in self.sensor_list])

    # --- Distance Computation ---

    def _compute_distances(self, pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix between two position arrays."""
        return np.linalg.norm(pos_a[:, np.newaxis, :] - pos_b[np.newaxis, :, :], axis=2)

    # --- Connectivity ---

    def check_connectivity(self, bs: BaseStation, entity: Entity) -> bool:
        """Check if entity-BS connection is valid based on SNR threshold."""
        snr = self.channel.snr(bs, entity)
        return snr > entity.snr_threshold

    # --- Association ---

    def _associate_entities_to_bs(
        self,
        entities: List,
        positions: np.ndarray,
        connections: Dict,
        set_connected_bs: bool = False
    ) -> None:
        """Associate entities to their closest BS."""
        if not entities or not self.env.stations:
            return

        bs_list = self.bs_list

        # distances[i, j] = distance from entity i to BS j
        distances = self._compute_distances(positions, self.bs_positions)

        # For each entity, find the index of the closest BS
        closest_indices = np.argmin(distances, axis=1)

        for bs in connections:
            connections[bs].clear()

        # Map each entity to its closest BS
        for entity_idx, bs_idx in enumerate(closest_indices):
            entity = entities[entity_idx]
            bs = bs_list[bs_idx]
            if set_connected_bs:
                entity.connected_bs = bs
            connections[bs].add(entity)

    def associate_ues_to_bs(self) -> None:
        """Associate each UE to the closest BS."""
        self._associate_entities_to_bs(self.ue_list, self.ue_positions, self.connections_ue)

    def associate_sensors_to_bs(self) -> None:
        """Associate each sensor to its closest BS."""
        self._associate_entities_to_bs(self.sensor_list, self.sensor_positions, self.connections_sensor, set_connected_bs=True)

    def associate_ues_to_sensors(self) -> None:
        """Associate each UE to its closest sensor."""
        ue_list = self.ue_list
        sensor_list = self.sensor_list

        if not ue_list or not sensor_list:
            return

        distances = self._compute_distances(self.ue_positions, self.sensor_positions)
        closest_indices = np.argmin(distances, axis=1)

        for ue_idx, sensor_idx in enumerate(closest_indices):
            ue_list[ue_idx].connected_sensor = sensor_list[sensor_idx]

    # --- Validation ---

    def _validate_connections(self, connections: Dict) -> None:
        """Filter connections based on SNR threshold."""
        # For each BS, keep only entities whose SNR exceeds their threshold
        updated = {
            bs: {entity for entity in entities if self.check_connectivity(bs, entity)}
            for bs, entities in connections.items()
        }
        # Replace old connections with the filtered ones
        connections.clear()
        connections.update(updated)

    def validate_connections(self) -> None:
        """Validate all connections based on SNR threshold."""
        self._validate_connections(self.connections_ue)
        self._validate_connections(self.connections_sensor)

    # --- Queries ---

    def get_connected_ues(self, bs: BaseStation) -> Set[UserEquipment]:
        """Get all UEs connected to a base station."""
        return self.connections_ue.get(bs, set())

    def get_connected_sensors(self, bs: BaseStation) -> Set[Sensor]:
        """Get all sensors connected to a base station."""
        return self.connections_sensor.get(bs, set())

    def get_bs_for_entity(self, entity: Entity) -> Optional[BaseStation]:
        """Get the base station an entity is connected to."""
        connections = self.connections_ue if isinstance(entity, UserEquipment) else self.connections_sensor
        for bs, entities in connections.items():
            if entity in entities:
                return bs
        return None

    def get_num_connections(self) -> Dict[str, int]:
        """Get total number of connections by type."""
        return {
            'ue': sum(len(ues) for ues in self.connections_ue.values()),
            'sensor': sum(len(sensors) for sensors in self.connections_sensor.values())
        }

    def get_available_bs(self, entity: Entity) -> Set[BaseStation]:
        """Get set of base stations an entity could connect to."""
        return {bs for bs in self.env.stations.values() if self.check_connectivity(bs, entity)}

    # --- Main Update ---

    def update_all(self) -> None:
        """
        Perform full association update cycle.

        1. Associate UEs to closest BS
        2. Associate sensors to closest BS
        3. Associate UEs to closest sensors
        4. Validate connections based on SNR
        """
        self.associate_ues_to_bs()
        self.associate_sensors_to_bs()
        self.associate_ues_to_sensors()
        self.validate_connections()
