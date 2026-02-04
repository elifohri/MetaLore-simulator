"""
Abstract Association base class for MetaLore.

Defines the interface for entity association models (UE-BS, Sensor-BS, UE-Sensor).
"""

from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

from metalore.core.entities.base_station import BaseStation
from metalore.core.entities.user_equipment import UserEquipment
from metalore.core.entities.sensor import Sensor

Entity = Union[UserEquipment, Sensor]


class Association:

    def __init__(self, **kwargs):
        # Connection storage
        self.connections_ue: Dict[BaseStation, Set[UserEquipment]] = defaultdict(set)
        self.connections_sensor: Dict[BaseStation, Set[Sensor]] = defaultdict(set)

    @abstractmethod
    def reset(self) -> None:
        """Reset all connections and internal state."""
        pass

    @abstractmethod
    def associate_ues_to_bs(self, stations: Dict, users: Dict) -> None:
        """Associate UEs to base stations."""
        pass

    @abstractmethod
    def associate_sensors_to_bs(self, stations: Dict, sensors: Dict) -> None:
        """Associate sensors to base stations."""
        pass

    @abstractmethod
    def update_nearest_sensor(self, users: Dict, sensors: Dict) -> None:
        """Update each UE's nearest_sensor with the closest sensor."""
        pass

    @abstractmethod
    def update_association(self, stations: Dict, users: Dict, sensors: Dict) -> None:
        """Perform full association update cycle."""
        pass

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
