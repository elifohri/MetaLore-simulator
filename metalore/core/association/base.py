"""
Abstract Association base class for MetaLore.

Defines the interface for entity association models (UE-BS, Sensor-BS, UE-Sensor).
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Set, Optional

from metalore.core.entities.base_station import BaseStation
from metalore.core.entities.user_equipment import UserEquipment
from metalore.core.entities.sensor import Sensor


class Association(ABC):
    def __init__(self, env):
        self.env = env

        # Connection storage
        self.connections_ue: Dict[BaseStation, Set[UserEquipment]] = defaultdict(set)
        self.connections_sensor: Dict[BaseStation, Set[Sensor]] = defaultdict(set)

    @abstractmethod
    def reset(self) -> None:
        """Reset all connections and internal state."""
        pass

    @abstractmethod
    def associate_ues_to_bs(self) -> None:
        """Associate UEs to base stations."""
        pass

    @abstractmethod
    def associate_sensors_to_bs(self) -> None:
        """Associate sensors to base stations."""
        pass

    @abstractmethod
    def associate_ues_to_sensors(self) -> None:
        """Associate UEs to sensors."""
        pass

    @abstractmethod
    def update_all(self) -> None:
        """Perform full association update cycle."""
        pass

    @abstractmethod
    def get_connected_ues(self, bs: BaseStation) -> Set[UserEquipment]:
        """Get all UEs connected to a base station."""
        pass

    @abstractmethod
    def get_connected_sensors(self, bs: BaseStation) -> Set[Sensor]:
        """Get all sensors connected to a base station."""
        pass

    @abstractmethod
    def get_num_connections(self) -> Dict[str, int]:
        """Get total number of connections by type."""
        pass
