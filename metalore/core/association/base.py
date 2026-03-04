"""
Abstract Association base class for MetaLore.

Defines the interface for entity association models (UE-BS, Sensor-BS, UE-Sensor).
"""

from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Optional, Set, Union
import numpy as np

from metalore.core.entities.base_station import BaseStation
from metalore.core.entities.user_equipment import UserEquipment
from metalore.core.entities.sensor import Sensor

Entity = Union[UserEquipment, Sensor]


class Association:

    def __init__(
        self,
        seed: int,
        reset_rng_episode: bool,
        **kwargs
    ):
        self.reset_rng_episode = reset_rng_episode
        self.seed = seed
        self.rng = None

        # Mappings of BS to connected UEs and sensors, and UE to nearest sensor
        self.connections_ue: Dict[BaseStation, Set[UserEquipment]] = defaultdict(set)
        self.connections_sensor: Dict[BaseStation, Set[Sensor]] = defaultdict(set)
        self.nearest_sensor: Dict[UserEquipment, Sensor] = {}

    def reset(self) -> None:
        """Reset state after episode ends."""
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def update_association(self, stations: Dict, users: Dict, sensors: Dict) -> None:
        """
        Perform full association update cycle for ues and sensors.

        Args:
            stations: Dictionary mapping station id to BaseStation.
            users: Dictionary mapping UE id to UserEquipment.
            sensors: Dictionary mapping sensor id to Sensor.
        """
        pass

    def get_connected_ues(self, bs: BaseStation) -> Set[UserEquipment]:
        """Return the set of UEs currently connected to bs."""
        return self.connections_ue.get(bs, set())

    def get_connected_sensors(self, bs: BaseStation) -> Set[Sensor]:
        """Return the set of sensors currently connected to bs."""
        return self.connections_sensor.get(bs, set())

    def get_bs_for_entity(self, entity: Entity) -> Optional[BaseStation]:
        """Return the base station the entity is connected to, or None."""
        connections = self.connections_ue if isinstance(entity, UserEquipment) else self.connections_sensor
        for bs, entities in connections.items():
            if entity in entities:
                return bs
        return None