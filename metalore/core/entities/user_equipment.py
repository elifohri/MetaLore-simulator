"""
User Equipment (UE) - Mobile devices in the network.

UEs move around the simulation area, generate service requests, and connect to base stations for communication.
"""

import math
from typing import Tuple


class UserEquipment:

    DEVICE_TYPE = 'UE'
    
    def __init__(
        self,
        ue_id: int,
        velocity: float,
        height: float,
        snr_threshold: float,
        noise: float,
    ) -> None:
        self._id = ue_id
        self._x, self._y = None, None
        self._velocity = velocity
        self._height = height
        self._snr_threshold = snr_threshold
        self._noise = noise

        # Associated sensor for synchronization
        #self._connected_sensor: 'Sensor' = None


    # ===================== Identity ========================

    @property
    def id(self) -> int:
        """User equipment identifier."""
        return self._id
        
    @property
    def device_type(self) -> str:
        return self.DEVICE_TYPE
    
    # ===================== Position =====================

    @property
    def x(self) -> float:
        return self._x
    
    @property
    def y(self) -> float:
        return self._y
    
    @property
    def position(self) -> Tuple[float, float]:
        """Base station position as (x, y) coordinates."""
        return (self._x, self._y)
    
    @position.setter
    def position(self, pos: Tuple[float, float]) -> None:
        """Set position as (x, y) coordinates."""
        self._x, self._y = pos
    
    @property
    def distance_to(self, x: float, y: float) -> float:
        """Calculate Euclidean distance from device to a point (x, y)."""
        return math.sqrt((self._x - x) ** 2 + (self._y - y) ** 2)
    
    #@property
    #def distance_to_base_station(self, base_station: BaseStation) -> float:
    #    """Calculate Euclidean distance from device to a base station."""
    #    return self.distance_to(base_station.x, base_station.y)
    
    #@property
    #def distance_to_sensor(self, sensor: Sensor) -> float:
    #    """Calculate Euclidean distance from device to another device or sensor."""
    #    return self.distance_to(sensor.x, sensor.y)

    # ===================== Physical ========================

    @property
    def height(self) -> float:
        """Antenna height in meters."""
        return self._height

    @property
    def snr_threshold(self) -> float:
        """Minimum SNR for connectivity."""
        return self._snr_threshold

    @property
    def noise(self) -> float:
        """Receiver noise power in Watts."""
        return self._noise

    # ===================== Mobility ========================

    @property
    def velocity(self) -> float:
        """Device velocity in m/s."""
        return self._velocity
    
    @property
    def is_mobile(self) -> bool:
        """Indicates if the device is mobile."""
        return self._velocity > 0
    
    # ===================== Sensor Connection ===============

    #@property
    #def connected_sensor(self) -> Optional['Sensor']:
    #    """The sensor to which this UE is currently connected, if any."""
    #    return self._connected_sensor
    
    #@connected_sensor.setter
    #def connected_sensor(self, sensor: Optional['Sensor']) -> None:
    #    self._connected_sensor = sensor

    # ===================== Reset ===========================

    #def reset(Self) -> None:
    #    """Reset UE state for a new simulation episode."""
    #    self._connected_sensor = None

    # =======================================================

    def __str__(self) -> str:
        return f"UE-{self._id}"
        
    def __repr__(self) -> str:
        return f"UserEquipment(id={self._id}, pos=({self._x, self._y}))"