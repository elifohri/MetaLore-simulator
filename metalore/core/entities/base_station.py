"""
Base Station - Cellular tower with Mobile Edge Computing (MEC).

Base stations provide wireless connectivity and edge computing resources.
"""

import math
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING


class BaseStation:

    def __init__(
        self,
        bs_id: int,
        pos: Tuple[float, float],
        bandwidth: float,
        frequency: float,
        tx_power: float,
        height: float,
        compute_capacity: float,
    ) -> None:
        self._id = bs_id
        self._x, self._y  = pos
        self._bandwidth = bandwidth                  # in Hz
        self._frequency = frequency                  # in MHz
        self._tx_power = tx_power                    # in dBm
        self._height = height                        # in meters
        self._compute_capacity = compute_capacity    # in units (CPU cycles per second)
        
    # ===================== Identity =====================

    @property
    def id(self) -> int:
        """Base station identifier."""
        return self._id

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
    
    @property
    def height(self) -> float:
        """Antenna height in meters."""
        return self._height
    
    @property
    def distance_to(self, x: float, y: float) -> float:
        """Calculate Euclidean distance from base station to a point (x, y)."""
        return math.sqrt((self._x - x) ** 2 + (self._y - y) ** 2)
    
    #@property
    #def distance_to_device(self, ue: UserEquipment) -> float:
    #    """Calculate Euclidean distance from base station to a device."""
    #    return self.distance_to(ue.x, ue.y)
    
    # ===================== Resources =====================

    @property
    def bandwidth(self) -> float:
        """Bandwidth in Hz."""
        return self._bandwidth
    
    @property
    def frequency(self) -> float:
        """Operating frequency in MHz."""
        return self._frequency
    
    @property
    def tx_power(self) -> float:
        """Transmission power in dBm."""
        return self._tx_power
    
    @property
    def compute_capacity(self) -> float:
        """Compute capacity in units."""
        return self._compute_capacity
    
    # =====================================================

    def __str__(self) -> str:
        return f"BaseStation(id={self._id})"
    
    def __repr__(self) -> str:
        return(
            f"BaseStation(id={self._id}, position=({self._x}, {self._y}), "
            f"bandwidth={self._bandwidth}MHz, compute_capacity={self._compute_capacity})"
        )