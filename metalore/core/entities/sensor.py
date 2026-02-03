"""
Sensor - Stationary entity that collects data from the environment.

Sensors are fixed-position devices that periodically transmit environmental data to maintain digital twin synchronization.
"""

from typing import Optional, Tuple


class Sensor:

    DEVICE_TYPE = 'SENSOR'

    def __init__(
        self,
        sensor_id: int,
        velocity: float,
        height: float,
        snr_threshold: float,
        noise: float,
        sensing_range: float,
        update_interval: int,
    ) -> None:
        self._id = sensor_id
        self._x, self._y = None, None
        self._velocity = velocity
        self._height = height
        self._snr_threshold = snr_threshold
        self._noise = noise
        self._sensing_range = sensing_range
        self._update_interval = update_interval

        # Arrival and departure times
        self.stime: Optional[int] = None
        self.extime: Optional[int] = None


    # --- Identity ---

    @property
    def id(self) -> int:
        """Sensor identifier."""
        return self._id
    
    @property
    def device_type(self) -> str:
        return self.DEVICE_TYPE
    
    @property
    def x(self) -> float:
        return self._x
    
    @property
    def y(self) -> float:
        return self._y
    
    @property
    def position(self) -> Tuple[float, float]:
        """Sensor position as (x, y) coordinates."""
        return (self._x, self._y)
    
    @position.setter
    def position(self, pos: Tuple[float, float]) -> None:
        """Set position as (x, y) coordinates."""
        self._x, self._y = pos

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

    @property   
    def sensing_range(self) -> float:
        """Sensing range in meters."""
        return self._sensing_range
    
    @property
    def update_interval(self) -> int:
        """Data update interval in seconds."""
        return self._update_interval    

    @property
    def velocity(self) -> float:
        """Velocity of the sensor in meters/second."""
        return self._velocity

    def __str__(self) -> str:
        return f"Sensor(id={self._id})"
    
    def __repr__(self) -> str:
        return f"Sensor(id={self._id}, position=({self._x}, {self._y}))"