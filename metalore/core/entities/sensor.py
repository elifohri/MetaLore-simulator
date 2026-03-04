"""
Sensor - Stationary entity that collects data from the environment.

Sensors are fixed-position devices that periodically transmit environmental data to maintain digital twin synchronization.
"""

from typing import Optional, Tuple

from metalore.core.jobs.queue import TxQueue


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

        # Transmission queue: holds jobs waiting to be sent to the BS
        self.tx_queue: TxQueue = TxQueue()


    # --- Identity ---

    @property
    def id(self) -> int:
        """Sensor identifier."""
        return self._id

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
    def velocity(self) -> float:
        """Velocity of the sensor."""
        return self._velocity

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
        """Data update interval in timesteps."""
        return self._update_interval
    
    @property
    def is_mobile(self) -> bool:
        """Indicates if the device is mobile."""
        return self._velocity > 0

    def reset_queue(self) -> None:
        """Clear queue state for a new episode."""
        self.tx_queue.clear()

    def __str__(self) -> str:
        return f"Sensor(id={self._id})"
    
    def __repr__(self) -> str:
        return f"Sensor(id={self._id}, position=({self._x}, {self._y}))"