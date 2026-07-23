"""
ISACVehicle - Automated vehicle with a single RF chain for sensing or communication.

In each slot the vehicle operates in one of two modes:
  - Sensing mode: observes its current sector and uploads the observation.
  - Communication mode: issues a service request for its target sector ahead.
"""

from typing import Optional, Tuple

from metalore.core.jobs.queue import TxQueue


class ISACVehicle:

    DEVICE_TYPE = 'ISAC'

    def __init__(
        self,
        vehicle_id: int,
        velocity: float,
        height: float,
        snr_threshold: float,
        noise: float,
        sensing_range: float,
    ) -> None:
        self._id = vehicle_id
        self._x, self._y = None, None
        self._velocity = velocity
        self._height = height
        self._snr_threshold = snr_threshold
        self._noise = noise
        self._sensing_range = sensing_range

        # Arrival and departure times
        self.stime: Optional[int] = None
        self.extime: Optional[int] = None

        # Single transmission queue shared by sensing observations and service requests
        self.tx_queue: TxQueue = TxQueue()

        # Mode set once per time slot (True = sense, False = communicate)
        self.sensing_mode: bool = False


    # --- Identity ---

    @property
    def id(self) -> int:
        """ISAC vehicle identifier."""
        return self._id

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def position(self) -> Tuple[float, float]:
        """ISAC vehicle position as (x, y) coordinates."""
        return (self._x, self._y)

    @position.setter
    def position(self, pos: Tuple[float, float]) -> None:
        """Set position as (x, y) coordinates."""
        self._x, self._y = pos

    @property
    def velocity(self) -> float:
        """Device velocity."""
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
        return self._sensing_range

    @property
    def is_mobile(self) -> bool:
        """Indicates if the device is mobile."""
        return self._velocity > 0

    @property
    def sensing_mode(self) -> bool:
        """True when the vehicle is in sensing mode, False for communication."""
        return self._sensing_mode

    @sensing_mode.setter
    def sensing_mode(self, value: bool) -> None:
        """Set the vehicle's mode for the current time slot."""
        self._sensing_mode = value

    def reset_queue(self) -> None:
        """Clear queue state for a new episode."""
        self.tx_queue.clear()

    def __str__(self) -> str:
        return f"ISACVehicle(id={self._id})"

    def __repr__(self) -> str:
        mode = "sense" if self._sensing_mode else "comm"
        return (f"ISACVehicle(id={self._id}, position=({self._x}, {self._y}), mode={mode})")