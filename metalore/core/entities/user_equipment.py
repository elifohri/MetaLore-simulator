"""
User Equipment (UE) - Mobile devices in the network.

UEs move around the simulation area, generate service requests and connect to base stations for communication.
"""

from typing import Optional, Tuple

from metalore.core.jobs.queue import TxQueue


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

        # Arrival and departure times
        self.stime: Optional[int] = None
        self.extime: Optional[int] = None

        # Transmission queue: holds jobs waiting to be sent to the BS
        self.tx_queue: TxQueue = TxQueue()


    # --- Identity ---

    @property
    def id(self) -> int:
        """User equipment identifier."""
        return self._id

    @property
    def x(self) -> float:
        return self._x
    
    @property
    def y(self) -> float:
        return self._y
    
    @property
    def position(self) -> Tuple[float, float]:
        """User equipment position as (x, y) coordinates."""
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
    def is_mobile(self) -> bool:
        """Indicates if the device is mobile."""
        return self._velocity > 0
    
    def reset_queue(self) -> None:
        """Clear queue state for a new episode."""
        self.tx_queue.clear()

    def __str__(self) -> str:
        return f"UE(id={self._id})"
        
    def __repr__(self) -> str:
        return f"UE(id={self._id}, position=({self._x}, {self._y}))"