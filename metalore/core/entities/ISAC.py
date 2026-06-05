from typing import Optional, Tuple
from metalore.core.jobs.queue import TxQueue

class ISACQueues:
    """ISAC Queues with an UE TxQueue and a Sensor TxQueue"""
    def __init__(self, ue_q: TxQueue, sen_q: TxQueue):
        self.ue_q = ue_q
        self.sen_q = sen_q
        
    @property
    def length(self) -> int:
        return self.ue_q.length + self.sen_q.length
        
    @property
    def total_bits(self) -> float:
        return self.ue_q.total_bits + self.sen_q.total_bits
        
    def enqueue(self, job) -> None:
        if job.job_type == 'UE':
            self.ue_q.enqueue(job)
        else:
            self.sen_q.enqueue(job)
            
    def clear(self) -> None:
        self.ue_q.clear()
        self.sen_q.clear()

class ISAC:
    DEVICE_TYPE = 'ISAC'

    def __init__(
        self,
        sensor_id: int,
        velocity: float,
        height: float,
        snr_threshold: float,
        noise: float,
        sensing_range: float,
        update_interval: int,
        isac_range_area: float,
        communication_prob:float,
    ) -> None:
        self._id = sensor_id
        self._x, self._y = None, None
        self._velocity = velocity
        self._height = height
        self._snr_threshold = snr_threshold
        self._noise = noise
        self._sensing_range = sensing_range
        self._communication_prob = communication_prob
        self.isac_range_area = isac_range_area
        self.current_mode = 'UE'
        self._update_interval = update_interval

        # Arrival and departure times
        self.stime: Optional[int] = None
        self.extime: Optional[int] = None

        self.tx_queue_ue: TxQueue = TxQueue()
        self.tx_queue_sensor: TxQueue = TxQueue()
        
        self.tx_queue = ISACQueues(self.tx_queue_ue, self.tx_queue_sensor)
        

    # --- Identity ---
    @property
    def id(self) -> int:
        return self._id

    @property
    def x(self) -> float:
        return self._x
        
    @property
    def y(self) -> float:
        return self._y
        
    @property
    def position(self) -> Tuple[float, float]:
        return (self._x, self._y)
        
    @position.setter
    def position(self, pos: Tuple[float, float]) -> None:
        self._x, self._y = pos

    @property
    def velocity(self) -> float:
        return self._velocity

    @property
    def height(self) -> float:
        return self._height

    @property
    def snr_threshold(self) -> float:
        return self._snr_threshold

    @property
    def noise(self) -> float:
        return self._noise

    @property       
    def sensing_range(self) -> float:
        return self._sensing_range
        
    @property
    def update_interval(self) -> int:
        return self._update_interval
        
    @property
    def is_mobile(self) -> bool:
        return self._velocity > 0

    def reset_queue(self) -> None:
        self.tx_queue.clear()

    def __str__(self) -> str:
        return f"ISAC(id={self._id})"
        
    def __repr__(self) -> str:
        return f"ISAC(id={self._id}, position=({self._x}, {self._y}))"