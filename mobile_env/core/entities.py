from typing import Tuple
from shapely.geometry import Point
from mobile_env.core.job_queue import JobQueue


class BaseStation:
    def __init__(
        self,
        bs_id: int,
        pos: Tuple[float, float],
        bw: float,
        freq: float,
        tx: float,
        height: float,
        computational_power: float,
    ):
        # BS ID should be final, i.e., BS ID must be unique
        self.bs_id = bs_id
        self.x, self.y = pos
        self.bw = bw  # in Hz
        self.frequency = freq  # in MHz
        self.tx_power = tx  # in dBm
        self.height = height  # in m
        self.computational_power = computational_power  # units
        self.computation_request_ue: float = 0.0
        self.computation_request_sensor: float = 0.0

        self.transferred_jobs_ue = self._init_job_queue()
        self.transferred_jobs_sensor = self._init_job_queue()
        self.accomplished_jobs_ue = self._init_job_queue()
        self.accomplished_jobs_sensor = self._init_job_queue()

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"BS: {self.bs_id}"

    def _init_job_queue(self) -> JobQueue:
        return JobQueue()
    
    def update_computation_request_ue(self, comp_request: float) -> None:
        self.computation_request_ue += comp_request

    def update_computation_request_sensor(self, comp_request: float) -> None:
        self.computation_request_sensor += comp_request

class UserEquipment:
    def __init__(
        self,
        ue_id: int,
        velocity: float,
        snr_tr: float,
        noise: float,
        height: float,
    ):
        # UE ID should be final, i.e., UE ID must be unique
        # NOTE: cannot use Final typing due to support for Python 3.7
        self.ue_id = ue_id
        self.velocity: float = velocity
        self.snr_threshold = snr_tr
        self.noise = noise
        self.height = height

        self.x: float = None
        self.y: float = None
        self.stime: int = None
        self.extime: int = None
        self.data_buffer_uplink = self._init_job_queue()
        self.total_traffic_request: float = 0.0
        self.total_computation_request: float = 0.0
        self.connected_bs: BaseStation = None
        self.connected_sensor: Sensor = None

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"UE: {self.ue_id}"
    
    def _init_job_queue(self) -> JobQueue:
        return JobQueue()
    
    def update_traffic_requests(self, traffic_request: float) -> None:
        self.total_traffic_request += traffic_request

    def update_computation_requests(self, computation_request: float) -> None:
        self.total_computation_request += computation_request

class Sensor:
    def __init__(
        self,
        sensor_id: int,
        height: float,
        snr_tr: float,
        noise: float,
    ):
        self.sensor_id = sensor_id
        self.x: float = None 
        self.y: float = None
        self.height = height
        self.snr_threshold = snr_tr
        self.noise = noise
        self.data_buffer_uplink = self._init_job_queue()
        self.total_traffic_request: float = 0.0
        self.total_computation_request: float = 0.0
        self.connected_bs: BaseStation = None

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"Sensor: {self.sensor_id}"
    
    def _init_job_queue(self) -> JobQueue:
        return JobQueue()
    
    def update_traffic_requests(self, traffic_request: float) -> None:
        self.total_traffic_request += traffic_request

    def update_computation_requests(self, computation_request: float) -> None:
        self.total_computation_request += computation_request