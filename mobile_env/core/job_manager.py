import logging
import numpy as np
from typing import Dict, Optional, Union
from mobile_env.core.entities import UserEquipment, Sensor
from mobile_env.core.constants import SENSOR, USER_DEVICE, UE_JOB, SENSOR_JOB, COMM_LAMBDA, COMP_LAMBDA, PROBABILITY


Device = Union[UserEquipment, Sensor]
Job = Dict[str, Optional[Union[float, int]]]

class JobGenerationManager:
    """
    The JobGenerationManager class is responsible for generating computation and communication jobs
    for user equipment (UE) and sensor devices in the simulation environment.
    """

    def __init__(self, env) -> None:
        self.env = env
        self.logger = env.logger
        self.config = env.default_config()
        self.job_counter: int = 0

    def _generate_index(self) -> int:
        self.job_counter += 1
        return self.job_counter

    def _generate_request(self, job_type: str, key: str) -> float:
        poisson_lambda = self.config[job_type][key]
        return float(max(np.random.poisson(lam=poisson_lambda), 1.0))

    def _generate_job(self, time: float, device_id: int, device_type: str, job_type: str) -> Job:
        job_index = self._generate_index()
        communication_request_size = self._generate_request(job_type, COMM_LAMBDA)
        computation_request_size = self._generate_request(job_type, COMP_LAMBDA)

        job = {
            'packet_id': job_index,
            'device_type': device_type,
            'device_id': device_id,
            'computation_request': computation_request_size,
            'initial_request_size': communication_request_size,
            'remaining_request_size': communication_request_size,
            'creation_time': time,
            'transfer_time_start': None,
            'transfer_time_end': None,
            'total_transfer_time': None,
            'processing_time_start': None,
            'processing_time_end': None,
            'total_processing_time': None,
            'total_accomplishment_time': None,
            'device_queue_waiting_time': None,
            'bs_queue_waiting_time': None,
        }

        return job
    
    def generate_job_ue(self, ue: UserEquipment) -> None:
        if  np.random.rand() < self.config[UE_JOB][PROBABILITY]:
            job = self._generate_job(self.env.time, ue.ue_id, USER_DEVICE, UE_JOB)
            ue.data_buffer_uplink.enqueue_job(job)
            self.log_generated_job(job)

    def generate_job_sensor(self, sensor: Sensor) -> None:
        job = self._generate_job(self.env.time, sensor.sensor_id, SENSOR, SENSOR_JOB)
        sensor.data_buffer_uplink.enqueue_job(job)
        self.log_generated_job(job)
            
    def log_generated_job(self, job: Job) -> None:
        self.logger.log_simulation(
            f"Time step: {self.env.time} Job generated: {job['packet_id']} by {job['device_type']} {job['device_id']} "
            f"with initial request of size {job['initial_request_size']} MB and computational request of {job['computation_request']} units"
        )