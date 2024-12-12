import numpy as np
import pandas as pd
import random
from typing import Dict, Optional, Union
from mobile_env.core.entities import UserEquipment, Sensor

# Type alias for job data
Job = Dict[str, Optional[Union[float, int]]]

# Constants for device types
SENSOR = 'sensor'
USER_DEVICE = 'user_device'

class JobGenerator:
    def __init__(self, env) -> None:
        self.env = env
        self.logger = env.logger
        self.config = env.default_config()
        self.job_counter: int = 0
        self.packet_df_ue = pd.DataFrame(columns=[
            'packet_id', 'device_type', 'device_id', 'is_transferred', 'is_accomplished', 
            'creation_time', 'arrival_time', 'accomplished_time', 'e2e_delay_threshold'
        ])
        self.packet_df_sensor = pd.DataFrame(columns=[
            'packet_id', 'device_type', 'device_id', 'is_transferred','is_accomplished', 
            'creation_time', 'arrival_time', 'accomplished_time', 'e2e_delay_threshold'
        ])

    def _generate_index(self) -> int:
        # Generate a unique index for each job.
        self.job_counter += 1
        return self.job_counter

    def _generate_communication_request(self, device_type: str) -> float:
        # Generate data size for communication request based on device type
        if device_type == SENSOR:
            poisson_lambda = self.config["sensor_job"]["communication_job_lambda_value"]
        elif device_type == USER_DEVICE:
            poisson_lambda = self.config["ue_job"]["communication_job_lambda_value"]  
        else:
            raise ValueError("Unknown device category. Expected 'sensor' or 'user_device'.")
    
        # Ensure a non-zero value for request size
        return float(max(np.random.poisson(lam=poisson_lambda), 1.0))      # Ensure a non-zero value (1 MB) for request size
    
    def _generate_computation_request(self, device_type: str) -> float:
        # Generate the computational requirement for a device based on its category.
        if device_type == SENSOR:
            poisson_lambda = self.config["sensor_job"]["computation_job_lambda_value"]
        elif device_type == USER_DEVICE:
            poisson_lambda = self.config["ue_job"]["computation_job_lambda_value"]
        else:
            raise ValueError("Unknown device category. Expected 'sensor' or 'user_device'.")
    
        return float(max(np.random.poisson(lam=poisson_lambda), 1.0))     # Ensure a non-zero value (1 unit) for computational request

    def _generate_job(self, time: float, device_id: int, device_type: str) -> Job:
        # Generate jobs for devices
        job_index = self._generate_index()
        communication_request_size = self._generate_communication_request(device_type)
        computation_request = self._generate_computation_request(device_type)

        # Create a new job record
        job = {
            'packet_id': job_index,
            'device_type': device_type,
            'device_id': device_id,
            'serving_bs': None,
            'computation_request': computation_request,
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

        # For reward computation, create a data frame
        packet = {
            'packet_id': job_index,
            'device_type': device_type,
            'device_id': device_id,
            'is_transferred': False,  
            'is_accomplished': False,  
            'creation_time': time,
            'arrival_time': None,
            'accomplished_time': None,
            'e2e_delay_threshold': self.config["e2e_delay_threshold"]
        }

        if device_type == SENSOR:
            if time not in self.env.sensor_traffic_logs:
                self.env.sensor_traffic_logs[time] = 0
            self.env.sensor_traffic_logs[time] += communication_request_size
            if time not in self.env.sensor_computation_logs:
                self.env.sensor_computation_logs[time] = 0
            self.env.sensor_computation_logs[time] += computation_request
        elif device_type == USER_DEVICE:
            if time not in self.env.ue_traffic_logs:
                self.env.ue_traffic_logs[time] = 0
            self.env.ue_traffic_logs[time] += communication_request_size
            if time not in self.env.ue_computation_logs:
                self.env.ue_computation_logs[time] = 0
            self.env.ue_computation_logs[time] += computation_request

        # Convert job to data frame and concatenate with existing data frame
        # TODO: use another way, more optimized way, to add the generated packet to df
        packet_df = pd.DataFrame([packet])

        if device_type == SENSOR:
            if self.packet_df_sensor.empty:
                self.packet_df_sensor = packet_df
            else:
                self.packet_df_sensor = pd.concat([self.packet_df_sensor, packet_df], ignore_index=True)
        elif device_type == USER_DEVICE:
            if self.packet_df_ue.empty:
                self.packet_df_ue = packet_df
            else:
                self.packet_df_ue = pd.concat([self.packet_df_ue, packet_df], ignore_index=True)
        else:
            raise ValueError("Unknown device category. Expected 'sensor' or 'user_device'.")

        return job
    
    def generate_job_ue(self, ue: UserEquipment) -> None:
        # Generate jobs for user equipments for device updates
        if random.random() < self.config["ue_job"]["job_generation_probability"]:
            job = self._generate_job(self.env.time, ue.ue_id, USER_DEVICE)
            if ue.data_buffer_uplink.enqueue_job(job):
                self.log_generated_job(job)

    def generate_job_sensor(self, sensor: Sensor) -> None:
        # Generate jobs for sensors for environmental updates
        job = self._generate_job(self.env.time, sensor.sensor_id, SENSOR)
        if sensor.data_buffer_uplink.enqueue_job(job):
            self.log_generated_job(job)
            

    def log_generated_job(self, job: Job) -> None:
        # Log the generated job for the specific device
        self.logger.log_simulation(
            f"Time step: {self.env.time} Job generated: {job['packet_id']} by {job['device_type']} {job['device_id']} "
            f"with initial request size of {job['initial_request_size']} MB and computational request of {job['computation_request']} units"
        )

    def log_df_ue(self) -> None:
        # Log the data frame of UEs at the current time step
        self.logger.log_reward(f"{self.packet_df_ue}")
    
    def log_df_sensor(self) -> None:
        # Log the data frame of sensors at the current time step
        self.logger.log_reward(f"{self.packet_df_sensor}")
        
    def reset_data_frames(self):
        # Reset the job data frames
        self.packet_df_ue.drop(self.packet_df_ue.index, inplace=True)
        self.packet_df_sensor.drop(self.packet_df_sensor.index, inplace=True)