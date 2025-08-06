import logging
from typing import Dict, Union, Optional
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.job_queue import JobQueue
from mobile_env.core.constants import USER_DEVICE, SENSOR


Device = Union[UserEquipment, Sensor]
Job = Dict[str, Optional[Union[float, int]]]

class JobProcessManager:
    """
    Manages the processing of jobs at base stations using available computational resources
    within the simulation environment.
    """

    def __init__(self, env, data_frame):
        self.env = env
        self.logger = env.logger
        self.data_frame = data_frame
        self.processed_data_ue = {ue.ue_id: 0.0 for ue in self.env.users.values()}
        self.processed_data_sensor = {sensor.sensor_id: 0.0 for sensor in self.env.sensors.values()}

    
    def reset_processed_data(self) -> None:
        self.processed_data_ue = {ue.ue_id: 0.0 for ue in self.env.users.values()}
        self.processed_data_sensor = {sensor.sensor_id: 0.0 for sensor in self.env.sensors.values()}

    def process_data_for_mec(self, ue_computational_power: float, sensor_computational_power: float) -> None:
        self.reset_processed_data()
        for bs in self.env.stations.values():
            self._process_jobs_in_base_station(bs, ue_computational_power, sensor_computational_power)
            
    def _process_jobs_in_base_station(self, bs: BaseStation, ue_computational_power: float, sensor_computational_power: float) -> None:
        self._process_jobs_in_queue(bs.transferred_jobs_ue, bs.accomplished_jobs_ue, ue_computational_power)
        self._process_jobs_in_queue(bs.transferred_jobs_sensor, bs.accomplished_jobs_sensor, sensor_computational_power)

    def _process_jobs_in_queue(self, transferred_jobs_queue: JobQueue, accomplished_jobs_queue: JobQueue, computational_power: float) -> None:
        # Process jobs while computational power is available and there are jobs to process.        
        while computational_power > 0 and not transferred_jobs_queue.is_empty():
            job = transferred_jobs_queue.peek_job()  

            if job['processing_time_start'] is None: 
                self.data_frame.update_before_processing(job)
                if job['device_type'] == USER_DEVICE and job['synched_sensor_job'] is None:
                    break
                else:
                    job['processing_time_start'] = self.env.time

            units_to_process = min(job['remaining_computation_request'], computational_power)
            computational_power -= units_to_process
            self._update_job_computation_size(job, units_to_process)
            self.data_frame.update_during_processing(job)

            # Update the transferred data tracking
            self._update_device_processed_data(job['device_type'], job['device_id'], units_to_process)

            if job['remaining_computation_request'] <= 0:
                job = transferred_jobs_queue.dequeue_job()
                self._update_job_timing(job)
                accomplished_jobs_queue.enqueue_job(job)
                self.data_frame.update_after_processing(job)
                #self.log_processed_job(job, units_to_process)

                if computational_power <= 0:
                    #self.logger.log_simulation(f"Time step: {self.env.time} MEC server computational power exhausted.")
                    break
            else:
                #self.log_processed_job(job, units_to_process)
                break

    def _update_job_computation_size(self, job: Dict[str, Union[int, float]], bits_processed: float) -> None:
        job['remaining_computation_request'] -= bits_processed
        if job['remaining_computation_request'] < 0:
            job['remaining_computation_request'] = 0

    def _update_job_timing(self, job: Job) -> None:
        job['processing_time_end'] = self.env.time
        job['total_processing_time'] = job['processing_time_end'] - job['processing_time_start']
        job['total_accomplishment_time'] = job['processing_time_end'] - job['creation_time']
        job['bs_queue_waiting_time'] = job['processing_time_start'] - job['transfer_time_end']

    def _update_device_computational_request(self, job: Job) -> None:
        device = (self.env.users[job['device_id']] if job['device_type'] == 'user_device' else self.env.sensors[job['device_id']])
        device.total_computation_request -= job['computation_request']

    def _update_device_processed_data(self, device_type: str, device_id: int, units_processed: float) -> None:
        if device_type == USER_DEVICE:
            self.processed_data_ue[device_id] += units_processed
        elif device_type == SENSOR:
            self.processed_data_sensor[device_id] += units_processed

    def log_processed_job(self, job: Job, units_processed: float) -> None:
        self.logger.log_simulation(
            f"Time step: {self.env.time} Job: {job['packet_id']} from {job['device_type']} {job['device_id']} "
            f"processed with computational requirement {job['initial_computation_request']} with units processed {units_processed}, remaining {job['remaining_computation_request']}."
        )

    def log_job_details(self, job: Job) -> None:
        self.logger.log_simulation(
            f"  Time step: {self.env.time}\n"
            f"  Job {job['packet_id']}:\n"
            f"  Creation Time: {job.get('creation_time', 'N/A')}\n"
            f"  Transfer Time Start: {job.get('transfer_time_start', 'N/A')}\n"
            f"  Transfer Time End: {job.get('transfer_time_end', 'N/A')}\n"
            f"  Total Transfer Time: {job.get('total_transfer_time', 'N/A')}\n"
            f"  Processing Time Start: {job.get('processing_time_start', 'N/A')}\n"
            f"  Processing Time End: {job.get('processing_time_end', 'N/A')}\n"
            f"  Total Processing Time: {job.get('total_processing_time', 'N/A')}\n"
            f"  Total Accomplishment Time: {job.get('total_accomplishment_time', 'N/A')}\n"
            f"  Device Queue Waiting Time: {job.get('device_queue_waiting_time', 'N/A')}\n"
            f"  BS Queue Waiting Time: {job.get('bs_queue_waiting_time', 'N/A')}"
        )