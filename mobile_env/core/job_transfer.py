import logging
from typing import Dict, Union, List, Tuple, Optional
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.job_queue import JobQueue
from mobile_env.core.constants import SENSOR, USER_DEVICE


Device = Union[UserEquipment, Sensor]
Job = Dict[str, Optional[Union[float, int]]]

class JobTransferManager:
    """
    Manages the uplink transfer of jobs from devices (UserEquipment and Sensors) to base stations
    within the simulation environment.
    """

    def __init__(self, env):
        self.env = env
        self.logger = env.logger

    def transfer_data_uplink(self) -> None:
        for bs in self.env.stations.values():
            self._transfer_job_to_bs(bs, self.env.connections.get(bs, []))
            self._transfer_job_to_bs(bs, self.env.connections_sensor.get(bs, []))

    def _transfer_job_to_bs(self, bs: BaseStation, devices: List[Device]) -> None:
        for device in devices:
            self._transfer_uplink(device, bs)

    def _transfer_uplink(self, src: Device, dst: BaseStation) -> None:
        data_transfer_rate = self._get_data_rate(src, dst)
        src_buffer, dst_buffer = self._get_buffers(src, dst)

        # Transfer jobs while bandwidth is available and there are jobs to transfer.
        while data_transfer_rate > 0 and not src_buffer.is_empty():
            job = src_buffer.peek_job()
            
            if job['transfer_time_start'] is None:
                job['transfer_time_start'] = self.env.time

            bits_to_send = min(job['remaining_request_size'], data_transfer_rate)
            data_transfer_rate -= bits_to_send
            self._update_job_request_size(job, bits_to_send)

            if job['remaining_request_size'] <= 0:
                job = src_buffer.dequeue_job()
                self._update_job_timing(job)
                dst_buffer.enqueue_job(job)
                self.log_transferred_job(src, dst, job, bits_to_send, full_transfer=True)
            else:
                self.log_transferred_job(src, dst, job, bits_to_send, full_transfer=False)
                break

    def _get_data_rate(self, src: Device, dst: BaseStation) -> float:
        return self.env.datarates.get((dst, src), 1e6) if isinstance(src, UserEquipment) else self.env.datarates_sensor.get((dst, src), 1e6)

    def _get_buffers(self, src: Device, dst: BaseStation) -> Tuple[JobQueue, JobQueue]:
        src_buffer = src.data_buffer_uplink
        dst_buffer = dst.transferred_jobs_ue if isinstance(src, UserEquipment) else dst.transferred_jobs_sensor
        return src_buffer, dst_buffer
    
    def _update_job_request_size(self, job: Dict[str, Union[int, float]], bits_to_send: float) -> None:
        job['remaining_request_size'] -= bits_to_send
        if job['remaining_request_size'] < 0:
            job['remaining_request_size'] = 0
    
    def _update_job_timing(self, job: Job) -> None:    
        job['transfer_time_end'] = self.env.time
        job['total_transfer_time'] = job['transfer_time_end'] - job['transfer_time_start']
        job['device_queue_waiting_time'] = job['transfer_time_end'] - job['creation_time']

    def log_transferred_job(self, src: Device, dst: BaseStation, job: Job, bits_to_send: float, full_transfer: bool) -> None:
            if full_transfer:
                self.logger.log_simulation(f"Time step: {self.env.time} Job: {job['packet_id']} fully transferred from {src} to {dst}.")
            else:
                self.logger.log_simulation(f"Time step: {self.env.time} Job: {job['packet_id']} partially transferred from {src} to {dst}, bits send {bits_to_send}.")
            
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