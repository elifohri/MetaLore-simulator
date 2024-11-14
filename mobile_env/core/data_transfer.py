from typing import Dict, Union, List, Tuple, Optional
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.buffers import JobQueue

Device = Union[UserEquipment, Sensor]
Job = Dict[str, Optional[Union[float, int]]]

# Constants for device types
SENSOR = 'sensor'
USER_DEVICE = 'user_device'

class DataTransferManager:
    def __init__(self, env):
        self.env = env
        self.logger = env.logger
        self.job_generator = env.job_generator

        # Initialize throughput history to store data for plotting
        self.throughput_bs_ue_logs: Dict[str, List[float]] = {}
        self.throughput_bs_sensor_logs: Dict[str, List[float]] = {}
        
    def transfer_data_uplink(self) -> None:
        # Set initial throughput for all devices to zero at the start of each time step
        self._initialize_throughput_trackers()
        
        # Transfers data from UEs and sensors to base stations according to data rates.
        for bs in self.env.stations.values():
            self._transfer_data_to_bs(bs, self.env.connections.get(bs, []))
            self._transfer_data_to_bs(bs, self.env.connections_sensor.get(bs, []))

    def _transfer_data_to_bs(self, bs: BaseStation, devices: List[Device]) -> None:
        for device in devices:
            self._transfer_data(device, bs)

    def _transfer_data(self, src: Device, dst: BaseStation) -> None:
        """ Handles the actual transfer of data from a device to the base station."""
        data_transfer_rate = self._get_remaining_data_rate(src, dst)
        src_buffer, dst_buffer = self._get_buffers(src, dst)

        if data_transfer_rate <= 0 or src_buffer.data_queue.empty():
            self.logger.log_simulation(
                f"Time step: {self.env.time} Queue is empty or no data rate for uplink connection from {src} "
                f"to {dst}. Packet transmission aborted."
            )
            return
    
        # Track total data transferred for throughput calculation
        total_data_transferred = 0

        # Transfer jobs while bandwidth is available and there are jobs to transfer.
        while data_transfer_rate > 0 and not src_buffer.data_queue.empty():
            # Peek at the job without removing it
            job = src_buffer.peek_job()
            
            # Log job times before transfer
            #self._log_job_times(job, "Before Transfer")

            # Update start of transfer time of job
            if job['transfer_time_start'] is None:
                job['transfer_time_start'] = self.env.time

            # Update remaining request size of job
            bits_to_send = min(job['remaining_request_size'], data_transfer_rate)
            self._update_job_request_size(job, bits_to_send)

            # Update data transfer rate
            data_transfer_rate -= bits_to_send
            total_data_transferred += bits_to_send

            # Check if there is still bits to send
            if job['remaining_request_size'] <= 0:
                self._transfer_job(src_buffer, dst_buffer)
                
                # Log job times after transfer
                #self._log_job_times(job, "After Transfer")

                # Log full job transfer
                #self._log_transfer(src, dst, job, bits_to_send, full_transfer=True)

            else:
                # Log partial job transfer
                #self._log_transfer(src, dst, job, bits_to_send, full_transfer=False)
                break
            
        # Update throughput records for src and dst
        self._update_throughput(dst, total_data_transferred, src)

    def _get_remaining_data_rate(self, src: Device, dst: BaseStation) -> float:
        if isinstance(src, UserEquipment):
            return self.env.datarates.get((dst, src), 1e6)
        elif isinstance(src, Sensor):
            return self.env.datarates_sensor.get((dst, src), 1e6)
        else:
            raise ValueError(f"Invalid Device")

    def _get_buffers(self, src: Device, dst: BaseStation) -> Tuple[JobQueue, JobQueue]:
        src_buffer = src.data_buffer_uplink
        dst_buffer = dst.transferred_jobs_ue if isinstance(src, UserEquipment) else dst.transferred_jobs_sensor

        return src_buffer, dst_buffer
    
    def _update_job_request_size(self, job: Dict[str, Union[int, float]], bits_to_send: float) -> None:
        job['remaining_request_size'] -= bits_to_send
        if job['remaining_request_size'] < 0:
            job['remaining_request_size'] = 0
    
    def _transfer_job(self, src_buffer: JobQueue, dst_buffer: JobQueue) -> None:
        """Helper method to transfer a job and update its time metrics."""
        job = src_buffer.dequeue_job()
        # Calculate and update job timing information
        self._update_job_times(job)
        # Add the job to the accomplished buffer
        dst_buffer.enqueue_job(job)
        # Update job properties in the corresponding data frame
        self._update_data_frame(job)
        
    def _update_job_times(self, job: Job) -> None:
        """Update the transfer and waiting times of the job."""        
        job['transfer_time_end'] = self.env.time
        job['total_transfer_time'] = job['transfer_time_end'] - job['transfer_time_start']
        job['device_queue_waiting_time'] = job['transfer_time_end'] - job['creation_time']
        
    def _update_data_frame(self, job: Job) -> None:
        """ Update arrival_time of the job in the data frame."""
        if job['device_type'] == USER_DEVICE:
            self.job_generator.packet_df_ue.loc[
                self.job_generator.packet_df_ue['packet_id'] == job['packet_id'], 
                ['is_transferred','arrival_time']
            ] = [True, self.env.time]
        elif job['device_type'] == SENSOR:
            self.job_generator.packet_df_sensor.loc[
                self.job_generator.packet_df_sensor['packet_id'] == job['packet_id'], 
                ['is_transferred','arrival_time']
            ] = [True, self.env.time]
        else:
            self.logger.log_simulation(f"Time step: {self.env.time} Unknown device type {job['device_type']}. Arrival time not updated.")
            
    def _log_job_transfer(self, src: Device, dst: BaseStation, job: Job, bits_to_send: float) -> None:
        self.logger.log_simulation(
            f"Time step: {self.env.time} from {src} to {dst}, job index: {job['packet_id']}, "
            f"Data sent: {bits_to_send}, Remaining size: {job['remaining_request_size']}"
        )
    
    def _log_transfer(self, src: Device, dst: BaseStation, job: Job, bits_to_send: float, full_transfer: bool) -> None:
            """Logs the transfer details."""
            if full_transfer:
                self.logger.log_simulation(f"Time step: {self.env.time} Job {job['packet_id']} fully transferred from {src} to {dst}.")
            else:
                self.logger.log_simulation(f"Time step: {self.env.time} Job {job['packet_id']} partially transferred from {src} to {dst}.")

            
    def _log_job_times(self, job: Job, log_stage: str) -> None:
        """Log the timing details of the job before and after processing."""
        self.logger.log_simulation(
            f"  Time step: {self.env.time} - {log_stage}\n"
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
        
    def _initialize_throughput_trackers(self) -> None:
        """Initialize or reset throughput for each device and base station to zero at the beginning of each time step."""
        for bs in self.env.stations.values():
            self.throughput_bs_ue_logs.setdefault(bs.bs_id, []).append(0)
            self.throughput_bs_sensor_logs.setdefault(bs.bs_id, []).append(0)
    
    def _update_throughput(self, device: Device, data_transferred: float, src: Optional[Device] = None) -> None:
        """Updates throughput for a device or base station."""
        if isinstance(device, BaseStation) and src:
            # Separate cumulative throughput tracking at the base station
            if isinstance(src, UserEquipment):
                self.throughput_bs_ue_logs[device.bs_id][-1] += data_transferred
            elif isinstance(src, Sensor):
                self.throughput_bs_sensor_logs[device.bs_id][-1] += data_transferred