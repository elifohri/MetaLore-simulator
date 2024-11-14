from typing import Dict, Union, List, Tuple, Optional
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.buffers import JobQueue

Device = Union[UserEquipment, Sensor]
Job = Dict[str, Optional[Union[float, int]]]

# Constants for device types
SENSOR = 'sensor'
USER_DEVICE = 'user_device'

class DataProcessManager:
    def __init__(self, env):
        self.env = env
        self.logger = env.logger
        self.job_generator = env.job_generator

    def process_data_mec(self, ue_computational_power: float, sensor_computational_power: float) -> None:
        """ Process data in MEC servers and write processed jobs into downlink queues."""
        for bs in self.env.stations.values():
            self._process_data_for_bs(bs, ue_computational_power, sensor_computational_power)
            
    def _process_data_for_bs(self, bs: BaseStation, ue_computational_power: float, sensor_computational_power: float) -> None:
        self._process_data(bs.transferred_jobs_ue, bs.accomplished_jobs_ue, ue_computational_power)
        self.logger.log_simulation(f"Time step: {self.env.time} UE jobs are processed.")
        self._process_data(bs.transferred_jobs_sensor, bs.accomplished_jobs_sensor, sensor_computational_power)
        self.logger.log_simulation(f"Time step: {self.env.time} Sensor jobs are processed.")

    def _process_data(self, transferred_jobs_queue: JobQueue, accomplished_jobs_queue: JobQueue, computational_power: float) -> None:
        """ Process jobs based on the computational power available at the base station."""
        if computational_power <= 0 or transferred_jobs_queue.data_queue.empty():
            self.logger.log_simulation(f"Time step: {self.env.time} No computational power or queue is empty.")
            return
        
        # Process jobs while computational power is available and there are jobs to process.
        while not transferred_jobs_queue.data_queue.empty() and computational_power > 0:
            # Peek at the job without removing it
            job = transferred_jobs_queue.peek_job()  

            # Log job times before processing
            #self._log_job_times(job, "Before Processing")
        
            # Check if job can be processed with available computational power.
            if job and job['computation_request'] <= computational_power:
                self._process_job(transferred_jobs_queue, accomplished_jobs_queue)
                computational_power -= job['computation_request']
                
                # Log job times after processing
                #self._log_job_times(job, "After Processing")
                
                # Check if computational power is exhausted.
                if computational_power <= 0:
                    self.logger.log_simulation(f"Time step: {self.env.time} MEC server computational power exhausted.")
                    break  # Exit the while loop if computational power is exhausted
            else:
                self.logger.log_simulation(f"Time step: {self.env.time} Job {job['packet_id']} requires more computational power than available. Skipping job.")
                break
            
    def _process_job(self, transferred_jobs_queue: JobQueue, accomplished_jobs_queue: JobQueue) -> None:
        """Helper method to process a job and update its time metrics."""
        job = transferred_jobs_queue.dequeue_job()
        # Calculate and update job timing information
        self._update_job_times(job)
        # Add the job to the accomplished buffer
        accomplished_jobs_queue.enqueue_job(job)
        # Update job properties in the corresponding data frame
        self._update_data_frame(job)
        # Log processed job
        #self._log_processed_job(job)

    def _log_processed_job(self, job: Job) -> None:
        """ Log the processed job."""
        self.logger.log_simulation(
            f"Time step: {self.env.time} job {job['packet_id']} from {job['device_type']} {job['device_id']} "
            f"processed with computational requirement {job['computation_request']}."
        )

    def _update_job_times(self, job: Job) -> None:
        """Update the processing and waiting times of the job."""
        job['processing_time_start'] = self.env.time
        job['processing_time_end'] = self.env.time
        job['total_processing_time'] = job['processing_time_end'] - job['processing_time_start']
        job['total_accomplishment_time'] = job['processing_time_end'] - job['creation_time']
        job['bs_queue_waiting_time'] = job['processing_time_start'] - job['transfer_time_end']

    def _update_data_frame(self, job: Job) -> None:
        """Helper function to update job status in the data frames."""
        if job['device_type'] == USER_DEVICE:
            self.job_generator.packet_df_ue.loc[
                self.job_generator.packet_df_ue['packet_id'] == job['packet_id'],
                ['is_accomplished', 'accomplished_time']
            ] = [True, self.env.time]
        elif job['device_type'] == SENSOR:
            self.job_generator.packet_df_sensor.loc[
                self.job_generator.packet_df_sensor['packet_id'] == job['packet_id'],
                ['is_accomplished', 'accomplished_time']
            ] = [True, self.env.time]
        else:
            self.logger.log_simulation(f"Time step: {self.env.time} Unknown device type {job['device_type']}. Computing time not updated.")

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