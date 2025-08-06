import pandas as pd
from typing import Dict, Optional, Union
from mobile_env.core.entities import UserEquipment, Sensor
from mobile_env.core.constants import USER_DEVICE, SENSOR, DEVICE_TYPE, E2E_THRESHOLD

Device = Union[UserEquipment, Sensor]
Job = Dict[str, Optional[Union[float, int]]]


class JobDataFrame:
    """
    Logs and tracks the jobs as data frames for reward calculation and RL.
    """
        
    def __init__(self, env):
        self.env = env
        self.logger = env.logger
        self.config = env.default_config()

        # Main DataFrames to store all packets for reward computation
        self.df_ue_packets = pd.DataFrame(columns=[
            'packet_id', 'device_type', 'device_id', 
            'initial_request_size', 'remaining_request_size', 'initial_computation_request', 'remaining_computation_request',
            'is_transferred', 'is_accomplished', 'creation_time', 
            'transfer_time_start', 'transfer_time_end', 'total_transfer_time',
            'processing_time_start', 'processing_time_end', 'total_processing_time',
            'total_accomplishment_time', 'device_queue_waiting_time', 'bs_queue_waiting_time',
            'e2e_delay_threshold', 'e2e_delay', 'synched_sensor_device', 'synched_sensor_job', 'synch_delay', 'synch_reward'
        ])

        self.df_sensor_packets = pd.DataFrame(columns=[
            'packet_id', 'device_type', 'device_id', 
            'initial_request_size', 'remaining_request_size', 'initial_computation_request', 'remaining_computation_request',
            'is_transferred', 'is_accomplished', 'creation_time', 
            'transfer_time_start', 'transfer_time_end', 'total_transfer_time',
            'processing_time_start', 'processing_time_end', 'total_processing_time',
            'total_accomplishment_time', 'device_queue_waiting_time', 'bs_queue_waiting_time',
            'e2e_delay_threshold', 'e2e_delay', 'synched_sensor_device', 'synched_sensor_job', 'synch_delay', 'synch_reward'
        ])

    def update_after_transfer(self, job: Job) -> None:
        # Update the respective data frame after transfer
        packet = {
            'packet_id': job['packet_id'],
            'device_type': job['device_type'],
            'device_id': job['device_id'],
            'initial_request_size': job['initial_request_size'],
            'remaining_request_size': job['remaining_request_size'],
            'initial_computation_request': job['initial_computation_request'],
            'remaining_computation_request': job['remaining_computation_request'],
            'is_transferred': True,
            'is_accomplished': False,
            'creation_time': job['creation_time'],
            'transfer_time_start': job['transfer_time_start'],
            'transfer_time_end': job['transfer_time_end'],
            'total_transfer_time': job['total_transfer_time'],
            'processing_time_start': None,
            'processing_time_end': None,
            'total_processing_time': None,
            'total_accomplishment_time': None,
            'device_queue_waiting_time': job['device_queue_waiting_time'],
            'bs_queue_waiting_time': None,
            'e2e_delay_threshold': self.config[E2E_THRESHOLD],
            'e2e_delay': None,
            'synched_sensor_device': None,
            'synched_sensor_job': None,
            'synch_delay': None,
            'synch_reward': None
        }

        # Ensure the df_ue_packets is not None or empty before concatenation
        if job[DEVICE_TYPE] == USER_DEVICE:
            if self.df_ue_packets.empty or self.df_ue_packets.isna().all().all():
                self.df_ue_packets = pd.DataFrame([packet])
            else:
                self.df_ue_packets = pd.concat([self.df_ue_packets, pd.DataFrame([packet])], ignore_index=True)

        # Ensure the df_sensor_packets is not None or empty before concatenation
        elif job[DEVICE_TYPE] == SENSOR:
            if self.df_sensor_packets.empty or self.df_sensor_packets.isna().all().all():
                self.df_sensor_packets = pd.DataFrame([packet])
            else:
                self.df_sensor_packets = pd.concat([self.df_sensor_packets, pd.DataFrame([packet])], ignore_index=True)
    
    def update_synched_sensor_device(self, job: Job) -> None:
        # Find the sensor that the UE is synching and update the dataframe
        packet_id = job['packet_id']
        device_id = job['device_id']

        ue = self.env.users.get(device_id)
        connected_sensor = ue.connected_sensor
        sensor_id = connected_sensor.sensor_id
        job['synched_sensor_device'] = sensor_id

        df = self.df_ue_packets
        packet_index = df.index[df['packet_id'] == packet_id]
        if not packet_index.empty:
            idx = packet_index[0]
            self.df_ue_packets.at[idx, 'synched_sensor_device'] = sensor_id
        else:
            self.logger.log_simulation(f"Time step: {self.env.time} Could not find job {packet_id} in dataframe.")

    def update_synched_sensor_job(self, job: Job) -> None:
        # Find the sensor job that the UE is synching and update the dataframe
        packet_id = job['packet_id']
        synched_sensor_id = job['synched_sensor_device']

        synched_job_id = self._get_latest_accomplished_job_from_sensor(synched_sensor_id)
        if synched_job_id is None:
            return None

        job['synched_sensor_job'] = synched_job_id

        df = self.df_ue_packets
        idx_series = df.index[df['packet_id'] == packet_id]
        if not idx_series.empty:
            idx = idx_series[0]
            self.df_ue_packets.at[idx, 'synched_sensor_job'] = synched_job_id
        else:
            self.logger.log_simulation(f"Time step: {self.env.time} Could not find job {packet_id} in dataframe.")
    
    def _get_latest_accomplished_job_from_sensor(self, sensor_id: int) -> Optional[Union[int, str]]:
        # Return the packet_id of the most recently accomplished sensor job for synchronization.
        df = self.df_sensor_packets
        accomplished_sensor_jobs = df[(df['is_accomplished'] == True) & (df['device_id'] == sensor_id)]

        if accomplished_sensor_jobs.empty:
            return None

        latest_time = accomplished_sensor_jobs['processing_time_end'].max()
        latest_jobs = accomplished_sensor_jobs[accomplished_sensor_jobs['processing_time_end'] == latest_time]
        most_recent = latest_jobs.loc[latest_jobs['creation_time'].idxmax()]

        return most_recent['packet_id']

    def update_before_processing(self, job: Job) -> None:
        if job['device_type'] != USER_DEVICE:
            return

        if job['synched_sensor_device'] is None or job['synched_sensor_job'] is None:
            self.update_synched_sensor_device(job)
        if job['synched_sensor_job'] is None:
            self.update_synched_sensor_job(job)
            
    def update_during_processing(self, job: Job) -> None:
        df = self.df_ue_packets if job['device_type'] == USER_DEVICE else self.df_sensor_packets
        packet_index = df.index[df['packet_id'] == job['packet_id']]
        
        if not packet_index.empty:
            idx = packet_index[0]
            df.at[idx, 'remaining_computation_request'] = job['remaining_computation_request']
            df.at[idx, 'processing_time_start'] = job['processing_time_start']
        else:
            self.logger.log_simulation(f"Time step: {self.env.time} WARNING: No packet with ID {job['packet_id']} found.")

    def update_after_processing(self, job: Job) -> None:
        # Update the respective data frame for reward and penalty calculation accomplished jobs
        df = self.df_ue_packets if job['device_type'] == USER_DEVICE else self.df_sensor_packets
        packet_index = df.index[df['packet_id'] == job['packet_id']]
        
        if not packet_index.empty:
            idx = packet_index[0]
            df.at[idx, 'is_accomplished'] = True
            df.at[idx, 'remaining_computation_request'] = job['remaining_computation_request']
            df.at[idx, 'processing_time_start'] = job['processing_time_start']
            df.at[idx, 'processing_time_end'] = job['processing_time_end']
            df.at[idx, 'total_processing_time'] = job['total_processing_time']
            df.at[idx, 'total_accomplishment_time'] = job['total_accomplishment_time']
            df.at[idx, 'bs_queue_waiting_time'] = job['bs_queue_waiting_time']
            df.at[idx, 'e2e_delay'] = job['processing_time_end'] - job['creation_time']
        else:
            self.logger.log_simulation(f"Time step: {self.env.time} WARNING: No packet with ID {job['packet_id']} found.")

    def compute_synchronization_delay(self) -> None:
        # Calculates synchronization delay between the UE packet and the synched sensor packet

        # Step 1: Filter UE packets that are accomplished at this timestep
        accomplished_ue_packets = self.df_ue_packets[self.df_ue_packets['is_accomplished'] & (self.df_ue_packets['processing_time_end'] == self.env.time)].copy()

        if accomplished_ue_packets.empty:
            return
        
        # Step 2: Get mapping from sensor packet_id (synched_job) → sensor creation_time
        sensor_job_creation_time_map = self.df_sensor_packets.set_index('packet_id')['creation_time'].to_dict()

        # Step 3: Map synched sensor job creation time into UE dataframe
        accomplished_ue_packets['sensor_job_creation_time'] = accomplished_ue_packets['synched_sensor_job'].map(sensor_job_creation_time_map)

        # Step 4: Compute synchronization delay (absolute difference)
        accomplished_ue_packets['synch_delay'] = (accomplished_ue_packets['creation_time'] - accomplished_ue_packets['sensor_job_creation_time']).abs()

        # Step 5: Update original df_ue_packets
        self.df_ue_packets.loc[accomplished_ue_packets.index, 'synch_delay'] = accomplished_ue_packets['synch_delay']

    def compute_processed_throughput_ue(self) -> float:
        # Computes the throughput of all UE packets that are accomplished at this time step
        accomplished_ue_packets = self.df_ue_packets[self.df_ue_packets['is_accomplished'] & (self.df_ue_packets['processing_time_end'] == self.env.time)].copy()
        processed_throughput_ue = accomplished_ue_packets['initial_request_size'].sum()

        return processed_throughput_ue
    
    def compute_processed_throughput_sensor(self) -> float:
        # Computes the throughput of all sensor packets that are accomplished at this time step
        accomplished_sensor_packets = self.df_sensor_packets[self.df_sensor_packets['is_accomplished'] & (self.df_sensor_packets['processing_time_end'] == self.env.time)].copy()
        processed_throughput_sensor = accomplished_sensor_packets['initial_request_size'].sum()

        return processed_throughput_sensor

    def log_dataframe(self, df: pd.DataFrame, label: str) -> None:
        self.logger.log_reward(f"Time step: {self.env.time} Data frame {label}:")
        if df.empty:
            self.logger.log_reward(f"Time step: {self.env.time} Empty DataFrame")
        else:
            self.logger.log_reward(f"Time step: {self.env.time} Columns: {list(df.columns)}")
            for _, row in df.iterrows():
                self.logger.log_reward(f"Time step: {self.env.time} Row: {row.to_dict()}")

    def log_ue_packets(self) -> None:
        self.log_dataframe(self.df_ue_packets, USER_DEVICE)
    
    def log_sensor_packets(self) -> None:
        self.log_dataframe(self.df_sensor_packets, SENSOR)
        
    def reset_dataframes(self) -> None:
        self.df_ue_packets = pd.DataFrame(columns=self.df_ue_packets.columns)
        self.df_sensor_packets = pd.DataFrame(columns=self.df_sensor_packets.columns)