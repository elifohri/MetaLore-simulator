import logging
import pandas as pd
import numpy as np
from typing import Union, Dict, Tuple

class DelayManager:
    """Handles delay calculations and related metrics for UE and sensor jobs."""
    
    def __init__(self, env):
        self.env = env

    def get_accomplished_ue_packets(self) -> pd.DataFrame:
        """Retrieve all accomplished UE packets at the current timestep."""
        return self.env.job_dataframe.df_ue_packets[
            self.env.job_dataframe.df_ue_packets['accomplished_time'] == self.env.time
        ].copy()

    def get_accomplished_ue_packets_with_synch_delay(self) -> pd.DataFrame:
        """Retrieve all accomplished UE packets at the current timestep."""
        return self.env.job_dataframe.df_ue_packets[
            self.env.job_dataframe.df_ue_packets['synch_delay'].notnull()
        ].copy()
    
    def get_accomplished_ue_packets_with_null_synch_delay(self) -> pd.DataFrame:
        """Retrieve all accomplished UE packets at the current timestep."""
        return self.env.job_dataframe.df_ue_packets[
            self.env.job_dataframe.df_ue_packets['synch_delay'].isnull()
        ].copy()
    
    def get_accomplished_ue_packets_with_null_synch_reward(self) -> pd.DataFrame:
        """Retrieve all accomplished UE packets at the current timestep."""
        return self.env.job_dataframe.df_ue_packets[
            self.env.job_dataframe.df_ue_packets['synch_reward'].isnull()
        ].copy()

    def get_accomplished_sensor_packets(self) -> pd.DataFrame:
        """Retrieve all accomplished sensor packets."""
        return self.env.job_dataframe.df_sensor_packets[
            self.env.job_dataframe.df_sensor_packets['accomplished_time'].notnull()
        ].copy()
    
    def get_currrent_accomplished_sensor_packets(self) -> pd.DataFrame:
        """Retrieve all accomplished sensor packets."""
        return self.env.job_dataframe.df_sensor_packets[
            self.env.job_dataframe.df_sensor_packets['accomplished_time'] == self.env.time
        ].copy()

    
    def get_latest_accomplished_sensor_packet(self, accomplished_sensor_packets: pd.DataFrame) -> Union[pd.Series]:
        """Retrieve the latest accomplished sensor packet with highest creation time."""
        latest_accomplished_time = accomplished_sensor_packets['accomplished_time'].max()
        latest_accomplished_packets = accomplished_sensor_packets[accomplished_sensor_packets['accomplished_time'] == latest_accomplished_time]
        latest_accomplsihed_sensor_packet = latest_accomplished_packets.loc[latest_accomplished_packets['creation_time'].idxmax()]
        
        return latest_accomplsihed_sensor_packet
    
    def get_latest_accomplished_sensor_packet_before_ue(self, accomplished_sensor_packets: pd.DataFrame, ue_packet: pd.Series) -> pd.DataFrame:
        """Retrieve the latest accomplished sensor packets generated before the corresponding UE packet."""
        ue_creation_time = ue_packet['creation_time']
        filtered_sensor_packets = accomplished_sensor_packets[accomplished_sensor_packets['creation_time'] <= ue_creation_time]

        if filtered_sensor_packets.empty:
            return None
        
        latest_accomplsihed_sensor_packet_before_ue = filtered_sensor_packets.loc[filtered_sensor_packets['creation_time'].idxmax()]

        return latest_accomplsihed_sensor_packet_before_ue


    def compute_absolute_synch_delay(self):
        """Computes absolute delay between the latest accomplished sensor and UE packets."""
        accomplished_ue_packets = self.get_accomplished_ue_packets_with_null_synch_delay()
        accomplished_sensor_packets = self.get_accomplished_sensor_packets()

        if accomplished_ue_packets.empty or accomplished_sensor_packets.empty:
            self.env.logger.log_reward(f"Time step: {self.env.time} No accomplished UE packets or sensor packets found.")
            return None
                
        latest_accomplished_sensor_packet = self.get_latest_accomplished_sensor_packet(accomplished_sensor_packets)

        sensor_creation_time = latest_accomplished_sensor_packet["creation_time"]
        accomplished_ue_packets['synch_delay'] = np.abs(accomplished_ue_packets['creation_time'] - sensor_creation_time)

        self.env.job_dataframe.df_ue_packets.update(accomplished_ue_packets[['packet_id', 'synch_delay']])


    def compute_positive_synch_delay(self):
        """Computes the positive delay between the latest accomplished sensor packet generated before the corresponding UE packet."""
        accomplished_ue_packets = self.get_accomplished_ue_packets_with_null_synch_delay()
        accomplished_sensor_packets = self.get_accomplished_sensor_packets()

        if accomplished_ue_packets.empty or accomplished_sensor_packets.empty:
            self.env.logger.log_reward(f"Time step: {self.env.time} No accomplished UE packets or sensor packets found.")
            return None
        
        # TODO: Optimize this code, vectorize the operations
        delays = []
        for _, ue_packet in accomplished_ue_packets.iterrows():
            valid_sensor_packet = self.get_latest_accomplished_sensor_packet_before_ue(accomplished_sensor_packets, ue_packet)
            delay = None if valid_sensor_packet is None else ue_packet['creation_time'] - valid_sensor_packet['creation_time']
            delays.append(delay)

        accomplished_ue_packets['synch_delay'] = delays
        
        self.env.job_dataframe.df_ue_packets.update(accomplished_ue_packets[['packet_id', 'synch_delay']])
    