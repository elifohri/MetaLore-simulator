import logging
import pandas as pd
from typing import Union, Dict, Tuple

class DelayManager:
    """Handles delay calculations and related metrics for UE and sensor jobs."""
    
    def __init__(self, env):
        self.env = env

    def get_accomplished_sensor_packets(self) -> pd.DataFrame:
        """Retrieve all accomplished sensor packets."""
        return self.env.job_dataframe.df_sensor_packets[
            self.env.job_dataframe.df_sensor_packets['accomplished_time'].notnull()
        ].copy()

    def get_accomplished_ue_packets(self) -> pd.DataFrame:
        """Retrieve all accomplished UE packets at the current timestep."""
        return self.env.job_dataframe.df_ue_packets[
            self.env.job_dataframe.df_ue_packets['accomplished_time'] == self.env.time
        ].copy()
    
    def get_latest_accomplished_sensor_packet(self, accomplished_sensor_packets: pd.DataFrame) -> Union[pd.Series]:
        """Retrieve the latest accomplished sensor packet."""
        latest_accomplished_time = accomplished_sensor_packets['accomplished_time'].max()
        latest_accomplished_packets = accomplished_sensor_packets[accomplished_sensor_packets['accomplished_time'] == latest_accomplished_time]
        latest_accomplsihed_sensor_packet = latest_accomplished_packets.loc[latest_accomplished_packets['creation_time'].idxmax()]
        
        return latest_accomplsihed_sensor_packet

    def compute_absolute_delay(self):
        """Compute absolute delay between the latest sensor and UE packets."""
        accomplished_ue_packets = self.get_accomplished_ue_packets()
        accomplished_sensor_packets = self.get_accomplished_sensor_packets()

        if accomplished_ue_packets.empty or accomplished_sensor_packets.empty:
            self.env.logger.log_reward(f"Time step: {self.env.time} No accomplished UE or sensor packets found.")
            return None
        
        latest_accomplished_sensor_packet = self.get_latest_accomplished_sensor_packet(accomplished_sensor_packets)

        sensor_creation_time = latest_accomplished_sensor_packet["creation_time"]
        accomplished_ue_packets['synch_delay'] = abs(accomplished_ue_packets['creation_time'] - sensor_creation_time)

        self.env.job_dataframe.df_ue_packets.update(accomplished_ue_packets[['packet_id', 'synch_delay']])


    def get_latest_accomplished_sensor_packet_before_ue(self, accomplished_sensor_packets: pd.DataFrame, ue_packet: pd.Series) -> pd.DataFrame:
        """Filter sensor packets generated before the corresponding UE packet."""
        ue_creation_time = ue_packet['creation_time']
        return accomplished_sensor_packets[accomplished_sensor_packets['creation_time'] <= ue_creation_time]

    def compute_positive_delay(self):
        """Computes the positive delay between the latest accomplished sensor packet generated before the corresponding UE packet."""
        accomplished_ue_packets = self.get_accomplished_ue_packets()
        accomplished_sensor_packets = self.get_accomplished_sensor_packets()

        if accomplished_ue_packets.empty or accomplished_sensor_packets.empty:
            self.env.logger.log_reward(f"Time step: {self.env.time} No accomplished UE or sensor packets found.")
            return None
        
        positive_delays = []
        for _, ue_packet in accomplished_ue_packets.iterrows():
            valid_sensor_packets = self.get_latest_accomplished_sensor_packet_before_ue(accomplished_sensor_packets, ue_packet)
            
            if valid_sensor_packets.empty:
                positive_delays.append(None)
                continue

            latest_sensor_packet = valid_sensor_packets.loc[valid_sensor_packets['creation_time'].idxmax()]
            delay = ue_packet['creation_time'] - latest_sensor_packet['creation_time']
            positive_delays.append(delay)

        accomplished_ue_packets['positive_delay'] = positive_delays
        
        self.env.job_dataframe.df_ue_packets.update(accomplished_ue_packets[['packet_id', 'positive_delay']])
    