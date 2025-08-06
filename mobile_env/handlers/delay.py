import logging
import pandas as pd
import numpy as np
from typing import Union, Dict, Tuple

class DelayManager:
    """
    Handles delay calculations and related metrics for UE and sensor jobs.
    """
    
    def __init__(self, env):
        self.env = env

    def get_accomplished_ue_packets(self) -> pd.DataFrame:
        """Retrieve all accomplished UE packets at the current timestep."""
        return self.env.job_dataframe.df_ue_packets[
            self.env.job_dataframe.df_ue_packets['processing_time_end'] == self.env.time
        ].copy()

    def get_accomplished_ue_packets_with_synch_delay(self) -> pd.DataFrame:
        """Retrieve all accomplished UE packets wth a synch delay at the current timestep."""
        return self.env.job_dataframe.df_ue_packets[
            self.env.job_dataframe.df_ue_packets['synch_delay'].notnull()
        ].copy()
    
    def get_accomplished_ue_packets_with_null_synch_delay(self) -> pd.DataFrame:
        """Retrieve all accomplished UE packets without a synch delay at the current timestep."""
        return self.env.job_dataframe.df_ue_packets[
            self.env.job_dataframe.df_ue_packets['synch_delay'].isnull()
        ].copy()
    
    def get_accomplished_ue_packets_with_null_synch_reward(self) -> pd.DataFrame:
        """Retrieve all accomplished UE packets without a synch reward at the current timestep."""
        return self.env.job_dataframe.df_ue_packets[
            self.env.job_dataframe.df_ue_packets['synch_reward'].isnull()
        ].copy()
    
    def get_accomplished_ue_packets_with_synch_delay_and_null_synch_reward(self) -> pd.DataFrame:
        """Retrieve all accomplished UE packets without a synch reward at the current timestep."""
        return self.env.job_dataframe.df_ue_packets[
            self.env.job_dataframe.df_ue_packets['synch_reward'].isnull() &
            self.env.job_dataframe.df_ue_packets['synch_delay'].notnull()
        ].copy()

    def get_accomplished_sensor_packets(self) -> pd.DataFrame:
        """Retrieve all accomplished sensor packets."""
        return self.env.job_dataframe.df_sensor_packets[
            self.env.job_dataframe.df_sensor_packets['processing_time_end'].notnull()
        ].copy()
    
    def get_currrent_accomplished_sensor_packets(self) -> pd.DataFrame:
        """Retrieve all accomplished sensor packets."""
        return self.env.job_dataframe.df_sensor_packets[
            self.env.job_dataframe.df_sensor_packets['processing_time_end'] == self.env.time
        ].copy()