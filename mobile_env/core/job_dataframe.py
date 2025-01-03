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
            'packet_id', 'device_type', 'device_id', 'is_transferred', 'is_accomplished', 
            'creation_time', 'arrival_time', 'accomplished_time', 'e2e_delay_threshold', 'e2e_delay', 'synch_delay'
        ])
        self.df_sensor_packets = pd.DataFrame(columns=[
            'packet_id', 'device_type', 'device_id', 'is_transferred', 'is_accomplished', 
            'creation_time', 'arrival_time', 'accomplished_time', 'e2e_delay_threshold', 'e2e_delay', 'synch_delay'
        ])

    def update_after_processing(self, job: Job) -> None:
        # Update the respective data frame for reward and penalty calculation accomplished jobs
        packet = {
            'packet_id': job['packet_id'],
            'device_type': job['device_type'],
            'device_id': job['device_id'],
            'is_transferred': True,
            'is_accomplished': True,
            'creation_time': job['creation_time'],
            'arrival_time': job['transfer_time_end'],
            'accomplished_time': self.env.time,
            'e2e_delay_threshold': self.config[E2E_THRESHOLD],
            'e2e_delay': job['total_accomplishment_time'],
            'synch_delay': None
        }
 
        if job[DEVICE_TYPE] == USER_DEVICE:
            self.df_ue_packets = pd.concat(
                [self.df_ue_packets, pd.DataFrame([packet])], ignore_index=True
            )
        elif job[DEVICE_TYPE] == SENSOR:
            self.df_sensor_packets = pd.concat(
                [self.df_sensor_packets, pd.DataFrame([packet])], ignore_index=True
            )

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
        
    def reset_packet_dataframes(self) -> None:
        self.df_ue_packets = pd.DataFrame(columns=self.df_ue_packets.columns)
        self.df_sensor_packets = pd.DataFrame(columns=self.df_sensor_packets.columns)