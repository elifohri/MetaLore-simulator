from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
import pandas as pd

from mobile_env.handlers.handler import Handler


class MComSmartCityHandler(Handler):

    features = ["queue_lengths"]

    def __init__(self, env):
        self.env = env
        self.logger = env.logger  

    @classmethod
    def obs_size(cls, env) -> int:
        return sum(env.feature_sizes[ftr] for ftr in cls.features)

    @classmethod
    def action_space(cls, env) -> spaces.Box:
        """Define continuous action space for bandwidth allocation and computational power allocation"""
        return spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    @classmethod
    def observation_space(cls, env) -> spaces.Box:
        """Define observation space"""
        size = cls.obs_size(env)
        #env.logger.log_reward(f"Time step: {env.time} Observation Space: size is {size}")        
        return spaces.Box(low=0.0, high=np.inf, shape=(size,), dtype=np.float32)

    @classmethod
    def action(cls, env, actions: Tuple[float, float]) -> Tuple[float, float]:
        """Transform action to expected shape of core environment."""
        assert len(actions) == 2, "Action must have two elements: bandwidth allocation and computational power allocation."

        bandwidth_allocation, computational_allocation = actions

        # Ensure actions are within valid range
        bandwidth_allocation = max(0.0, min(1.0, bandwidth_allocation))
        computational_allocation = max(0.0, min(1.0, computational_allocation))

        # Log action
        env.logger.log_reward(f"Time step: {env.time} Action: {bandwidth_allocation}, {computational_allocation}")

        return bandwidth_allocation, computational_allocation
    
    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Compute system-wide observations for the RL agent."""
        
        # Gather the queue lengths (from base station)
        queue_lengths = np.array(env.get_queue_lengths()).ravel()
        #env.logger.log_reward(f"Time step: {env.time} Queue lengths: {queue_lengths}")    
        
        # Define maximum queue sizes for normalization
        max_queue_lengths = np.array([500, 2000, 500, 2000])  # Adjust these values as needed

        # Normalize queue lengths
        normalized_queue_lengths = queue_lengths / max_queue_lengths  
        env.logger.log_reward(f"Time step: {env.time} Queue lengths: {normalized_queue_lengths}")

        if normalized_queue_lengths.shape != (4,):
            raise ValueError(f"Unexpected shapes: queue_lengths {normalized_queue_lengths.shape}")

        # Concatenate all observations into a single array
        observation = np.concatenate([
            normalized_queue_lengths,              # 4 values
        ]).astype(np.float32)
        
        return observation

    @classmethod
    def reward(cls, env) -> float:
        """Process UE packets: apply penalties, rewards, and update the data frame."""
        total_reward = 0
        config = env.default_config()["reward_calculation"]
        penalty = config["ue_penalty"]
        base_reward = config["base_reward"]
        discount_factor = config["discount_factor"]

        accomplished_packets = env.job_generator.packet_df_ue[
            (env.job_generator.packet_df_ue['is_accomplished']) &
            (env.job_generator.packet_df_ue['accomplished_time'] == env.time)
        ].copy()

        if accomplished_packets.empty:
            return total_reward

        # Compute delays and penalties
        accomplished_packets['delay'] = env.time - accomplished_packets['creation_time']
        penalties = (accomplished_packets['delay'] > accomplished_packets['e2e_delay_threshold']) * penalty

        # Compute rewards for valid packets
        valid_packets = accomplished_packets[
            accomplished_packets['delay'] <= accomplished_packets['e2e_delay_threshold']
        ]

        if not valid_packets.empty:
            valid_ue_packets = valid_packets.copy()

            # Compute 'computed_delay' column
            valid_ue_packets['computed_delay'] = valid_ue_packets.apply(
                lambda row: cls.compute_absolute_delay(cls, env, row), axis=1
            )

            valid_ue_packets['reward'] = base_reward * (discount_factor ** valid_ue_packets['computed_delay'])
            total_reward += valid_ue_packets['reward'].sum()

        total_reward += penalties.sum()

        # Group delays by UE
        delay_logs_per_user = accomplished_packets.groupby('device_id')['delay'].mean().to_dict()

        # Append delay logs and current time
        env.aosi_logs['time'].append(env.time)
        env.aosi_logs['aosi_logs'].append(delay_logs_per_user)

        env.job_generator.packet_df_ue.drop(accomplished_packets.index, inplace=True)
        env.logger.log_reward(f"Time step: {env.time} Total reward applied at this time step: {total_reward}.")

        return total_reward
    
    def compute_absolute_delay(cls, env, ue_packet: pd.Series) -> float:
        """Computes the delay between the latest accomplished sensor packet and the UE packet."""

        # Find all accomplished sensor packets
        accomplished_sensor_packets = env.job_generator.packet_df_sensor[
            (env.job_generator.packet_df_sensor['is_accomplished']) &
            (env.job_generator.packet_df_sensor['accomplished_time'].notnull())
        ].copy()

        if accomplished_sensor_packets.empty:
            return None

        # Find the latest accomplished_time
        latest_accomplished_time = accomplished_sensor_packets['accomplished_time'].max()

        # Filter packets with the highest accomplished_time
        latest_packets = accomplished_sensor_packets[
            accomplished_sensor_packets['accomplished_time'] == latest_accomplished_time
        ].copy()

        # If there are multiple packets with the same accomplished_time, choose the one with the highest creation_time
        latest_sensor_packet = latest_packets.loc[latest_packets['creation_time'].idxmax()]

        # Calculate the delay
        sensor_generating_time = latest_sensor_packet['creation_time']
        ue_generating_time = ue_packet['creation_time']
        delay = abs(ue_generating_time - sensor_generating_time)
        
        #env.logger.log_reward(f"Time step: {env.time} Positive delay for UE packet {ue_packet['packet_id']} from device {ue_packet['device_id']}: {delay}")

        return delay
    
    def compute_positive_delay(cls, env, ue_packet: pd.Series) -> float:
        """Computes the positive delay between the latest accomplished sensor packet
        (generated before the corresponding UE packet) and the UE packet."""

        # Find all accomplished sensor packets that have been completed
        accomplished_sensor_packets = env.job_generator.packet_df_sensor[
            (env.job_generator.packet_df_sensor['is_accomplished']) &
            (env.job_generator.packet_df_sensor['accomplished_time'].notnull())
        ]

        if accomplished_sensor_packets.empty:
            return None

        # Filter for sensor packets created before the UE packet creation time
        ue_generating_time = ue_packet['creation_time']
        valid_sensor_packets = accomplished_sensor_packets[
            accomplished_sensor_packets['creation_time'] <= ue_generating_time
        ]

        if valid_sensor_packets.empty:
            return None

        # If there are multiple packets with the same accomplished_time, choose the one with the highest creation_time
        latest_sensor_packet = valid_sensor_packets.loc[valid_sensor_packets['creation_time'].idxmax()]

        # Calculate the positive delay
        sensor_generating_time = latest_sensor_packet['creation_time']
        positive_delay = ue_generating_time - sensor_generating_time  # Positive delay since sensor is generated before UE packet

        #env.logger.log_reward(f"Time step: {env.time} Positive delay for UE packet {ue_packet['packet_id']} from device {ue_packet['device_id']}: {positive_delay}")
        
        return positive_delay
    
    @classmethod
    def aori_per_user(cls, env) -> None:
        """Computes the Age of Information (AoI) for UE packets."""

        # Filter for accomplished packets at the current timestep
        accomplished_packets = env.job_generator.packet_df_ue[
            (env.job_generator.packet_df_ue['is_accomplished']) &
            (env.job_generator.packet_df_ue['accomplished_time'] == env.time)
        ].copy()

        # Skip if no packets were accomplished at this timestep
        if accomplished_packets.empty:
            return None

        # Compute AoI (accomplished_time - creation_time)
        accomplished_packets['AoI'] = (
            accomplished_packets['accomplished_time'] - accomplished_packets['creation_time']
        )

        # Group AoI by user
        aoi_logs_per_user = accomplished_packets.groupby('device_id')['AoI'].mean().to_dict()

        # Log AoI values for each user
        for ue_id, aoi in aoi_logs_per_user.items():
            env.logger.log_reward(f"Time step: {env.time}, UE: {ue_id}, AoI: {aoi}")

        return aoi_logs_per_user

    @classmethod
    def aosi_per_user(cls, env) -> None:
        """Logs age of information (AoI) per user device at each timestep."""

        # Find all accomplished sensor packets
        accomplished_sensor_packets = env.job_generator.packet_df_sensor[
            (env.job_generator.packet_df_sensor['is_accomplished']) &
            (env.job_generator.packet_df_sensor['accomplished_time'].notnull())
        ].copy()

        if accomplished_sensor_packets.empty:
            return None

        # Find the latest accomplished_time
        latest_accomplished_time = accomplished_sensor_packets['accomplished_time'].max()

        # Filter packets with the highest accomplished_time
        latest_packets = accomplished_sensor_packets[
            accomplished_sensor_packets['accomplished_time'] == latest_accomplished_time
        ].copy()

        # If there are multiple packets with the same accomplished_time, choose the one with the highest creation_time
        latest_sensor_packet = latest_packets.loc[latest_packets['creation_time'].idxmax()]

        # Calculate the delay
        sensor_generating_time = latest_sensor_packet['creation_time']
        ue_generating_time = ue_packet['creation_time']
        delay = abs(ue_generating_time - sensor_generating_time)

        return aosi_logs_per_user
    
    @classmethod
    def check(cls, env) -> None:
        """Check if handler is applicable to simulation configuration."""
        assert all(
            ue.stime <= 0.0 and ue.extime >= env.EP_MAX_TIME
            for ue in env.users.values()
        ), "Central environment cannot handle a changing number of UEs."

    @classmethod
    def info(cls, env) -> Dict:
        """Compute information for feedback loop."""
        return {}
    
    @classmethod
    def info(cls, env) -> Dict:
        """Compute information for feedback loop."""
        return {
            "time": env.time,                                 # Current timestep
            "cumulative_reward": env.episode_reward,          # Total reward for the episode
            "num_users": len(env.users),                      # Number of users
            "num_sensors": len(env.sensors),                  # Number of sensors
            "mean_data_rate": np.mean(list(env.datarates.values())) if env.datarates else 0,
            "delayed_ue_jobs": env.delayed_ue_jobs,        
            "delayed_sensor_jobs": env.delayed_sensor_jobs,
        }
