from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
import pandas as pd

from mobile_env.handlers.handler import Handler


class MComSmartCityHandler(Handler):

    features = ["queue_lengths", "resource_utilization"]

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

        return bandwidth_allocation, computational_allocation
    
    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Compute system-wide observations for the RL agent."""
        
        # Gather the queue lengths (from base station)
        queue_lengths = np.array(env.get_queue_lengths()).ravel()
        #env.logger.log_reward(f"Time step: {env.time} Queue lengths: {queue_lengths}")      

        # Get resource utilization (bandwidth and CPU)
        resource_utilization = np.array(env.get_resource_utilization()).ravel()
        #env.logger.log_reward(f"Time step: {env.time} Resource utilization: {resource_utilization}")   
        
        if queue_lengths.shape != (4,) or resource_utilization.shape != (2,):
            raise ValueError(f"Unexpected shapes: queue_lengths {queue_lengths.shape}, resource_utilization {resource_utilization.shape}")

        # Concatenate all observations into a single array
        observation = np.concatenate([
            queue_lengths,              # 4 values
            resource_utilization        # 2 values
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

        # List of packets that exceeded the delay constraint or were accomplished
        indices_to_remove_ue_jobs = []

        for index, row in env.job_generator.packet_df_ue.iterrows():
            # Check if packet is accomplished at this timestep
            if row['is_accomplished'] and row['accomplished_time'] == env.time:
                # Packet is accomplished in this time step
                dt = env.time - row['creation_time']
                
                if dt > row['e2e_delay_threshold']:
                    # Packet exceeds threshold, penalty applied
                    total_reward += penalty
                    #env.logger.log_reward(f"Time step: {env.time} Packet {row['packet_id']} from UE {row['device_id']} failed due to delay. Penalty applied: {penalty}.")
                else:
                    # Packet succeeded within the time threshold, compute delay and reward
                    delay = cls.compute_delay(cls, env, row)
                    
                    # Skip reward computation if no accomplished sensor packets are found
                    if delay is None:
                        #env.logger.log_reward(f"Time step: {env.time} No accomplished sensor packets found for Packet {row['packet_id']} from UE {row['device_id']}. Skipping reward calculation.")
                        continue
                    
                    reward = base_reward * (discount_factor ** delay)        
                    total_reward += reward
                    #env.logger.log_reward(f"Time step: {env.time} Packet {row['packet_id']} from UE {row['device_id']} succeeded within time threshold. Reward applied: {reward}.")

                # Mark as processed and remove from the data frame
                indices_to_remove_ue_jobs.append(index)

        # Drop processed UE packets
        if indices_to_remove_ue_jobs:
            env.job_generator.packet_df_ue.drop(indices_to_remove_ue_jobs, inplace=True)

        env.logger.log_reward(f"Time step: {env.time} Total reward applied at this time step: {total_reward}.")
        
        return total_reward
    
    def compute_delay(cls, env, ue_packet: pd.Series) -> float:
        """Computes the delay between the latest accomplished sensor packet and the UE packet."""

        # Find all accomplished sensor packets
        accomplished_sensor_packets = env.job_generator.packet_df_sensor[
            (env.job_generator.packet_df_sensor['is_accomplished']) &
            (env.job_generator.packet_df_sensor['accomplished_time'].notnull())
        ]

        if accomplished_sensor_packets.empty:
            #env.logger.log_reward(f"Time step: {env.time} No accomplished sensor packets found.")
            return None

        # Find the latest accomplished_time
        latest_accomplished_time = accomplished_sensor_packets['accomplished_time'].max()

        # Filter packets with the highest accomplished_time
        latest_packets = accomplished_sensor_packets[
            accomplished_sensor_packets['accomplished_time'] == latest_accomplished_time
        ]

        # If there are multiple packets with the same accomplished_time, choose the one with the highest creation_time
        latest_sensor_packet = latest_packets.loc[latest_packets['creation_time'].idxmax()]

        # Calculate the delay
        sensor_generating_time = latest_sensor_packet['creation_time']
        ue_generating_time = ue_packet['creation_time']
        delay = abs(ue_generating_time - sensor_generating_time)
        
        #env.logger.log_reward(f"Time step: {env.time} Positive delay for UE packet {ue_packet['packet_id']} from device {ue_packet['device_id']}: {delay}")

        return delay
    
    
    def compute_positive_delay(cls, env, ue_packet: pd.Series) -> float:
        """
        Computes the positive delay between the latest accomplished sensor packet
        (generated before the corresponding UE packet) and the UE packet. Only
        sensor data generated before the UE packet is considered.
        """

        # Find all accomplished sensor packets that have been completed
        accomplished_sensor_packets = env.job_generator.packet_df_sensor[
            (env.job_generator.packet_df_sensor['is_accomplished']) &
            (env.job_generator.packet_df_sensor['accomplished_time'].notnull())
        ]

        if accomplished_sensor_packets.empty:
            #env.logger.log_reward(f"Time step: {env.time} No accomplished sensor packets found.")
            return None

        # Filter for sensor packets created before the UE packet creation time
        ue_generating_time = ue_packet['creation_time']
        valid_sensor_packets = accomplished_sensor_packets[
            accomplished_sensor_packets['creation_time'] <= ue_generating_time
        ]

        if valid_sensor_packets.empty:
            #env.logger.log_reward(f"Time step: {env.time} No sensor packets generated before UE packet {ue_packet['packet_id']}.")
            return None

        # If there are multiple packets with the same accomplished_time, choose the one with the highest creation_time
        latest_sensor_packet = valid_sensor_packets.loc[valid_sensor_packets['creation_time'].idxmax()]

        # Calculate the positive delay
        sensor_generating_time = latest_sensor_packet['creation_time']
        positive_delay = ue_generating_time - sensor_generating_time  # Positive delay since sensor is generated before UE packet

        #env.logger.log_reward(f"Time step: {env.time} Positive delay for UE packet {ue_packet['packet_id']} from device {ue_packet['device_id']}: {positive_delay}")
        
        return positive_delay
    
    @classmethod
    def aoi_per_user(cls, env) -> None:
        """Logs age of information per user device at each timestep."""
        aoi_logs_per_user = {}
        
        for _, row in env.job_generator.packet_df_ue.iterrows():
            if row['is_accomplished'] and row['accomplished_time'] == env.time:
                aoi = cls.compute_delay(cls, env, row)
                
                if aoi is not None:
                    device_id = row['device_id']
                    aoi_logs_per_user[device_id] = aoi_logs_per_user.get(device_id, 0) + aoi

        # If a device has no packets at this timestep, set age of information to zero
        for device_id in env.users.keys():
            if device_id not in aoi_logs_per_user:
                aoi_logs_per_user[device_id] = 0
            
        # Log the delays
        #env.logger.log_reward(f"Time step: {env.time} Delays per device: {aoi_logs_per_user}")
        
        return aoi_logs_per_user

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