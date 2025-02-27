from typing import Dict, Tuple
import numpy as np
from gymnasium import spaces
import logging
from mobile_env.handlers.delay import DelayManager
from mobile_env.handlers.handler import Handler
from mobile_env.core import metrics


class MComSmartCityHandler(Handler):

    features = ["queue_lengths"]

    def __init__(self, env):
        self.env = env

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
        return spaces.Box(low=0.0, high=1000.0, shape=(size,), dtype=np.float32)

    @classmethod
    def action(cls, env, actions: Tuple[float, float]) -> Tuple[float, float]:
        """Transform action to expected shape of core environment."""
        assert len(actions) == 2, "Action must have two elements: bandwidth allocation and computational power allocation."

        bandwidth_allocation = np.clip(np.float32(actions[0]), 0.0, 1.0)
        computational_allocation = np.clip(np.float32(actions[1]), 0.0, 1.0)

        env.logger.log_reward(f"Time step: {env.time} Action: {bandwidth_allocation:.3f}, {computational_allocation:.3f}")

        return bandwidth_allocation, computational_allocation
    
    @classmethod
    def get_queue_lengths(cls, env) -> np.ndarray:
        """Return queue lengths from the base station for transferred jobs and accomplished jobs."""
        queue_sizes = np.array([
            list(metrics.get_bs_transferred_ue_queue_size(env).values()),
            list(metrics.get_bs_transferred_sensor_queue_size(env).values()),
            list(metrics.get_bs_accomplished_ue_queue_size(env).values()),
            list(metrics.get_bs_accomplished_sensor_queue_size(env).values())
        ], dtype=np.float32).squeeze()

        if queue_sizes.shape != (4,):
            raise ValueError(f"Unexpected queue sizes shape: {queue_sizes.shape}, expected (4,)")
        
        # Flatten the array
        return queue_sizes.ravel()

    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Compute observations for the RL agent."""
        obs = cls.get_queue_lengths(env)
        observation = np.clip(obs, env.observation_space.low, env.observation_space.high)

        env.logger.log_reward(f"Time step: {env.time} Observation: {observation}")
        
        return observation
    
    
    @classmethod
    def reward(cls, env) -> float:
        """Process UE packets: apply penalties, rewards, and update the data frame."""
        # Initialize reward components
        tx_reward = 0
        synch_reward = 0        
        total_penalty = 0  
        total_sensor_penalty = 0
        total_reward = 0

        # Get reward configuration
        config = env.default_config()["reward_calculation"]
        penalty = config["ue_penalty"]
        penalty_sensor = config["sensor_penalty"]
        base_reward = config["base_reward"]
        synch_base_reward = config["synch_base_reward"]
        discount_factor = config["discount_factor"]

        # Find all accomplished UE packets at that timestep and cache
        accomplished_ue_packets = env.delay_manager.get_accomplished_ue_packets()

        # Step 1: Transmission Reward
        # Compute reward for accomplished UE packets
        if not accomplished_ue_packets.empty:
            tx_reward = len(accomplished_ue_packets) * base_reward
            total_reward += tx_reward
            
        env.logger.log_reward(f"Time step: {env.time} Transmission reward: {tx_reward}.")
        
        
        # Step 2: Synchronization Reward
        # Compute reward for synchronization of accomplished UE packets
        # Find all accomplished UE packets that is synchronized with the latest accomplished sensor packet
        valid_ue_packets = env.delay_manager.get_accomplished_ue_packets_with_null_synch_reward()

        # Compute reward for synchronization of packets
        if not valid_ue_packets.empty:
            synch_rewards = synch_base_reward * (discount_factor ** valid_ue_packets['synch_delay'])
            valid_ue_packets['synch_reward'] = synch_rewards
            synch_reward = synch_rewards.sum()
            total_reward += synch_reward
            
            # Update dataframe with new synchronization rewards
            env.job_dataframe.df_ue_packets.update(valid_ue_packets[['packet_id', 'synch_reward']])
            
        env.logger.log_reward(f"Time step: {env.time} Synchronization reward: {synch_reward:.2f}.")


        # Step 3: Delay Penalty for UE Packets
        # Compute penalty for late transmission of accomplished UE packets
        if not accomplished_ue_packets.empty:
            delayed_packets = (accomplished_ue_packets['e2e_delay'] > accomplished_ue_packets['e2e_delay_threshold'])
            total_penalty = delayed_packets.sum() * penalty
            total_reward += total_penalty
            
        env.logger.log_reward(f"Time step: {env.time} Delayed transmission penalty: {total_penalty}.")


        # Step 4: Delay Penalty for Sensor Packets
        # Compute penalty for late transmission of accomplished sensor packets
        valid_sensor_packets = env.delay_manager.get_currrent_accomplished_sensor_packets()

        if not valid_sensor_packets.empty:
            delayed_sensor_packets = (valid_sensor_packets['e2e_delay'] > valid_sensor_packets['e2e_delay_threshold'])
            total_sensor_penalty = delayed_sensor_packets.sum() * penalty_sensor
            total_reward += total_sensor_penalty

        env.logger.log_reward(f"Time step: {env.time} Delayed transmission penalty for sensors: {total_sensor_penalty}.")

        env.logger.log_reward(f"Time step: {env.time} Total reward applied: {total_reward:.2f}.")

        return total_reward


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
        return {
            "time": env.time,
            "reward": metrics.get_reward(env),
            "delayed UE jobs": metrics.delayed_ue_packets(env),
            "aori": metrics.get_aori(env),
            "aosi": metrics.get_aosi(env),
            "bs trans. ue": metrics.get_bs_transferred_ue_queue_size(env),
            "bs trans. ss": metrics.get_bs_transferred_sensor_queue_size(env),
            "bs accomp. us": metrics.get_bs_accomplished_ue_queue_size(env),
            "bs accomp. ss": metrics.get_bs_accomplished_sensor_queue_size(env),
        }
