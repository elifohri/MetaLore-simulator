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
        return spaces.Box(low=0.0, high=1e4, shape=(size,), dtype=np.float32)

    @classmethod
    def action(cls, env, actions: Tuple[float, float]) -> Tuple[float, float]:
        """Transform action to expected shape of core environment."""
        assert len(actions) == 2, "Action must have two elements: bandwidth allocation and computational power allocation."

        bandwidth_allocation = np.clip(np.float32(actions[0]), 0.0, 1.0)
        computational_allocation = np.clip(np.float32(actions[1]), 0.0, 1.0)

        #env.logger.log_reward(f"Time step: {env.time} Action: {bandwidth_allocation:.3f}, {computational_allocation:.3f}")

        return bandwidth_allocation, computational_allocation
    
    @classmethod
    def get_queue_lengths(cls, env) -> np.ndarray:
        """Return queue lengths from the base station for transferred jobs and accomplished jobs.""" 
        queue_sizes = np.array([
            list(metrics.get_bs_transferred_ue_jobs_queue_size(env).values()),
            list(metrics.get_bs_transferred_sensor_jobs_queue_size(env).values()),
        ], dtype=np.float32).squeeze()

        if queue_sizes.shape != (2,):            
            raise ValueError(f"Unexpected queue sizes shape: {queue_sizes.shape}, expected (2,)")
        
        # Flatten the array
        return queue_sizes.ravel()

    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Compute observations for the RL agent."""
        obs = cls.get_queue_lengths(env)
        observation = np.clip(obs, env.observation_space.low, env.observation_space.high)

        #env.logger.log_reward(f"Time step: {env.time} Observation: {observation}")
        
        return observation
    
    @classmethod
    def reward(cls, env) -> float:
        """Process UE packets: apply penalties, rewards, and update the data frame."""
        # Initialize reward components
        synch_reward = 0        
        e2e_delay_penalty = 0  
        total_reward = 0

        # Get reward configuration
        config = env.default_config()["reward_calculation"]
        penalty = config["ue_penalty"]
        synch_base_reward = config["synch_base_reward"]
        discount_factor = config["discount_factor"]
        
        # Step 1: Synchronization Reward
        # Compute reward for synchronization of accomplished UE packets
        # Find all accomplished UE packets that is synchronized with the latest accomplished sensor packet
        valid_ue_packets = env.delay_manager.get_accomplished_ue_packets_with_synch_delay_and_null_synch_reward()

        # Compute reward for synchronization of packets
        if not valid_ue_packets.empty:
            synch_rewards = synch_base_reward * (discount_factor ** valid_ue_packets['synch_delay'])
            valid_ue_packets['synch_reward'] = synch_rewards
            synch_reward = synch_rewards.sum()
            total_reward += synch_reward
            
            # Update dataframe with new synchronization rewards
            env.job_dataframe.df_ue_packets.update(valid_ue_packets[['packet_id', 'synch_reward']])
            
        #env.logger.log_reward(f"Time step: {env.time} Synchronization reward: {synch_reward:.2f}.")

        # Step 2: Delay Penalty for UE Packets
        # Compute penalty for late transmission of accomplished UE packets
        # Find all accomplished UE packets at that timestep and cache
        accomplished_ue_packets = env.delay_manager.get_accomplished_ue_packets()

        # Compute penalty for end-to-end delay of packets
        if not accomplished_ue_packets.empty:
            delayed_packets = (accomplished_ue_packets['e2e_delay'] > accomplished_ue_packets['e2e_delay_threshold'])
            e2e_delay_penalty = delayed_packets.sum() * penalty
            total_reward += e2e_delay_penalty
            
        #env.logger.log_reward(f"Time step: {env.time} E2E Delay penalty: {e2e_delay_penalty}.")

        #env.logger.log_reward(f"Time step: {env.time} Total reward applied: {total_reward:.2f}.")

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
            "timestep reward": env.timestep_reward,
            "total reward": env.episode_reward,

            "total UE jobs generated": metrics.get_cumulative_ue_packets_generated(env),
            "total sensor jobs generated": metrics.get_cumulative_sensor_packets_generated(env),
            "total UE jobs served": env.total_episode_ue_packets_served,
            "total sensor jobs served": env.total_episode_sensor_packets_served,
            "total UE jobs delayed": env.total_episode_ue_packets_delayed,
            "total sensor jobs delayed": env.total_episode_sensor_packets_delayed,

            "cumulative transmission throughput ue": env.total_episode_transmission_throughput_ue,
            "cumulative transmission throughput sensor": env.total_episode_transmission_throughput_sensor,
            "cumulative processed data ue": env.total_episode_processed_data_ue,
            "cumulative processed data sensor": env.total_episode_processed_data_sensor,

            "total episode processed throughput ue": env.processed_throughput_ue,
            "total episode processed throughput sensor": env.processed_throughput_sensor,

            "avg e2e delay": metrics.get_e2e_delay(env),
            "avg synchronization delay": metrics.get_synchronization_delay(env),

            "bs trans. ue size": metrics.get_bs_transferred_ue_jobs_queue_size(env),
            "bs trans. ss size": metrics.get_bs_transferred_sensor_jobs_queue_size(env),
            "bs accomp. ue size": metrics.get_bs_accomplished_ue_jobs_queue_size(env),
            "bs accomp. ss size": metrics.get_bs_accomplished_sensor_jobs_queue_size(env),
        }