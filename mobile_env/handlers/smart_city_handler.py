from typing import Dict, Tuple
import numpy as np
from gymnasium import spaces
import logging
from mobile_env.handlers.delay import DelayCalculator
from mobile_env.handlers.handler import Handler


class MComSmartCityHandler(Handler):

    features = ["queue_lengths"]

    def __init__(self, env, logger: logging.Logger,):
        self.env = env
        self.logger = logger  

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
        #env.logger.log_reward(f"Time step: {env.time} Observation Space size is: {size}")        
        return spaces.Box(low=0.0, high=np.inf, shape=(size,), dtype=np.float32)

    @classmethod
    def action(cls, env, actions: Tuple[float, float]) -> Tuple[float, float]:
        """Transform action to expected shape of core environment."""
        assert len(actions) == 2, "Action must have two elements: bandwidth allocation and computational power allocation."

        bandwidth_allocation, computational_allocation = actions
        bandwidth_allocation = max(0.0, min(1.0, bandwidth_allocation))
        computational_allocation = max(0.0, min(1.0, computational_allocation))

        env.logger.log_reward(f"Time step: {env.time} Action: {bandwidth_allocation}, {computational_allocation}")

        return bandwidth_allocation, computational_allocation
    
    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Compute observations for the RL agent."""
        
        queue_lengths = np.array(env.get_queue_lengths()).ravel()
        env.logger.log_reward(f"Time step: {env.time} Queue lengths: {queue_lengths}")    
        
        # Define maximum queue sizes for normalization
        max_queue_lengths = np.array([500, 2000, 500, 2000])

        # Normalize queue lengths
        normalized_queue_lengths = queue_lengths / max_queue_lengths  
        env.logger.log_reward(f"Time step: {env.time} Normalized queue lengths: {normalized_queue_lengths}")

        if normalized_queue_lengths.shape != (4,):
            raise ValueError(f"Unexpected shapes: queue_lengths {normalized_queue_lengths.shape}")

        observation = np.concatenate([
            normalized_queue_lengths
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

        accomplished_packets = env.metrics_logger.df_ue_packets[
            (env.metrics_logger.df_ue_packets['is_accomplished']) &
            (env.metrics_logger.df_ue_packets['accomplished_time'] == env.time)
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
                lambda row: DelayCalculator.compute_absolute_delay(env, row), axis=1
            )

            valid_ue_packets['reward'] = base_reward * (discount_factor ** valid_ue_packets['computed_delay'])
            total_reward += valid_ue_packets['reward'].sum()

        total_reward += penalties.sum()

        # Group delays by UE
        delay_logs_per_user = accomplished_packets.groupby('device_id')['delay'].mean().to_dict()

        # Append delay logs and current time
        env.aosi_logs['time'].append(env.time)
        env.aosi_logs['aosi_logs'].append(delay_logs_per_user)

        env.metrics_logger.df_ue_packets.drop(accomplished_packets.index, inplace=True)
        env.logger.log_reward(f"Time step: {env.time} Total reward applied at this time step: {total_reward}.")

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
        return {}
    
    @classmethod
    def info(cls, env) -> Dict:
        """Compute information for feedback loop."""
        return {
            "time": env.time,
            "cumulative_reward": env.episode_reward,
            "num_users": len(env.users),
            "num_sensors": len(env.sensors),
            "mean_data_rate": np.mean(list(env.datarates.values())) if env.datarates else 0,
            "delayed_ue_jobs": env.delayed_ue_jobs,        
            "delayed_sensor_jobs": env.delayed_sensor_jobs,
        }
