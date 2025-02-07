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
        return spaces.Box(low=0.0, high=2000.0, shape=(size,), dtype=np.float32)

    @classmethod
    def action(cls, env, actions: Tuple[float, float]) -> Tuple[float, float]:
        """Transform action to expected shape of core environment."""
        assert len(actions) == 2, "Action must have two elements: bandwidth allocation and computational power allocation."

        bandwidth_allocation, computational_allocation = actions
        bandwidth_allocation = max(0.0, min(1.0, bandwidth_allocation))
        computational_allocation = max(0.0, min(1.0, computational_allocation))

        env.logger.log_reward(f"Time step: {env.time} Action: {bandwidth_allocation:.3f}, {computational_allocation:.3f}")

        return bandwidth_allocation, computational_allocation
    
    @classmethod
    def get_queue_lengths(cls, env) -> np.ndarray:
        """Return queue lengths from the base station for transferred jobs and accomplished jobs."""
        transferred_ue_queue_size = np.array(list(metrics.get_bs_transferred_ue_queue_size(env).values()))
        transferred_sensor_queue_size = np.array(list(metrics.get_bs_transferred_sensor_queue_size(env).values()))
        accomplished_ue_queue_size = np.array(list(metrics.get_bs_accomplished_ue_queue_size(env).values()))
        accomplished_sensor_queue_size = np.array(list(metrics.get_bs_accomplished_sensor_queue_size(env).values()))

        queue_lengths = np.concatenate([transferred_ue_queue_size, transferred_sensor_queue_size,
                                        accomplished_ue_queue_size, accomplished_sensor_queue_size]).astype(np.float32)

        return queue_lengths

    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Compute observations for the RL agent."""
        obs = np.array(cls.get_queue_lengths(env)).ravel()
        if obs.shape != (4,):
            raise ValueError(f"Unexpected shapes: queue_lengths {obs.shape}")
        
        observation = np.clip(obs, env.observation_space.low, env.observation_space.high)

        env.logger.log_reward(f"Time step: {env.time} Observation: {observation}")
        
        return observation
    
    
    @classmethod
    def reward(cls, env) -> float:
        """Process UE packets: apply penalties, rewards, and update the data frame."""
        total_reward = 0
        penalties = 0
        config = env.default_config()["reward_calculation"]
        penalty = config["ue_penalty"]
        base_reward = config["base_reward"]
        discount_factor = config["discount_factor"]

        # Find all accomplished UE packets at that timestep
        accomplished_ue_packets = env.delay_manager.get_accomplished_ue_packets()

        if accomplished_ue_packets.empty:
            env.logger.log_reward(f"Time step: {env.time} There are no accomplished UE packets.")
            return total_reward
                
        # Compute penalty for packets that have delayed the threshold
        penalties = (accomplished_ue_packets['e2e_delay'] > accomplished_ue_packets['e2e_delay_threshold']) * penalty
        total_reward += penalties.sum()

        env.logger.log_reward(f"Time step: {env.time} Total penalty applied: {penalties.sum():.3f}.")

        valid_ue_packets = accomplished_ue_packets[accomplished_ue_packets['e2e_delay'] <= accomplished_ue_packets['e2e_delay_threshold']].copy()

        # Compute reward for packets that haven't delayed the threshold
        if not valid_ue_packets.empty:
            valid_ue_packets['reward'] = base_reward * (discount_factor ** valid_ue_packets['synch_delay'])
            total_reward += valid_ue_packets['reward'].sum()

        env.logger.log_reward(f"Time step: {env.time} Total reward applied: {total_reward:.3f}.")

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
            "aori": metrics.compute_aori(env),
            "aosi": metrics.compute_aosi(env),
            "bs trans. ue": metrics.get_bs_transferred_ue_queue_size(env),
            "bs trans. ss": metrics.get_bs_transferred_sensor_queue_size(env),
            "bs accomp. us": metrics.get_bs_accomplished_ue_queue_size(env),
            "bs accomp. ss": metrics.get_bs_accomplished_sensor_queue_size(env),
        }
