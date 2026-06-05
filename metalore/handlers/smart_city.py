"""
Smart City Handler for MetaLore Environments.

Defines the RL interface for smart city scenarios:
    - Action: [bandwidth_allocation, compute_allocation]
    - Observation: [ue_queue_length, sensor_queue_length]
    - Reward: Based on synchronization and delay penalties
"""

from typing import Dict, Tuple
import numpy as np
from gymnasium import spaces

from metalore.handlers.handler import Handler

class SmartCityHandler(Handler):

    @classmethod
    def action_space(cls, env) -> spaces.Space:
        """Define continuous action space for bandwidth allocation and computational power allocation"""
        # Action: [bandwidth_allocation, compute_allocation, alpha_isac[1], alpha_isac[2], ..., alpha_isac[n]]
        num_isacs = env.num_isacs
        dim = 2 + num_isacs

        low = np.zeros(dim, dtype=np.float32)
        high = np.ones(dim, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    @classmethod
    def observation_space(cls, env) -> spaces.Space:
        """Defines observation space."""
        # Observation: [ue_queue_length, sensor_queue_length, ISAC_queue_length]
        low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1e4, 1e4, 1e4], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
        """ low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1e5, 1e5, 1e4, 1e4], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32) """

    @classmethod
    def action(cls, env, actions) -> Tuple[float, float]:
        """Process agent action into environment action."""
        bw_split = float(np.clip(actions[0], 0.0, 1.0))
        comp_split = float(np.clip(actions[1], 0.0, 1.0))

        num_isacs = env.num_isacs
        if num_isacs > 0:
            alphas = {i: float(np.clip(actions[2+i], 0.0, 1.0)) for i in range(num_isacs)}
        else:
            alphas = None

        return bw_split, comp_split, alphas 

    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Computes observations for agent."""

        ue_pending = sum(bs.proc_queues['UE'].length for bs in env.stations.values())
        sensor_pending = sum(bs.proc_queues['SENSOR'].length for bs in env.stations.values())
        
        return np.array([float(sensor_pending), float(ue_pending)], dtype=np.float32)

    @classmethod
    def reward(cls, env) -> float:
        """Computes rewards for agent."""
        reward_cfg = env.config['reward']
        delay_threshold = reward_cfg['e2e_delay_threshold']
        delay_penalty   = reward_cfg['delay_penalty']
        sync_base_reward = reward_cfg['sync_base_reward']
        discount_factor  = reward_cfg['discount_factor']

        # UE jobs fully processed this timestep
        step_ue_jobs = [
            job for job in env.job_tracker.step_completed_jobs
            if job.job_type == 'UE'
        ]

        # Part 1: delay penalty — applied per job that exceeded the e2e threshold
        reward = sum(
            delay_penalty
            for job in step_ue_jobs
            if job.aori > delay_threshold
        )

        # Part 2: sync reward — discounted by how stale the sensor data was at job birth
        reward += sum(
            sync_base_reward * (discount_factor ** job.aosi)
            for job in step_ue_jobs
        )

        return float(reward)

    @classmethod
    def check(cls, env) -> None:
        """Check if handler is applicable to simulation configuration."""
        pass
    
    @classmethod
    def info(cls, env) -> Dict:
        """Compute information for feedback loop."""
        info_dict = {
            'time': env.time,
            'num_bs': env.num_bs,
            'num_ues': env.num_ues,
            'num_sensors': env.num_sensors,
            'num_isacs': env.num_isacs,
            'num_active_users': len(env.active_ues),
            'num_active_sensors': len(env.active_sensors),
            'num_active_isacs': len(env.active_isacs),
            'ue_datarates': {ue.id: rate for (_, ue), rate in env.datarates_ue.items()},
            'sensor_datarates': {s.id: rate for (_, s), rate in env.datarates_sensor.items()},
            'isac_datarates': {isac.id: rate for (_, isac), rate in env.datarates_isac.items()},
        }
        

        return info_dict
    