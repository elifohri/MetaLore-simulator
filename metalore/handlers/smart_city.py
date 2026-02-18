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

    features = ["queue_lengths"]

    @classmethod
    def action_space(cls, env) -> spaces.Space:
        """Define continuous action space for bandwidth allocation and computational power allocation"""
        # Action: [bandwidth_allocation, compute_allocation]
        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    @classmethod
    def observation_space(cls, env) -> spaces.Space:
        """Defines observation space."""
        # Observation: [ue_queue_length, sensor_queue_length]
        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([1e4, 1e4], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @classmethod
    def action(cls, env, actions) -> Tuple[float, float]:
        """Process agent action into environment action. Returns (bandwidth_allocation, compute_allocation)."""
        return (
            float(np.clip(actions[0], 0.0, 1.0)),
            float(np.clip(actions[1], 0.0, 1.0)),
        )

    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Computes observations for agent."""
        return np.array([0.0, 0.0], dtype=np.float32)

    @classmethod
    def reward(cls, env) -> float:
        """Computes rewards for agent."""
        return 1.0  

    @classmethod
    def check(cls, env) -> None:
        """Check if handler is applicable to simulation configuration."""
        pass
    
    @classmethod
    def info(cls, env) -> Dict:
        """Compute information for feedback loop."""
        ue_rates = {ue.id: rate for (_, ue), rate in env.datarates_ue.items()}
        sensor_rates = {s.id: rate for (_, s), rate in env.datarates_sensor.items()}

        return {
            'time': env.time,
            'num_bs': env.num_bs,
            'num_ues': env.num_ues,
            'num_sensors': env.num_sensors,
            'num_active_users': len(env.active_ues),
            'num_active_sensors': len(env.active_sensors),
            'ue_datarates': ue_rates,
            'sensor_datarates': sensor_rates,
        }
    