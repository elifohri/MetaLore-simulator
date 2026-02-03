"""
Base Handler for MetaLore Environments.

Handlers define how the RL agent interacts with the environment.
"""

from abc import abstractmethod
from typing import Any, Dict
import numpy as np
from gymnasium.spaces import Space

class Handler:

    @classmethod
    @abstractmethod
    def action_space(cls, env) -> Space:
        """Defines action space for passed environment."""
        pass

    @classmethod
    @abstractmethod
    def observation_space(cls, env) -> Space:
        """Defines observation space for passed environment."""
        pass

    @classmethod
    @abstractmethod
    def action(cls, env, action) -> Any:
        """Process agent action into environment action."""
        pass

    @classmethod
    @abstractmethod
    def observation(cls, env) -> np.ndarray:
        """Computes observations for agent."""
        pass

    @classmethod
    @abstractmethod
    def reward(cls, env) -> float:
        """Computes rewards for agent."""
        pass

    @classmethod
    def check(cls, env) -> None:
        """Check if handler is applicable to simulation configuration."""
        pass

    @classmethod
    def info(cls, env) -> Dict:
        """Compute information for feedback loop."""
        return {}