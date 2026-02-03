"""
Base Handler for MetaLore Environments.

Handlers define how the RL agent interacts with the environment:
"""

from abc import ABC, abstractmethod
from typing import Any
from gymnasium.spaces import Space

class Handler(ABC):

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
    def observation(cls, env):
        """Computes observations for agent."""
        pass

    @classmethod
    @abstractmethod
    def reward(cls, env):
        """Computes rewards for agent."""
        pass

    @classmethod
    def check(cls, env):
        """Check if handler is applicable to simulation configuration."""
        pass

    @classmethod
    def info(cls, env):
        """Compute information for feedback loop."""
        return {}