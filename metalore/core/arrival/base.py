"""
Base Arrival class for MetaLore.

Abstract base class that defines the interface for arrival/departure patterns of entities.
"""

from abc import abstractmethod
import numpy as np


class Arrival():

    def __init__(
        self,
        ep_time: int,
        seed: int = None,
        reset_rng_episode: bool = False,
        **kwargs
    ):
        self.ep_time = ep_time
        self.reset_rng_episode = reset_rng_episode
        self.seed = seed
        self.rng = None

    def reset(self) -> None:
        """Reset state after episode ends."""
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def arrival(self, entity) -> int:
        """
        Get arrival time for an entity.
        
        Args:
            entity: UE or Sensor
            
        Returns:
            Timestep when entity becomes active
        """
        pass

    @abstractmethod
    def departure(self, entity) -> int:
        """
        Get departure time for an entity.
        
        Args:
            entity: UE or Sensor
            
        Returns:
            Timestep when entity becomes inactive.
        """
        pass