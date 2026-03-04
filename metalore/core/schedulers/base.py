"""
Base Scheduler class for MetaLore.

Abstract base class that defines the interface for resource scheduling.
"""

from abc import abstractmethod
from typing import List
import numpy as np

from metalore.core.entities.base_station import BaseStation


class Scheduler:

    def __init__(
        self, 
        seed: int,
        reset_rng_episode: bool,
        **kwargs
    ):
        self.reset_rng_episode = reset_rng_episode
        self.seed = seed
        self.rng = None

    def reset(self) -> None:
        """Reset state after episode ends."""
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def share(self, bs: BaseStation, conns: List, total_resources: float) -> List[float]:
        """
        Allocate resources among connected entities.

        Args:
            bs: Base station
            conns: List of connected entities
            total_resources: Total resources to allocate

        Returns:
            List of resource allocations per entity
        """
        pass