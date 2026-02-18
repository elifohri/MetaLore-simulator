"""
Base Arrival class for MetaLore.

Abstract base class that defines the interface for arrival/departure patterns of entities.
"""

from abc import abstractmethod
from typing import Dict

import numpy as np


class Arrival:

    def __init__(
        self,
        ep_time: int,
        seed: int,
        reset_rng_episode: bool,
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
    def arrival(self, entities: Dict) -> None:
        """Assign arrival times (stime) on each entity.

        Args:
            entities: Dictionary mapping entity id to entity object.
        """
        pass

    @abstractmethod
    def departure(self, entities: Dict) -> None:
        """Assign departure times (extime) on each entity.

        Args:
            entities: Dictionary mapping entity id to entity object.
        """
        pass
