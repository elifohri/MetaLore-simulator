"""
Base Movement class for MetaLore.

Abstract base class that defines the interface for movement patterns.
"""

from abc import abstractmethod
from typing import Tuple
import numpy as np


class Movement:

    def __init__(
        self,
        width: float,
        height: float,
        seed: int,
        reset_rng_episode: bool,
        **kwargs
    ):
        self.width = width
        self.height = height

        self.reset_rng_episode = reset_rng_episode
        self.seed = seed
        self.rng = None

    def reset(self) -> None:
        """Reset state of movement object after episode ends."""
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def move(self, entity) -> Tuple[float, float]:
        """
        Move entity one timestep.

        Args:
            entity: Entity to move (UE or sensor).

        Returns:
            New (x, y) position of the entity.
        """
        pass

