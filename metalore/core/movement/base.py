"""
Base Movement class for MetaLore.

Abstract base class that defines the interface for movement patterns.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Movement(ABC):

    def __init__(
        self,
        width: float,
        height: float,
        seed: int = None,
        reset_rng_episode: bool = False,
        **kwargs
    ):
        self.width = width
        self.height = height

        # RNG for movement and initial positions
        self.seed = seed
        self.rng = None
        self.reset_rng_episode = reset_rng_episode

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

