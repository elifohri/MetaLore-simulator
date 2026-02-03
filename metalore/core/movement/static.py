"""
Static Movement Model for MetaLore.

Entities stay in their initial position and don't move.
"""

from typing import Tuple

from metalore.core.movement.base import Movement


class StaticMovement(Movement):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self) -> None:
        """Reset for new episode."""
        super().reset()

    def move(self, entity) -> Tuple[float, float]:
        """Return current position (no movement)."""
        return entity.x, entity.y