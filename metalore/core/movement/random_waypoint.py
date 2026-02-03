"""
Random Waypoint Movement Model for MetaLore.

Entities move towards randomly generated waypoints.
When a waypoint is reached, a new random waypoint is generated.
"""

from typing import Tuple, Dict
import numpy as np

from metalore.core.movement.base import Movement


class RandomWaypointMovement(Movement):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.waypoints: Dict[int, Tuple[float, float]] = None

    def reset(self) -> None:
        """Reset for new episode."""
        super().reset()
        self.waypoints = {}

    def move(self, entity) -> Tuple[float, float]:
        """Move entity one step towards a waypoint."""
        # Generate random waypoint if entity has none
        if entity.id not in self.waypoints:
            wx = self.rng.uniform(0, self.width)
            wy = self.rng.uniform(0, self.height)
            self.waypoints[entity.id] = (wx, wy)

        # Distance to waypoint
        position = np.array([entity.x, entity.y])
        waypoint = np.array(self.waypoints[entity.id])
        distance = np.linalg.norm(position - waypoint)

        # If close enough, move directly to waypoint
        if distance <= entity.velocity:
            # Remove waypoint after reaching it (new one generated next step)
            waypoint = self.waypoints.pop(entity.id)
            return waypoint

        # Move by velocity towards waypoint
        direction = waypoint - position
        direction = direction / np.linalg.norm(direction)
        new_position = position + direction * entity.velocity

        return tuple(new_position)