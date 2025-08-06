from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np

from mobile_env.core.entities import Sensor

class SensorPlacement:
    def __init__(
        self, width: float, height: float, seed: int, reset_rng_episode: str, **kwargs
    ):
        self.width, self.height = width, height
        self.reset_rng_episode = reset_rng_episode
        self.seed = seed
        self.rng = None

        self.initial: Dict[Sensor, Tuple[float, float]] = None
        self.max_distance = kwargs.get("max_distance", 90)
        self.min_distance = kwargs.get("min_distance", 20)
        self.station_positions = [(100, 100)]

    def reset(self):
        """Reset state of sensor object after episode ends."""
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        self.initial = {}

    def initial_position(self, sensor: Sensor) -> Tuple[float, float]:
        """Return valid initial position of a sensor, satisfying placement constraints."""
        if sensor not in self.initial:
            while True:
                x = self.rng.uniform(0, self.width)
                y = self.rng.uniform(0, self.height)
                existing_positions = list(self.initial.values())

                if self.is_valid_position(x, y, self.station_positions, existing_positions, self.max_distance, self.min_distance):
                    self.initial[sensor] = (x, y)
                    break

        sensor.x, sensor.y = self.initial[sensor]
        return sensor.x, sensor.y

    def is_valid_position(self, x, y, station_positions, existing_sensor_positions, max_distance, min_distance):
        return (
            self.is_within_max_distance_from_stations(x, y, station_positions, max_distance)
            and self.is_far_enough_from_others(x, y, existing_sensor_positions, min_distance)
            and self.is_far_enough_from_stations(x, y, station_positions, min_distance)
        )

    def is_within_max_distance_from_stations(self, x, y, station_positions, max_dist):
        return any((x - sx) ** 2 + (y - sy) ** 2 <= max_dist ** 2 for sx, sy in station_positions)

    def is_far_enough_from_others(self, x, y, existing_sensor_positions, min_dist):
        return all((x - ox) ** 2 + (y - oy) ** 2 >= min_dist ** 2 for ox, oy in existing_sensor_positions)

    def is_far_enough_from_stations(self, x, y, station_positions, min_dist):
        return all((x - sx) ** 2 + (y - sy) ** 2 >= min_dist ** 2 for sx, sy in station_positions)
