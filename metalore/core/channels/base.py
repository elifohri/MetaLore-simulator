"""
Base Channel class for MetaLore.

Abstract base class that defines the interface for channel models.
"""

from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np

from metalore.core.entities.base_station import BaseStation

EPSILON = 1e-16


class Channel:

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
    def power_loss(self, bs: BaseStation, entity) -> float:
        """
        Calculate power loss for transmission between BS and a device.

        Args:
            bs: Base station
            entity: UE or sensor

        Returns:
            Power loss in dB
        """
        pass

    def snr(self, bs: BaseStation, entity) -> float:
        """
        Calculate SNR for transmission between BS and a device.

        Args:
            bs: Base station
            entity: UE or sensor

        Returns:
            Signal-to-noise ratio
        """
        loss = self.power_loss(bs, entity)              # Path loss in dB
        power = 10 ** ((bs.tx_power - loss) / 10)       # Convert dBm to Watts
        return power / entity.noise

    def check_connectivity(self, bs: BaseStation, entity) -> bool:
        """
        Check if entity-BS connection is valid based on SNR threshold.

        Args:
            bs: Base station
            entity: UE or sensor

        Returns:
            True if SNR exceeds entity's minimum threshold
        """
        return self.snr(bs, entity) > entity.snr_threshold

    def isoline(self, bs: BaseStation, entity_class, entity_config: Dict, map_bounds: Tuple[float, float], dthresh: float, num: int = 32) -> Tuple:
        """
        Compute isoline where devices receive at least dthresh max data rate.

        Args:
            bs: Base station
            entity_class: Entity class to use (UserEquipment or Sensor)
            entity_config: Configuration for dummy device
            map_bounds: (width, height) of the map
            dthresh: Data rate threshold
            num: Number of points on the isoline

        Returns:
            Tuple of (xs, ys) coordinates
        """
        width, height = map_bounds

        dummy = entity_class(**entity_config)

        isoline = []

        for theta in np.linspace(EPSILON, 2 * np.pi, num=num):
            # Calculate collision point with map boundary
            x1, y1 = self.boundary_collision(theta, bs.x, bs.y, width, height)

            # Points on line between BS and collision with map
            slope = (y1 - bs.y) / (x1 - bs.x + EPSILON)
            xs = np.linspace(bs.x, x1, num=100)
            ys = slope * (xs - bs.x) + bs.y

            # Compute data rate for each point
            def drate(point):
                dummy.position = point
                snr = self.snr(bs, dummy)
                return self.datarate(dummy, snr, bs.bandwidth)

            points = zip(xs.tolist(), ys.tolist())
            datarates = np.asarray(list(map(drate, points)))

            # Find largest x coordinate where drate is exceeded
            (idx,) = np.where(datarates > dthresh)
            if len(idx) > 0:
                idx = np.max(idx)
                isoline.append((xs[idx], ys[idx]))

        if not isoline:
            return [], []

        xs, ys = zip(*isoline)
        return xs, ys

    @classmethod
    def datarate(cls, entity, snr: float, bandwidth: float) -> float:
        """
        Calculate data rate using Shannon capacity formula.

        Args:
            entity: UE or sensor
            snr: Signal-to-noise ratio
            bandwidth: Allocated bandwidth in Hz

        Returns:
            Data rate in Mbps
        """
        if snr > entity.snr_threshold and bandwidth != 0:
            return bandwidth * np.log2(1 + snr) / 1e6         # Convert to Mbps
        return 0.0

    @classmethod
    def boundary_collision(cls, theta: float, x0: float, y0: float, width: float, height: float) -> Tuple[float, float]:
        """
        Find point on map boundaries with angle theta from position (x0, y0).

        Args:
            theta: Angle in radians
            x0, y0: Starting position
            width, height: Map dimensions

        Returns:
            (x, y) collision point on boundary
        """
        # Collision with right boundary
        rgt_x1, rgt_y1 = width, np.tan(theta) * (width - x0) + y0
        # Collision with upper boundary
        upr_x1, upr_y1 = ((-1) * np.tan(theta - 0.5 * np.pi) * (height - y0) + x0, height)
        # Collision with left boundary
        lft_x1, lft_y1 = 0.0, np.tan(theta) * (0.0 - x0) + y0
        # Collision with lower boundary
        lwr_x1, lwr_y1 = np.tan(theta - 0.5 * np.pi) * (y0 - 0.0) + x0, 0.0

        if theta == 0.0:
            return width, y0

        elif 0.0 < theta < 0.5 * np.pi:
            x1 = np.min((rgt_x1, upr_x1, width))
            y1 = np.min((rgt_y1, upr_y1, height))
            return x1, y1

        elif theta == 0.5 * np.pi:
            return x0, height

        elif 0.5 * np.pi < theta < np.pi:
            x1 = np.max((lft_x1, upr_x1, 0.0))
            y1 = np.min((lft_y1, upr_y1, height))
            return x1, y1

        elif theta == np.pi:
            return 0.0, y0

        elif np.pi < theta < 1.5 * np.pi:
            return np.max((lft_x1, lwr_x1, 0.0)), np.max((lft_y1, lwr_y1, 0.0))

        elif theta == 1.5 * np.pi:
            return x0, 0.0

        else:
            x1 = np.min((rgt_x1, lwr_x1, width))
            y1 = np.max((rgt_y1, lwr_y1, 0.0))
            return x1, y1