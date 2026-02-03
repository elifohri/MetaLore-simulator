"""
Okumura-Hata Channel Model for MetaLore.

Implements the Okumura-Hata propagation model for urban environments.
"""

import numpy as np

from metalore.core.channels.base import Channel, EPSILON
from metalore.core.entities.base_station import BaseStation


class OkumuraHata(Channel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def power_loss(self, bs: BaseStation, entity) -> float:

        # Distance in km
        distance = np.sqrt((bs.x - entity.x) ** 2 + (bs.y - entity.y) ** 2) / 1000

        # Correction factor for mobile antenna height
        ch = (0.8 + (1.1 * np.log10(bs.frequency) - 0.7) * entity.height - 1.56 * np.log10(bs.frequency))

        # Base path loss calculation
        tmp_1 = (69.55 - ch + 26.16 * np.log10(bs.frequency) - 13.82 * np.log10(bs.height))
        tmp_2 = 44.9 - 6.55 * np.log10(bs.height)

        # Add small epsilon to avoid log(0) if distance = 0
        path_loss = tmp_1 + tmp_2 * np.log10(distance + EPSILON)

        return path_loss