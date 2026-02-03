"""
Base Scheduler class for MetaLore.

Abstract base class that defines the interface for resource scheduling.
"""

from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from metalore.core.entities.base_station import BaseStation


class Scheduler:

    def __init__(self, seed: int = None, **kwargs):
        self.seed = seed

    def reset(self) -> None:
        """Reset scheduler state."""
        pass

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

    def compute_rates(self, bs: BaseStation, entities: List, bandwidth: float, channel) -> Dict[Tuple, float]:
        """
        Compute data rates for entities connected to a base station.

        Combines bandwidth scheduling with Shannon capacity to produce rates.

        Args:
            bs: Base station
            entities: List of connected entities (UEs or sensors)
            bandwidth: Total bandwidth to allocate
            channel: Channel model for SNR computation

        Returns:
            Dict mapping (bs, entity) to data rate in Mbps
        """
        if not entities:
            return {}

        snrs = [channel.snr(bs, e) for e in entities]
        allocated_bw = self.share(bs, entities, bandwidth)
        rates = [bw * np.log2(1 + snr) / 1e6 for bw, snr in zip(allocated_bw, snrs)]

        return {(bs, e): rate for e, rate in zip(entities, rates)}