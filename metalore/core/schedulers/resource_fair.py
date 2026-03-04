"""
Resource Fair Scheduler for MetaLore.

Allocates bandwidth evenly among all connected devices.
"""

from typing import List

from metalore.core.schedulers.base import Scheduler
from metalore.core.entities.base_station import BaseStation


class ResourceFair(Scheduler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def share(self, bs: BaseStation, conns: List, total_resources: float) -> List[float]:
        """Allocate bandwidth evenly among entities."""
        if not conns:
            return []
        return [total_resources / len(conns) for _ in conns]