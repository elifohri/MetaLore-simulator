"""
Dynamic Arrival Pattern for MetaLore.

Assigns UE arrival and departure times based on a fixed traffic profile.
Profiles are defined in metalore/config/profiles.py and selected at config creation time.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from metalore.core.arrival.base import Arrival


@dataclass
class TrafficProfile:
    """All parameters that define one traffic profile."""
    pool_size:        int
    job_data_size:    int
    job_compute_size: int
    job_gen_prob:     float
    arrival_window:   tuple          # (start, end) as fractions of episode length
    mean_sojourn:     float          # mean sojourn in timesteps (exponential)
    arrival_type:     str  = 'uniform'  # 'uniform' or 'gaussian'
    burst_fraction:   float = 0.0   # fraction of pool in the Gaussian sub-group
    burst_center:     float = 0.5   # centre of Gaussian arrivals as fraction of T
    burst_std:        float = 0.08  # spread of Gaussian arrivals as fraction of T
    burst_sojourn:    float = 0.0   # mean sojourn for the Gaussian sub-group


class DynamicArrival(Arrival):
    """Assigns UE arrivals and departures according to a fixed traffic profile."""

    def __init__(
        self,
        profiles: Dict[str, TrafficProfile],
        profile:  str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.profiles        = profiles
        self.current_profile = profile

    def reset(self) -> None:
        super().reset()

    # ── Arrival / Departure ───────────────────────────────────────────────

    def arrival(self, entities: Dict) -> None:
        """Assign stime to each UE based on the current profile's arrival pattern."""
        p   = self.profiles[self.current_profile]
        ues = list(entities.values())
        if p.arrival_type == 'gaussian':
            self._gaussian_arrival(ues, p)
        else:
            self._uniform_arrival(ues, p)

    def departure(self, entities: Dict) -> None:
        """Assign extime to each UE by drawing a sojourn from the current profile."""
        T   = self.ep_max_time
        p   = self.profiles[self.current_profile]
        ues = list(entities.values())

        if p.arrival_type == 'gaussian':
            n_burst = max(1, int(len(ues) * p.burst_fraction))
            for ue in ues[:n_burst]:
                ue.extime = min(T, ue.stime + max(1, int(self.rng.exponential(p.burst_sojourn))))
            for ue in ues[n_burst:]:
                ue.extime = min(T, ue.stime + max(1, int(self.rng.exponential(p.mean_sojourn))))
        else:
            for ue in ues:
                ue.extime = min(T, ue.stime + max(1, int(self.rng.exponential(p.mean_sojourn))))


    @classmethod
    def with_profile(cls, profile: str, profiles: Dict[str, TrafficProfile]) -> type:
        """Return a subclass with the given profile fixed for every episode."""
        _profiles = profiles
        class _Fixed(cls):
            def __init__(self, **kwargs):
                super().__init__(profile=profile, profiles=_profiles, **kwargs)
        _Fixed.__name__ = f"DynamicArrival[{profile}]"
        return _Fixed

    # ── Private helpers ───────────────────────────────────────────────────

    def _uniform_arrival(self, ues: List, p: TrafficProfile) -> None:
        T = self.ep_max_time
        w_start, w_end = p.arrival_window
        for ue in ues:
            ue.stime = int(self.rng.uniform(T * w_start, T * w_end))

    def _gaussian_arrival(self, ues: List, p: TrafficProfile) -> None:
        T       = self.ep_max_time
        n_burst = max(1, int(len(ues) * p.burst_fraction))
        center  = int(T * p.burst_center)
        std     = T * p.burst_std
        w_start, w_end = p.arrival_window
        for ue in ues[:n_burst]:
            ue.stime = int(np.clip(self.rng.normal(center, std), 0, T - 1))
        for ue in ues[n_burst:]:
            ue.stime = int(self.rng.uniform(T * w_start, T * w_end))
