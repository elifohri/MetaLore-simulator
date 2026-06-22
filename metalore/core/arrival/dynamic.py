"""
Dynamic Arrival Pattern for MetaLore.

Each episode randomly selects one of several traffic profiles,
creating non-stationary load conditions for RL training.

Profiles and approximate concurrent UE counts (pool = 15, episode = 100 steps):
    low       ~4–6  concurrent — sparse arrivals, short sojourn (~30 steps)
    medium    ~9    concurrent — moderate load, sojourn fills most of episode
    high      ~12   concurrent — early arrivals, long sojourn (~80 steps)
    ramp_up         0 → 12   — arrivals spread across second two-thirds
    ramp_down       12 → 3   — early arrivals drain away by episode end
    burst     ~5 + spike     — quiet background + Gaussian surge at midpoint

All parameters can be overridden per instance via __init__ kwargs,
or per subclass by overriding the class-level defaults.
"""

from typing import Dict, List, Optional

import numpy as np

from metalore.core.arrival.base import Arrival


class DynamicArrival(Arrival):
    """
    Randomly selects a traffic profile at each episode reset.

    arrival()   — assigns stime to each UE based on the profile's arrival window.
    departure() — assigns extime to each UE by drawing a sojourn from an
                  exponential distribution parameterised per profile.

    Parameters (all optional, fall back to class-level defaults)
    -----------------------------------------------------------
    profiles        : list of profile names to sample from
    profile_weights : sampling weights aligned with profiles (None = uniform)
    arrival_window  : dict mapping profile → (start, end) as fractions of T
    mean_sojourn    : dict mapping profile → mean sojourn in timesteps
    burst_fraction  : fraction of UEs assigned to the burst group (default 0.5)
    burst_center    : centre of burst arrivals as fraction of T (default 0.5)
    burst_std       : std of burst arrivals as fraction of T (default 0.08)
    burst_sojourn   : mean sojourn for burst-group UEs in timesteps (default 50)
    """

    # ── Class-level defaults ──────────────────────────────────────────────

    PROFILES: List[str] = ['low', 'medium', 'high', 'ramp_up', 'ramp_down', 'burst']

    # Number of UEs in the pool per profile.
    # make_env reads this to set cfg['environment']['num_ues'] before building the env.
    POOL_SIZE: Dict[str, int] = {
        'low':       20,
        'medium':    30,
        'high':      40,
        'ramp_up':   30,
        'ramp_down': 30,
        'burst':     30,
    }

    # Per-profile UE job characteristics.
    # Profiles differ in traffic type, not just load level, so each has a distinct
    # bandwidth-to-compute ratio that shifts the optimal (bw_split, comp_split).
    JOB_DATA_SIZE: Dict[str, int] = {
        'low':         300_000,   # 300 Kbits  — light IoT, small payloads
        'medium':      500_000,   # 500 Kbits  — balanced (default)
        'high':        500_000,   # 500 Kbits  — same data but compute-heavy
        'ramp_up':   1_000_000,   # 1 Mbit     — large uploads, bandwidth-heavy
        'ramp_down':   600_000,   # 600 Kbits  — moderate
        'burst':       300_000,   # 300 Kbits  — small data, compute-intensive
    }

    JOB_COMPUTE_SIZE: Dict[str, int] = {
        'low':        50_000_000,   #  50 Mcycles — light compute
        'medium':    100_000_000,   # 100 Mcycles — balanced (default)
        'high':      200_000_000,   # 200 Mcycles — heavy MEC offload
        'ramp_up':    80_000_000,   #  80 Mcycles — bandwidth bottleneck, not compute
        'ramp_down': 120_000_000,   # 120 Mcycles — moderate compute
        'burst':     250_000_000,   # 250 Mcycles — inference-type, very compute-heavy
    }

    JOB_GEN_PROB: Dict[str, float] = {
        'low':       0.5,   # sparse generation
        'medium':    0.7,   # default
        'high':      0.9,   # dense — almost every step a job is generated
        'ramp_up':   0.7,
        'ramp_down': 0.8,
        'burst':     0.8,
    }

    # Arrival window per profile: (start, end) as fractions of episode length.
    # UE stime is drawn uniformly from [T*start, T*end).
    ARRIVAL_WINDOW: Dict[str, tuple] = {
        'low':       (0.0, 0.7),
        'medium':    (0.0, 0.4),
        'high':      (0.0, 0.15),
        'ramp_up':   (0.3, 0.9),
        'ramp_down': (0.0, 0.15),
        'burst':     (0.0, 0.8), 
    }

    # Mean sojourn duration (exponential scale) per profile, in timesteps.
    MEAN_SOJOURN: Dict[str, float] = {
        'low':        30.0,
        'medium':     60.0,
        'high':       80.0,
        'ramp_up':    70.0,
        'ramp_down':  40.0,
        'burst':      40.0, 
    }

    # Burst profile sub-group defaults
    BURST_FRACTION: float = 0.6   # fraction of UE pool in the burst sub-group
    BURST_CENTER:   float = 0.5   # arrival centre as fraction of T
    BURST_STD:      float = 0.08  # arrival spread as fraction of T
    BURST_SOJOURN:  float = 40.0  # mean sojourn for burst sub-group (timesteps)

    # ── Constructor ───────────────────────────────────────────────────────

    def __init__(
        self,
        profiles:        Optional[List[str]]         = None,
        profile_weights: Optional[List[float]]       = None,
        pool_size:       Optional[Dict[str, int]]    = None,
        arrival_window:  Optional[Dict[str, tuple]]  = None,
        mean_sojourn:    Optional[Dict[str, float]]  = None,
        burst_fraction:  Optional[float]             = None,
        burst_center:    Optional[float]             = None,
        burst_std:       Optional[float]             = None,
        burst_sojourn:   Optional[float]             = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.profiles        = list(profiles       or self.PROFILES)
        self.profile_weights = list(profile_weights) if profile_weights is not None else None
        self.pool_size       = dict(pool_size      or self.POOL_SIZE)
        self.arrival_window  = dict(arrival_window or self.ARRIVAL_WINDOW)
        self.mean_sojourn    = dict(mean_sojourn   or self.MEAN_SOJOURN)
        self.burst_fraction  = burst_fraction if burst_fraction is not None else self.BURST_FRACTION
        self.burst_center    = burst_center   if burst_center   is not None else self.BURST_CENTER
        self.burst_std       = burst_std      if burst_std      is not None else self.BURST_STD
        self.burst_sojourn   = burst_sojourn  if burst_sojourn  is not None else self.BURST_SOJOURN

        self.current_profile: Optional[str] = None
        self._burst_ue_ids: set = set()   # populated in arrival(), read in departure()

    # ── Arrival / Departure ───────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        weights = None
        if self.profile_weights is not None:
            w = np.array(self.profile_weights, dtype=float)
            weights = w / w.sum()
        self.current_profile = str(self.rng.choice(self.profiles, p=weights))
        self._burst_ue_ids = set()

    def arrival(self, entities: Dict) -> None:
        """Assign stime to each UE."""
        T       = self.ep_max_time
        ues     = list(entities.values())
        profile = self.current_profile

        if profile == 'burst':
            n_burst = max(1, int(len(ues) * self.burst_fraction))
            w_start, w_end = self.arrival_window['burst']
            center = int(T * self.burst_center)
            std    = T * self.burst_std

            for ue in ues[n_burst:]:
                ue.stime = int(self.rng.uniform(T * w_start, T * w_end))

            for ue in ues[:n_burst]:                        # burst sub-group
                ue.stime = int(np.clip(self.rng.normal(center, std), 0, T - 1))
                self._burst_ue_ids.add(ue.id)
        else:
            w_start, w_end = self.arrival_window[profile]
            for ue in ues:
                ue.stime = int(self.rng.uniform(T * w_start, T * w_end))

    def departure(self, entities: Dict) -> None:
        """Assign extime to each UE."""
        T       = self.ep_max_time
        profile = self.current_profile

        for ue in entities.values():
            if profile == 'burst' and ue.id in self._burst_ue_ids:
                mean = self.burst_sojourn
            else:
                mean = self.mean_sojourn[profile]
            sojourn   = max(1, int(self.rng.exponential(mean)))
            ue.extime = min(T, ue.stime + sojourn)
